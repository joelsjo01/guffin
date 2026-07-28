#!/usr/bin/env python3
"""Run product_conversation_v1 against a verified pinned llama-server.

The comparator is standard-library-only development tooling. It never enters
the local engine's production inference path. When paired local JSONL is
provided, the exact local prompt-token workload is submitted to llama-server;
the independently rendered llama.cpp prompt and every mismatch remain recorded.
"""

import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

REFERENCE_COMMIT = "d48a56effbba99944a677938d4beb63a0065ecdf"
SCHEMA_VERSION = 3
SCENARIO_VERSION = 1
TURNS = (
    ("Hello, how are you?", 32),
    ("Okay, that is good.", 32),
    ("Tell me about Gothenburg in about 120 words.", 160),
    ("What was the second place you mentioned, and why might a visitor choose it?", 128),
)


def post_json(base_url, path, payload, timeout=None):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def erase_slot(base_url):
    try:
        with post_json(base_url, "/slots/0?action=erase", {}, timeout=10) as response:
            if response.status not in (200, 204):
                raise RuntimeError("llama-server did not erase slot 0")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"pinned comparison requires the slot erase endpoint: HTTP {error.code}: {detail}"
        ) from error


def wait_ready(base_url, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited before becoming ready (status {process.returncode})"
            )
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    raise RuntimeError("llama-server did not become ready")


def require_available_launch_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise RuntimeError(
                f"refusing to launch on occupied or unavailable {host}:{port}"
            ) from error


def verify_server_binary(server):
    result = subprocess.run(
        [server, "--version"], check=True, capture_output=True, text=True, timeout=30
    )
    version = result.stdout + result.stderr
    if REFERENCE_COMMIT not in version and REFERENCE_COMMIT[:9] not in version:
        raise RuntimeError(
            "llama-server binary is not the pinned commit " + REFERENCE_COMMIT
            + "; reported version: " + version.strip()
        )
    return version.strip()


def render_and_tokenize(base_url, messages):
    with post_json(base_url, "/apply-template", {
        "messages": messages,
        "add_generation_prompt": True,
    }) as response:
        rendered = json.load(response)
    prompt = rendered.get("prompt", rendered.get("content"))
    if not isinstance(prompt, str):
        raise RuntimeError("llama-server /apply-template omitted rendered prompt")
    with post_json(base_url, "/tokenize", {"content": prompt, "add_special": True}) as response:
        tokenized = json.load(response)
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError("llama-server /tokenize omitted token IDs")
    return prompt, tokens


def stream_completion(base_url, prompt, max_tokens):
    started = time.monotonic_ns()
    first_visible = None
    text = ""
    selected_ids = []
    final_event = {}
    with post_json(base_url, "/completion", {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
        "seed": 42,
        "cache_prompt": True,
        "id_slot": 0,
        "stream": True,
        "return_tokens": True,
    }, timeout=None) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            event = json.loads(payload)
            content = event.get("content", "")
            if isinstance(content, str) and content:
                if first_visible is None:
                    first_visible = time.monotonic_ns()
                text += content
            event_tokens = event.get("tokens")
            if isinstance(event_tokens, list) and all(isinstance(token, int) for token in event_tokens):
                selected_ids.extend(event_tokens)
            final_event = event
    completed = time.monotonic_ns()
    timings = final_event.get("timings", {})
    return {
        "external_observed_ttft_ns": None if first_visible is None else first_visible - started,
        "complete_turn_ns": completed - started,
        "final_text": text,
        "selected_token_ids": selected_ids or None,
        "cache_n": timings.get("cache_n", final_event.get("tokens_cached")),
        "prompt_n": timings.get("prompt_n", final_event.get("tokens_evaluated")),
        "predicted_n": timings.get("predicted_n", final_event.get("tokens_predicted")),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "stop": final_event.get("stop"),
        "stopped_eos": final_event.get("stopped_eos"),
        "stopped_limit": final_event.get("stopped_limit"),
    }


def summarize(values):
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not finite:
        return {"count": 0, "median": None, "min": None, "max": None, "stddev": None, "mad": None}
    median = statistics.median(finite)
    return {
        "count": len(finite),
        "median": median,
        "min": finite[0],
        "max": finite[-1],
        "stddev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "mad": statistics.median(abs(value - median) for value in finite),
    }


def load_local_records(path):
    prompts = {}
    reused = {}
    selected = {}
    identity = None
    if not path:
        return prompts, reused, selected, identity
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_kind") == "setup":
                identity = record
            if record.get("record_kind") != "turn" or record.get("memory_audit"):
                continue
            key = (bool(record["warmup"]), int(record["repetition_index"]), int(record["turn_index"]))
            prompts[key] = record["rendered_prompt_token_ids"]
            reused[key] = record["reused_tokens"]
            selected[key] = record["selected_token_ids"]
    return prompts, reused, selected, identity


def emit(record):
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", default="/opt/homebrew/bin/llama-server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--connect", action="store_true", help="use an already-running, unverified server")
    parser.add_argument("--allow-unverified-connect", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--local-jsonl", help="paired local JSONL for exact prompt/cache checks")
    parser.add_argument(
        "--workload-view", choices=("product", "paired"), default="product",
        help="use llama.cpp's own history, or submit exact local prompt-token IDs",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be nonnegative and repetitions must be positive")
    if args.connect and not args.allow_unverified_connect:
        parser.error("--connect is not a trustworthy pinned run; pass --allow-unverified-connect explicitly")
    if args.workload_view == "paired" and not args.local_jsonl:
        parser.error("--workload-view paired requires --local-jsonl")

    parsed_url = urllib.parse.urlparse(args.server_url)
    if parsed_url.hostname not in ("127.0.0.1", "localhost"):
        parser.error("the pinned comparator only launches/connects to localhost")
    port = parsed_url.port or 80
    if not args.connect:
        require_available_launch_port("127.0.0.1", port)
    binary_version = verify_server_binary(args.server)
    model_path = os.path.realpath(args.model)
    model_stat = os.stat(model_path)
    model_identity = f"{model_path}:size={model_stat.st_size}:mtime={model_stat.st_mtime_ns}"
    slot_directory = None if args.connect else tempfile.TemporaryDirectory(
        prefix="local-gguf-llama-slots-"
    )
    command = [
        args.server, "--model", model_path, "--ctx-size", str(args.ctx_size),
        "--n-gpu-layers", "99", "--flash-attn", "on", "--parallel", "1",
        "--batch-size", "128", "--ubatch-size", "128", "--no-context-shift",
        "--cache-ram", "0", "--slots",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if slot_directory is not None:
        command.extend(["--slot-save-path", slot_directory.name])
    local_prompts, local_reused, local_selected, local_identity = load_local_records(
        args.local_jsonl
    )
    process = None
    records = []
    conversation_records = []
    errors = []
    if args.local_jsonl and local_identity is None:
        errors.append("paired local JSONL has no setup record")
    if local_identity is not None and local_identity.get("model_identity") != model_identity:
        errors.append("paired local and reference model identities differ")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "scenario_version": SCENARIO_VERSION,
        "engine": "llama.cpp",
        "engine_build": REFERENCE_COMMIT,
        "model": model_path,
        "model_identity": model_identity,
        "backend": "metal",
        "os": platform.platform(),
        "hardware": platform.machine(),
        "frontend": "HTTP SSE /completion",
        "workload_view": args.workload_view,
    }
    sampler = {
        "mode": "greedy", "seed": 42, "temperature": 0.0,
        "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0,
    }
    try:
        if not args.connect:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=sys.stderr)
        wait_ready(args.server_url, 180.0, process)
        emit({
            **identity, **sampler, "record_kind": "reference_setup", "binary_version": binary_version,
            "configuration_verified": not args.connect, "server_url": args.server_url,
            "command": command if not args.connect else None,
            "local_model_identity_match": None if local_identity is None
                else local_identity.get("model_identity") == model_identity,
        })
        for warmup, count in ((True, args.warmup), (False, args.repetitions)):
            for repetition in range(count):
                erase_slot(args.server_url)
                messages = []
                conversation_ttfts = []
                conversation_duration = 0
                for turn_index, (user_text, limit) in enumerate(TURNS):
                    messages.append({"role": "user", "content": user_text})
                    prompt, prompt_ids = render_and_tokenize(args.server_url, messages)
                    key = (warmup, repetition, turn_index)
                    paired_prompt_ids = local_prompts.get(key)
                    if args.local_jsonl and paired_prompt_ids is None:
                        errors.append(
                            f"paired local JSONL is missing warmup={warmup} "
                            f"repetition={repetition} turn={turn_index}"
                        )
                    submitted_prompt = prompt
                    paired_prompt_used = False
                    if args.workload_view == "paired" and paired_prompt_ids is not None:
                        submitted_prompt = paired_prompt_ids
                        paired_prompt_used = True
                    result = stream_completion(args.server_url, submitted_prompt, limit)
                    prompt_match = None if key not in local_prompts else local_prompts[key] == prompt_ids
                    cache_match = None if key not in local_reused or result["cache_n"] is None \
                        else local_reused[key] == result["cache_n"]
                    selected_match = None if key not in local_selected \
                        else local_selected[key] == result["selected_token_ids"]
                    record = {
                        **identity, **sampler, **result,
                        "record_kind": "reference_turn", "warmup": warmup,
                        "repetition_index": repetition, "turn_index": turn_index,
                        "user_text": user_text, "generated_token_limit": limit,
                        "rendered_prompt_token_ids": prompt_ids,
                        "submitted_prompt_token_ids": paired_prompt_ids
                            if paired_prompt_used else prompt_ids,
                        "paired_local_prompt_used": paired_prompt_used,
                        "local_prompt_ids_match": prompt_match,
                        "local_reused_tokens_match": cache_match,
                        "local_reused_tokens": local_reused.get(key),
                        "cache_reuse_delta": None if key not in local_reused or result["cache_n"] is None
                            else result["cache_n"] - local_reused[key],
                        "local_selected_ids_match": selected_match,
                    }
                    emit(record)
                    records.append(record)
                    conversation_ttfts.append(result["external_observed_ttft_ns"])
                    conversation_duration += result["complete_turn_ns"]
                    messages.append({"role": "assistant", "content": result["final_text"]})
                conversation_record = {
                    **identity, **sampler, "record_kind": "reference_conversation",
                    "warmup": warmup, "repetition_index": repetition, "turn_index": None,
                    "complete_conversation_ns": conversation_duration,
                    "per_turn_external_observed_ttft_ns": conversation_ttfts,
                }
                emit(conversation_record)
                conversation_records.append(conversation_record)
        measured = [record for record in records if not record["warmup"]]
        for turn_index in range(len(TURNS)):
            turns = [record for record in measured if record["turn_index"] == turn_index]
            emit({
                **identity, **sampler, "record_kind": "reference_summary",
                "warmup": False, "repetition_index": None, "turn_index": turn_index,
                "external_observed_ttft_ns": summarize(
                    record["external_observed_ttft_ns"] for record in turns
                ),
                "complete_turn_ns": summarize(record["complete_turn_ns"] for record in turns),
                "matched_prompt_count": sum(record["local_prompt_ids_match"] is True for record in turns),
                "matched_cache_count": sum(record["local_reused_tokens_match"] is True for record in turns),
                "matched_selected_ids_count": sum(
                    record["local_selected_ids_match"] is True for record in turns
                ),
            })
        measured_conversations = [record for record in conversation_records if not record["warmup"]]
        emit({
            **identity, **sampler, "record_kind": "reference_summary",
            "warmup": False, "repetition_index": None, "turn_index": None,
            "complete_conversation_ns": summarize(
                record["complete_conversation_ns"] for record in measured_conversations
            ),
        })
        emit({
            **identity, **sampler, "record_kind": "reference_validation",
            "warmup": False, "repetition_index": None, "turn_index": None,
            "valid": not errors, "errors": errors,
        })
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if slot_directory is not None:
            slot_directory.cleanup()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

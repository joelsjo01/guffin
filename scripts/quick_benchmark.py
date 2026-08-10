#!/usr/bin/env python3
"""Benchmark the real local two-turn chat product and local kernels."""

import argparse
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import time

from chat_benchmark_common import (
    INTER_TURN_PAUSE_SECONDS,
    PROMPTS,
    SEED,
    TURN_NAMES,
    checked_rate,
    print_turn_table,
    run_process,
)


LOCAL_RATE = re.compile(
    r"\[ Prompt: (?P<prompt>[0-9.]+) t/s \| Generation: "
    r"(?P<generation>[0-9.]+) t/s \| TTFT: (?P<ttft>[0-9.]+) s \]"
)
LOCAL_DETAIL = re.compile(
    r"\[ Detail: Evaluated: (?P<evaluated>[0-9]+) \| Cached: (?P<cached>[0-9]+)"
    r" \| Decoded: (?P<decoded>[0-9]+) \| Prompt: (?P<prompt_seconds>[0-9.]+) s"
    r" \| Generation: (?P<generation_seconds>[0-9.]+) s \]"
)
LOCAL_DECODE = re.compile(r"^\[ Decode: (?P<json>\{.*\}) \]$", re.MULTILINE)

KERNEL_SCHEMA_VERSION = 2
KERNEL_MEASURED_COMMANDS = 3
KERNEL_SUSTAINED_DISPATCHES = 42
KERNEL_MODE_DISPATCHES = {"isolated": 1, "sustained": KERNEL_SUSTAINED_DISPATCHES}
ENVIRONMENT_SCHEMA_VERSION = 1
DEFAULT_COOLDOWN_SECONDS = 240.0


def parse_local(output):
    rates = list(LOCAL_RATE.finditer(output))
    details = list(LOCAL_DETAIL.finditer(output))
    decode_matches = list(LOCAL_DECODE.finditer(output))
    if (
        len(rates) != len(PROMPTS)
        or len(details) != len(PROMPTS)
        or len(decode_matches) != len(PROMPTS)
    ):
        raise RuntimeError("local chat did not report exactly two complete timing and attribution records")
    turns = []
    for name, rate, detail, decode_match in zip(TURN_NAMES, rates, details, decode_matches):
        try:
            attribution = json.loads(decode_match["json"])
        except json.JSONDecodeError as error:
            raise RuntimeError(f"local {name} decode attribution is invalid JSON") from error
        turn = {
            "name": name,
            "evaluated": int(detail["evaluated"]),
            "cached": int(detail["cached"]),
            "decoded": int(detail["decoded"]),
            "ttft_ns": round(float(rate["ttft"]) * 1e9),
            "prompt_ns": round(float(detail["prompt_seconds"]) * 1e9),
            "generation_ns": round(float(detail["generation_seconds"]) * 1e9),
            "prompt_tps": float(rate["prompt"]),
            "generation_tps": float(rate["generation"]),
            "decode_attribution": attribution,
        }
        checked_rate(turn["evaluated"], turn["prompt_ns"], turn["prompt_tps"], f"local {name} prompt")
        checked_rate(turn["decoded"], turn["generation_ns"], turn["generation_tps"], f"local {name} generation")
        validate_decode_attribution(turn, name)
        turns.append(turn)
    return turns


def validate_decode_attribution(turn, name):
    record = turn.get("decode_attribution")
    if not isinstance(record, dict) or record.get("schema_version") != 3:
        raise RuntimeError(f"local {name} decode attribution schema mismatch")
    token_hash = record.get("selected_token_hash")
    if not isinstance(token_hash, str) or len(token_hash) != 16 \
            or any(character not in "0123456789abcdef" for character in token_hash):
        raise RuntimeError(f"local {name} selected-token hash is invalid")
    buckets = record.get("proposal_buckets")
    if not isinstance(buckets, list):
        raise RuntimeError(f"local {name} proposal buckets are invalid")
    bucket_cycles = 0
    bucket_matches = 0
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise RuntimeError(f"local {name} proposal bucket is invalid")
        for field in ("match_length", "support", "confidence"):
            if not isinstance(bucket.get(field), str):
                raise RuntimeError(f"local {name} proposal bucket {field} is invalid")
        for field in ("cycles", "matches"):
            if isinstance(bucket.get(field), bool) or not isinstance(bucket.get(field), int) \
                    or bucket[field] < 0:
                raise RuntimeError(f"local {name} proposal bucket {field} is invalid")
        if bucket["matches"] > bucket["cycles"]:
            raise RuntimeError(f"local {name} proposal bucket matches exceed cycles")
        bucket_cycles += bucket["cycles"]
        bucket_matches += bucket["matches"]
    if bucket_cycles != record.get("speculative_cycles") \
            or bucket_matches != record.get("speculative_matches"):
        raise RuntimeError(f"local {name} proposal bucket totals mismatch")
    required_nonnegative = (
        "selected_tokens", "decoded_tokens", "initial_cache_position",
        "final_cache_position", "evaluated_prompt_tokens", "reused_cache_tokens",
        "decode_wall_ns", "gpu_ns", "selection_ns", "wall_minus_gpu_ns",
        "resident_weights_bytes", "resident_kv_bytes", "resident_scratch_bytes",
        "warmup_kv_clear_passes", "warmup_host_cleared_kv_bytes",
        "command_buffers", "commits", "waits", "compute_encoders",
        "compute_dispatches", "blit_encoders", "barriers", "transient_allocations",
        "selector_dispatches", "max_candidates", "compact_readback_bytes",
        "attention_dispatches", "attention_partitions", "attention_tokens",
        "output_head_elisions", "output_elided_weight_bytes",
        "scheduled_bytes_total",
    )
    for field in required_nonnegative:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"local {name} decode attribution {field} is invalid")
    if record["selected_tokens"] != turn["decoded"]:
        raise RuntimeError(f"local {name} selected-token attribution mismatch")
    if record["evaluated_prompt_tokens"] != turn["evaluated"]:
        raise RuntimeError(f"local {name} evaluated-token attribution mismatch")
    if record["reused_cache_tokens"] != turn["cached"]:
        raise RuntimeError(f"local {name} cached-token attribution mismatch")
    if record["decoded_tokens"] != record["selected_tokens"]:
        raise RuntimeError(f"local {name} selected/decode attribution mismatch")
    expected_final = (
        record["initial_cache_position"]
        + record["evaluated_prompt_tokens"]
        + record["decoded_tokens"]
    )
    if record["final_cache_position"] != expected_final:
        raise RuntimeError(f"local {name} final cache attribution mismatch")
    if record["gpu_ns"] > record["decode_wall_ns"]:
        raise RuntimeError(f"local {name} GPU attribution exceeds decode wall time")
    if record["wall_minus_gpu_ns"] != record["decode_wall_ns"] - record["gpu_ns"]:
        raise RuntimeError(f"local {name} decode residual attribution mismatch")
    if record["warmup_kv_clear_passes"] != 0 or record["warmup_host_cleared_kv_bytes"] != 0:
        raise RuntimeError(f"local {name} warm-up materialized KV with a host clear")
    if record["decoded_tokens"] > 0 and (
        record["command_buffers"] == 0
        or record["compute_dispatches"] == 0
        or record["gpu_ns"] == 0
    ):
        raise RuntimeError(f"local {name} decoded tokens have no GPU work")
    if record["output_head_elisions"] > record["decoded_tokens"]:
        raise RuntimeError(f"local {name} output-head elisions exceed decoded tokens")
    byte_categories = record.get("scheduled_bytes_per_token")
    if not isinstance(byte_categories, dict):
        raise RuntimeError(f"local {name} scheduled byte categories are missing")
    category_names = (
        "embedding", "attention_projections", "ffn_gate_up", "ffn_down",
        "output_head", "norms",
    )
    for field in (*category_names, "total"):
        value = byte_categories.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"local {name} scheduled byte category {field} is invalid")
    if sum(byte_categories[field] for field in category_names) != byte_categories["total"]:
        raise RuntimeError(f"local {name} scheduled byte categories do not sum")
    minimum_elided = byte_categories["output_head"] * record["output_head_elisions"]
    if record["output_elided_weight_bytes"] < minimum_elided:
        raise RuntimeError(f"local {name} output elision byte accounting is too small")
    expected_scheduled = byte_categories["total"] * record["decoded_tokens"] \
        - record["output_elided_weight_bytes"]
    if record["scheduled_bytes_total"] != expected_scheduled:
        raise RuntimeError(f"local {name} scheduled byte total mismatch")


def run_local_chat(binary, model, cooldown, speculative=True):
    if cooldown:
        print(f"cooling down for {cooldown:g}s before local chat...", file=sys.stderr)
        time.sleep(cooldown)
    command = [
        binary, model, "--chat", "--backend", "metal", "--seed", str(SEED),
        "--decode-diagnostics",
    ]
    if not speculative:
        command.append("--no-speculative")
    stdout, stderr = run_process(
        command,
        "[ Detail:",
    )
    return parse_local(stdout + stderr)


def run_kernel_diagnostics(binary, model):
    result = subprocess.run(
        [binary, model, "--json"], check=True, capture_output=True, text=True
    )
    diagnostics = json.loads(result.stdout)
    validate_kernel_diagnostics(diagnostics)
    return diagnostics


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"kernel diagnostics {name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"kernel diagnostics {name} must be finite and positive")


def validate_kernel_diagnostics(diagnostics):
    if diagnostics.get("schema_version") != KERNEL_SCHEMA_VERSION:
        raise RuntimeError("kernel diagnostics schema version mismatch")
    if diagnostics.get("measured_commands_per_mode") != KERNEL_MEASURED_COMMANDS:
        raise RuntimeError("kernel diagnostics measured-command count mismatch")
    if diagnostics.get("warmup_commands_per_mode") != 1:
        raise RuntimeError("kernel diagnostics warm-up count mismatch")
    if diagnostics.get("sustained_dispatches_per_command") != KERNEL_SUSTAINED_DISPATCHES:
        raise RuntimeError("kernel diagnostics sustained dispatch count mismatch")
    kernels = diagnostics.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise RuntimeError("kernel diagnostics must contain kernels")
    for kernel in kernels:
        if not isinstance(kernel, dict) or not isinstance(kernel.get("name"), str):
            raise RuntimeError("kernel diagnostics contains an invalid kernel")
        measurements = kernel.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != len(KERNEL_MODE_DISPATCHES):
            raise RuntimeError(f"kernel diagnostics {kernel['name']} must contain two modes")
        seen_modes = set()
        for measurement in measurements:
            if not isinstance(measurement, dict):
                raise RuntimeError(f"kernel diagnostics {kernel['name']} measurement is invalid")
            mode = measurement.get("mode")
            if mode not in KERNEL_MODE_DISPATCHES or mode in seen_modes:
                raise RuntimeError(f"kernel diagnostics {kernel['name']} has invalid or duplicate mode")
            seen_modes.add(mode)
            expected_dispatches = KERNEL_MODE_DISPATCHES[mode]
            if measurement.get("dispatches_per_command") != expected_dispatches:
                raise RuntimeError(
                    f"kernel diagnostics {kernel['name']} {mode} dispatch count mismatch"
                )
            raw = measurement.get("raw_command_ns")
            if not isinstance(raw, list) or len(raw) != KERNEL_MEASURED_COMMANDS:
                raise RuntimeError(
                    f"kernel diagnostics {kernel['name']} {mode} raw sample count mismatch"
                )
            for duration in raw:
                _positive_number(duration, f"{kernel['name']} {mode} raw duration")
            for field in (
                "median_per_dispatch_ns",
                "min_per_dispatch_ns",
                "max_per_dispatch_ns",
                "effective_gbps",
            ):
                _positive_number(measurement.get(field), f"{kernel['name']} {mode} {field}")
            sorted_raw = sorted(raw)
            expected_statistics = {
                "median_per_dispatch_ns": sorted_raw[len(raw) // 2] / expected_dispatches,
                "min_per_dispatch_ns": sorted_raw[0] / expected_dispatches,
                "max_per_dispatch_ns": sorted_raw[-1] / expected_dispatches,
            }
            for field, expected in expected_statistics.items():
                if not math.isclose(
                    measurement[field], expected, rel_tol=1e-12, abs_tol=1e-6
                ):
                    raise RuntimeError(
                        f"kernel diagnostics {kernel['name']} {mode} {field} normalization mismatch"
                    )
        if seen_modes != set(KERNEL_MODE_DISPATCHES):
            raise RuntimeError(f"kernel diagnostics {kernel['name']} is missing a mode")


def _command_text(command, runner=subprocess.run):
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_environment(model, local_binary, kernel_binary, kernels, runner=subprocess.run):
    model_path = os.path.realpath(model)
    try:
        model_size = os.path.getsize(model_path)
        with open(model_path, "rb") as model_file:
            header = model_file.read(24)
        magic, version, tensor_count, metadata_count = struct.unpack("<4sIQQ", header)
        if magic != b"GGUF":
            raise RuntimeError("benchmark model does not have a GGUF header")
    except (OSError, struct.error) as error:
        raise RuntimeError(f"could not identify benchmark model: {error}") from error

    display_json = _command_text(["system_profiler", "-json", "SPDisplaysDataType"], runner)
    metal_device = None
    if display_json is not None:
        try:
            displays = json.loads(display_json).get("SPDisplaysDataType", [])
            if displays:
                metal_device = displays[0].get("sppci_model") or displays[0].get("_name")
        except (json.JSONDecodeError, AttributeError):
            metal_device = None
    machine_model = _command_text(["sysctl", "-n", "hw.model"], runner)
    os_version = _command_text(["sw_vers", "-productVersion"], runner)
    os_build = _command_text(["sw_vers", "-buildVersion"], runner)
    environment = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "engine_build": kernels.get("engine_build"),
        "diagnostic_schema": kernels.get("schema_version"),
        "model": {
            "path": model_path,
            "size_bytes": model_size,
            "gguf_version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
        },
        "hardware": {
            "architecture": platform.machine() or None,
            "machine_model": machine_model,
            "metal_device": metal_device,
            "runtime_counter_capabilities": "not_probed_by_canonical_benchmark",
        },
        "os": {
            "name": platform.system() or None,
            "product_version": os_version,
            "build": os_build,
        },
        "executables": {
            "local": os.path.realpath(local_binary),
            "kernels": os.path.realpath(kernel_binary),
            "build_mode": "release" if "release" in pathlib_parts(local_binary) else "unknown",
        },
    }
    validate_environment(environment)
    return environment


def pathlib_parts(path):
    return tuple(part for part in os.path.normpath(path).split(os.sep) if part)


def validate_environment(environment):
    if not isinstance(environment, dict) or environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise RuntimeError("benchmark environment schema mismatch")
    if not isinstance(environment.get("engine_build"), str) or not environment["engine_build"]:
        raise RuntimeError("benchmark environment engine build is missing")
    if environment.get("diagnostic_schema") != KERNEL_SCHEMA_VERSION:
        raise RuntimeError("benchmark environment diagnostic schema mismatch")
    model = environment.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("path"), str):
        raise RuntimeError("benchmark environment model identity is missing")
    for field in ("size_bytes", "gguf_version", "tensor_count", "metadata_count"):
        if isinstance(model.get(field), bool) or not isinstance(model.get(field), int) or model[field] <= 0:
            raise RuntimeError(f"benchmark environment model {field} is invalid")
    hardware = environment.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("architecture"):
        raise RuntimeError("benchmark environment architecture is missing")
    if not isinstance(environment.get("os"), dict) or not environment["os"].get("name"):
        raise RuntimeError("benchmark environment operating system is missing")


def print_human(local, kernels, terminal_rendered):
    print("Local two-turn chat (fresh process, seed 42, natural stop)")
    print("local startup warm-up: enabled (16-token prefill + 1-token decode + width-two verifier)")
    stream_mode = "terminal alternate screen" if terminal_rendered else "captured pipes"
    print(f"chat streams: {stream_mode}; fixed 5s input pause between turns")
    print_turn_table("local-gguf-engine", local)
    print("\nProduction decode attribution (same chat path; logical scheduled weight bytes)")
    print("Turn       GPU/token ms  Wall/token ms  Residual/token ms  Scheduled GB/s  Dispatch/token  Attention tokens")
    for turn in local:
        record = turn["decode_attribution"]
        tokens = record["decoded_tokens"]
        print(
            f"{turn['name']:<10} {record['gpu_ns'] / tokens / 1e6:>12.2f}"
            f" {record['decode_wall_ns'] / tokens / 1e6:>14.2f}"
            f" {record['wall_minus_gpu_ns'] / tokens / 1e6:>18.2f}"
            f" {record['scheduled_bytes_total'] / record['gpu_ns']:>15.1f}"
            f" {record['compute_dispatches'] / tokens:>15.1f}"
            f" {record['attention_tokens']:>17}"
        )
    print("\nKernel diagnostics (separate process, after headline chat)")
    print("sustained mode: 42 serial, buffer-barrier-separated dispatches per command")
    print(
        "Kernel          Shape                    Mode       Dispatches  "
        "Median/dispatch ms  Range/dispatch ms  Effective GB/s"
    )
    for kernel in kernels["kernels"]:
        for measurement in kernel["measurements"]:
            print(
                f"{kernel['name']:<15} {kernel['shape']:<24} {measurement['mode']:<10}"
                f" {measurement['dispatches_per_command']:>10}"
                f" {measurement['median_per_dispatch_ns'] / 1e6:>19.2f}"
                f"  {measurement['min_per_dispatch_ns'] / 1e6:>6.2f}-"
                f"{measurement['max_per_dispatch_ns'] / 1e6:<6.2f}"
                f" {measurement['effective_gbps']:>15.1f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--local", default="build/release/local-gguf-generate")
    parser.add_argument("--kernels", default="build/release/local-gguf-bench")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-speculative", action="store_true")
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error("cooldown must be nonnegative")

    terminal_rendered = sys.stderr.isatty()
    local = run_local_chat(args.local, args.model, args.cooldown, not args.no_speculative)
    if args.cooldown:
        print(f"cooling down for {args.cooldown:g}s before kernel diagnostics...", file=sys.stderr)
        time.sleep(args.cooldown)
    kernels = run_kernel_diagnostics(args.kernels, args.model)
    environment = collect_environment(args.model, args.local, args.kernels, kernels)
    result = {
        "protocol": {
            "prompts": PROMPTS,
            "seed": SEED,
            "repetitions": 1,
            "response_limit": None,
            "local_startup_warmup": True,
            "local_startup_warmup_prefill_tokens": 16,
            "local_startup_warmup_decode_tokens": 1,
            "local_startup_warmup_verifier_tokens": 2,
            "speculative_decode": not args.no_speculative,
            "cooldown_seconds": args.cooldown,
            "inter_turn_pause_seconds": INTER_TURN_PAUSE_SECONDS,
            "terminal_rendered": terminal_rendered,
            "chat_stream_mode": "terminal_alternate_screen" if terminal_rendered else "captured_pipes",
            "decode_attribution_schema": 3,
            "kernel_diagnostic_schema": KERNEL_SCHEMA_VERSION,
            "local_executable": os.path.realpath(args.local),
            "kernel_executable": os.path.realpath(args.kernels),
        },
        "environment": environment,
        "local": local,
        "kernels": kernels,
    }
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print_human(local, kernels, terminal_rendered)


if __name__ == "__main__":
    main()

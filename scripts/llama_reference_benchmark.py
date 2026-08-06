#!/usr/bin/env python3
"""Benchmark the pinned llama-cli two-turn reference independently."""

import argparse
import json
import pathlib
import re
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


REFERENCE_COMMIT = "d48a56effbba99944a677938d4beb63a0065ecdf"
LLAMA_RATE = re.compile(
    r"\[ Prompt: (?P<prompt>[0-9.]+) t/s \| Generation: (?P<generation>[0-9.]+) t/s \]"
)
LLAMA_PROMPT = re.compile(
    r"prompt eval time =\s*(?P<duration>[0-9.]+) ms /\s*(?P<count>[0-9]+) tokens"
)
LLAMA_GENERATION = re.compile(
    r"(?<!prompt )eval time =\s*(?P<duration>[0-9.]+) ms /\s*(?P<count>[0-9]+) tokens"
)
LLAMA_TOTAL_PROMPT = re.compile(r"new prompt, .*task\.n_tokens = (?P<count>[0-9]+)")


def verify_reference_install(binary):
    result = subprocess.run(
        [binary, "--version"], check=True, capture_output=True, text=True
    )
    version = result.stdout + result.stderr
    if REFERENCE_COMMIT[:9] not in version:
        raise RuntimeError(f"llama-cli is not pinned to {REFERENCE_COMMIT}")
    receipt_path = pathlib.Path(binary).resolve().parent.parent / "INSTALL_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        installed_revision = receipt["source"]["scm_revision"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot verify full llama-cli revision from {receipt_path}"
        ) from error
    if installed_revision != REFERENCE_COMMIT:
        raise RuntimeError(
            f"llama-cli receipt revision {installed_revision} does not match {REFERENCE_COMMIT}"
        )
    return version.strip().splitlines()[0]


def parse_llama(output):
    rates = list(LLAMA_RATE.finditer(output))
    prompt_records = list(LLAMA_PROMPT.finditer(output))
    generation_records = list(LLAMA_GENERATION.finditer(output))
    total_prompts = list(LLAMA_TOTAL_PROMPT.finditer(output))
    expected = len(PROMPTS)
    if not all(len(records) == expected for records in (rates, prompt_records, generation_records, total_prompts)):
        raise RuntimeError(
            "llama-cli did not expose exactly two rate, raw timing, and prompt-cache records; "
            "the pinned CLI must run with --log-verbosity 4"
        )
    turns = []
    for name, rate, prompt, generation, total in zip(
        TURN_NAMES, rates, prompt_records, generation_records, total_prompts
    ):
        evaluated = int(prompt["count"])
        task_tokens = int(total["count"])
        prompt_ns = round(float(prompt["duration"]) * 1e6)
        generation_ns = round(float(generation["duration"]) * 1e6)
        turn = {
            "name": name,
            "evaluated": evaluated,
            "cached": task_tokens - evaluated,
            "decoded": int(generation["count"]),
            "ttft_ns": prompt_ns,
            "prompt_ns": prompt_ns,
            "generation_ns": generation_ns,
            "prompt_tps": float(rate["prompt"]),
            "generation_tps": float(rate["generation"]),
        }
        if turn["cached"] < 0:
            raise RuntimeError(f"llama.cpp {name} reported an impossible cache position")
        checked_rate(evaluated, prompt_ns, turn["prompt_tps"], f"llama.cpp {name} prompt")
        checked_rate(turn["decoded"], generation_ns, turn["generation_tps"], f"llama.cpp {name} generation")
        turns.append(turn)
    return turns


def run_reference(binary, model, cooldown):
    if cooldown:
        print(f"cooling down for {cooldown:g}s before llama.cpp...", file=sys.stderr)
        time.sleep(cooldown)
    stdout, stderr = run_process(
        [
            binary,
            "-m", model,
            "--seed", str(SEED),
            "--simple-io",
            "--color", "off",
            "--log-colors", "off",
            "--log-verbosity", "4",
        ],
        "[ Prompt:",
    )
    return parse_llama(stdout + stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", default="/opt/homebrew/bin/llama-cli")
    parser.add_argument("--cooldown", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error("cooldown must be nonnegative")

    terminal_rendered = sys.stderr.isatty()
    reference_version = verify_reference_install(args.reference)
    reference = run_reference(args.reference, args.model, args.cooldown)
    result = {
        "protocol": {
            "prompts": PROMPTS,
            "seed": SEED,
            "repetitions": 1,
            "response_limit": None,
            "llama_commit": REFERENCE_COMMIT,
            "llama_startup_warmup": True,
            "cooldown_seconds": args.cooldown,
            "inter_turn_pause_seconds": INTER_TURN_PAUSE_SECONDS,
            "terminal_rendered": terminal_rendered,
        },
        "reference": reference,
    }
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print("Pinned llama.cpp two-turn reference (fresh process, seed 42, natural stop)")
        print(f"llama.cpp pin: {REFERENCE_COMMIT} ({reference_version})")
        print("llama-cli startup warm-up: enabled (its normal interactive default)")
        stream_mode = "terminal alternate screen" if terminal_rendered else "captured pipes"
        print(f"chat streams: {stream_mode}; fixed 5s input pause between turns")
        print_turn_table("llama.cpp", reference)


if __name__ == "__main__":
    main()

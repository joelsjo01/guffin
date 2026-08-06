#!/usr/bin/env python3
"""Benchmark the real local two-turn chat product and local kernels."""

import argparse
import json
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


LOCAL_RATE = re.compile(
    r"\[ Prompt: (?P<prompt>[0-9.]+) t/s \| Generation: "
    r"(?P<generation>[0-9.]+) t/s \| TTFT: (?P<ttft>[0-9.]+) s \]"
)
LOCAL_DETAIL = re.compile(
    r"\[ Detail: Evaluated: (?P<evaluated>[0-9]+) \| Cached: (?P<cached>[0-9]+)"
    r" \| Decoded: (?P<decoded>[0-9]+) \| Prompt: (?P<prompt_seconds>[0-9.]+) s"
    r" \| Generation: (?P<generation_seconds>[0-9.]+) s \]"
)


def parse_local(output):
    rates = list(LOCAL_RATE.finditer(output))
    details = list(LOCAL_DETAIL.finditer(output))
    if len(rates) != len(PROMPTS) or len(details) != len(PROMPTS):
        raise RuntimeError("local chat did not report exactly two complete timing records")
    turns = []
    for name, rate, detail in zip(TURN_NAMES, rates, details):
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
        }
        checked_rate(turn["evaluated"], turn["prompt_ns"], turn["prompt_tps"], f"local {name} prompt")
        checked_rate(turn["decoded"], turn["generation_ns"], turn["generation_tps"], f"local {name} generation")
        turns.append(turn)
    return turns


def run_local_chat(binary, model, cooldown):
    if cooldown:
        print(f"cooling down for {cooldown:g}s before local chat...", file=sys.stderr)
        time.sleep(cooldown)
    stdout, stderr = run_process(
        [binary, model, "--chat", "--backend", "metal", "--seed", str(SEED)],
        "[ Detail:",
    )
    return parse_local(stdout + stderr)


def run_kernel_diagnostics(binary, model):
    result = subprocess.run(
        [binary, model, "--json"], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def print_human(local, kernels, terminal_rendered):
    print("Local two-turn chat (fresh process, seed 42, natural stop)")
    print("local startup warm-up: enabled (16-token prefill + 1-token decode)")
    stream_mode = "terminal alternate screen" if terminal_rendered else "captured pipes"
    print(f"chat streams: {stream_mode}; fixed 5s input pause between turns")
    print_turn_table("local-gguf-engine", local)
    print("\nKernel diagnostics (separate process, after headline chat)")
    print("Kernel          Shape                    Median ms  Range ms         Effective GB/s")
    for kernel in kernels["kernels"]:
        print(
            f"{kernel['name']:<15} {kernel['shape']:<24} {kernel['median_ns'] / 1e6:>9.1f}"
            f"  {kernel['min_ns'] / 1e6:>5.1f}-{kernel['max_ns'] / 1e6:<5.1f}"
            f" {kernel['effective_gbps']:>20.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--local", default="build/release/local-gguf-generate")
    parser.add_argument("--kernels", default="build/release/local-gguf-bench")
    parser.add_argument("--cooldown", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error("cooldown must be nonnegative")

    terminal_rendered = sys.stderr.isatty()
    local = run_local_chat(args.local, args.model, args.cooldown)
    if args.cooldown:
        print(f"cooling down for {args.cooldown:g}s before kernel diagnostics...", file=sys.stderr)
        time.sleep(args.cooldown)
    kernels = run_kernel_diagnostics(args.kernels, args.model)
    result = {
        "protocol": {
            "prompts": PROMPTS,
            "seed": SEED,
            "repetitions": 1,
            "response_limit": None,
            "local_startup_warmup": True,
            "local_startup_warmup_prefill_tokens": 16,
            "local_startup_warmup_decode_tokens": 1,
            "cooldown_seconds": args.cooldown,
            "inter_turn_pause_seconds": INTER_TURN_PAUSE_SECONDS,
            "terminal_rendered": terminal_rendered,
        },
        "local": local,
        "kernels": kernels,
    }
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print_human(local, kernels, terminal_rendered)


if __name__ == "__main__":
    main()

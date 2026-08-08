#!/usr/bin/env python3
"""Paired production-chat comparison for F16 and Q8_0 output heads."""

import argparse
import json
import os
import statistics
import sys
import time

from chat_benchmark_common import PROMPTS, SEED, run_process
from quick_benchmark import parse_local


SCHEMA_VERSION = 1


def run_chat(binary, model, policy):
    stdout, stderr = run_process(
        [
            binary, model, "--chat", "--backend", "metal", "--seed", str(SEED),
            "--decode-diagnostics", "--output-head", policy,
        ],
        "[ Detail:",
    )
    turns = parse_local(stdout + stderr)
    for turn in turns:
        record = turn["decode_attribution"]
        if record.get("output_head_policy") != policy:
            raise RuntimeError(f"{policy} run reported the wrong output-head policy")
        if policy == "q8_0" and record.get("q8_decode", 0) <= 0:
            raise RuntimeError("Q8_0 run did not execute the Q8 output kernel")
        if policy == "f16" and record.get("q8_decode", 0) != 0:
            raise RuntimeError("F16 control unexpectedly executed the Q8 output kernel")
    return turns


def combined_generation_rate(turns):
    tokens = sum(turn["decoded"] for turn in turns)
    duration = sum(turn["generation_ns"] for turn in turns)
    return tokens * 1e9 / duration


def summarize(samples):
    rates = [combined_generation_rate(sample) for sample in samples]
    return {
        "median_combined_generation_tps": statistics.median(rates),
        "combined_generation_tps": rates,
        "median_first_generation_tps": statistics.median(sample[0]["generation_tps"] for sample in samples),
        "median_followup_generation_tps": statistics.median(sample[1]["generation_tps"] for sample in samples),
        "median_first_prompt_tps": statistics.median(sample[0]["prompt_tps"] for sample in samples),
        "median_followup_prompt_tps": statistics.median(sample[1]["prompt_tps"] for sample in samples),
        "median_first_ttft_ns": statistics.median(sample[0]["ttft_ns"] for sample in samples),
        "median_followup_ttft_ns": statistics.median(sample[1]["ttft_ns"] for sample in samples),
    }


def decide(samples):
    f16_rates = [combined_generation_rate(sample) for sample in samples["f16"]]
    q8_rates = [combined_generation_rate(sample) for sample in samples["q8_0"]]
    paired_deltas = [(q8 / f16 - 1.0) * 100.0 for f16, q8 in zip(f16_rates, q8_rates)]
    f16_summary = summarize(samples["f16"])
    q8_summary = summarize(samples["q8_0"])
    median_delta = (
        q8_summary["median_combined_generation_tps"]
        / f16_summary["median_combined_generation_tps"] - 1.0
    ) * 100.0
    pair_wins = sum(delta > 0.0 for delta in paired_deltas)
    prompt_regressions = []
    ttft_regressions = []
    for turn in ("first", "followup"):
        prompt_regressions.append(
            1.0 - q8_summary[f"median_{turn}_prompt_tps"] / f16_summary[f"median_{turn}_prompt_tps"]
        )
        ttft_regressions.append(
            q8_summary[f"median_{turn}_ttft_ns"] / f16_summary[f"median_{turn}_ttft_ns"] - 1.0
        )
    return {
        "paired_delta_percent": paired_deltas,
        "median_generation_delta_percent": median_delta,
        "paired_wins": pair_wins,
        "worst_prompt_regression_percent": max(prompt_regressions) * 100.0,
        "worst_ttft_regression_percent": max(ttft_regressions) * 100.0,
        "retain": median_delta >= 2.0 and pair_wins >= 4
            and max(prompt_regressions) <= 0.02 and max(ttft_regressions) <= 0.02,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="build/release/local-gguf-generate")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.cooldown < 0:
        parser.error("repetitions must be positive and cooldown must be nonnegative")

    samples = {"f16": [], "q8_0": []}
    order = []
    for repetition in range(args.repetitions):
        pair = ("f16", "q8_0") if repetition % 2 == 0 else ("q8_0", "f16")
        for policy in pair:
            if order and args.cooldown:
                print(f"cooling down for {args.cooldown:g}s...", file=sys.stderr)
                time.sleep(args.cooldown)
            print(f"running {policy} pair {repetition + 1}...", file=sys.stderr)
            samples[policy].append(run_chat(args.binary, args.model, policy))
            order.append(policy)

    result = {
        "schema_version": SCHEMA_VERSION,
        "model": os.path.realpath(args.model),
        "binary": os.path.realpath(args.binary),
        "protocol": {
            "prompts": PROMPTS,
            "seed": SEED,
            "repetitions": args.repetitions,
            "cooldown_seconds": args.cooldown,
            "order": order,
        },
        "samples": samples,
        "summary": {policy: summarize(values) for policy, values in samples.items()},
        "decision": decide(samples),
    }
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
        return
    for policy in ("f16", "q8_0"):
        summary = result["summary"][policy]
        print(
            f"{policy:<5} combined={summary['median_combined_generation_tps']:.2f} tok/s "
            f"first/follow-up={summary['median_first_generation_tps']:.1f}/"
            f"{summary['median_followup_generation_tps']:.1f} tok/s"
        )
    decision = result["decision"]
    print(
        f"delta={decision['median_generation_delta_percent']:+.2f}% "
        f"wins={decision['paired_wins']}/{args.repetitions} "
        f"retain={'yes' if decision['retain'] else 'no'}"
    )


if __name__ == "__main__":
    main()

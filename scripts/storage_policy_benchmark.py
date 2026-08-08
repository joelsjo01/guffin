#!/usr/bin/env python3
"""Paired immutable-weight Metal policy screen; benchmark output stays on stdout."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


CONTROL = "shared-default-tracked"
CLASSES = ("Q4_K gate/up", "Q4_K down", "Q6_K down", "F16 output")


def parse_result(text: str, expected_policy: str) -> dict:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"storage probe returned malformed JSON: {error}") from error
    if result.get("schema_version") != 1 or result.get("policy") != expected_policy:
        raise ValueError("storage probe schema or policy does not match the request")
    kernels = result.get("kernels")
    if not isinstance(kernels, list) or [kernel.get("name") for kernel in kernels] != list(CLASSES):
        raise ValueError("storage probe representative classes are missing or reordered")
    for kernel in kernels:
        if not isinstance(kernel.get("scheduled_class_weight_bytes"), int) \
                or kernel["scheduled_class_weight_bytes"] <= 0:
            raise ValueError("storage probe scheduled class bytes must be positive")
        measurements = kernel.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != 2:
            raise ValueError("storage probe must report isolated and sustained measurements")
        sustained = next((value for value in measurements if value.get("mode") == "sustained"), None)
        if sustained is None or not isinstance(sustained.get("median_per_dispatch_ns"), (int, float)) \
                or sustained["median_per_dispatch_ns"] <= 0:
            raise ValueError("storage probe sustained duration must be positive")
    return result


def weighted_sustained_gbps(result: dict) -> float:
    total_bytes = 0
    total_ns = 0.0
    for kernel in result["kernels"]:
        class_bytes = kernel["scheduled_class_weight_bytes"]
        duration_ns = next(
            value["median_per_dispatch_ns"]
            for value in kernel["measurements"]
            if value["mode"] == "sustained"
        )
        total_bytes += class_bytes
        total_ns += class_bytes * duration_ns / kernel["weight_bytes"]
    if total_bytes <= 0 or total_ns <= 0:
        raise ValueError("storage probe weighted summary is invalid")
    return total_bytes / total_ns


def class_changes(control: dict, candidate: dict) -> dict[str, float]:
    changes = {}
    for control_kernel, candidate_kernel in zip(control["kernels"], candidate["kernels"], strict=True):
        control_ns = next(value["median_per_dispatch_ns"] for value in control_kernel["measurements"]
                          if value["mode"] == "sustained")
        candidate_ns = next(value["median_per_dispatch_ns"] for value in candidate_kernel["measurements"]
                            if value["mode"] == "sustained")
        changes[control_kernel["name"]] = (control_ns / candidate_ns - 1.0) * 100.0
    return changes


def summarize_pairs(pairs: list[tuple[dict, dict]]) -> dict:
    if not pairs:
        raise ValueError("storage policy summary requires at least one pair")
    improvements = []
    per_class: dict[str, list[float]] = {name: [] for name in CLASSES}
    for control, candidate in pairs:
        control_gbps = weighted_sustained_gbps(control)
        candidate_gbps = weighted_sustained_gbps(candidate)
        improvements.append((candidate_gbps / control_gbps - 1.0) * 100.0)
        for name, change in class_changes(control, candidate).items():
            per_class[name].append(change)
    medians = {name: statistics.median(values) for name, values in per_class.items()}
    median_improvement = statistics.median(improvements)
    return {
        "pair_improvement_percent": improvements,
        "median_weighted_improvement_percent": median_improvement,
        "paired_wins": sum(value > 0.0 for value in improvements),
        "class_median_improvement_percent": medians,
        "class_wins": sum(value > 0.0 for value in medians.values()),
        "worst_class_regression_percent": min(medians.values()),
        "passes_three_pair_screen": len(pairs) == 3
            and median_improvement >= 2.0
            and sum(value > 0.0 for value in improvements) >= 2
            and sum(value > 0.0 for value in medians.values()) >= 3
            and min(medians.values()) >= -2.0,
    }


def run_probe(binary: Path, model: Path, policy: str) -> dict:
    completed = subprocess.run(
        [str(binary), str(model), "--storage-policy", policy, "--json"],
        check=True, capture_output=True, text=True,
    )
    return parse_result(completed.stdout, policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/release/local-gguf-bench"))
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.candidate == CONTROL:
        parser.error("candidate must differ from the control policy")
    if args.pairs <= 0 or args.cooldown < 0:
        parser.error("pairs must be positive and cooldown must be nonnegative")

    pairs = []
    for pair_index in range(args.pairs):
        order = (CONTROL, args.candidate) if pair_index % 2 == 0 else (args.candidate, CONTROL)
        measured = {}
        for order_index, policy in enumerate(order):
            if pair_index != 0 or order_index != 0:
                print(f"cooling down for {args.cooldown:g}s before {policy}...", flush=True)
                time.sleep(args.cooldown)
            measured[policy] = run_probe(args.binary, args.model, policy)
        pairs.append((measured[CONTROL], measured[args.candidate]))
    summary = summarize_pairs(pairs)
    output = {"control": CONTROL, "candidate": args.candidate, "pair_count": args.pairs, **summary}
    if args.json:
        print(json.dumps(output, separators=(",", ":"), sort_keys=True))
    else:
        print(f"{args.candidate}: weighted median {summary['median_weighted_improvement_percent']:+.2f}%")
        print(f"paired wins {summary['paired_wins']}/{args.pairs}; class wins {summary['class_wins']}/4; "
              f"worst class {summary['worst_class_regression_percent']:+.2f}%")
        print("PASS" if summary["passes_three_pair_screen"] else "REJECT")


if __name__ == "__main__":
    main()

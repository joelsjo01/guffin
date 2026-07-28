#!/usr/bin/env python3
"""Apply the product-conversation retention gate to two local JSONL runs."""

import argparse
import json
import math
import statistics


def load(path):
    turns = {}
    memory_turns = {}
    conversations = {}
    setup = None
    validation = None
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            kind = record.get("record_kind")
            if kind == "setup":
                setup = record
            elif kind == "validation":
                validation = record
            elif kind == "turn" and record.get("timing_eligible"):
                turns[(record["repetition_index"], record["turn_index"])] = record
            elif kind == "turn" and record.get("memory_audit"):
                memory_turns[record["turn_index"]] = record
            elif kind == "conversation" and record.get("timing_eligible"):
                conversations[record["repetition_index"]] = record
    if setup is None or validation is None:
        raise ValueError(f"{path}: missing setup or validation record")
    if not validation.get("valid"):
        raise ValueError(f"{path}: benchmark declared invalid: {validation.get('errors', [])}")
    return setup, turns, conversations, memory_turns


def median(values):
    return statistics.median(values)


def mad(values):
    center = median(values)
    return statistics.median(abs(value - center) for value in values)


def relative_improvement(baseline, candidate):
    return (baseline - candidate) / baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline_setup, baseline_turns, baseline_conversations, baseline_memory = load(args.baseline)
    candidate_setup, candidate_turns, candidate_conversations, candidate_memory = load(args.candidate)
    errors = []

    identity_fields = (
        "schema_version", "scenario_version", "model_identity", "backend",
        "mode", "seed", "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
    )
    for field in identity_fields:
        if baseline_setup.get(field) != candidate_setup.get(field):
            errors.append(f"setup field {field} differs")
    if set(baseline_turns) != set(candidate_turns):
        errors.append("measured repetition/turn keys differ")
    if set(baseline_conversations) != set(candidate_conversations):
        errors.append("measured conversation keys differ")

    exact_fields = (
        "rendered_prompt_token_ids", "evaluated_suffix_token_ids", "selected_token_ids",
        "visible_token_ids", "final_text", "stop_reason", "stop_token_id", "interrupted",
        "initial_context_length", "final_context_length", "reused_tokens", "trimmed_turns",
        "rebuilt_context", "configured_stop_token_ids", "configured_reverse_stop_strings",
        "resident_weight_bytes", "resident_kv_bytes", "resident_scratch_bytes", "plan_revision",
    )
    for key in sorted(set(baseline_turns) & set(candidate_turns)):
        for field in exact_fields:
            if baseline_turns[key].get(field) != candidate_turns[key].get(field):
                errors.append(f"repetition={key[0]} turn={key[1]} field {field} differs")
    if set(baseline_memory) != set(candidate_memory):
        errors.append("memory-audit turn keys differ")
    for turn_index in sorted(set(baseline_memory) & set(candidate_memory)):
        for field in (
            "metal_current_allocated_bytes", "process_physical_footprint_bytes",
            "process_physical_footprint_peak_bytes",
        ):
            baseline_value = baseline_memory[turn_index].get(field)
            candidate_value = candidate_memory[turn_index].get(field)
            if baseline_value is not None and candidate_value is not None \
                    and candidate_value > baseline_value * 1.02:
                errors.append(f"turn={turn_index} {field} regressed by more than 2%")

    repetitions = sorted(set(baseline_conversations) & set(candidate_conversations))
    baseline_primary = []
    candidate_primary = []
    for repetition in repetitions:
        baseline_values = baseline_conversations[repetition]["per_turn_frontend_complete_ttft_ns"]
        candidate_values = candidate_conversations[repetition]["per_turn_frontend_complete_ttft_ns"]
        if any(value is None for value in baseline_values + candidate_values):
            errors.append(f"repetition={repetition} has unavailable primary TTFT")
            continue
        baseline_primary.append(sum(baseline_values))
        candidate_primary.append(sum(candidate_values))
    if len(baseline_primary) < 5:
        errors.append("retention gate requires at least five paired measured conversations")

    report = {"valid": False, "errors": errors}
    if not errors:
        baseline_median = median(baseline_primary)
        candidate_median = median(candidate_primary)
        noise_threshold = max(0.02, 2.0 * mad(baseline_primary) / baseline_median)
        improvement = relative_improvement(baseline_median, candidate_median)
        paired_wins = sum(c < b for b, c in zip(baseline_primary, candidate_primary))
        if improvement <= noise_threshold:
            errors.append(
                f"primary improvement {improvement:.4%} does not exceed {noise_threshold:.4%}"
            )
        if paired_wins < 4:
            errors.append(f"only {paired_wins} paired conversations improved; four are required")

        turn_indices = sorted({turn for _, turn in baseline_turns})
        for turn_index in turn_indices:
            for field in ("frontend_complete_ttft_ns", "total_ns"):
                baseline_values = [
                    baseline_turns[(repetition, turn_index)][field] for repetition in repetitions
                ]
                candidate_values = [
                    candidate_turns[(repetition, turn_index)][field] for repetition in repetitions
                ]
                if None in baseline_values or None in candidate_values:
                    continue
                regression = -relative_improvement(median(baseline_values), median(candidate_values))
                if regression > noise_threshold:
                    errors.append(
                        f"turn={turn_index} {field} regressed {regression:.4%}"
                    )

        report.update({
            "valid": not errors,
            "baseline_primary_median_ns": baseline_median,
            "candidate_primary_median_ns": candidate_median,
            "baseline_primary_mad_ns": mad(baseline_primary),
            "required_improvement_fraction": noise_threshold,
            "observed_improvement_fraction": improvement,
            "paired_wins": paired_wins,
            "errors": errors,
        })
    print(json.dumps(report, separators=(",", ":")))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize a saved release-chat run without executing either inference engine."""

import argparse
import csv
import html
import hashlib
import io
import json
import math
import pathlib
import statistics
import sys
import tempfile


CSV_FIELDS = (
    "configuration_id", "engine", "mode", "model_sha256", "case_id", "category",
    "suite", "conversation_status", "turn_index", "turn_position", "turn_status",
    "prompt_tokens", "cached_tokens", "selected_tokens", "generation_steps",
    "prompt_ns", "internal_ttft_ns", "generation_ns", "engine_turn_ns", "wall_turn_ns",
    "generation_tps", "native_generation_tps", "prompt_tps", "peak_physical_footprint_bytes", "stop_reason",
)
METRIC_FIELDS = CSV_FIELDS[CSV_FIELDS.index("prompt_tokens"):CSV_FIELDS.index("peak_physical_footprint_bytes")]
NOTES = (
    "One pass per planned conversation; no repetitions, confidence intervals or stability claim.",
    "Natural answers and follow-up histories can differ. These are workload observations, not identical-work or quality-equivalence results.",
    "Generation rate uses summed generation_steps / summed generation_ns. Selected tokens are reported separately; proposed and rejected draft tokens are excluded.",
    "Internal TTFT and external wall-turn latency have separate clocks. Startup, cooldowns and time waiting for the next prompt are excluded from generation rates.",
    "Physical footprint uses the macOS native lifetime maximum at the last successful observation. RSS is the maximum sampled resident size. Unavailable measurements and collection errors remain explicit.",
    "Frontend completion does not establish EOS. Unknown stop reasons stay unknown; known limited or interrupted replies are excluded from complete-conversation aggregates.",
    "Headline comparisons and charts use only jointly complete cases planned for both engines with matching model hash, mode and suite. Independent configuration aggregates retain their own completed workloads and coverage.",
    "The core MT-Bench workload and project supplements have separate suite summaries; their combined rates are not a core-workload headline.",
    "Gemma 2 comparisons must account for Guffin's Q6-down to Q4 and F16-head to Q8 numerical policies; the same GGUF hash does not establish identical executed weights.",
)
CAPPED_REASONS = {"capped", "length", "limit", "token_limit", "max_new_tokens", "context_limit", "context_full"}
STOPPED_REASONS = {"interrupted", "callback_stop", "reverse_stop"}


def _number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return value


def _rate(count, duration):
    return count * 1e9 / duration if count is not None and duration and duration > 0 else None


def _aggregate(rows):
    result = {"turns": len(rows)}
    for field in ("prompt_tokens", "cached_tokens", "selected_tokens", "generation_steps", "prompt_ns", "generation_ns", "engine_turn_ns", "wall_turn_ns"):
        values = [row.get(field) for row in rows]
        result[field] = sum(values) if values and all(value is not None for value in values) else None
    result["generation_tps"] = _rate(result["generation_steps"], result["generation_ns"])
    result["prompt_tps"] = _rate(result["prompt_tokens"], result["prompt_ns"])
    ttfts = [row["internal_ttft_ns"] for row in rows if row.get("internal_ttft_ns") is not None]
    result["median_internal_ttft_ns"] = statistics.median(ttfts) if ttfts else None
    result["ttft_samples"] = len(ttfts)
    footprints = [row["peak_physical_footprint_bytes"] for row in rows if row.get("peak_physical_footprint_bytes") is not None]
    result["peak_physical_footprint_bytes"] = max(footprints) if footprints else None
    reasons = [row.get("stop_reason") or "unknown" for row in rows]
    result["termination_counts"] = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    return result


def _planned_jobs(manifest):
    if "jobs" in manifest:
        return {(job["configuration_id"], job["case_id"]) for job in manifest["jobs"]}
    cases = [case["id"] for case in manifest["cases"]]
    return {(config["id"], case_id) for config in manifest["configurations"] for case_id in config.get("case_ids", cases)}


def summarize_run(manifest, receipts):
    """Normalize receipts, retaining failures outside complete-conversation rates."""
    campaign_id = manifest.get("campaign_id", manifest.get("run_id"))
    if manifest.get("schema_version") != 1 or not campaign_id:
        raise ValueError("manifest must have schema_version 1 and a campaign_id")
    configs = {config["id"]: config for config in manifest["configurations"]}
    cases = {case["id"]: case for case in manifest["cases"]}
    if len(configs) != len(manifest["configurations"]) or len(cases) != len(manifest["cases"]):
        raise ValueError("manifest contains duplicate configuration or case IDs")
    for config in configs.values():
        model_hash = config.get("model_sha256")
        if not isinstance(model_hash, str) or len(model_hash) != 64 or any(character not in "0123456789abcdef" for character in model_hash):
            raise ValueError("configuration must have a lowercase SHA-256 model identity")
        if config.get("engine") not in {"guffin", "llama"} or config.get("mode", "default") not in {"default", "speculative"}:
            raise ValueError("configuration has an unknown engine or mode")
    planned = _planned_jobs(manifest)
    if any(config_id not in configs or case_id not in cases for config_id, case_id in planned):
        raise ValueError("manifest jobs refer to an unknown configuration or case")
    job_ids = {(job["configuration_id"], job["case_id"]): job.get("run_id") for job in manifest.get("jobs", [])}
    if "jobs" in manifest and (len(job_ids) != len(manifest["jobs"]) or not all(job_ids.values())
                                or len(set(job_ids.values())) != len(job_ids)):
        raise ValueError("manifest jobs must have unique nonempty run IDs and case/configuration pairs")
    seen, rows, conversations, failures = set(), [], [], []
    for receipt in receipts:
        if receipt.get("schema_version") != 1:
            raise ValueError("receipt belongs to a different schema")
        config_id, case_id = receipt["configuration"]["id"], receipt["case"]["id"]
        key = (config_id, case_id)
        if key not in planned:
            raise ValueError(f"unplanned receipt: {config_id}/{case_id}")
        if receipt.get("run_id") != job_ids.get(key, campaign_id) or receipt.get("campaign_id", campaign_id) != campaign_id:
            raise ValueError("receipt belongs to a different run or campaign")
        if key in seen:
            raise ValueError(f"duplicate receipt: {config_id}/{case_id}")
        seen.add(key)
        config, case = configs[config_id], cases[case_id]
        if receipt["configuration"] != config or receipt["case"] != case:
            raise ValueError(f"receipt identity mismatch: {config_id}/{case_id}")
        status = receipt.get("status")
        if status not in {"complete", "failed", "interrupted"}:
            raise ValueError(f"invalid conversation status: {status}")
        turns = receipt.get("turns", [])
        if len(turns) > len(case["turns"]):
            raise ValueError(f"receipt has more turns than planned: {config_id}/{case_id}")
        if status == "complete" and len(turns) != len(case["turns"]):
            raise ValueError(f"complete receipt is missing turns: {config_id}/{case_id}")
        conversation_rows = []
        indexes = set()
        for turn in turns:
            index = turn["index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(case["turns"]) or index in indexes:
                raise ValueError(f"invalid or duplicate turn index: {config_id}/{case_id}")
            indexes.add(index)
            if turn.get("prompt") != case["turns"][index]:
                raise ValueError(f"submitted prompt mismatch: {config_id}/{case_id}")
            if "answer_sha256" in turn:
                answer = turn.get("answer")
                if not isinstance(answer, str) or hashlib.sha256(answer.encode()).hexdigest() != turn["answer_sha256"]:
                    raise ValueError(f"answer hash mismatch: {config_id}/{case_id}")
            metrics = turn.get("metrics", {})
            stop_reason = turn.get("stop_reason", metrics.get("stop_reason"))
            turn_status = turn.get("status", "complete")
            if turn.get("capped") or stop_reason in CAPPED_REASONS:
                turn_status = "capped"
            elif stop_reason in STOPPED_REASONS:
                turn_status = "stopped_short"
            row = {
                "configuration_id": config_id, "engine": config.get("engine"), "mode": config.get("mode", "default"),
                "model_sha256": config.get("model_sha256"), "case_id": case_id,
                "category": case.get("category", "uncategorized"), "suite": case.get("suite", ""),
                "conversation_status": status, "turn_index": index,
                "turn_position": "first" if index == 0 else "follow-up", "turn_status": turn_status, "stop_reason": stop_reason,
            }
            for field in METRIC_FIELDS:
                row[field] = _number(metrics.get(field), field)
                if (row[field] is not None and not field.endswith("_tps")
                        and not isinstance(row[field], int)):
                    raise ValueError(f"{field} must be an integer count or duration")
            if (row["selected_tokens"] is not None and row["generation_steps"] is not None
                    and row["generation_steps"] != max(row["selected_tokens"] - 1, 0)):
                raise ValueError(f"generation steps disagree with selected tokens: {config_id}/{case_id}")
            row["peak_physical_footprint_bytes"] = _number(turn.get("memory", {}).get("peak_physical_footprint_bytes"), "peak_physical_footprint_bytes")
            # Do not use an engine's rounded/native rate in the common comparison.
            row["generation_tps"] = _rate(row["generation_steps"], row["generation_ns"])
            row["prompt_tps"] = _rate(row["prompt_tokens"], row["prompt_ns"])
            conversation_rows.append(row)
        if status == "complete" and any(row["turn_status"] != "complete" for row in conversation_rows):
            statuses = {row["turn_status"] for row in conversation_rows}
            status = next((value for value in ("capped", "stopped_short") if value in statuses), "partial")
        for row in conversation_rows:
            row["conversation_status"] = status
        rows.extend(conversation_rows)
        memory = receipt.get("memory", {})
        footprint = _number(memory.get("peak_physical_footprint_bytes"), "peak_physical_footprint_bytes")
        rss = _number(memory.get("peak_rss_bytes"), "peak_rss_bytes")
        conversation = {
            "configuration_id": config_id, "engine": config.get("engine"), "mode": config.get("mode", "default"),
            "model_sha256": config.get("model_sha256"), "case_id": case_id,
            "category": case.get("category", "uncategorized"), "suite": case.get("suite", ""),
            "status": status, "recorded_turns": len(turns), "planned_turns": len(case["turns"]),
            "metrics": _aggregate(conversation_rows),
            "memory": {**memory, "peak_physical_footprint_bytes": footprint, "peak_rss_bytes": rss},
        }
        conversations.append(conversation)
        if status != "complete":
            failures.append({"configuration_id": config_id, "case_id": case_id, "status": status,
                             "recorded_turns": len(turns), "error": receipt.get("error")})
    for config_id, case_id in sorted(planned - seen):
        failures.append({"configuration_id": config_id, "case_id": case_id, "status": "unattempted", "recorded_turns": 0, "error": None})
    aggregates = []
    for config_id, config in configs.items():
        complete_rows = [row for row in rows if row["configuration_id"] == config_id and row["conversation_status"] == "complete"]
        config_conversations = [conv for conv in conversations if conv["configuration_id"] == config_id]
        complete = [conv for conv in config_conversations if conv["status"] == "complete"]
        category_rows = [{"category": category, **_aggregate([row for row in complete_rows if row["category"] == category])}
                         for category in sorted({row["category"] for row in complete_rows})]
        suites = []
        for suite in sorted({case.get("suite", "") for case in cases.values()}):
            suite_cases = {case_id for case_id, case in cases.items() if case.get("suite", "") == suite}
            suite_planned = sum(job[0] == config_id and job[1] in suite_cases for job in planned)
            if not suite_planned:
                continue
            suite_complete = [conv for conv in complete if conv["suite"] == suite]
            suites.append({"suite": suite, "planned_conversations": suite_planned, "complete_conversations": len(suite_complete),
                           **_aggregate([row for row in complete_rows if row["suite"] == suite]),
                           "peak_physical_footprint_bytes": max((conv["memory"]["peak_physical_footprint_bytes"] for conv in suite_complete
                                                                 if conv["memory"]["peak_physical_footprint_bytes"] is not None), default=None)})
        aggregates.append({
            "configuration_id": config_id, "engine": config.get("engine"), "mode": config.get("mode", "default"),
            "model_sha256": config.get("model_sha256"), "planned_conversations": sum(job[0] == config_id for job in planned),
            "complete_conversations": len(complete), "failed_conversations": sum(conv["status"] != "complete" for conv in config_conversations),
            "unattempted_conversations": sum(item["configuration_id"] == config_id and item["status"] == "unattempted" for item in failures),
            "configuration": config, "metrics": _aggregate(complete_rows), "suites": suites, "categories": category_rows,
            "turn_positions": [{"position": position, **_aggregate([row for row in complete_rows if row["turn_position"] == position])}
                               for position in ("first", "follow-up")],
            "peak_physical_footprint_bytes": max((conv["memory"]["peak_physical_footprint_bytes"] for conv in complete
                                                 if conv["memory"]["peak_physical_footprint_bytes"] is not None), default=None),
        })
    comparisons, paired_suites = [], []
    by_key = {(item["configuration_id"], item["case_id"]): item for item in conversations}
    for candidate_id, candidate in configs.items():
        if candidate["engine"] != "guffin":
            continue
        for baseline_id, baseline in configs.items():
            if baseline["engine"] != "llama" or candidate["model_sha256"] != baseline["model_sha256"] or candidate.get("mode", "default") != baseline.get("mode", "default"):
                continue
            shared = {case_id for config_id, case_id in planned if config_id == candidate_id} & {case_id for config_id, case_id in planned if config_id == baseline_id}
            for suite in sorted({cases[case_id].get("suite", "") for case_id in shared}):
                suite_planned = sorted(case_id for case_id in shared if cases[case_id].get("suite", "") == suite)
                complete_ids = [case_id for case_id in suite_planned
                                if all(by_key.get((config_id, case_id), {}).get("status") == "complete" for config_id in (candidate_id, baseline_id))]
                metrics = []
                for config_id in (candidate_id, baseline_id):
                    group = _aggregate([row for row in rows if row["configuration_id"] == config_id and row["case_id"] in complete_ids])
                    group["peak_physical_footprint_bytes"] = max((by_key[(config_id, case_id)]["memory"]["peak_physical_footprint_bytes"]
                                                                 for case_id in complete_ids if by_key[(config_id, case_id)]["memory"]["peak_physical_footprint_bytes"] is not None), default=None)
                    metrics.append(group)
                a, b = metrics[0]["generation_tps"], metrics[1]["generation_tps"]
                paired_suites.append({"candidate_configuration_id": candidate_id, "baseline_configuration_id": baseline_id,
                                      "model_sha256": candidate["model_sha256"], "mode": candidate.get("mode", "default"), "suite": suite,
                                      "planned_case_ids": suite_planned, "complete_case_ids": complete_ids,
                                      "planned_conversations": len(suite_planned), "complete_conversations": len(complete_ids),
                                      "candidate_metrics": metrics[0], "baseline_metrics": metrics[1],
                                      "generation_rate_ratio": a / b if a is not None and b else None})
                for case_id in complete_ids:
                    a = by_key[(candidate_id, case_id)]["metrics"]["generation_tps"]
                    b = by_key[(baseline_id, case_id)]["metrics"]["generation_tps"]
                    comparisons.append({"case_id": case_id, "category": cases[case_id].get("category", "uncategorized"), "suite": suite,
                                        "mode": candidate.get("mode", "default"), "model_sha256": candidate["model_sha256"],
                                        "baseline_configuration_id": baseline_id, "candidate_configuration_id": candidate_id,
                                        "generation_rate_ratio": a / b if a is not None and b else None})
    return {"schema_version": 1, "campaign_id": campaign_id, "run_id": campaign_id, "notes": list(NOTES),
            "timing_conventions": manifest.get("timing_conventions", {}), "configurations": aggregates,
            "conversations": conversations, "turns": rows, "comparisons": comparisons, "paired_suites": paired_suites, "failures": failures,
            "memory_errors": [{"configuration_id": item["configuration_id"], "case_id": item["case_id"], "error": item["memory"]["error"]}
                              for item in conversations if item["memory"].get("error")]}


def _display(value, scale=1.0, suffix=""):
    return "—" if value is None else f"{value / scale:.2f}{suffix}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows])


def render_markdown(summary):
    text = [f"# Release chat run {_cell(summary['run_id'])}", "", "Single-pass actual-chat observations. Frontend-completed conversations contribute to the rates; known limited, interrupted and incomplete conversations remain listed below. Unknown stop reasons are not assumed to be EOS.", ""]
    text.extend(["## Paired workload comparisons", "", "Each row compares the same jointly complete case IDs within one model, mode and suite. Coverage is jointly complete / planned for both engines; cases completed by only one engine do not enter these rates or charts.", "",
                 _table(["Suite", "Mode", "Configurations (Guffin / llama.cpp)", "Paired cases", "Guffin t/s", "llama.cpp t/s", "Rate ratio"],
                        [[item["suite"] or "unspecified", item["mode"], f"{item['candidate_configuration_id']} / {item['baseline_configuration_id']}",
                          f"{item['complete_conversations']} / {item['planned_conversations']}", _display(item["candidate_metrics"]["generation_tps"]),
                          _display(item["baseline_metrics"]["generation_tps"]), _display(item["generation_rate_ratio"], suffix="×")]
                         for item in summary["paired_suites"]]), ""])
    for item in summary["configurations"]:
        text.extend([f"## {_cell(item['configuration_id'])}", "", f"Engine: `{item['engine']}`; mode: `{item['mode']}`; model SHA-256: `{item['model_sha256']}`.", "",
                     "Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.", "",
                     _table(["Suite", "Complete / planned", "Generation t/s", "Selected tokens", "Median internal TTFT ms", "Peak footprint GiB"],
                            [[row["suite"] or "unspecified", f"{row['complete_conversations']} / {row['planned_conversations']}", _display(row["generation_tps"]),
                              row["selected_tokens"] if row["selected_tokens"] is not None else "—", _display(row["median_internal_ttft_ns"], 1e6),
                              _display(row["peak_physical_footprint_bytes"], 2**30)] for row in item["suites"]]), "",
                     _table(["Category", "Turns", "Generation t/s", "Prompt t/s", "Median internal TTFT ms"],
                            [[row["category"], row["turns"], _display(row["generation_tps"]), _display(row["prompt_tps"]), _display(row["median_internal_ttft_ns"], 1e6)] for row in item["categories"]]), "",
                     _table(["Turn position", "Turns", "Generation t/s", "Median internal TTFT ms"],
                            [[row["position"], row["turns"], _display(row["generation_tps"]), _display(row["median_internal_ttft_ns"], 1e6)] for row in item["turn_positions"]]), ""])
    text.extend(["## Cases", "", _table(["Case", "Suite", "Category", "Configuration", "Status", "Stop reasons", "Turns", "Generation t/s", "Selected tokens", "Wall reply seconds"],
                 [[item["case_id"], item["suite"], item["category"], item["configuration_id"], item["status"],
                   ", ".join(f"{reason}: {count}" for reason, count in item["metrics"]["termination_counts"].items()) or "—",
                   f"{item['recorded_turns']} / {item['planned_turns']}",
                   _display(item["metrics"]["generation_tps"]), item["metrics"]["selected_tokens"] if item["metrics"]["selected_tokens"] is not None else "—",
                   _display(item["metrics"]["wall_turn_ns"], 1e9)] for item in summary["conversations"]]), ""])
    if summary["comparisons"]:
        text.extend(["## Per-case generation rate ratios", "", "Guffin / llama.cpp; above 1 means higher measured generation rate. Modes and source-model hashes match, but natural outputs can differ.", "",
                     _table(["Case", "Mode", "Configurations", "Rate ratio"],
                            [[row["case_id"], row["mode"], f"{row['candidate_configuration_id']} / {row['baseline_configuration_id']}", _display(row["generation_rate_ratio"], suffix="×")] for row in summary["comparisons"]]), ""])
    text.extend(["## Incomplete or unattempted conversations", ""])
    if summary["failures"]:
        text.extend([_table(["Configuration", "Case", "Status", "Recorded turns", "Error"],
                            [[row["configuration_id"], row["case_id"], row["status"], row["recorded_turns"], row["error"] or "—"] for row in summary["failures"]]), ""])
    else:
        text.extend(["None.", ""])
    if summary["memory_errors"]:
        text.extend(["## Memory collection gaps", "", _table(["Configuration", "Case", "Error"],
                     [[item["configuration_id"], item["case_id"], item["error"]] for item in summary["memory_errors"]]), ""])
    text.extend(["## Measurement boundaries", ""] + ["- " + note for note in summary["notes"]] + ["", "Charts: [generation](generation.svg), [internal TTFT](ttft.svg), [memory](memory.svg). Raw turn rows: [CSV](summary.csv); complete summary: [JSON](summary.json).", ""])
    return "\n".join(text)


def render_svg(title, unit, rows):
    """Render measured bars with labels; no synthetic values or uncertainty bands."""
    width, left, right, top, spacing = 1040, 390, 120, 95, 30
    height = max(160, top + len(rows) * spacing + 60)
    plot_width = width - left - right
    maximum = max((value for _, value, _ in rows if value is not None), default=0) or 1
    escape = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>', '<desc>Single-pass chat measurements. Missing values have no bar. No uncertainty or stability estimate.</desc>',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<g font-family="system-ui, sans-serif" fill="#17212b">',
             f'<text x="24" y="31" font-size="22" font-weight="600">{escape(title)}</text>',
             f'<text x="24" y="56" font-size="13">Single pass · jointly complete conversations · {escape(unit)} · natural outputs can differ</text>']
    for tick in range(5):
        x = left + plot_width * tick / 4
        parts.extend([f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 48}" stroke="#e2e8f0"/>',
                      f'<text x="{x:.1f}" y="{height - 25}" text-anchor="middle" font-size="11">{maximum * tick / 4:.2f}</text>'])
    for index, (label, value, engine) in enumerate(rows):
        y = top + index * spacing
        color = "#087f8c" if engine == "guffin" else "#ca5d22"
        parts.append(f'<text x="{left - 12}" y="{y + 13}" text-anchor="end" font-size="12">{escape(label)}</text>')
        if value is not None:
            length = value / maximum * plot_width
            parts.extend([f'<rect x="{left}" y="{y}" width="{length:.2f}" height="18" rx="3" fill="{color}"/>',
                          f'<text x="{left + length + 8:.2f}" y="{y + 13}" font-size="12">{value:.2f}</text>'])
        else:
            parts.append(f'<text x="{left + 8}" y="{y + 13}" font-size="12">unavailable</text>')
    if not rows:
        parts.append('<text x="24" y="112" font-size="14">No jointly complete conversation measurements.</text>')
    parts.extend(['</g>', '</svg>'])
    return "\n".join(parts) + "\n"


def _write_text(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + ".", encoding="utf-8", newline="", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(summary, output):
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_text(output / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in summary["turns"])
    _write_text(output / "summary.csv", stream.getvalue())
    paired_keys = {(config_id, case_id) for pair in summary["paired_suites"] for case_id in pair["complete_case_ids"]
                   for config_id in (pair["candidate_configuration_id"], pair["baseline_configuration_id"])}
    complete = sorted((item for item in summary["conversations"] if (item["configuration_id"], item["case_id"]) in paired_keys), key=lambda item: (item["case_id"], item["mode"], item["configuration_id"]))
    generation = [(f"{item['case_id']} · {item['configuration_id']}", item["metrics"]["generation_tps"], item["engine"]) for item in complete]
    ttft = [(f"{row['case_id']} · turn {row['turn_index'] + 1} · {row['configuration_id']}", row["internal_ttft_ns"] / 1e6 if row["internal_ttft_ns"] is not None else None, row["engine"])
            for row in sorted(summary["turns"], key=lambda row: (row["case_id"], row["turn_index"], row["mode"], row["configuration_id"])) if (row["configuration_id"], row["case_id"]) in paired_keys]
    memory = [(f"{item['case_id']} · {item['configuration_id']}", item["memory"]["peak_physical_footprint_bytes"] / 2**30 if item["memory"]["peak_physical_footprint_bytes"] is not None else None, item["engine"]) for item in complete]
    for filename, title, unit, chart_rows in (("generation.svg", "Generation throughput by conversation", "tokens/s; higher is faster", generation),
                                               ("ttft.svg", "Internal time to first token by turn", "milliseconds; lower is faster", ttft),
                                               ("memory.svg", "Native lifetime peak physical footprint", "GiB; lower uses less memory", memory)):
        _write_text(output / filename, render_svg(title, unit, chart_rows))
    _write_text(output / "summary.md", render_markdown(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="report directory (default: RUN_DIRECTORY/report)")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
        receipts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.run_directory / "receipts").glob("*.json"))]
        summary = summarize_run(manifest, receipts)
        output = args.output or args.run_directory / "report"
        write_report(summary, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(output / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

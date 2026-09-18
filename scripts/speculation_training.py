#!/usr/bin/env python3
"""One entry point for immutable trace preparation and faithful training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.speculation.manifest import publish_manifests
from training.speculation.registry import load_registry


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-corpus")
    prepare.add_argument(
        "--source-registry", type=Path,
        default=Path("training/speculation/corpus_sources.json"),
    )
    prepare.add_argument(
        "--cache", type=Path,
        default=Path("traces/learned_speculation/source-cache-v1"),
    )
    prepare.add_argument(
        "--output", type=Path,
        default=Path("traces/learned_speculation/sources-v1.jsonl"),
    )
    prepare.add_argument("--conversations-per-workload", type=int, default=400)
    manifest = commands.add_parser("publish-manifests")
    manifest.add_argument("--sources", type=Path, required=True)
    manifest.add_argument("--registry", type=Path, default=Path("training/speculation/registry.json"))
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--split-salt", required=True)
    manifest.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    plan = commands.add_parser("plan")
    plan.add_argument("--registry", type=Path, default=Path("training/speculation/registry.json"))
    plan.add_argument("--family", action="append", default=[])
    plan.add_argument("--unit", action="append", default=[])
    plan.add_argument("--show-cells", action="store_true")
    plan.add_argument("--limit", type=int, default=100)
    validate = commands.add_parser("validate-traces")
    validate.add_argument("--registry", type=Path, default=Path("training/speculation/registry.json"))
    validate.add_argument("--traces", type=Path, required=True)
    validate.add_argument("--split-receipt", type=Path, required=True)
    validate.add_argument("--model", type=Path)
    validate.add_argument("--auditor", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--registry", type=Path, default=Path("training/speculation/registry.json"))
    train.add_argument("--traces", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--split-receipt", type=Path, required=True)
    train.add_argument("--run-contract", type=Path, required=True)
    train.add_argument("--auditor", type=Path)
    train.add_argument("--scale", type=int, required=True)
    train.add_argument("--family", action="append", default=[])
    train.add_argument("--unit", action="append", default=[])
    train.add_argument("--cell", action="append", default=[])
    train.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    train.add_argument("--allow-cpu-diagnostic", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "prepare-corpus":
        from training.speculation.corpus_builder import prepare_public_corpus
        receipt = prepare_public_corpus(
            arguments.source_registry,
            arguments.cache,
            arguments.output,
            arguments.conversations_per_workload,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if arguments.command == "publish-manifests":
        receipt = publish_manifests(
            arguments.sources,
            arguments.output,
            arguments.registry,
            arguments.split_salt,
            arguments.near_duplicate_threshold,
        )
        print(receipt["receipt_sha256"])
        return 0
    registry = load_registry(arguments.registry)
    if arguments.command == "plan":
        if arguments.limit <= 0:
            raise ValueError("--limit must be positive")
        unknown_families = set(arguments.family) - {
            str(unit["family"]) for unit in registry.raw["training_units"]
        }
        unknown_units = set(arguments.unit) - {
            str(unit["id"]) for unit in registry.raw["training_units"]
        }
        if unknown_families or unknown_units:
            raise ValueError(
                f"unknown families/units: {sorted(unknown_families)}, {sorted(unknown_units)}"
            )
        counts: dict[str, int] = {}
        cells = [
            cell for cell in registry.cells()
            if (not arguments.family or cell.family in set(arguments.family))
            and (not arguments.unit or cell.unit_id in set(arguments.unit))
        ]
        for cell in cells:
            counts[cell.family] = counts.get(cell.family, 0) + 1
        result = {
            "registry_sha256": registry.digest,
            "training_cell_count": sum(counts.values()),
            "cells_by_family": counts,
            "source_exact_blockers": registry.raw["source_exact_blockers"],
        }
        if arguments.show_cells:
            result["cells"] = [
                {"cell_id": cell.identifier, "unit_id": cell.unit_id,
                 "family": cell.family, "parameters": dict(cell.parameters)}
                for cell in cells[:arguments.limit]
            ]
            result["cells_truncated"] = len(cells) > arguments.limit
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "validate-traces":
        from training.speculation.corpus import validate_trace_corpus
        print(json.dumps(validate_trace_corpus(
            arguments.traces, registry, arguments.model, arguments.auditor,
            arguments.split_receipt,
        ), sort_keys=True))
        return 0
    if arguments.command == "train":
        from training.speculation.train import train_cells
        results = train_cells(
            REPOSITORY_ROOT, registry, arguments.traces, arguments.output,
            arguments.model, arguments.scale, set(arguments.family),
            set(arguments.unit), set(arguments.cell), arguments.device,
            arguments.allow_cpu_diagnostic, arguments.split_receipt, arguments.auditor,
            arguments.run_contract,
        )
        print(json.dumps({"completed_cells": len(results)}, sort_keys=True))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)

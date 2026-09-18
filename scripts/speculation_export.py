#!/usr/bin/env python3
"""Export completed recurrent training checkpoints for the native engine."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.export_recurrent import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "receipt", "registry", "run-contract", "model", "trace", "output", "split-receipt"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args.checkpoint,args.receipt,args.registry,args.run_contract,
                        args.model,args.trace,args.output,args.split_receipt)
        print(json.dumps(result,sort_keys=True))
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        print(f"speculation export: {error}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

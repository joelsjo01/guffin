#!/usr/bin/env python3
"""Capture exactly one warmed production Gemma 2 decode command buffer."""

import argparse
import os
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local", default="build/release/guffin")
    args = parser.parse_args()

    output = pathlib.Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not output.parent.is_dir():
        parser.error(f"output parent does not exist: {output.parent}")
    environment = os.environ.copy()
    environment["MTL_CAPTURE_ENABLED"] = "1"
    command = [
        args.local,
        args.model,
        "--interactive",
        "--backend",
        "metal",
        "--greedy",
        "--max-new-tokens",
        "2",
        "--decode-diagnostics",
        "--metal-capture",
        str(output),
    ]
    result = subprocess.run(
        command,
        input="Explain why the sky is blue.\n/exit\n",
        text=True,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    artifact_nonempty = output.is_file() and output.stat().st_size > 0
    if output.is_dir():
        artifact_nonempty = next(output.iterdir(), None) is not None
    if not artifact_nonempty:
        raise SystemExit("capture command succeeded without a nonempty .gputrace artifact")
    if f"[ Metal capture: {output} ]" not in result.stdout:
        raise SystemExit("capture command did not report one completed production decode capture")
    print(output)


if __name__ == "__main__":
    main()

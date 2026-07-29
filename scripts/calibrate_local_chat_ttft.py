#!/usr/bin/env python3
"""Calibrate in-process TTFT against bytes observed from the real chat CLI."""

import argparse
import json
import os
import pty
import re
import select
import subprocess
import termios
import time


TTFT_PATTERN = re.compile(rb"Frontend TTFT: ([0-9]+) ns")


def read_until(master, marker, timeout):
    deadline = time.monotonic() + timeout
    data = bytearray()
    while marker not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"timed out waiting for {marker!r}")
        readable, _, _ = select.select([master], [], [], remaining)
        if not readable:
            continue
        chunk = os.read(master, 65536)
        if not chunk:
            raise RuntimeError("chat CLI closed before calibration completed")
        data.extend(chunk)
    return bytes(data)


def parse_frontend_ttft(output):
    match = TTFT_PATTERN.search(output)
    if match is None:
        raise RuntimeError("chat CLI omitted in-process frontend TTFT")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="build/release/local-gguf-generate")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.repetitions <= 0:
        parser.error("token limit and repetitions must be positive")

    master, slave = pty.openpty()
    attributes = termios.tcgetattr(slave)
    attributes[3] &= ~termios.ECHO
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    command = [
        args.binary, args.model, "--chat", "--greedy",
        "--max-new-tokens", str(args.max_new_tokens), "--backend", "metal",
    ]
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    try:
        read_until(master, b"> ", args.timeout)
        for repetition in range(args.repetitions):
            if repetition:
                os.write(master, b"/clear\n")
                read_until(master, b"> ", args.timeout)
            started = time.monotonic_ns()
            os.write(master, args.prompt.encode("utf-8") + b"\n")
            readable, _, _ = select.select([master], [], [], args.timeout)
            if not readable:
                raise RuntimeError("timed out waiting for first assistant bytes")
            first_observed = time.monotonic_ns()
            first_chunk = os.read(master, 65536)
            remainder = read_until(master, b"> ", args.timeout)
            full_output = first_chunk + remainder
            in_process = parse_frontend_ttft(full_output)
            external = first_observed - started
            print(json.dumps({
                "schema_version": 4,
                "record_kind": "local_frontend_calibration",
                "repetition_index": repetition,
                "prompt": args.prompt,
                "generated_token_limit": args.max_new_tokens,
                "external_observed_ttft_ns": external,
                "in_process_frontend_complete_ttft_ns": in_process,
                "external_minus_in_process_ns": external - in_process,
                "command": command,
            }, separators=(",", ":")), flush=True)
        os.write(master, b"/exit\n")
    finally:
        os.close(master)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare real chat startup and process memory without changing inference."""

import argparse
import json
import os
import re
import resource
import select
import statistics
import subprocess
import sys
import time


SCHEMA_VERSION = 1
DECODE_RECORD = re.compile(rb"^\[ Decode: (?P<json>\{.*\}) \]$", re.MULTILINE)


def parse_ps_snapshot(output):
    fields = output.split()
    if len(fields) != 2:
        raise RuntimeError("ps did not return RSS and virtual size")
    rss_kib, virtual_kib = (int(value) for value in fields)
    if rss_kib <= 0 or virtual_kib <= 0:
        raise RuntimeError("ps returned nonpositive process memory")
    return {"ready_rss_bytes": rss_kib * 1024, "ready_virtual_bytes": virtual_kib * 1024}


def parse_decode_record(output, require_stage1=True):
    matches = list(DECODE_RECORD.finditer(output))
    if len(matches) != 1:
        raise RuntimeError("chat did not emit exactly one decode attribution record")
    record = json.loads(matches[0].group("json"))
    if not require_stage1:
        return record
    for name in (
        "resident_weights_bytes", "resident_kv_bytes", "resident_scratch_bytes",
        "warmup_kv_clear_passes", "warmup_host_cleared_kv_bytes",
    ):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"decode attribution {name} is invalid")
    if record["warmup_kv_clear_passes"] != 0 or record["warmup_host_cleared_kv_bytes"] != 0:
        raise RuntimeError("decode attribution reports a warm-up KV clear")
    return record


def _read_ready(streams, stdout, stderr, marker, deadline):
    while marker not in stdout:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("timed out waiting for chat readiness")
        ready, _, _ = select.select(list(streams), [], [], min(remaining, 1.0))
        for descriptor in ready:
            stream, destination = streams[descriptor]
            chunk = os.read(descriptor, 65536)
            if not chunk:
                raise RuntimeError("chat exited before becoming ready")
            destination.extend(chunk)


def _drain_process(process, streams, stdout, stderr, deadline):
    sent_exit = False
    while streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            raise RuntimeError("timed out waiting for memory benchmark chat")
        ready, _, _ = select.select(list(streams), [], [], min(remaining, 1.0))
        for descriptor in ready:
            stream, destination = streams[descriptor]
            chunk = os.read(descriptor, 65536)
            if not chunk:
                stream.close()
                del streams[descriptor]
                continue
            destination.extend(chunk)
        if not sent_exit and DECODE_RECORD.search(stdout):
            process.stdin.write(b"/exit\n")
            process.stdin.close()
            sent_exit = True
    pid, status, usage = os.wait4(process.pid, 0)
    if pid != process.pid:
        raise RuntimeError("wait4 returned an unexpected process")
    process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != 0:
        diagnostic = (stderr or stdout)[-2000:].decode(errors="replace")
        raise RuntimeError(f"chat exited with {process.returncode}:\n{diagnostic}")
    return usage


def measure(
    binary, model, require_stage1=True, output_head_policy=None,
    ps_runner=subprocess.run, timeout=900,
):
    started = time.monotonic_ns()
    command = [binary, model, "--chat", "--max-new-tokens", "1", "--greedy", "--decode-diagnostics"]
    if output_head_policy is not None:
        command.extend(("--output-head", output_head_policy))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = bytearray()
    stderr = bytearray()
    streams = {
        process.stdout.fileno(): (process.stdout, stdout),
        process.stderr.fileno(): (process.stderr, stderr),
    }
    deadline = time.monotonic() + timeout
    try:
        _read_ready(streams, stdout, stderr, b"> ", deadline)
        ready_ns = time.monotonic_ns()
        ps_result = ps_runner(
            ["ps", "-o", "rss=,vsz=", "-p", str(process.pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        snapshot = parse_ps_snapshot(ps_result.stdout)
        process.stdin.write(b"Hello\n")
        process.stdin.flush()
        usage = _drain_process(process, streams, stdout, stderr, deadline)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise

    attribution = parse_decode_record(bytes(stdout), require_stage1=require_stage1)
    return {
        "startup_to_ready_ns": ready_ns - started,
        **snapshot,
        "peak_rss_bytes": usage.ru_maxrss,
        "page_reclaims": usage.ru_minflt,
        "page_faults": usage.ru_majflt,
        "block_input_operations": usage.ru_inblock,
        "block_output_operations": usage.ru_oublock,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "decode_attribution": attribution,
    }


def summarize(samples):
    fields = ("startup_to_ready_ns", "ready_rss_bytes", "peak_rss_bytes", "page_reclaims", "page_faults")
    return {f"median_{name}": statistics.median(sample[name] for sample in samples) for name in fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", default="build/release/local-gguf-generate")
    parser.add_argument("--baseline-output-head", choices=("f16", "q8_0"))
    parser.add_argument("--candidate-output-head", choices=("f16", "q8_0"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.cooldown < 0:
        parser.error("repetitions must be positive and cooldown must be nonnegative")

    samples = {"baseline": [], "candidate": []}
    order = []
    for repetition in range(args.repetitions):
        pair = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
        for name in pair:
            if order and args.cooldown:
                print(f"cooling down for {args.cooldown:g}s...", file=sys.stderr)
                time.sleep(args.cooldown)
            binary = args.baseline if name == "baseline" else args.candidate
            print(f"measuring {name} run {repetition + 1}...", file=sys.stderr)
            policy = (
                args.baseline_output_head if name == "baseline"
                else args.candidate_output_head
            )
            sample = measure(
                binary, args.model, require_stage1=name == "candidate",
                output_head_policy=policy,
            )
            samples[name].append(sample)
            order.append(name)

    result = {
        "schema_version": SCHEMA_VERSION,
        "model": os.path.realpath(args.model),
        "executables": {
            "baseline": os.path.realpath(args.baseline),
            "candidate": os.path.realpath(args.candidate),
        },
        "protocol": {
            "repetitions": args.repetitions,
            "cooldown_seconds": args.cooldown,
            "order": order,
            "prompt": "Hello",
            "max_new_tokens": 1,
            "selection": "greedy",
            "output_head_policies": {
                "baseline": args.baseline_output_head,
                "candidate": args.candidate_output_head,
            },
            "filesystem_cache": "uncontrolled",
        },
        "samples": samples,
        "summary": {name: summarize(values) for name, values in samples.items()},
    }
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
        return
    for name in ("baseline", "candidate"):
        summary = result["summary"][name]
        print(
            f"{name:<9} startup={summary['median_startup_to_ready_ns'] / 1e9:.3f}s "
            f"ready-rss={summary['median_ready_rss_bytes'] / 2**30:.3f}GiB "
            f"peak-rss={summary['median_peak_rss_bytes'] / 2**30:.3f}GiB "
            f"reclaims={summary['median_page_reclaims']:.0f} faults={summary['median_page_faults']:.0f}"
        )


if __name__ == "__main__":
    main()

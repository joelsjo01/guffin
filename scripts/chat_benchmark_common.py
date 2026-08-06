"""Shared real-chat process driver for local and reference benchmarks."""

import os
import re
import select
import subprocess
import sys
import time


PROMPTS = (
    "Hello, is AI getting smarter?",
    "Okay, so will it always progress this fast?",
)
TURN_NAMES = ("First", "Follow-up")
SEED = 42
INTER_TURN_PAUSE_SECONDS = 5.0
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def clean_output(text):
    return ANSI_ESCAPE.sub("", text).replace("\r", "")


def run_process(
    command,
    completion_marker,
    timeout=900,
    inter_turn_pause=INTER_TURN_PAUSE_SECONDS,
):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.stdin.write((PROMPTS[0] + "\n").encode())
    process.stdin.flush()
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + timeout
    terminal_rendered = sys.stderr.isatty()
    if terminal_rendered:
        sys.stderr.write("\x1b[?1049h\x1b[2J\x1b[H")
        sys.stderr.write(PROMPTS[0] + "\n")
        sys.stderr.flush()
    handled_completions = 0
    try:
        streams = {
            process.stdout.fileno(): (process.stdout, stdout, True),
            process.stderr.fileno(): (process.stderr, stderr, False),
        }
        while streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                raise RuntimeError(f"timed out waiting for natural stop: {command[0]}")
            ready, _, _ = select.select(list(streams), [], [], min(remaining, 1.0))
            for descriptor in ready:
                stream, destination, is_chat_stdout = streams[descriptor]
                chunk = os.read(descriptor, 4096)
                if not chunk:
                    stream.close()
                    del streams[descriptor]
                    continue
                destination.extend(chunk)
                if terminal_rendered and is_chat_stdout:
                    sys.stderr.buffer.write(chunk)
                    sys.stderr.flush()
                if is_chat_stdout:
                    completion_count = stdout.count(completion_marker.encode())
                    while handled_completions < completion_count:
                        handled_completions += 1
                        if handled_completions < len(PROMPTS):
                            time.sleep(inter_turn_pause)
                            next_prompt = PROMPTS[handled_completions]
                            if terminal_rendered:
                                sys.stderr.write(next_prompt + "\n")
                                sys.stderr.flush()
                            process.stdin.write((next_prompt + "\n").encode())
                            process.stdin.flush()
                        elif handled_completions == len(PROMPTS):
                            process.stdin.write(b"/exit\n")
                            process.stdin.close()
        return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if not process.stdout.closed:
            process.stdout.close()
        if not process.stderr.closed:
            process.stderr.close()
        if terminal_rendered:
            sys.stderr.write("\x1b[?1049l")
            sys.stderr.flush()
    cleaned_stdout = clean_output(stdout.decode(errors="replace"))
    cleaned_stderr = clean_output(stderr.decode(errors="replace"))
    if return_code != 0:
        diagnostic = cleaned_stderr or cleaned_stdout
        raise RuntimeError(f"{command[0]} exited with {return_code}:\n{diagnostic[-2000:]}")
    if handled_completions != len(PROMPTS):
        raise RuntimeError(
            f"{command[0]} exited after {handled_completions} of {len(PROMPTS)} completed turns"
        )
    return cleaned_stdout, cleaned_stderr


def checked_rate(count, duration_ns, displayed, label):
    calculated = count * 1e9 / duration_ns
    if abs(calculated - displayed) > 0.11:
        raise RuntimeError(
            f"{label} raw counters imply {calculated:.3f} t/s, not displayed {displayed:.1f} t/s"
        )


def print_turn_table(engine, turns):
    print(
        "Engine             Turn       Eval  Cache  Decode  TTFT ms  Prompt t/s  Gen t/s"
        "  Prompt ms  Gen ms"
    )
    for turn in turns:
        print(
            f"{engine:<18} {turn['name']:<9} {turn['evaluated']:>5} {turn['cached']:>6}"
            f" {turn['decoded']:>7} {turn['ttft_ns'] / 1e6:>8.1f}"
            f" {turn['prompt_tps']:>11.1f} {turn['generation_tps']:>8.1f}"
            f" {turn['prompt_ns'] / 1e6:>10.1f} {turn['generation_ns'] / 1e6:>7.1f}"
        )

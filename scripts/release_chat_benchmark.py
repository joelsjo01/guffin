#!/usr/bin/env python3
"""Collect one pass of default CLI conversations, with resumable raw receipts."""

import argparse
import codecs
from contextlib import ExitStack
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import tempfile
import time
import uuid

from release_chat_metrics import (
    GuffinMetricsParser, LlamaMetricsParser, clean_terminal_text,
    encode_prompt, extract_answer, is_prompt_ready,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmarks/chat/corpus.json"
SCHEMA_VERSION = 1


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    info = path.stat()
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": info.st_size}


def command_text(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=30, check=True)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def executable_identity(name):
    resolved = shutil.which(name)
    if resolved is None:
        raise ValueError(f"executable not found: {name}")
    identity = file_identity(resolved)
    identity["version"] = command_text([resolved, "--version"])
    identity["help"] = command_text([resolved, "--help"])
    if not identity["version"] or not identity["help"]:
        raise ValueError(f"cannot identify CLI version and defaults: {resolved}")
    # Homebrew can link its ggml dependency separately from the llama binary.
    if platform.system() == "Darwin":
        identity["linked_libraries"] = command_text(["otool", "-L", str(resolved)])
        identity["homebrew_versions"] = command_text(["brew", "list", "--versions", "llama.cpp", "ggml"])
    return identity


def system_snapshot():
    if platform.system() != "Darwin":
        return {}
    return {
        "power": command_text(["pmset", "-g", "batt"]),
        "thermal_report": command_text(["pmset", "-g", "therm"]),
        "swap": command_text(["sysctl", "vm.swapusage"]),
        "vm_stat": command_text(["vm_stat"]),
    }


def collect_environment():
    environment = {
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(),
        "source_commit": command_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "source_status": command_text(["git", "-C", str(ROOT), "status", "--short"]),
        "inference_environment": {
            key: value for key, value in os.environ.items()
            if key.startswith("LLAMA_ARG_")
            and not any(secret in key for secret in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
        },
    }
    if platform.system() == "Darwin":
        environment.update(
            macos=command_text(["sw_vers"]),
            hardware=command_text(["sysctl", "hw.model", "hw.memsize", "hw.ncpu"]),
            power_settings=command_text(["pmset", "-g", "custom"]),
        )
    return environment


def load_cases(path, suite="all", selected_ids=None):
    raw = Path(path).read_bytes()
    corpus = json.loads(raw)
    if corpus.get("schema_version") != 1 or not isinstance(corpus.get("conversations"), list):
        raise ValueError("unsupported chat corpus")
    cases, seen = [], set()
    for case in corpus["conversations"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("corpus IDs must be unique nonempty strings")
        seen.add(case_id)
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t for t in turns):
            raise ValueError(f"invalid conversation turns: {case_id}")
        for prompt in turns:
            # Both adapters must preserve exactly the same user text.
            encode_prompt("guffin", prompt)
            encode_prompt("llama", prompt)
        if suite == "core" and case.get("suite") != "core":
            continue
        if selected_ids and case_id not in selected_ids:
            continue
        cases.append(case)
    if selected_ids and set(selected_ids) - {case["id"] for case in cases}:
        raise ValueError("requested case is absent from the selected suite")
    if not cases:
        raise ValueError("the selected corpus is empty")
    return corpus, cases, hashlib.sha256(raw).hexdigest()


def build_configurations(guffin, llama, model, draft=None):
    configurations = []
    mode = "speculative" if draft else "default"
    for engine, executable in (("guffin", guffin), ("llama", llama)):
        command = [executable["path"], "-m", model["path"]]
        if draft:
            command.extend(["--draft-model" if engine == "guffin" else "--model-draft", draft["path"]])
            if engine == "llama":
                # Current llama-cli loads a drafter without enabling it.
                command.extend(["--spec-type", "draft-simple"])
        # These observe the real chat; no seed, sampler, context, output limit,
        # verifier horizon, offload or warmup setting is overridden.
        measurement_options = (["--decode-diagnostics"] if engine == "guffin"
                               else ["--log-verbosity", "4"])
        configurations.append({
            "id": f"{engine}-{mode}", "engine": engine, "mode": mode,
            "model_path": model["path"], "model_sha256": model["sha256"],
            "draft_sha256": draft["sha256"] if draft else None,
            "executable": executable, "command": command + measurement_options,
            "measurement_options": measurement_options,
        })
    return configurations


def build_jobs(cases, configurations):
    jobs = []
    for index, case in enumerate(cases):
        # Each case is a block. Rotate the first configuration so one engine is
        # not always earlier in the campaign. This schedule contains no repeats.
        shift = index % len(configurations)
        order = configurations[shift:] + configurations[:shift]
        for configuration in order:
            identity = {"configuration_id": configuration["id"], "case_id": case["id"]}
            identity["run_id"] = canonical_hash(identity)[:20]
            jobs.append(identity)
    return jobs


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def prepare_manifest(directory, specification, environment):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    contract = canonical_hash(specification)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("contract_sha256") != contract:
            raise ValueError("existing campaign has different models, binaries, corpus or timing policy; use a new output directory")
        previous = manifest.get("environment", {})
        for field in ("platform", "hardware", "power_settings", "inference_environment"):
            if previous.get(field) != environment.get(field):
                raise ValueError(f"{field} changed; use a new output directory")
        (directory / "receipts").mkdir(exist_ok=True)
        (directory / "logs").mkdir(exist_ok=True)
        return manifest
    if any(directory.iterdir()):
        raise ValueError("output directory is nonempty and has no benchmark manifest")
    manifest = {
        "schema_version": SCHEMA_VERSION, "campaign_id": str(uuid.uuid4()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_sha256": contract, "environment": environment, **specification,
    }
    atomic_json(path, manifest)
    (directory / "receipts").mkdir()
    (directory / "logs").mkdir()
    return manifest


def completed_jobs(directory, manifest):
    planned = {job["run_id"]: job for job in manifest["jobs"]}
    completed = {}
    configurations = {config["id"]: config for config in manifest["configurations"]}
    cases = {case["id"]: case for case in manifest["cases"]}
    for path in sorted((Path(directory) / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        if not isinstance(receipt, dict):
            raise ValueError(f"invalid conversation receipt: {path}")
        job = planned.get(receipt.get("run_id"))
        if job is None or receipt.get("campaign_id") != manifest["campaign_id"]:
            raise ValueError(f"receipt does not belong to this campaign: {path}")
        case = cases[job["case_id"]]
        if (receipt.get("schema_version") != SCHEMA_VERSION
                or receipt.get("configuration") != configurations[job["configuration_id"]]
                or receipt.get("case") != case):
            raise ValueError(f"receipt does not match its planned conversation: {path}")
        turns = receipt.get("turns")
        if (not isinstance(turns, list) or len(turns) > len(case["turns"])
                or any(not isinstance(turn, dict) or turn.get("index") != index or turn.get("prompt") != case["turns"][index]
                       for index, turn in enumerate(turns))
                or (receipt.get("status") == "complete" and len(turns) != len(case["turns"]))):
            raise ValueError(f"receipt has inconsistent conversation turns: {path}")
        if receipt.get("status") not in ("complete", "failed", "interrupted"):
            raise ValueError(f"invalid receipt status: {path}")
        if receipt["run_id"] in completed:
            raise ValueError("duplicate conversation receipt")
        completed[receipt["run_id"]] = receipt
    return completed


class ProcessMemory:
    """macOS process footprint; RSS and footprint remain distinct measures."""

    class RUsageV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16),
                    ("values", ctypes.c_uint64 * 35)]

    def __init__(self, pid):
        self.pid = pid
        self.last = None
        self.peak_physical_footprint_bytes = None
        self.peak_rss_bytes = None
        self.samples = 0
        self.error = None
        self.library = None
        if platform.system() == "Darwin":
            try:
                self.library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                self.library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                self.library.proc_pid_rusage.restype = ctypes.c_int
            except OSError as error:
                self.error = str(error)
        else:
            self.error = "physical footprint collection requires macOS"

    def sample(self):
        if self.library is None:
            return None
        usage = self.RUsageV4()
        if self.library.proc_pid_rusage(self.pid, 4, ctypes.byref(usage)) != 0:
            self.error = f"proc_pid_rusage: errno {ctypes.get_errno()}"
            return None
        fields = usage.values
        self.last = {
            "physical_footprint_bytes": fields[7], "resident_bytes": fields[6],
            "lifetime_peak_physical_footprint_bytes": fields[28], "pageins": fields[4],
        }
        self.peak_physical_footprint_bytes = max(self.peak_physical_footprint_bytes or 0,
                                                fields[7], fields[28])
        self.peak_rss_bytes = max(self.peak_rss_bytes or 0, fields[6])
        self.samples += 1
        return self.last

    def summary(self):
        return {"peak_physical_footprint_bytes": self.peak_physical_footprint_bytes,
                "peak_rss_bytes": self.peak_rss_bytes, "sample_count": self.samples,
                "rss_basis": "maximum sampled resident size",
                "footprint_basis": "macOS lifetime maximum physical footprint",
                "error": self.error}


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


class ChatProcess:
    """Drive actual terminal chat while continuously draining both output streams."""

    def __init__(self, configuration, log_prefix, timeout):
        self.configuration = configuration
        self.timeout = timeout
        self.parser = GuffinMetricsParser() if configuration["engine"] == "guffin" else LlamaMetricsParser()
        self.stdout = ""
        self.stderr = ""
        self.streams = {}
        self.decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in ("stdout", "stderr")}
        # A failed terminal/process setup must release every acquired resource.
        # Successful construction transfers their lifetime to close().
        with ExitStack() as cleanup:
            self.logs = {
                name: cleanup.enter_context(Path(str(log_prefix) + f".{name}.log").open("wb"))
                for name in ("stdout", "stderr")
            }
            master, slave = pty.openpty()
            cleanup.callback(os.close, master)
            try:
                settings = termios.tcgetattr(slave)
                # Canonical terminals truncate long lines. The CLIs own line
                # assembly; disabling kernel echo avoids duplicating UI echo.
                settings[3] &= ~(termios.ICANON | termios.ECHO)
                settings[6][termios.VMIN] = 1
                settings[6][termios.VTIME] = 0
                termios.tcsetattr(slave, termios.TCSANOW, settings)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
                self.started_ns = time.monotonic_ns()
                self.process = subprocess.Popen(configuration["command"], stdin=slave,
                                                stdout=slave, stderr=subprocess.PIPE,
                                                start_new_session=True)
                cleanup.callback(self.process.stderr.close)
                cleanup.callback(stop_process, self.process)
            finally:
                os.close(slave)
            self.master = master
            os.set_blocking(master, False)
            os.set_blocking(self.process.stderr.fileno(), False)
            self.streams = {master: "stdout", self.process.stderr.fileno(): "stderr"}
            self.memory = ProcessMemory(self.process.pid)
            self.next_memory_sample = 0.0
            self.active_turn = False
            cleanup.pop_all()

    def pump(self, timeout=0.1, outgoing=None):
        writers = [self.master] if outgoing else []
        ready, writable, _ = select.select(list(self.streams), writers, [], timeout)
        written = 0
        if writable:
            try:
                written = os.write(self.master, outgoing[:8192])
            except BlockingIOError:
                pass
        for descriptor in ready:
            try:
                data = os.read(descriptor, 65536)
            except BlockingIOError:
                continue
            except OSError as error:
                if error.errno != errno.EIO or descriptor != self.master:
                    raise
                data = b""
            stream = self.streams[descriptor]
            if not data:
                del self.streams[descriptor]
                continue
            self.logs[stream].write(data)
            text = self.decoders[stream].decode(data)
            if stream == "stdout":
                self.stdout += text
            else:
                self.stderr += text
            if self.active_turn:
                self.parser.feed(stream, text)
        now = time.monotonic()
        if now >= self.next_memory_sample:
            self.memory.sample()
            self.next_memory_sample = now + 1.0
        return written

    def wait_ready(self):
        deadline = time.monotonic() + self.timeout
        while not is_prompt_ready(self.stdout):
            if self.process.poll() is not None:
                raise RuntimeError(f"chat exited before readiness: {self.stderr[-1500:]}")
            if time.monotonic() >= deadline:
                raise TimeoutError("chat startup timed out")
            self.pump()
        self.ready_ns = time.monotonic_ns()
        self.memory.sample()
        return self.ready_ns - self.started_ns

    def turn(self, prompt, index, previous_completed_ns=None):
        outgoing = encode_prompt(self.configuration["engine"], prompt)
        stdout_start, stderr_start = len(self.stdout), len(self.stderr)
        started_ns = time.monotonic_ns()
        self.active_turn = True
        deadline = time.monotonic() + self.timeout
        pending = outgoing
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"turn {index + 1} timed out; no output limit was imposed")
            sent = self.pump(outgoing=pending)
            pending = pending[sent:]
            turn_stdout = self.stdout[stdout_start:]
            if not pending and self.parser.can_finish_turn and is_prompt_ready(turn_stdout):
                # Drain already-arrived diagnostics before sealing optional
                # acceptance statistics. Completion requires native timings AND
                # the CLI's next input prompt, never a substring in the answer.
                self.pump(timeout=0)
                completed_ns = time.monotonic_ns()
                metrics = self.parser.end_turn()
                self.active_turn = False
                metrics["wall_turn_ns"] = completed_ns - started_ns
                self.memory.sample()
                answer = extract_answer(self.configuration["engine"], self.stdout[stdout_start:], prompt)
                return {
                    "index": index, "prompt": prompt, "answer": answer,
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    "metrics": metrics, "memory": {**(self.memory.last or {}), **self.memory.summary()},
                    "started_monotonic_ns": started_ns,
                    "completed_monotonic_ns": completed_ns,
                    "inter_turn_idle_ns": (started_ns - previous_completed_ns) if previous_completed_ns is not None else None,
                    "terminal_text": clean_terminal_text(self.stdout[stdout_start:]),
                }
            diagnostic = clean_terminal_text(turn_stdout + self.stderr[stderr_start:])
            if self.process.poll() is not None:
                raise RuntimeError(f"chat exited during turn {index + 1}: {diagnostic[-1500:]}")

    def idle(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("chat exited during the requested inter-turn pause")
            self.pump(timeout=min(0.25, max(0, deadline - time.monotonic())))

    def close(self):
        self.active_turn = False
        try:
            if self.process.poll() is None:
                outgoing = b"/exit\n"
                deadline = time.monotonic() + 5
                while self.process.poll() is None and time.monotonic() < deadline:
                    sent = self.pump(outgoing=outgoing)
                    outgoing = outgoing[sent:]
                stop_process(self.process)
            drain_deadline = time.monotonic() + 5
            while self.streams and time.monotonic() < drain_deadline:
                self.pump(timeout=0.1)
        finally:
            os.close(self.master)
            self.process.stderr.close()
            for log in self.logs.values():
                log.close()


def run_conversation(configuration, case, job, directory, campaign_id, turn_pause, timeout):
    prefix = Path(directory) / "logs" / job["run_id"]
    receipt = {
        "schema_version": SCHEMA_VERSION, "campaign_id": campaign_id, "run_id": job["run_id"],
        "configuration": configuration, "case": case, "status": "failed", "turns": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logs": {stream: str(Path("logs") / (job["run_id"] + f".{stream}.log")) for stream in ("stdout", "stderr")},
        "system_before": system_snapshot(),
    }
    receipt["error"] = "unfinished attempt; a hard stop must not trigger an automatic repeat"
    atomic_json(Path(directory) / "receipts" / (job["run_id"] + ".json"), receipt)
    chat = None
    try:
        chat = ChatProcess(configuration, prefix, timeout)
        receipt["startup_to_ready_ns"] = chat.wait_ready()
        previous_completed_ns = None
        for index, prompt in enumerate(case["turns"]):
            if index:
                chat.idle(turn_pause)
            turn = chat.turn(prompt, index, previous_completed_ns)
            receipt["turns"].append(turn)
            previous_completed_ns = turn["completed_monotonic_ns"]
            # Persist completed turns as they happen; failure or interruption can
            # never erase a preceding answer or masquerade as natural completion.
            atomic_json(Path(directory) / "receipts" / (job["run_id"] + ".json"), receipt)
        chat.parser.finish()
        receipt["status"] = "complete"
        receipt.pop("error", None)
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["error"] = "interrupted by user; this conversation will not be automatically repeated"
        raise
    except (OSError, RuntimeError, ValueError) as error:
        receipt["error"] = str(error)
    finally:
        if chat is not None:
            receipt["memory"] = chat.memory.summary()
            receipt["partial_terminal_text"] = clean_terminal_text(chat.stdout) if receipt["status"] != "complete" else None
            chat.close()
        receipt["system_after"] = system_snapshot()
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(Path(directory) / "receipts" / (job["run_id"] + ".json"), receipt)
    return receipt


def sleep_between_conversations(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def run_campaign(directory, manifest):
    completed = completed_jobs(directory, manifest)
    configurations = {c["id"]: c for c in manifest["configurations"]}
    cases = {case["id"]: case for case in manifest["cases"]}
    policy = manifest["protocol"]
    for number, job in enumerate(manifest["jobs"], 1):
        if job["run_id"] in completed:
            continue
        print(f"[{number}/{len(manifest['jobs'])}] {job['configuration_id']} / {job['case_id']}", flush=True)
        print(f"  waiting {policy['conversation_pause_seconds']:g}s before starting the conversation", flush=True)
        sleep_between_conversations(policy["conversation_pause_seconds"])
        receipt = run_conversation(
            configurations[job["configuration_id"]], cases[job["case_id"]], job, directory,
            manifest["campaign_id"], policy["turn_pause_seconds"], policy["timeout_seconds"],
        )
        completed[job["run_id"]] = receipt
        print(f"  {receipt['status']}: {len(receipt['turns'])}/{len(receipt['case']['turns'])} turns", flush=True)
        if receipt["status"] != "complete":
            raise RuntimeError(f"conversation failed and was saved: {receipt.get('error')}; fix the issue before resuming")
    return completed


def interrupt_on_termination(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True, help="same local target GGUF for both CLIs")
    parser.add_argument("--draft-model", help="compare speculative chat only; omit for ordinary chat")
    parser.add_argument("-o", "--output", required=True, type=Path, help="campaign directory; rerun this command to resume")
    parser.add_argument("--guffin", default="guffin", help="installed CLI or explicit release executable")
    parser.add_argument("--llama", default="llama-cli", help="installed llama-cli")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--suite", choices=("all", "core"), default="all")
    parser.add_argument("--case", action="append", help="select a corpus case ID; repeat this option to select more")
    parser.add_argument("--conversation-pause", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument("--turn-pause", type=float, default=120.0, metavar="SECONDS")
    parser.add_argument("--timeout", type=float, default=900.0, metavar="SECONDS", help="startup/turn safety deadline, not a token limit")
    parser.add_argument("--plan", action="store_true", help="show the schedule without launching models or writing results")
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) for value in (args.conversation_pause, args.turn_pause, args.timeout)) or args.conversation_pause < 0 or args.turn_pause < 0 or args.timeout <= 0:
        parser.error("pauses must be nonnegative and timeout positive")
    directory = args.output.expanduser().resolve()
    try:
        corpus, cases, corpus_sha256 = load_cases(args.corpus, args.suite, args.case)
        guffin, llama = executable_identity(args.guffin), executable_identity(args.llama)
        model = file_identity(args.model)
        draft = file_identity(args.draft_model) if args.draft_model else None
        configurations = build_configurations(guffin, llama, model, draft)
        jobs = build_jobs(cases, configurations)
        specification = {
            "corpus": {"path": str(args.corpus.resolve()), "sha256": corpus_sha256,
                       "description": corpus.get("description"), "sources": corpus.get("sources")},
            "model": model, "draft_model": draft, "cases": cases,
            "configurations": configurations, "jobs": jobs,
            "measurement_scripts": {
                name: file_identity(Path(__file__).with_name(name))["sha256"]
                for name in ("release_chat_benchmark.py", "release_chat_metrics.py")
            },
            "protocol": {
                "comparison": "default CLI chat; measurement logging only",
                "passes": 1, "seed": "each CLI's unmodified random default",
                "conversation_pause_seconds": args.conversation_pause,
                "turn_pause_seconds": args.turn_pause, "timeout_seconds": args.timeout,
                "fresh_process_per_conversation": True, "warmup": "each CLI's default",
                "output_limit": "each CLI's default; no benchmark token cap",
                "internal_ttft": "after tokenization through first target token selection",
                "generation_rate": "(selected tokens - 1) / first-to-last selection duration",
                "terminal": "PTY input/output, kernel echo disabled, native CLI UI unchanged",
                "memory": "macOS process physical footprint; includes target and draft",
                "scope": "same user messages; sampled outputs and follow-up histories may differ",
            },
        }
        if args.plan:
            waits = len(jobs) * args.conversation_pause + sum(len(case["turns"]) - 1 for case in cases) * len(configurations) * args.turn_pause
            print(json.dumps({"conversations": len(cases), "user_turns_per_configuration": sum(len(c["turns"]) for c in cases),
                              "configurations": [{"id": c["id"], "command": c["command"]} for c in configurations],
                              "schedule": [case["id"] + ": " + ", ".join(
                                  job["configuration_id"] for job in jobs if job["case_id"] == case["id"])
                                  for case in cases],
                              "sleep_hours": waits / 3600, "protocol": specification["protocol"]}, indent=2))
            return 0
        environment = collect_environment()
        directory.mkdir(parents=True, exist_ok=True)
        # A kernel-backed lock prevents two campaigns competing on this Air;
        # stale PID files are unnecessary and cannot block a resumed campaign.
        lock_path = Path(tempfile.gettempdir()) / f"guffin-release-chat-{os.getuid()}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another benchmark campaign is already running") from error
            manifest = prepare_manifest(directory, specification, environment)
            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_termination)
            try:
                outcomes = run_campaign(directory, manifest)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
                from release_chat_report import main as report_main
                report_status = report_main([str(directory)])
            if report_status != 0:
                raise RuntimeError("campaign report could not be generated; raw receipts are preserved")
            if any(receipt["status"] != "complete" for receipt in outcomes.values()):
                raise RuntimeError("campaign contains failed or interrupted conversations; report marks them incomplete")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed/attempted conversations are saved; the same command resumes pending cases.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

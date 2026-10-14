import collections
import contextlib
import copy
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("release_chat_benchmark", SCRIPTS / "release_chat_benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def configurations(speculative=False):
    guffin = {"path": "/fixture/guffin", "sha256": "g" * 64}
    llama = {"path": "/fixture/llama-cli", "sha256": "l" * 64}
    target = {"path": "/fixture/target.gguf", "sha256": "t" * 64}
    draft = {"path": "/fixture/draft.gguf", "sha256": "d" * 64}
    return BENCH.build_configurations(guffin, llama, target, draft if speculative else None)


def specification(case_count=2, speculative=False):
    cases = [{"id": f"case-{index}", "category": "conversation", "turns": ["First", "Follow-up"]}
             for index in range(case_count)]
    configs = configurations(speculative)
    return {"cases": cases, "configurations": configs,
            "jobs": BENCH.build_jobs(cases, configs),
            "model": {"sha256": "target-v1"},
            "corpus": {"sha256": "corpus-v1"},
            "measurement_scripts": {"runner": "runner-v1", "parser": "parser-v1"},
            "protocol": {"conversation_pause_seconds": 300.0,
                         "turn_pause_seconds": 120.0, "timeout_seconds": 900.0}}


def environment():
    return {"platform": "fixture-os", "hardware": "fixture-hardware",
            "power_settings": "fixture-power", "inference_environment": {}}


def receipt_for(manifest, job, status="complete"):
    configuration = next(config for config in manifest["configurations"]
                         if config["id"] == job["configuration_id"])
    case = next(case for case in manifest["cases"] if case["id"] == job["case_id"])
    turns = [{"index": index, "prompt": prompt, "answer": "fixture"}
             for index, prompt in enumerate(case["turns"])] if status == "complete" else []
    return {"schema_version": 1, "campaign_id": manifest["campaign_id"],
            "run_id": job["run_id"], "status": status, "turns": turns,
            "configuration": copy.deepcopy(configuration), "case": copy.deepcopy(case)}


def save_receipt(directory, receipt, name=None):
    path = pathlib.Path(directory) / "receipts" / ((name or receipt["run_id"]) + ".json")
    BENCH.atomic_json(path, receipt)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return round(self.now * 1_000_000_000)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class CampaignTests(unittest.TestCase):
    def test_rotating_schedule_visits_each_pair_once(self):
        for speculative in (False, True):
            with self.subTest(speculative=speculative):
                spec = specification(5, speculative)
                jobs = spec["jobs"]
                config_ids = [config["id"] for config in spec["configurations"]]
                self.assertEqual(len(jobs), 10)
                self.assertEqual(len({job["run_id"] for job in jobs}), 10)
                pairs = collections.Counter((job["case_id"], job["configuration_id"]) for job in jobs)
                self.assertEqual(set(pairs.values()), {1})
                for index, case in enumerate(spec["cases"]):
                    block = jobs[index * 2:(index + 1) * 2]
                    self.assertTrue(all(job["case_id"] == case["id"] for job in block))
                    self.assertCountEqual([job["configuration_id"] for job in block], config_ids)
                    self.assertEqual(block[0]["configuration_id"], config_ids[index % 2])
                self.assertEqual(jobs, BENCH.build_jobs(spec["cases"], spec["configurations"]))

    def test_child_commands_keep_production_defaults_in_separate_campaigns(self):
        forbidden = {"--seed", "--greedy", "--temperature", "--top-k", "--top-p", "--min-p",
                     "--repeat-penalty", "--context-size", "--ctx-size", "-c", "-n",
                     "--max-new-tokens", "--draft-horizon", "--draft-max", "--no-warmup",
                     "--warmup", "--n-gpu-layers", "--threads", "--chat", "--conversation"}
        for speculative in (False, True):
            configs = configurations(speculative)
            mode = "speculative" if speculative else "default"
            self.assertEqual({config["id"] for config in configs}, {f"guffin-{mode}", f"llama-{mode}"})
            self.assertEqual(len(configs), 2)
            for config in configs:
                with self.subTest(config=config["id"]):
                    self.assertEqual(config["mode"], mode)
                    self.assertFalse(forbidden.intersection(config["command"]))
                    self.assertEqual(config["command"][0], config["executable"]["path"])
                    model_index = config["command"].index("-m")
                    self.assertEqual(config["command"][model_index + 1], "/fixture/target.gguf")
                    if speculative:
                        draft_flag = "--draft-model" if config["engine"] == "guffin" else "--model-draft"
                        draft_index = config["command"].index(draft_flag)
                        self.assertEqual(config["command"][draft_index + 1], "/fixture/draft.gguf")
                    else:
                        self.assertNotIn("/fixture/draft.gguf", config["command"])
                        self.assertNotIn("--draft-model", config["command"])
                        self.assertNotIn("--model-draft", config["command"])
                    if config["engine"] == "llama" and speculative:
                        activation = config["command"].index("--spec-type")
                        self.assertEqual(config["command"][activation + 1], "draft-simple")
                    else:
                        self.assertNotIn("--spec-type", config["command"])
                    measured = len(config["measurement_options"])
                    self.assertEqual(config["command"][-measured:], config["measurement_options"])

    def test_both_plans_keep_cooldowns_without_launch_or_result_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary).resolve()
            target = base / "identity-fixture.gguf"
            target.write_bytes(b"identity fixture; not a model")
            draft = base / "draft-fixture.gguf"
            draft.write_bytes(b"draft fixture; not a model")
            corpus = base / "corpus.json"
            corpus.write_text(json.dumps({"schema_version": 1, "conversations": [
                {"id": "case", "suite": "core", "turns": ["Hello", "Again"]}]}))
            fake_identity = {"path": sys.executable, "sha256": "fixture", "version": "fixture", "help": "fixture"}
            for speculative in (False, True):
                with self.subTest(speculative=speculative):
                    mode = "speculative" if speculative else "default"
                    output = base / mode
                    stdout = io.StringIO()
                    arguments = ["-m", str(target), "-o", str(output), "--corpus", str(corpus), "--plan"]
                    if speculative:
                        arguments.extend(["--draft-model", str(draft)])
                    with mock.patch.object(BENCH, "executable_identity", return_value=fake_identity), \
                            mock.patch.object(BENCH, "run_campaign", side_effect=AssertionError("plan launched chat")), \
                            mock.patch.object(BENCH, "collect_environment", side_effect=AssertionError("plan prepared campaign")), \
                            contextlib.redirect_stdout(stdout):
                        result = BENCH.main(arguments)
                    plan = json.loads(stdout.getvalue())
                    self.assertEqual(result, 0)
                    self.assertEqual(plan["protocol"]["conversation_pause_seconds"], 300.0)
                    self.assertEqual(plan["protocol"]["turn_pause_seconds"], 120.0)
                    self.assertEqual(plan["sleep_hours"], (2 * 300 + 2 * 120) / 3600)
                    self.assertEqual([config["id"] for config in plan["configurations"]],
                                     [f"guffin-{mode}", f"llama-{mode}"])
                    self.assertEqual(plan["schedule"], [f"case: guffin-{mode}, llama-{mode}"])
                    for config in plan["configurations"]:
                        command = config["command"]
                        self.assertEqual(command[command.index("-m") + 1], str(target))
                        if speculative:
                            draft_flag = "--draft-model" if config["id"].startswith("guffin") else "--model-draft"
                            self.assertEqual(command[command.index(draft_flag) + 1], str(draft))
                            if config["id"].startswith("llama"):
                                self.assertEqual(command[command.index("--spec-type") + 1], "draft-simple")
                        else:
                            self.assertNotIn(str(draft), command)
                    self.assertFalse(output.exists())

    def test_nonfinite_or_invalid_timing_policy_is_rejected_before_metadata(self):
        for flag, value in (("--conversation-pause", "nan"), ("--turn-pause", "inf"),
                            ("--timeout", "nan"), ("--conversation-pause", "-1"),
                            ("--timeout", "0")):
            with self.subTest(flag=flag, value=value), \
                    mock.patch.object(BENCH, "executable_identity", side_effect=AssertionError("invalid policy reached metadata")), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    BENCH.main(["-m", "fixture.gguf", "-o", "fixture-output", flag, value, "--plan"])
                self.assertEqual(raised.exception.code, 2)

    def test_sleep_and_live_idle_obey_exact_deadlines(self):
        clock = FakeClock()
        with mock.patch.object(BENCH.time, "monotonic", clock.monotonic), \
                mock.patch.object(BENCH.time, "sleep", clock.sleep):
            BENCH.sleep_between_conversations(300)
        self.assertEqual(clock.now, 300)
        self.assertTrue(all(0 < interval <= 1 for interval in clock.sleeps))
        chat = BENCH.ChatProcess.__new__(BENCH.ChatProcess)
        chat.process = mock.Mock()
        chat.process.poll.return_value = None
        pumps = []
        def pump(timeout):
            pumps.append(timeout)
            clock.now += timeout
        chat.pump = pump
        started = clock.now
        with mock.patch.object(BENCH.time, "monotonic", clock.monotonic):
            chat.idle(120)
        self.assertEqual(clock.now - started, 120)
        self.assertTrue(all(0 < interval <= 0.25 for interval in pumps))

    def test_campaign_waits_before_first_launch_and_between_turns(self):
        clock, launches, closes, idles = FakeClock(), [], [], []
        class FakeChat:
            def __init__(self, configuration, prefix, timeout):
                launches.append(clock.now)
                self.stdout = ""
                self.memory = mock.Mock(last=None)
                self.memory.summary.return_value = {"sample_count": 0}
                self.parser = mock.Mock()
            def wait_ready(self):
                clock.now += 3
                return 3_000_000_000
            def idle(self, seconds):
                idles.append(seconds)
                clock.now += seconds
            def turn(self, prompt, index, previous_completed_ns):
                started = clock.monotonic_ns()
                clock.now += 2
                return {"index": index, "prompt": prompt, "answer": "fixture",
                        "started_monotonic_ns": started,
                        "completed_monotonic_ns": clock.monotonic_ns(),
                        "inter_turn_idle_ns": (started - previous_completed_ns)
                        if previous_completed_ns is not None else None}
            def close(self):
                closes.append(clock.now)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
            with mock.patch.object(BENCH, "ChatProcess", FakeChat), \
                    mock.patch.object(BENCH, "system_snapshot", return_value={}), \
                    mock.patch.object(BENCH.time, "monotonic", clock.monotonic), \
                    mock.patch.object(BENCH.time, "sleep", clock.sleep), \
                    contextlib.redirect_stdout(io.StringIO()):
                receipts = BENCH.run_campaign(temporary, manifest)
            self.assertEqual(launches[0], 300)
            self.assertEqual(idles, [120] * 2)
            self.assertEqual(len(receipts), 2)
            for index in range(1, 2):
                self.assertEqual(launches[index] - closes[index - 1], 300)
            for receipt in receipts.values():
                self.assertEqual(receipt["turns"][1]["inter_turn_idle_ns"], 120_000_000_000)

    def test_resume_refuses_changed_identity_and_effective_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec, env = specification(), environment()
            manifest = BENCH.prepare_manifest(temporary, spec, env)
            before = (pathlib.Path(temporary) / "manifest.json").read_bytes()
            self.assertEqual(BENCH.prepare_manifest(temporary, spec, env)["campaign_id"], manifest["campaign_id"])
            for field in ("model", "corpus", "measurement_scripts", "protocol"):
                changed = copy.deepcopy(spec)
                changed[field]["fixture_change"] = True
                with self.subTest(field=field), self.assertRaises(ValueError):
                    BENCH.prepare_manifest(temporary, changed, env)
            for field, value in (("platform", "other-os"), ("hardware", "other-machine"),
                                 ("power_settings", "other-policy"),
                                 ("inference_environment", {"LLAMA_ARG_CTX_SIZE": "1024"})):
                changed = copy.deepcopy(env)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    BENCH.prepare_manifest(temporary, spec, changed)
            self.assertEqual((pathlib.Path(temporary) / "manifest.json").read_bytes(), before)

    def test_resume_refuses_switching_campaign_mode_without_changing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            ordinary = specification(2)
            manifest = BENCH.prepare_manifest(temporary, ordinary, environment())
            receipt = receipt_for(manifest, manifest["jobs"][0])
            save_receipt(temporary, receipt)
            before_manifest = (pathlib.Path(temporary) / "manifest.json").read_bytes()
            before_receipt = (pathlib.Path(temporary) / "receipts" / (receipt["run_id"] + ".json")).read_bytes()
            with self.assertRaisesRegex(ValueError, "use a new output directory"):
                BENCH.prepare_manifest(temporary, specification(2, speculative=True), environment())
            self.assertEqual((pathlib.Path(temporary) / "manifest.json").read_bytes(), before_manifest)
            self.assertEqual((pathlib.Path(temporary) / "receipts" / (receipt["run_id"] + ".json")).read_bytes(), before_receipt)
            self.assertEqual(BENCH.prepare_manifest(temporary, ordinary, environment())["campaign_id"],
                             manifest["campaign_id"])
            self.assertEqual(list(BENCH.completed_jobs(temporary, manifest)), [receipt["run_id"]])

    def test_resume_recovers_setup_directories_without_losing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = specification()
            original = BENCH.prepare_manifest(temporary, spec, environment())
            for name in ("logs", "receipts"):
                (pathlib.Path(temporary) / name).rmdir()
            resumed = BENCH.prepare_manifest(temporary, spec, environment())
            self.assertEqual(resumed["campaign_id"], original["campaign_id"])
            self.assertTrue((pathlib.Path(temporary) / "logs").is_dir())
            self.assertTrue((pathlib.Path(temporary) / "receipts").is_dir())

    def test_failure_stops_then_resume_never_retries_attempted_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = BENCH.prepare_manifest(temporary, specification(2), environment())
            calls = []
            def run(configuration, case, job, directory, campaign_id, turn_pause, timeout):
                calls.append(job["run_id"])
                status = "failed" if len(calls) == 2 else "complete"
                receipt = receipt_for(manifest, job, status)
                receipt["error"] = "fixture failure" if status == "failed" else None
                save_receipt(directory, receipt)
                return receipt
            with mock.patch.object(BENCH, "run_conversation", side_effect=run), \
                    mock.patch.object(BENCH, "sleep_between_conversations"), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "failed and was saved"):
                    BENCH.run_campaign(temporary, manifest)
                self.assertEqual(len(calls), 2)
                BENCH.run_campaign(temporary, manifest)
                BENCH.run_campaign(temporary, manifest)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(set(calls)), 4)
            self.assertEqual(len(BENCH.completed_jobs(temporary, manifest)), 4)

    def test_foreign_malformed_and_duplicate_receipts_are_rejected(self):
        changes = ({"campaign_id": "foreign"}, {"run_id": "unknown"},
                   {"configuration": {"id": "wrong"}}, {"case": {"id": "wrong"}},
                   {"status": "pending"})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
                receipt = receipt_for(manifest, manifest["jobs"][0])
                receipt.update(change)
                save_receipt(temporary, receipt, "fixture")
                with self.assertRaises(ValueError):
                    BENCH.completed_jobs(temporary, manifest)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
            receipt = receipt_for(manifest, manifest["jobs"][0], "interrupted")
            save_receipt(temporary, receipt)
            self.assertEqual(len(BENCH.completed_jobs(temporary, manifest)), 1)
            save_receipt(temporary, receipt, "duplicate")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                BENCH.completed_jobs(temporary, manifest)

    def test_receipt_admission_checks_immutable_identity_and_turn_sequence(self):
        changes = (
            lambda receipt: receipt.update(schema_version=2),
            lambda receipt: receipt["configuration"]["command"].append("--greedy"),
            lambda receipt: receipt["case"]["turns"].__setitem__(0, "altered source prompt"),
            lambda receipt: receipt["turns"][0].update(index=1),
            lambda receipt: receipt["turns"][0].update(prompt="altered submitted prompt"),
            lambda receipt: receipt["turns"].pop(),
            lambda receipt: receipt["turns"].append({"index": 2, "prompt": "extra"}),
            lambda receipt: receipt.update(turns=["malformed turn"]),
            lambda receipt: receipt.update(turns=None),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as temporary:
                manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
                receipt = receipt_for(manifest, manifest["jobs"][0])
                change(receipt)
                save_receipt(temporary, receipt)
                with self.assertRaises(ValueError):
                    BENCH.completed_jobs(temporary, manifest)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
            BENCH.atomic_json(pathlib.Path(temporary) / "receipts/malformed.json", [])
            with self.assertRaises(ValueError):
                BENCH.completed_jobs(temporary, manifest)

    def test_main_reports_incomplete_resumed_campaign_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            target = base / "identity-fixture.gguf"
            target.write_bytes(b"identity fixture; not a model")
            corpus = base / "corpus.json"
            corpus.write_text(json.dumps({"schema_version": 1, "conversations": [
                {"id": "case", "suite": "core", "turns": ["Hello", "Again"]}]}))
            report = types.SimpleNamespace(main=mock.Mock(return_value=0))
            identity = {"path": sys.executable, "sha256": "fixture", "version": "fixture", "help": "fixture"}
            with mock.patch.object(BENCH, "executable_identity", return_value=identity), \
                    mock.patch.object(BENCH, "collect_environment", return_value=environment()), \
                    mock.patch.object(BENCH.tempfile, "gettempdir", return_value=temporary), \
                    mock.patch.object(BENCH, "run_campaign", return_value={"old-attempt": {"status": "failed"}}), \
                    mock.patch.dict(sys.modules, {"release_chat_report": report}), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                result = BENCH.main(["-m", str(target), "-o", str(base / "campaign"), "--corpus", str(corpus)])
            self.assertEqual(result, 1)
            self.assertIn("incomplete", stderr.getvalue())
            report.main.assert_called_once_with([str((base / "campaign").resolve())])

    def test_report_failure_is_not_reported_as_campaign_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            target = base / "identity-fixture.gguf"
            target.write_bytes(b"identity fixture; not a model")
            corpus = base / "corpus.json"
            corpus.write_text(json.dumps({"schema_version": 1, "conversations": [
                {"id": "case", "suite": "core", "turns": ["Hello"]}]}))
            report = types.SimpleNamespace(main=mock.Mock(return_value=1))
            identity = {"path": sys.executable, "sha256": "fixture", "version": "fixture", "help": "fixture"}
            with mock.patch.object(BENCH, "executable_identity", return_value=identity), \
                    mock.patch.object(BENCH, "collect_environment", return_value=environment()), \
                    mock.patch.object(BENCH.tempfile, "gettempdir", return_value=temporary), \
                    mock.patch.object(BENCH, "run_campaign", return_value={"done": {"status": "complete"}}), \
                    mock.patch.dict(sys.modules, {"release_chat_report": report}), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                result = BENCH.main(["-m", str(target), "-o", str(base / "campaign"), "--corpus", str(corpus)])
            self.assertEqual(result, 1)
            self.assertIn("report could not be generated", stderr.getvalue())
            self.assertTrue((base / "campaign/manifest.json").exists())

    def test_attempt_is_durable_before_process_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = BENCH.prepare_manifest(temporary, specification(1), environment())
            job = manifest["jobs"][0]
            config, case = manifest["configurations"][0], manifest["cases"][0]
            def launch(*args):
                saved = json.loads((pathlib.Path(temporary) / "receipts" / (job["run_id"] + ".json")).read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertIn("unfinished attempt", saved["error"])
                raise OSError("fixture executable unavailable")
            with mock.patch.object(BENCH, "ChatProcess", side_effect=launch), \
                    mock.patch.object(BENCH, "system_snapshot", return_value={}):
                receipt = BENCH.run_conversation(config, case, job, temporary, manifest["campaign_id"], 120, 900)
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["turns"], [])
            self.assertIn("executable unavailable", receipt["error"])
            self.assertEqual(len(BENCH.completed_jobs(temporary, manifest)), 1)


_CHILD_SOURCE = r"""
import hashlib
import json
import os
import sys
import time

engine, mode = sys.argv[1:]
pid_path = os.environ.get('GUFFIN_BENCH_FIXTURE_PID')
if pid_path:
    with open(pid_path, 'w') as pid_file:
        pid_file.write(str(os.getpid()))
def emit(text, stream=1, split_unicode=False):
    data = text.encode('utf-8')
    if split_unicode and '世'.encode() in data:
        cut = data.index('世'.encode()) + 1
        os.write(stream, data[:cut])
        time.sleep(0.005)
        data = data[cut:]
    while data:
        data = data[os.write(stream, data):]

def exit_tail():
    emit('exit diagnostic: ' + 'z' * 150000 + '\nEXIT-TAIL-END\n', 2)

if mode == 'early-exit':
    emit('fixture startup failure\n', 2)
    raise SystemExit(7)
if mode == 'startup-timeout':
    emit('Loading fixture\n')
else:
    emit('Fixture ready\n> ')
parts = []
turn = 0
for line in sys.stdin:
    if line == '/exit\n':
        exit_tail()
        break
    if line.endswith('\\\n'):
        parts.append(line[:-2])
        if engine == 'guffin':
            emit('... ')
        continue
    parts.append(line[:-1])
    prompt = '\n'.join(parts)
    parts = []
    turn += 1
    if mode in ('turn-timeout', 'startup-timeout'):
        emit('partial answer only\n')
        continue
    if mode == 'second-turn-failure' and turn == 2:
        emit('partial second answer\n')
        emit('error: context: fixture exhausted\n', 2)
        raise SystemExit(3)
    if engine == 'llama':
        emit(prompt + '\n')
    emit('diagnostic: ' + 'x' * 150000 + '\n', 2)
    answer = 'Résumé 世界 — ' + hashlib.sha256(prompt.encode()).hexdigest() + '\n> '
    emit(answer, split_unicode=True)
    time.sleep(0.005)
    emit('still generating; Error: is answer content\n')
    if engine == 'guffin':
        emit('[ Prompt: 10.00 t/s | Generation: 15.00 t/s | TTFT: 1.000 s ]\n')
        emit('[ Detail: Evaluated: 10 | Cached: 2 | Decoded: 3 | Prompt: 1.000000 s | Generation: 0.200000 s ]\n')
        metrics = {'schema_version': 6, 'evaluated_prompt_tokens': 10, 'reused_cache_tokens': 2,
                   'selected_tokens': 3, 'decoded_tokens': 3,
                   'generation_completion_ns': 1250000000, 'full_turn_ns': 1300000000,
                   'first_text_ns': 1100000000}
        emit('[ Decode: ' + json.dumps(metrics) + ' ]\n', 2)
    else:
        prefix = 'slot print_timing: id 0 | task ' + str(turn) + ' | '
        emit('slot update_slots: id 0 | task ' + str(turn) + ' | new prompt, n_ctx_slot = 4096, n_keep = -1, task.n_tokens = 12\n', 2)
        emit(prefix + 'prompt eval time = 1000.00 ms / 10 tokens (100.00 ms per token, 10.00 tokens per second)\n', 2)
        emit(prefix + 'eval time = 200.00 ms / 3 tokens (100.00 ms per token, 10.00 tokens per second)\n', 2)
        emit(prefix + 'total time = 1200.00 ms / 13 tokens\n', 2)
        emit('\n[ Prompt: 10.00 t/s | Generation: 10.00 t/s ]\n')
    emit('\n> ')
"""


class PtyConversationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temporary.name)
        (self.directory / "logs").mkdir()
        (self.directory / "receipts").mkdir()
        self.child = self.directory / "fake_chat.py"
        self.child.write_text(_CHILD_SOURCE, encoding="utf-8")
        self.chats = []
        self.original_chat = BENCH.ChatProcess

    def tearDown(self):
        for chat in self.chats:
            if chat.process.poll() is None:
                BENCH.stop_process(chat.process)
        self.temporary.cleanup()

    def test_setup_failure_reaps_child_and_closes_terminal_and_logs(self):
        command = {"engine": "guffin", "command": [sys.executable, str(self.child), "guffin", "complete"]}
        original_popen = BENCH.subprocess.Popen
        original_openpty = BENCH.pty.openpty
        children, terminals = [], []
        def open_terminal():
            descriptors = original_openpty()
            terminals.extend(descriptors)
            return descriptors
        def launch(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            return child
        with mock.patch.object(BENCH.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(BENCH.pty, "openpty", side_effect=open_terminal), \
                mock.patch.object(BENCH.os, "set_blocking", side_effect=OSError("fixture nonblocking failure")):
            with self.assertRaisesRegex(OSError, "nonblocking failure"):
                self.original_chat(command, self.directory / "logs/setup-failure", 3)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stderr.closed)
        self.assertEqual(len(terminals), 2)
        for descriptor in terminals:
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        for name in ("stdout", "stderr"):
            path = self.directory / f"logs/setup-failure.{name}.log"
            self.assertTrue(path.is_file())

    def run_fixture(self, engine="guffin", mode="complete", turns=None, timeout=3):
        config = {"id": engine + "-fixture", "engine": engine,
                  "command": [sys.executable, "-u", str(self.child), engine, mode]}
        case = {"id": "fixture-case", "category": "conversation", "turns": turns or ["Hello", "Again"]}
        job = {"run_id": "fixture-" + engine + "-" + mode}
        def launch(*args):
            chat = self.original_chat(*args)
            self.chats.append(chat)
            return chat
        with mock.patch.object(BENCH, "ChatProcess", side_effect=launch), \
                mock.patch.object(BENCH, "system_snapshot", return_value={}):
            return BENCH.run_conversation(config, case, job, self.directory, "fixture-campaign", 0, timeout)

    def assert_closed(self):
        for chat in self.chats:
            self.assertIsNotNone(chat.process.poll())
            self.assertTrue(chat.process.stderr.closed)
            self.assertTrue(all(log.closed for log in chat.logs.values()))
            with self.assertRaises(OSError) as raised:
                os.fstat(chat.master)
            self.assertEqual(raised.exception.errno, errno.EBADF)

    def test_real_pty_preserves_large_multiline_unicode_and_drains_both_streams(self):
        prompts = ['Résumé 世界\n' + 'long input ' * 2300 + '\nlast line',
                   'Follow-up with literal Error: text and / in the middle.']
        for engine in ("guffin", "llama"):
            with self.subTest(engine=engine):
                receipt = self.run_fixture(engine=engine, turns=prompts)
                self.assertEqual(receipt["status"], "complete", receipt.get("error"))
                self.assertEqual(len(receipt["turns"]), 2)
                for index, turn in enumerate(receipt["turns"]):
                    digest = hashlib.sha256(prompts[index].encode()).hexdigest()
                    expected = 'Résumé 世界 — ' + digest + '\n> still generating; Error: is answer content\n'
                    self.assertEqual(turn["prompt"], prompts[index])
                    self.assertEqual(turn["answer"], expected)
                    self.assertEqual(turn["answer_sha256"], hashlib.sha256(expected.encode()).hexdigest())
                    self.assertEqual(turn["metrics"]["selected_tokens"], 3)
                    self.assertEqual(turn["metrics"]["generation_steps"], 2)
                    self.assertEqual(turn["metrics"]["generation_tps"], 10)
                    self.assertEqual(turn["metrics"]["cached_tokens"], 2)
                    self.assertIn(b'\xe4\xb8\x96\xe7\x95\x8c', (self.directory / receipt["logs"]["stdout"]).read_bytes())
                stderr = (self.directory / receipt["logs"]["stderr"]).read_bytes()
                self.assertGreater(len(stderr), 450000)
                self.assertTrue(stderr.endswith(b'EXIT-TAIL-END\n'))
                saved = json.loads((self.directory / "receipts" / (receipt["run_id"] + ".json")).read_text())
                self.assertEqual(saved, json.loads(json.dumps(receipt)))
        self.assert_closed()

    def test_turn_and_startup_timeouts_save_failure_and_close_process(self):
        for mode in ("turn-timeout", "startup-timeout", "early-exit"):
            with self.subTest(mode=mode):
                receipt = self.run_fixture(mode=mode, turns=["Hello"], timeout=0.15)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["turns"], [])
                self.assertTrue(receipt["error"])
                if mode != "early-exit":
                    self.assertIn("timed out", receipt["error"])
                else:
                    self.assertIn("startup failure", receipt["error"])
        self.assert_closed()

    def test_second_turn_exit_retains_completed_first_answer(self):
        receipt = self.run_fixture(mode="second-turn-failure")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(len(receipt["turns"]), 1)
        self.assertEqual(receipt["turns"][0]["prompt"], "Hello")
        self.assertIn("partial second answer", receipt["partial_terminal_text"])
        self.assertIn("context", receipt["error"])
        self.assert_closed()

    def test_sigterm_records_attempt_and_reaps_native_child(self):
        campaign = self.directory / "terminated-campaign"
        target = self.directory / "identity-fixture.gguf"
        target.write_bytes(b"identity fixture; not a model")
        corpus = self.directory / "corpus.json"
        corpus.write_text(json.dumps({"schema_version": 1, "conversations": [
            {"id": "termination-case", "suite": "core", "category": "conversation", "turns": ["Hello"]}]}))
        pid_path = self.directory / "native-child.pid"
        code = "\n".join([
            "import os, sys",
            f"sys.path.insert(0, {str(SCRIPTS)!r})",
            "import release_chat_benchmark as bench",
            f"os.environ['GUFFIN_BENCH_FIXTURE_PID'] = {str(pid_path)!r}",
            "bench.executable_identity = lambda name: {'path': sys.executable, 'sha256': 'fixture', 'version': 'fixture', 'help': 'fixture'}",
            f"bench.collect_environment = lambda: {environment()!r}",
            "bench.system_snapshot = lambda: {}",
            f"bench.tempfile.gettempdir = lambda: {str(self.directory)!r}",
            "bench.build_configurations = lambda guffin, llama, model, draft=None: [",
            "{'id': 'guffin-fixture', 'engine': 'guffin', 'mode': 'default',",
            " 'model_path': model['path'], 'model_sha256': model['sha256'], 'draft_sha256': None,",
            " 'executable': guffin, 'measurement_options': [],",
            f" 'command': [sys.executable, '-u', {str(self.child)!r}, 'guffin', 'startup-timeout']}}]",
            f"raise SystemExit(bench.main({['-m', str(target), '-o', str(campaign), '--corpus', str(corpus), '--conversation-pause', '0', '--turn-pause', '0', '--timeout', '30']!r}))",
        ])
        runner = subprocess.Popen([sys.executable, "-B", "-u", "-c", code],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=True)
        native_pid = None
        try:
            deadline = time.monotonic() + 5
            while not pid_path.exists() and time.monotonic() < deadline:
                if runner.poll() is not None:
                    stdout, stderr = runner.communicate()
                    self.fail(f"fixture runner exited before launch: {stdout!r} {stderr!r}")
                time.sleep(0.01)
            self.assertTrue(pid_path.exists(), "fake native child was not launched")
            native_pid = int(pid_path.read_text())
            self.assertEqual(len(list((campaign / "receipts").glob("*.json"))), 1)
            os.kill(runner.pid, signal.SIGTERM)
            stdout, stderr = runner.communicate(timeout=10)
            self.assertEqual(runner.returncode, 130, (stdout, stderr))
            manifest = json.loads((campaign / "manifest.json").read_text())
            receipts = BENCH.completed_jobs(campaign, manifest)
            self.assertEqual(len(receipts), 1)
            receipt = next(iter(receipts.values()))
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["turns"], [])
            self.assertIn("not be automatically repeated", receipt["error"])
            with self.assertRaises(ProcessLookupError):
                os.kill(native_pid, 0)
            self.assertTrue((campaign / receipt["logs"]["stderr"]).read_bytes().endswith(b"EXIT-TAIL-END\n"))
        finally:
            if runner.poll() is None:
                runner.kill()
            runner.communicate(timeout=5)
            if native_pid is not None:
                try:
                    os.killpg(native_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_interruption_retains_completed_turn_and_closes_chat(self):
        chat = mock.Mock(stdout="first answer and partial second answer")
        chat.wait_ready.return_value = 1
        chat.memory.summary.return_value = {"sample_count": 0}
        chat.turn.side_effect = [
            {"index": 0, "prompt": "first", "answer": "first answer", "completed_monotonic_ns": 1},
            KeyboardInterrupt(),
        ]
        configuration = {"id": "fixture", "engine": "guffin"}
        case = {"id": "fixture", "turns": ["first", "second"]}
        job = {"run_id": "interrupted"}
        with mock.patch.object(BENCH, "ChatProcess", return_value=chat), \
                mock.patch.object(BENCH, "system_snapshot", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                BENCH.run_conversation(configuration, case, job, self.directory, "campaign", 120, 900)
        receipt = json.loads((self.directory / "receipts/interrupted.json").read_text())
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(len(receipt["turns"]), 1)
        self.assertIn("not be automatically repeated", receipt["error"])
        chat.close.assert_called_once()
        chat.idle.assert_called_once_with(120)


class MemoryCollectionTests(unittest.TestCase):
    def test_sdk_v4_footprint_rss_and_lifetime_peak_remain_distinct(self):
        # Named ABI fields from macOS SDK sys/resource.h rusage_info_v4;
        # this fixture deliberately avoids the runner's numeric array indexes.
        names = (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
            "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
            "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
            "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
            "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
            "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time",
        )
        class SdkRusageV4(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in names]
        self.assertEqual(ctypes.sizeof(BENCH.ProcessMemory.RUsageV4), ctypes.sizeof(SdkRusageV4))
        sample = SdkRusageV4()
        sample.resident_size, sample.phys_footprint = 700, 400
        sample.lifetime_max_phys_footprint, sample.pageins = 900, 3
        def rusage(pid, flavor, destination):
            self.assertEqual((pid, flavor), (12345, 4))
            ctypes.memmove(destination, ctypes.byref(sample), ctypes.sizeof(sample))
            return 0
        function = mock.Mock(side_effect=rusage)
        library = types.SimpleNamespace(proc_pid_rusage=function)
        with mock.patch.object(BENCH.platform, "system", return_value="Darwin"), \
                mock.patch.object(BENCH.ctypes, "CDLL", return_value=library):
            memory = BENCH.ProcessMemory(12345)
        first = memory.sample()
        self.assertEqual(first["resident_bytes"], 700)
        self.assertEqual(first["physical_footprint_bytes"], 400)
        self.assertEqual(first["lifetime_peak_physical_footprint_bytes"], 900)
        self.assertEqual(first["pageins"], 3)
        sample.resident_size, sample.phys_footprint, sample.lifetime_max_phys_footprint = 600, 300, 950
        memory.sample()
        summary = memory.summary()
        self.assertEqual(summary["peak_rss_bytes"], 700)
        self.assertEqual(summary["peak_physical_footprint_bytes"], 950)
        self.assertEqual(summary["sample_count"], 2)
        ctypes.set_errno(errno.ESRCH)
        function.side_effect = None
        function.return_value = -1
        self.assertIsNone(memory.sample())
        self.assertEqual(memory.summary()["sample_count"], 2)
        self.assertIn("errno", memory.summary()["error"])


if __name__ == "__main__":
    unittest.main()

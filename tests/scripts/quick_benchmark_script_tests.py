import contextlib
import copy
import importlib.util
import io
import hashlib
import pathlib
import json
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "quick_benchmark", SCRIPTS / "quick_benchmark.py"
)
QUICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUICK)
REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "llama_reference_benchmark", SCRIPTS / "llama_reference_benchmark.py"
)
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)


def kernel_diagnostics_fixture():
    return {
        "engine_build": "test-build",
        "model": "test.gguf",
        "schema_version": 2,
        "measured_commands_per_mode": 3,
        "warmup_commands_per_mode": 1,
        "sustained_dispatches_per_command": 42,
        "kernels": [
            {
                "name": "Q4_K decode",
                "shape": "3584x14336, 1 row",
                "measurements": [
                    {
                        "mode": "isolated",
                        "dispatches_per_command": 1,
                        "median_per_dispatch_ns": 200,
                        "min_per_dispatch_ns": 100,
                        "max_per_dispatch_ns": 300,
                        "effective_gbps": 30.0,
                        "raw_command_ns": [300, 100, 200],
                    },
                    {
                        "mode": "sustained",
                        "dispatches_per_command": 42,
                        "median_per_dispatch_ns": 200,
                        "min_per_dispatch_ns": 100,
                        "max_per_dispatch_ns": 300,
                        "effective_gbps": 98.0,
                        "raw_command_ns": [12600, 4200, 8400],
                    },
                ],
            }
        ],
    }


def decode_attribution(evaluated, cached, decoded, initial_cache, attention_tokens=100):
    per_token = {
        "embedding": 1,
        "attention_projections": 2,
        "ffn_gate_up": 3,
        "ffn_down": 4,
        "output_head": 5,
        "norms": 6,
        "total": 21,
    }
    return {
        "schema_version": 3,
        "selected_tokens": decoded,
        "selected_token_hash": "0123456789abcdef",
        "decoded_tokens": decoded,
        "initial_cache_position": initial_cache,
        "final_cache_position": initial_cache + evaluated + decoded,
        "evaluated_prompt_tokens": evaluated,
        "reused_cache_tokens": cached,
        "decode_wall_ns": 9_000_000_000,
        "gpu_ns": 8_000_000_000,
        "selection_ns": 1000,
        "proposer_ns": 0,
        "speculative_cycles": 0,
        "speculative_matches": 0,
        "speculative_misses": 0,
        "speculative_fallbacks": 0,
        "speculative_rollbacks": 0,
        "proposal_buckets": [],
        "wall_minus_gpu_ns": 1_000_000_000,
        "resident_weights_bytes": 100,
        "resident_kv_bytes": 200,
        "resident_scratch_bytes": 300,
        "warmup_kv_clear_passes": 0,
        "warmup_host_cleared_kv_bytes": 0,
        "command_buffers": decoded,
        "commits": decoded,
        "waits": decoded,
        "compute_encoders": decoded,
        "compute_dispatches": decoded * 10,
        "blit_encoders": 0,
        "barriers": decoded * 5,
        "transient_allocations": 0,
        "selector_dispatches": decoded,
        "max_candidates": 40,
        "compact_readback_bytes": decoded * 320,
        "attention_dispatches": decoded * 42,
        "attention_partitions": decoded * 42,
        "attention_tokens": attention_tokens,
        "output_head_elisions": 1 if decoded else 0,
        "output_elided_weight_bytes": 5 if decoded else 0,
        "scheduled_bytes_per_token": per_token,
        "scheduled_bytes_total": per_token["total"] * decoded - (5 if decoded else 0),
    }


def standalone_attribution_v5():
    record = decode_attribution(12, 0, 10, 0)
    record.update(
        schema_version=5, generation_completion_ns=10_000_000_000,
        full_turn_ns=11_000_000_000, first_text_ns=1_000_000_000, learned_owned_bytes=1000,
        learned_horizon=3, learned_cycles=4, learned_accepted_edges=6,
        learned_numeric_refusals=0, learned_draft_ns=100, learned_draft_gpu_ns=90,
        learned_target_gpu_ns=1000, learned_cycle_ns=1200,
        scheduled_bytes_basis="one_row_equivalent_not_physical_bandwidth",
        speculative_draft_model="standalone_gemma3_tli", speculative_sampling="lossless",
        standalone_draft_sampler="explicit_or_target_scalars_native_history_v1",
        standalone_effective_sampler=dict(temperature=0.5, top_k=0, top_p=0.75,
                                          min_p=0.125, repetition_penalty=1.5),
        learned_draft_head="q8_0", learned_verification_schedule="full",
        standalone_prefill_ns=10, standalone_rebuild_ns=0,
        standalone_draft_uploaded_bytes=64, standalone_draft_readback_bytes=32,
        standalone_draft_wall_ns=110, standalone_draft_gpu_ns=95,
        standalone_draft_command_buffer_count=2, standalone_draft_compute_dispatch_count=40,
        learned_accepted_prefix_histogram=[1, 1, 1, 1] + [0] * 20,
        learned_rejection_depth_histogram=[1, 1] + [0] * 21,
        learned_censored_cycle_count=1, learned_full_accept_count=1,
        learned_horizon_limited_cycle_count=1,
    )
    return record


class QuickBenchmarkScriptTests(unittest.TestCase):
    def test_schema5_acceptance_accounting_distinguishes_rejection_stop_and_capacity(self):
        record = standalone_attribution_v5()
        turn = dict(evaluated=12, cached=0, decoded=10, decode_attribution=record)
        QUICK.validate_decode_attribution(turn, "First")
        # A full-accepted stop is not a horizon-limited continuation.
        record["learned_horizon_limited_cycle_count"] = 0
        QUICK.validate_decode_attribution(turn, "First")
        for horizon in (1, 23):
            boundary = standalone_attribution_v5()
            boundary.update(learned_horizon=horizon, learned_cycles=1,
                            learned_accepted_edges=horizon,
                            learned_accepted_prefix_histogram=[int(i == horizon) for i in range(24)],
                            learned_rejection_depth_histogram=[0] * 23,
                            learned_censored_cycle_count=0)
            QUICK.validate_acceptance_accounting(boundary, "Boundary")
        stage4 = standalone_attribution_v5()
        stage4.update(speculative_draft_model="none", learned_horizon=0, learned_cycles=0,
                      learned_accepted_edges=0, learned_accepted_prefix_histogram=[0] * 24,
                      learned_rejection_depth_histogram=[0] * 23, learned_censored_cycle_count=0,
                      learned_full_accept_count=0, learned_horizon_limited_cycle_count=0,
                      standalone_draft_sampler="none", standalone_effective_sampler=None)
        turn["decode_attribution"] = stage4
        QUICK.validate_decode_attribution(turn, "Stage4")

    def test_schema5_accounting_rejects_missing_malformed_and_inconsistent_counts(self):
        fixture = standalone_attribution_v5()
        fields = ("learned_accepted_prefix_histogram", "learned_rejection_depth_histogram",
                  "learned_censored_cycle_count", "learned_full_accept_count",
                  "learned_horizon_limited_cycle_count")
        for field in fields:
            bad = copy.deepcopy(fixture)
            del bad[field]
            with self.subTest(missing=field), self.assertRaises(RuntimeError):
                QUICK.validate_acceptance_accounting(bad, "First")
        malformed = [
            ("learned_accepted_prefix_histogram", [0] * 23),
            ("learned_accepted_prefix_histogram", [0] * 25),
            ("learned_accepted_prefix_histogram", tuple(fixture["learned_accepted_prefix_histogram"])),
            ("learned_rejection_depth_histogram", [0] * 22),
            ("learned_rejection_depth_histogram", [0] * 24),
            ("learned_rejection_depth_histogram", None),
            ("learned_cycles", 5), ("learned_accepted_edges", 7),
            ("learned_censored_cycle_count", 0), ("learned_full_accept_count", 0),
            ("learned_horizon_limited_cycle_count", 2),
            ("learned_horizon", 0), ("learned_horizon", 2), ("learned_horizon", 24),
        ]
        for field, value in malformed:
            bad = dict(fixture, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                QUICK.validate_acceptance_accounting(bad, "First")
        for field in fields:
            for value in (-1, 2**32, True, 1.0, "1", None):
                bad = copy.deepcopy(fixture)
                if isinstance(bad[field], list):
                    bad[field][0] = value
                else:
                    bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    QUICK.validate_acceptance_accounting(bad, "First")
        # Preserve global cycle totals but move rejection to a depth with no
        # matching accepted prefix; this must not masquerade as a valid run.
        bad = copy.deepcopy(fixture)
        bad["learned_rejection_depth_histogram"][:3] = [0, 0, 2]
        with self.assertRaises(RuntimeError):
            QUICK.validate_acceptance_accounting(bad, "First")
        bad = copy.deepcopy(fixture)
        bad["learned_rejection_depth_histogram"][:4] = [0, 1, 0, 1]
        with self.assertRaises(RuntimeError):
            QUICK.validate_acceptance_accounting(bad, "First")

    def test_schema5_requires_complete_valid_effective_sampler_but_keeps_schema4(self):
        fixture = standalone_attribution_v5()
        turn = dict(evaluated=12, cached=0, decoded=10, decode_attribution=fixture)
        QUICK.validate_decode_attribution(turn, "First")
        for effective in (None, {}, [], {"temperature": 1},
                          dict(fixture["standalone_effective_sampler"], seed=42)):
            bad = copy.deepcopy(turn)
            bad["decode_attribution"]["standalone_effective_sampler"] = effective
            with self.subTest(effective=effective), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")
        for field, value in (("temperature", float("nan")), ("temperature", -1),
                             ("top_k", True), ("top_k", 262145), ("top_k", 1.0),
                             ("top_p", 0), ("min_p", 1.1), ("repetition_penalty", 0)):
            bad = copy.deepcopy(turn)
            bad["decode_attribution"]["standalone_effective_sampler"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")
        bad = copy.deepcopy(turn)
        bad["decode_attribution"].update(speculative_draft_model="standalone_vicuna_tli", learned_draft_head="f16")
        bad["decode_attribution"]["standalone_effective_sampler"]["top_k"] = 32001
        with self.assertRaises(RuntimeError):
            QUICK.validate_decode_attribution(bad, "First")
        legacy = copy.deepcopy(turn)
        legacy["decode_attribution"].update(schema_version=4, standalone_draft_sampler="target_scalars_native_history_v1")
        for field in ("standalone_effective_sampler", "learned_accepted_prefix_histogram",
                      "learned_rejection_depth_histogram", "learned_censored_cycle_count",
                      "learned_full_accept_count", "learned_horizon_limited_cycle_count"):
            del legacy["decode_attribution"][field]
        QUICK.validate_decode_attribution(legacy, "Legacy")
        legacy["decode_attribution"]["schema_version"] = 5
        with self.assertRaises(RuntimeError):
            QUICK.validate_decode_attribution(legacy, "Missing new contract")

    def test_complete_generation_metrics_fail_closed(self):
        record = decode_attribution(12, 0, 2, 0)
        record.update(schema_version=4, generation_completion_ns=10_000_000_000,
                      full_turn_ns=11_000_000_000, first_text_ns=1_000_000_000, learned_owned_bytes=1000,
                      learned_horizon=3, learned_cycles=1, learned_accepted_edges=2,
                      learned_numeric_refusals=0, learned_draft_ns=100,
                      learned_draft_gpu_ns=90, learned_target_gpu_ns=1000,
                      learned_cycle_ns=1200,
                      scheduled_bytes_basis="one_row_equivalent_not_physical_bandwidth")
        turn = dict(evaluated=12, cached=0, decoded=2, decode_attribution=record)
        QUICK.validate_decode_attribution(turn, "First")
        execution = dict(learned_verification_schedule="conditional", learned_draft_head="q8_0")
        record.update(execution)
        QUICK.validate_decode_attribution(turn, "First")
        for field in execution:
            bad = copy.deepcopy(turn)
            del bad["decode_attribution"][field]
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")
            bad["decode_attribution"][field] = "invalid"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")
        for field, value in (("generation_completion_ns", 0), ("full_turn_ns", 1),
                             ("first_text_ns", 12_000_000_000),
                             ("learned_owned_bytes", True), ("learned_cycles", -1),
                             ("learned_accepted_edges", 4), ("scheduled_bytes_basis", None)):
            bad = copy.deepcopy(turn)
            bad["decode_attribution"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")

    def test_recurrent_chat_flags_reach_real_executable(self):
        with mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                mock.patch.object(QUICK, "parse_local", return_value=[]) as parse:
            QUICK.run_local_chat("engine", "model", 0, recurrent_weights="weights", retrieval=False)
            command = run.call_args.args[0]
            self.assertEqual(command[-3:], ["--development-recurrent-weights", "weights", "--no-retrieval"])
            self.assertNotIn("--no-speculative", command)
            parse.assert_called_once_with("outerr")

    def test_lossless_chat_mode_is_explicit_and_validated(self):
        with mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                mock.patch.object(QUICK, "parse_local", return_value=[]):
            QUICK.run_local_chat("engine", "model", 0, recurrent_weights="weights",
                                 speculative_sampling="lossless")
            self.assertEqual(run.call_args.args[0][-2:], ["--speculative-sampling", "lossless"])
        for options in ({"speculative_sampling": "lossless"},
                        {"speculative_sampling": "unknown", "recurrent_weights": "weights"},
                        {"speculative_sampling": "lossless", "recurrent_weights": "weights",
                         "speculative": False}):
            with self.subTest(options=options), mock.patch.object(QUICK, "run_process") as run:
                with self.assertRaises(ValueError):
                    QUICK.run_local_chat("engine", "model", 0, **options)
                run.assert_not_called()

    def test_standalone_tli_flags_reach_executable_for_full_capacity(self):
        for horizon in (1, 3, 23):
            for schedule in ("full", "conditional"):
                with self.subTest(horizon=horizon, schedule=schedule), \
                        mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                        mock.patch.object(QUICK, "parse_local", return_value=[]) as parse:
                    QUICK.run_local_chat("engine", "model", 0, draft_model="draft.gguf",
                                         draft_horizon=horizon, speculative_sampling="lossless",
                                         speculative_schedule=schedule)
                    command = run.call_args.args[0]
                    for flag, value in (("--draft-model", "draft.gguf"), ("--draft-horizon", str(horizon)),
                                        ("--speculative-sampling", "lossless"),
                                        ("--speculative-schedule", schedule)):
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertNotIn("--development-recurrent-weights", command)
                    self.assertNotIn("--draft-head", command)
                    parse.assert_called_once_with("outerr")

    def test_draft_sampling_flags_are_forwarded_without_mutating_target_flags_or_inputs(self):
        overrides = dict(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.25)
        previous = copy.deepcopy(overrides)
        with mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                mock.patch.object(QUICK, "parse_local", return_value=[]):
            QUICK.run_local_chat("engine", "target", 0, draft_model="draft.gguf",
                                 speculative_sampling="lossless", draft_sampling=overrides)
        command = run.call_args.args[0]
        self.assertEqual(overrides, previous)
        for field, flag in QUICK.DRAFT_SAMPLING_FLAGS.items():
            self.assertEqual(command.count(flag), 1)
            self.assertEqual(command[command.index(flag) + 1], str(overrides[field]))
        for target_flag in ("--temperature", "--top-k", "--top-p", "--min-p", "--repeat-penalty"):
            self.assertNotIn(target_flag, command)
        self.assertEqual(command[command.index("--seed") + 1], str(QUICK.SEED))
        self.assertNotIn("--draft-seed", command)
        self.assertNotIn("--draft-repetition-window", command)
        with mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                mock.patch.object(QUICK, "parse_local", return_value=[]):
            QUICK.run_local_chat("engine", "target", 0, draft_model="draft.gguf", speculative_sampling="lossless")
        self.assertFalse(set(QUICK.DRAFT_SAMPLING_FLAGS.values()).intersection(run.call_args.args[0]))

    def test_draft_sampling_invalid_values_and_contexts_fail_before_process_and_cooldown(self):
        invalid = [None, [], "temperature=1", {"seed": 1}, {"repetition_window": 2}]
        invalid += [{field: value} for field, value in (
            ("temperature", -1), ("temperature", float("nan")), ("temperature", float("inf")),
            ("temperature", 1e39), ("temperature", 1e-300), ("temperature", 10**400),
            ("top_k", -1), ("top_k", 262145), ("top_k", True), ("top_k", 1.0),
            ("top_k", "1"), ("top_k", 10**400),
            ("top_p", 0), ("top_p", 1.1), ("top_p", 1e-300),
            ("min_p", -0.1), ("min_p", 1.1), ("min_p", False),
            ("repetition_penalty", 0), ("repetition_penalty", -1),
            ("repetition_penalty", 1e-300), ("repetition_penalty", float("inf")),
        )]
        # None is the inheritance sentinel at run_local_chat, not a malformed
        # explicit scalar map. Its direct validation remains invalid.
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                QUICK.validate_draft_sampling(values)
            if values is None:
                continue
            with self.subTest(values=values), mock.patch.object(QUICK, "run_process") as run, \
                    mock.patch.object(QUICK.time, "sleep") as sleep:
                with self.assertRaises(ValueError):
                    QUICK.run_local_chat("engine", "target", 240, draft_model="draft",
                                         speculative_sampling="lossless", draft_sampling=values)
                run.assert_not_called()
                sleep.assert_not_called()
        for options in ({}, {"speculative": False},
                        {"recurrent_weights": "weights", "speculative_sampling": "lossless"},
                        {"draft_model": "draft", "speculative_sampling": "exact-match"},
                        {"draft_model": "draft", "speculative_sampling": "lossless", "speculative": False},
                        {"draft_model": "draft", "speculative_sampling": "lossless", "recurrent_weights": "weights"}):
            with self.subTest(options=options), mock.patch.object(QUICK, "run_process") as run, \
                    mock.patch.object(QUICK.time, "sleep") as sleep:
                with self.assertRaises(ValueError):
                    QUICK.run_local_chat("engine", "target", 240, draft_sampling={"top_k": 0}, **options)
                run.assert_not_called()
                sleep.assert_not_called()

    def test_draft_sampling_receipts_require_the_requested_f32_values(self):
        requested = dict(temperature=0.8, top_k=0, top_p=0.95, min_p=0.05, repetition_penalty=1.1)
        effective = {field: value if field == "top_k" else struct.unpack("<f", struct.pack("<f", value))[0]
                     for field, value in requested.items()}
        record = standalone_attribution_v5()
        record["standalone_effective_sampler"] = effective
        for field in requested:
            bad = copy.deepcopy(record)
            if field == "top_k":
                bad["standalone_effective_sampler"][field] = 1
            else:
                bits = struct.unpack("<I", struct.pack("<f", effective[field]))[0]
                bad["standalone_effective_sampler"][field] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
            with self.subTest(field=field), \
                    mock.patch.object(QUICK, "run_process", return_value=("out", "err")), \
                    mock.patch.object(QUICK, "parse_local", return_value=[{"decode_attribution": bad}]):
                with self.assertRaisesRegex(RuntimeError, "actual proposal execution"):
                    QUICK.run_local_chat("engine", "target", 0, draft_model="draft",
                                         speculative_sampling="lossless", draft_sampling=requested)
            del bad["standalone_effective_sampler"][field]
            with mock.patch.object(QUICK, "run_process", return_value=("out", "err")), \
                    mock.patch.object(QUICK, "parse_local", return_value=[{"decode_attribution": bad}]):
                with self.assertRaises(RuntimeError):
                    QUICK.run_local_chat("engine", "target", 0, draft_model="draft",
                                         speculative_sampling="lossless", draft_sampling=requested)
        with mock.patch.object(QUICK, "run_process", return_value=("out", "err")), \
                mock.patch.object(QUICK, "parse_local", return_value=[{"decode_attribution": record}]):
            turns = QUICK.run_local_chat("engine", "target", 0, draft_model="draft",
                                        speculative_sampling="lossless", draft_sampling=requested)
        self.assertEqual(turns[0]["decode_attribution"]["standalone_effective_sampler"], effective)

    def test_standalone_tli_conflicts_fail_before_process_or_cooldown(self):
        invalid = [
            {"draft_model": "draft"},
            {"draft_model": "draft", "speculative_sampling": "exact-match"},
            {"draft_model": "draft", "speculative_sampling": "lossless", "speculative": False},
            {"draft_model": "draft", "speculative_sampling": "lossless", "recurrent_weights": "weights"},
            {"draft_model": "draft", "speculative_sampling": "lossless", "draft_head": "f16"},
            {"draft_model": "draft", "speculative_sampling": "lossless", "draft_head": "q8_0"},
            {"draft_model": "", "speculative_sampling": "lossless"},
            {"draft_horizon": 3},
        ]
        invalid.extend({"draft_model": "draft", "speculative_sampling": "lossless", "draft_horizon": value}
                       for value in (0, -1, 24, True, 3.0, "3", 2**64))
        for options in invalid:
            with self.subTest(options=options), mock.patch.object(QUICK, "run_process") as run, \
                    mock.patch.object(QUICK.time, "sleep") as sleep:
                with self.assertRaises(ValueError):
                    QUICK.run_local_chat("engine", "model", 240, **options)
                run.assert_not_called()
                sleep.assert_not_called()

    def test_standalone_tli_benchmark_rejects_execution_fallback(self):
        record = {"speculative_draft_model": "standalone_vicuna_tli", "speculative_sampling": "lossless",
                  "learned_horizon": 3, "learned_verification_schedule": "conditional"}
        for field, replacement in (("speculative_draft_model", "recurrent"), ("learned_horizon", 1),
                                   ("speculative_sampling", "exact-match"),
                                   ("learned_verification_schedule", "full")):
            bad = dict(record, **{field: replacement})
            with self.subTest(field=field), \
                    mock.patch.object(QUICK, "run_process", return_value=("out", "err")), \
                    mock.patch.object(QUICK, "parse_local", return_value=[{"decode_attribution": bad}]):
                with self.assertRaisesRegex(RuntimeError, "does not match chat execution"):
                    QUICK.run_local_chat("engine", "model", 0, draft_model="draft",
                                         speculative_sampling="lossless", speculative_schedule="conditional")

    def test_standalone_tli_benchmark_records_artifact_identity_and_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = pathlib.Path(directory) / "draft.gguf"
            draft.write_bytes(b"test draft artifact")
            argv = ["quick_benchmark.py", "--model", "target.gguf", "--draft-model", str(draft),
                    "--draft-horizon", "23", "--speculative-sampling", "lossless",
                    "--speculative-schedule", "conditional", "--cooldown", "0", "--json"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(QUICK, "run_local_chat", return_value=[{
                        "decode_attribution": {"speculative_draft_model": "standalone_vicuna_tli"}
                    }]) as chat, \
                    mock.patch.object(QUICK, "run_kernel_diagnostics", return_value={}), \
                    mock.patch.object(QUICK, "collect_environment", return_value={}), \
                    contextlib.redirect_stdout(output):
                QUICK.main()
            protocol = json.loads(output.getvalue())["protocol"]
            self.assertEqual(protocol["standalone_draft_model_sha256"],
                             hashlib.sha256(b"test draft artifact").hexdigest())
            self.assertEqual(protocol["standalone_draft_horizon"], 23)
            self.assertEqual(protocol["speculative_draft_model"], "standalone_vicuna_tli")
            self.assertIsNone(protocol["development_recurrent_weights_sha256"])
            self.assertFalse(protocol["cross_mode_token_identity_required"])
            self.assertEqual(chat.call_args.args[-2:], (str(draft), 23))
            self.assertEqual(chat.call_args.kwargs, {"draft_sampling": None})

    def test_main_records_and_passes_draft_sampling_overrides_as_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = pathlib.Path(directory) / "draft.gguf"
            draft.write_bytes(b"test draft artifact")
            argv = ["quick_benchmark.py", "--model", "target.gguf", "--draft-model", str(draft),
                    "--draft-horizon", "23", "--speculative-sampling", "lossless",
                    "--draft-temperature", "0.5", "--draft-top-k", "0", "--draft-top-p", "1",
                    "--draft-min-p", "0", "--draft-repeat-penalty", "1.25", "--cooldown", "0", "--json"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(QUICK, "run_local_chat", return_value=[{
                        "decode_attribution": {"speculative_draft_model": "standalone_gemma3_tli"}
                    }]) as chat, \
                    mock.patch.object(QUICK, "run_kernel_diagnostics", return_value={}), \
                    mock.patch.object(QUICK, "collect_environment", return_value={}), \
                    contextlib.redirect_stdout(output):
                QUICK.main()
            expected = dict(temperature=0.5, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.25)
            protocol = json.loads(output.getvalue())["protocol"]
            self.assertEqual(protocol["standalone_draft_sampling_overrides"], expected)
            self.assertEqual(protocol["decode_attribution_schema"], 5)
            self.assertEqual(protocol["seed"], QUICK.SEED)
            self.assertEqual(chat.call_args.kwargs, {"draft_sampling": expected})
            self.assertEqual(chat.call_args.args[-2:], (str(draft), 23))

    def test_main_invalid_draft_scalars_fail_before_artifact_process_or_cooldown(self):
        for extra in (["--draft-temperature", "nan"], ["--draft-top-k", "-1"],
                      ["--draft-top-p", "0"], ["--draft-min-p", "1.1"],
                      ["--draft-repeat-penalty", "1e-300"]):
            argv = ["quick_benchmark.py", "--model", "target.gguf", "--draft-model", "/does-not-exist",
                    "--speculative-sampling", "lossless", *extra]
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(QUICK, "run_local_chat") as chat, \
                    mock.patch.object(QUICK, "run_process") as run, \
                    mock.patch.object(QUICK.time, "sleep") as sleep, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    QUICK.main()
                chat.assert_not_called()
                run.assert_not_called()
                sleep.assert_not_called()

    def test_standalone_tli_metrics_reject_wrong_probability_contract(self):
        record = decode_attribution(12, 0, 2, 0)
        record.update(schema_version=4, generation_completion_ns=10_000_000_000,
                      full_turn_ns=11_000_000_000, first_text_ns=1_000_000_000, learned_owned_bytes=1000,
                      learned_horizon=3, learned_cycles=1, learned_accepted_edges=2,
                      learned_numeric_refusals=0, learned_draft_ns=100, learned_draft_gpu_ns=90,
                      learned_target_gpu_ns=1000, learned_cycle_ns=1200,
                      scheduled_bytes_basis="one_row_equivalent_not_physical_bandwidth",
                      speculative_draft_model="standalone_vicuna_tli", speculative_sampling="lossless",
                      standalone_draft_sampler="target_scalars_native_history_v1",
                      learned_draft_head="f16", learned_verification_schedule="full",
                      standalone_prefill_ns=10, standalone_rebuild_ns=0,
                      standalone_draft_uploaded_bytes=64, standalone_draft_readback_bytes=32,
                      standalone_draft_wall_ns=110, standalone_draft_gpu_ns=95,
                      standalone_draft_command_buffer_count=2, standalone_draft_compute_dispatch_count=40)
        turn = dict(evaluated=12, cached=0, decoded=2, decode_attribution=record)
        QUICK.validate_decode_attribution(turn, "First")
        gemma = copy.deepcopy(turn)
        gemma["decode_attribution"].update(speculative_draft_model="standalone_gemma3_tli",
                                          learned_draft_head="q8_0")
        QUICK.validate_decode_attribution(gemma, "First")
        gemma["decode_attribution"]["learned_draft_head"] = "f16"
        with self.assertRaises(RuntimeError):
            QUICK.validate_decode_attribution(gemma, "First")
        for field, value in (("speculative_draft_model", "unknown"), ("speculative_sampling", "exact-match"),
                             ("learned_horizon", 24), ("learned_draft_head", "q8_0"),
                             ("standalone_draft_gpu_ns", 89), ("standalone_draft_wall_ns", 99),
                             ("standalone_draft_sampler", "unfiltered")):
            bad = copy.deepcopy(turn)
            bad["decode_attribution"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUICK.validate_decode_attribution(bad, "First")

        for field in (key for key in record if key.startswith("standalone_") and key != "standalone_draft_sampler"):
            for value in (None, -1, True, "1"):
                bad = copy.deepcopy(turn)
                bad["decode_attribution"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    QUICK.validate_decode_attribution(bad, "First")

    def test_human_report_labels_captured_stream_mode(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            QUICK.print_human([], {"kernels": []}, False)
        self.assertIn("chat streams: captured pipes;", output.getvalue())
        self.assertIn(
            "local startup warm-up: enabled (16-token prefill + 1-token decode + width-two verifier)",
            output.getvalue(),
        )

    def test_lossless_execution_options_are_independent_and_fail_closed(self):
        choices = {"speculative_schedule": ("conditional", "--speculative-schedule"),
                   "draft_head": ("q8_0", "--draft-head")}
        for field, (value, flag) in choices.items():
            with self.subTest(field=field), \
                    mock.patch.object(QUICK, "run_process", return_value=("out", "err")) as run, \
                    mock.patch.object(QUICK, "parse_local", return_value=[]):
                QUICK.run_local_chat("engine", "model", 0, recurrent_weights="weights",
                                     speculative_sampling="lossless", **{field: value})
                command = run.call_args.args[0]
                self.assertEqual(command[command.index(flag) + 1], value)
                for other, (_, other_flag) in choices.items():
                    if other != field:
                        self.assertNotIn(other_flag, command)
            for mode, option in ((None, value), ("exact-match", value), ("lossless", "invalid")):
                with self.subTest(field=field, mode=mode, option=option), \
                        mock.patch.object(QUICK, "run_process") as run:
                    with self.assertRaises(ValueError):
                        QUICK.run_local_chat("engine", "model", 0, recurrent_weights="weights",
                                             speculative_sampling=mode, **{field: option})
                    run.assert_not_called()

    def test_kernel_schema_accepts_and_prints_isolated_and_sustained_modes(self):
        diagnostics = kernel_diagnostics_fixture()
        QUICK.validate_kernel_diagnostics(diagnostics)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            QUICK.print_human([], diagnostics, False)
        report = output.getvalue()
        self.assertIn("Median/dispatch ms", report)
        self.assertIn("isolated", report)
        self.assertIn("sustained", report)
        self.assertIn("42 serial, buffer-barrier-separated dispatches", report)

    def test_kernel_schema_rejects_unknown_version(self):
        diagnostics = kernel_diagnostics_fixture()
        diagnostics["schema_version"] = 3
        with self.assertRaisesRegex(RuntimeError, "schema version mismatch"):
            QUICK.validate_kernel_diagnostics(diagnostics)

    def test_kernel_schema_rejects_missing_duplicate_and_wrong_dispatch_modes(self):
        missing = kernel_diagnostics_fixture()
        missing["kernels"][0]["measurements"].pop()
        with self.assertRaisesRegex(RuntimeError, "must contain two modes"):
            QUICK.validate_kernel_diagnostics(missing)

        duplicate = kernel_diagnostics_fixture()
        duplicate["kernels"][0]["measurements"][1]["mode"] = "isolated"
        with self.assertRaisesRegex(RuntimeError, "invalid or duplicate mode"):
            QUICK.validate_kernel_diagnostics(duplicate)

        wrong_dispatch = copy.deepcopy(kernel_diagnostics_fixture())
        wrong_dispatch["kernels"][0]["measurements"][1]["dispatches_per_command"] = 41
        with self.assertRaisesRegex(RuntimeError, "dispatch count mismatch"):
            QUICK.validate_kernel_diagnostics(wrong_dispatch)

    def test_kernel_schema_rejects_incorrect_median_normalization(self):
        diagnostics = kernel_diagnostics_fixture()
        diagnostics["kernels"][0]["measurements"][1]["median_per_dispatch_ns"] = 201
        with self.assertRaisesRegex(RuntimeError, "median_per_dispatch_ns normalization mismatch"):
            QUICK.validate_kernel_diagnostics(diagnostics)

    def test_process_driver_captures_real_stdio_path(self):
        stdout, stderr = QUICK.run_process(
            [
                sys.executable,
                "-c",
                "import sys; [print(line.strip() + ' [ Done ]', flush=True) for line in sys.stdin if line.strip() != '/exit']",
            ],
            "[ Done ]",
            timeout=2,
            inter_turn_pause=0,
        )
        self.assertEqual(
            stdout,
            "Hello, is AI getting smarter? [ Done ]\n"
            "Okay, so will it always progress this fast? [ Done ]\n",
        )
        self.assertEqual(stderr, "")

    def test_process_driver_reports_child_failure(self):
        with self.assertRaisesRegex(RuntimeError, "exited with 7"):
            QUICK.run_process(
                [sys.executable, "-c", "raise SystemExit(7)"],
                "[ Done ]",
                timeout=2,
            )

    def test_local_parser_uses_interactive_rates_and_raw_details(self):
        first = json.dumps(decode_attribution(12, 0, 140, 0), separators=(",", ":"))
        second = json.dumps(decode_attribution(13, 151, 135, 151), separators=(",", ":"))
        output = f"""
[ Prompt: 12.0 t/s | Generation: 14.0 t/s | TTFT: 1.000000 s ]
[ Detail: Evaluated: 12 | Cached: 0 | Decoded: 140 | Prompt: 1.000000 s | Generation: 10.000000 s ]
[ Decode: {first} ]
[ Prompt: 130.0 t/s | Generation: 13.5 t/s | TTFT: 0.100000 s ]
[ Detail: Evaluated: 13 | Cached: 151 | Decoded: 135 | Prompt: 0.100000 s | Generation: 10.000000 s ]
[ Decode: {second} ]
"""
        turns = QUICK.parse_local(output)
        self.assertEqual(turns[0]["cached"], 0)
        self.assertEqual(turns[1]["cached"], 151)
        self.assertEqual(turns[1]["generation_ns"], 10_000_000_000)
        self.assertEqual(turns[1]["generation_tps"], 13.5)

    def test_llama_parser_uses_cli_owned_raw_timings_and_cache_prefix(self):
        stdout = """
[ Prompt: 37.5 t/s | Generation: 14.3 t/s ]
[ Prompt: 136.8 t/s | Generation: 13.9 t/s ]
"""
        stderr = """
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 8192, n_keep = 0, task.n_tokens = 12
slot print_timings: prompt eval time =     320.00 ms /    12 tokens (26.67 ms per token, 37.50 tokens per second)
slot print_timings:        eval time =    7000.00 ms /   100 tokens (70.00 ms per token, 14.29 tokens per second)
slot update_slots: id  0 | task 1 | new prompt, n_ctx_slot = 8192, n_keep = 0, task.n_tokens = 111
slot print_timings: prompt eval time =      95.00 ms /    13 tokens (7.31 ms per token, 136.84 tokens per second)
slot print_timings:        eval time =   10800.00 ms /   150 tokens (72.00 ms per token, 13.89 tokens per second)
"""
        turns = REFERENCE.parse_llama(stdout + stderr)
        self.assertEqual(turns[0]["cached"], 0)
        self.assertEqual(turns[1]["cached"], 98)
        self.assertEqual(turns[0]["ttft_ns"], turns[0]["prompt_ns"])
        self.assertEqual(turns[1]["decoded"], 150)

    def test_parser_rejects_a_rate_not_backed_by_raw_counters(self):
        first = json.dumps(decode_attribution(12, 0, 140, 0), separators=(",", ":"))
        second = json.dumps(decode_attribution(13, 151, 135, 151), separators=(",", ":"))
        output = f"""
[ Prompt: 99.0 t/s | Generation: 14.0 t/s | TTFT: 1.000000 s ]
[ Detail: Evaluated: 12 | Cached: 0 | Decoded: 140 | Prompt: 1.000000 s | Generation: 10.000000 s ]
[ Decode: {first} ]
[ Prompt: 130.0 t/s | Generation: 13.5 t/s | TTFT: 0.100000 s ]
[ Detail: Evaluated: 13 | Cached: 151 | Decoded: 135 | Prompt: 0.100000 s | Generation: 10.000000 s ]
[ Decode: {second} ]
"""
        with self.assertRaisesRegex(RuntimeError, "raw counters imply"):
            QUICK.parse_local(output)

    def test_decode_attribution_rejects_inconsistent_bytes(self):
        turn = {
            "evaluated": 12,
            "cached": 0,
            "decoded": 2,
            "decode_attribution": decode_attribution(12, 0, 2, 0),
        }
        turn["decode_attribution"]["scheduled_bytes_total"] += 1
        with self.assertRaisesRegex(RuntimeError, "scheduled byte total mismatch"):
            QUICK.validate_decode_attribution(turn, "First")

    def test_environment_manifest_identifies_model_build_and_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "tiny.gguf"
            model.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 2, 4) + b"payload")

            def runner(command, **_kwargs):
                values = {
                    ("sysctl", "-n", "hw.model"): "Mac16,12\n",
                    ("sw_vers", "-productVersion"): "26.5.2\n",
                    ("sw_vers", "-buildVersion"): "25F84\n",
                    ("system_profiler", "-json", "SPDisplaysDataType"):
                        '{"SPDisplaysDataType":[{"sppci_model":"Apple M4"}]}',
                }
                return types.SimpleNamespace(stdout=values[tuple(command)])

            environment = QUICK.collect_environment(
                str(model), "build/release/local-gguf-generate",
                "build/release/local-gguf-bench", kernel_diagnostics_fixture(), runner,
            )
            self.assertEqual(environment["engine_build"], "test-build")
            self.assertEqual(environment["model"]["gguf_version"], 3)
            self.assertEqual(environment["model"]["tensor_count"], 2)
            self.assertEqual(environment["hardware"]["metal_device"], "Apple M4")
            self.assertEqual(environment["os"]["build"], "25F84")
            self.assertEqual(environment["executables"]["build_mode"], "release")

    def test_protocol_is_exact_two_turn_natural_sampled_chat(self):
        self.assertEqual(QUICK.DEFAULT_COOLDOWN_SECONDS, 240.0)
        self.assertEqual(
            QUICK.PROMPTS,
            (
                "Hello, is AI getting smarter?",
                "Okay, so will it always progress this fast?",
            ),
        )
        self.assertEqual(QUICK.SEED, 42)
        self.assertEqual(REFERENCE.REFERENCE_COMMIT, "d48a56effbba99944a677938d4beb63a0065ecdf")
        source = (SCRIPTS / "quick_benchmark.py").read_text()
        self.assertIn('"local_startup_warmup": True', source)
        self.assertIn('"local_startup_warmup_prefill_tokens": 16', source)
        self.assertIn('"local_startup_warmup_decode_tokens": 1', source)

    def test_local_and_reference_entrypoints_are_separate(self):
        local_source = (SCRIPTS / "quick_benchmark.py").read_text()
        reference_source = (SCRIPTS / "llama_reference_benchmark.py").read_text()
        self.assertNotIn("llama-cli", local_source)
        self.assertNotIn("local-gguf-generate", reference_source)


if __name__ == "__main__":
    unittest.main()

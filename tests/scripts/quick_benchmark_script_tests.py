import contextlib
import copy
import importlib.util
import io
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


class QuickBenchmarkScriptTests(unittest.TestCase):
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

    def test_human_report_labels_captured_stream_mode(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            QUICK.print_human([], {"kernels": []}, False)
        self.assertIn("chat streams: captured pipes;", output.getvalue())
        self.assertIn(
            "local startup warm-up: enabled (16-token prefill + 1-token decode + width-two verifier)",
            output.getvalue(),
        )

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

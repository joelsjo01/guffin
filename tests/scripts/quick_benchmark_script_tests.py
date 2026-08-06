import contextlib
import importlib.util
import io
import pathlib
import sys
import unittest


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


class QuickBenchmarkScriptTests(unittest.TestCase):
    def test_human_report_labels_captured_stream_mode(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            QUICK.print_human([], {"kernels": []}, False)
        self.assertIn("chat streams: captured pipes;", output.getvalue())

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
        output = """
[ Prompt: 12.0 t/s | Generation: 14.0 t/s | TTFT: 1.000000 s ]
[ Detail: Evaluated: 12 | Cached: 0 | Decoded: 140 | Prompt: 1.000000 s | Generation: 10.000000 s ]
[ Prompt: 130.0 t/s | Generation: 13.5 t/s | TTFT: 0.100000 s ]
[ Detail: Evaluated: 13 | Cached: 151 | Decoded: 135 | Prompt: 0.100000 s | Generation: 10.000000 s ]
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
        output = """
[ Prompt: 99.0 t/s | Generation: 14.0 t/s | TTFT: 1.000000 s ]
[ Detail: Evaluated: 12 | Cached: 0 | Decoded: 140 | Prompt: 1.000000 s | Generation: 10.000000 s ]
[ Prompt: 130.0 t/s | Generation: 13.5 t/s | TTFT: 0.100000 s ]
[ Detail: Evaluated: 13 | Cached: 151 | Decoded: 135 | Prompt: 0.100000 s | Generation: 10.000000 s ]
"""
        with self.assertRaisesRegex(RuntimeError, "raw counters imply"):
            QUICK.parse_local(output)

    def test_protocol_is_exact_two_turn_natural_sampled_chat(self):
        self.assertEqual(
            QUICK.PROMPTS,
            (
                "Hello, is AI getting smarter?",
                "Okay, so will it always progress this fast?",
            ),
        )
        self.assertEqual(QUICK.SEED, 42)
        self.assertEqual(REFERENCE.REFERENCE_COMMIT, "d48a56effbba99944a677938d4beb63a0065ecdf")

    def test_local_and_reference_entrypoints_are_separate(self):
        local_source = (SCRIPTS / "quick_benchmark.py").read_text()
        reference_source = (SCRIPTS / "llama_reference_benchmark.py").read_text()
        self.assertNotIn("llama-cli", local_source)
        self.assertNotIn("local-gguf-generate", reference_source)


if __name__ == "__main__":
    unittest.main()

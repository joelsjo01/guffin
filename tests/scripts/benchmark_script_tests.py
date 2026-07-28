import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LLAMA = load_module("compare_llama_conversation", ROOT / "scripts/compare_llama_conversation.py")
CALIBRATION = load_module("calibrate_local_chat_ttft", ROOT / "scripts/calibrate_local_chat_ttft.py")


class BenchmarkScriptTests(unittest.TestCase):
    def test_local_calibration_parses_in_process_ttft(self):
        self.assertEqual(
            CALIBRATION.parse_frontend_ttft(b"[ Prompt: 1 | Frontend TTFT: 12345 ns ]"),
            12345,
        )

    def test_reference_statistics_and_paired_local_prompt_loading(self):
        self.assertEqual(LLAMA.summarize([1, 2, 3, 4, 100])["median"], 3.0)
        self.assertEqual(LLAMA.summarize([1, 2, 3, 4, 100])["mad"], 1.0)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as output:
            output.write(json.dumps({"record_kind": "setup", "model_identity": "model"}) + "\n")
            for repetition in range(2):
                output.write(json.dumps({
                    "record_kind": "turn", "warmup": False,
                    "repetition_index": repetition, "turn_index": 0,
                    "memory_audit": False, "rendered_prompt_token_ids": [repetition],
                    "reused_tokens": repetition, "selected_token_ids": [100 + repetition],
                }) + "\n")
            output.flush()
            prompts, reused, selected, identity = LLAMA.load_local_records(output.name)
        self.assertEqual(prompts[(False, 0, 0)], [0])
        self.assertEqual(prompts[(False, 1, 0)], [1])
        self.assertEqual(reused[(False, 1, 0)], 1)
        self.assertEqual(selected[(False, 1, 0)], [101])
        self.assertEqual(identity["model_identity"], "model")

    def write_local_run(self, path, speed, changed_token=False):
        identity = {
            "schema_version": 3, "scenario_version": 1,
            "model_identity": "model", "backend": "metal", "mode": "greedy",
            "seed": 42, "temperature": 0.0, "top_k": 40, "top_p": 0.95,
            "min_p": 0.05, "repeat_penalty": 1.0,
        }
        records = [{**identity, "record_kind": "setup"}]
        for repetition in range(5):
            ttfts = []
            for turn in range(4):
                latency = int((1_000_000 + turn * 100_000 + repetition * 1_000) * speed)
                ttfts.append(latency)
                records.append({
                    **identity, "record_kind": "turn", "timing_eligible": True,
                    "memory_audit": False, "repetition_index": repetition, "turn_index": turn,
                    "rendered_prompt_token_ids": [turn], "evaluated_suffix_token_ids": [turn],
                    "selected_token_ids": [999 if changed_token and repetition == 0 and turn == 0 else turn],
                    "visible_token_ids": [turn], "final_text": str(turn), "stop_reason": "token_limit",
                    "stop_token_id": None, "interrupted": False, "initial_context_length": turn,
                    "final_context_length": turn + 1, "reused_tokens": turn, "trimmed_turns": 0,
                    "rebuilt_context": False, "configured_stop_token_ids": [1],
                    "configured_reverse_stop_strings": [], "resident_weight_bytes": 10,
                    "resident_kv_bytes": 20, "resident_scratch_bytes": 30, "plan_revision": 0,
                    "frontend_complete_ttft_ns": latency, "total_ns": latency * 2,
                })
            records.append({
                **identity, "record_kind": "conversation", "timing_eligible": True,
                "repetition_index": repetition,
                "per_turn_frontend_complete_ttft_ns": ttfts,
            })
        for turn in range(4):
            records.append({
                **identity, "record_kind": "turn", "timing_eligible": False,
                "memory_audit": True, "repetition_index": 0, "turn_index": turn,
                "metal_current_allocated_bytes": 100, "process_physical_footprint_bytes": 200,
                "process_physical_footprint_peak_bytes": 300,
            })
        records.append({**identity, "record_kind": "validation", "valid": True, "errors": []})
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_local_retention_gate_accepts_only_exact_faster_work(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            baseline = directory / "baseline.jsonl"
            candidate = directory / "candidate.jsonl"
            self.write_local_run(baseline, 1.0)
            self.write_local_run(candidate, 0.95)
            command = [str(ROOT / "scripts/compare_local_conversation.py"), str(baseline), str(candidate)]
            accepted = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.write_local_run(candidate, 0.95, changed_token=True)
            rejected = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("selected_token_ids", rejected.stdout)


if __name__ == "__main__":
    unittest.main()

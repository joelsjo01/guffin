import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("memory_benchmark", ROOT / "scripts" / "memory_benchmark.py")
MEMORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY)


class MemoryBenchmarkTests(unittest.TestCase):
    def test_parses_ps_snapshot(self):
        self.assertEqual(
            MEMORY.parse_ps_snapshot(" 1024 4096\n"),
            {"ready_rss_bytes": 1024 * 1024, "ready_virtual_bytes": 4096 * 1024},
        )
        with self.assertRaises(RuntimeError):
            MEMORY.parse_ps_snapshot("0 1")

    def test_parses_and_validates_decode_memory_accounting(self):
        record = {
            "resident_weights_bytes": 10,
            "resident_kv_bytes": 20,
            "resident_scratch_bytes": 30,
            "warmup_kv_clear_passes": 0,
            "warmup_host_cleared_kv_bytes": 0,
        }
        output = b"[ Decode: " + json.dumps(record, separators=(",", ":")).encode() + b" ]\n"
        self.assertEqual(MEMORY.parse_decode_record(output), record)
        record["warmup_kv_clear_passes"] = 1
        output = b"[ Decode: " + json.dumps(record).encode() + b" ]\n"
        with self.assertRaises(RuntimeError):
            MEMORY.parse_decode_record(output)

    def test_summarizes_raw_samples(self):
        samples = [
            {"startup_to_ready_ns": 3, "ready_rss_bytes": 30, "peak_rss_bytes": 50,
             "page_reclaims": 7, "page_faults": 1},
            {"startup_to_ready_ns": 1, "ready_rss_bytes": 10, "peak_rss_bytes": 40,
             "page_reclaims": 5, "page_faults": 0},
            {"startup_to_ready_ns": 2, "ready_rss_bytes": 20, "peak_rss_bytes": 60,
             "page_reclaims": 6, "page_faults": 2},
        ]
        self.assertEqual(MEMORY.summarize(samples)["median_ready_rss_bytes"], 20)
        self.assertEqual(MEMORY.summarize(samples)["median_startup_to_ready_ns"], 2)


if __name__ == "__main__":
    unittest.main()

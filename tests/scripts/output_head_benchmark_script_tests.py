import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("output_head_benchmark", SCRIPTS / "output_head_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def turn(name, rate, prompt=100.0, ttft=100):
    return {
        "name": name,
        "decoded": 100,
        "generation_ns": int(100e9 / rate),
        "generation_tps": rate,
        "prompt_tps": prompt,
        "ttft_ns": ttft,
    }


class OutputHeadBenchmarkTests(unittest.TestCase):
    def test_decision_requires_two_percent_and_four_wins(self):
        baseline = [[turn("First", 10.0), turn("Follow-up", 10.0)] for _ in range(5)]
        candidate = [[turn("First", 10.3), turn("Follow-up", 10.3)] for _ in range(5)]
        decision = MODULE.decide({"f16": baseline, "q8_0": candidate})
        self.assertTrue(decision["retain"])
        self.assertEqual(decision["paired_wins"], 5)

    def test_decision_rejects_prompt_regression(self):
        baseline = [[turn("First", 10.0), turn("Follow-up", 10.0)] for _ in range(5)]
        candidate = [[turn("First", 11.0, prompt=97.0), turn("Follow-up", 11.0)] for _ in range(5)]
        self.assertFalse(MODULE.decide({"f16": baseline, "q8_0": candidate})["retain"])


if __name__ == "__main__":
    unittest.main()

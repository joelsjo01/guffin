import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "storage_policy_benchmark", ROOT / "scripts" / "storage_policy_benchmark.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(policy, durations):
    return {
        "schema_version": 1,
        "policy": policy,
        "kernels": [
            {
                "name": name,
                "weight_bytes": 100,
                "scheduled_class_weight_bytes": weight,
                "measurements": [
                    {"mode": "isolated", "median_per_dispatch_ns": duration * 2},
                    {"mode": "sustained", "median_per_dispatch_ns": duration},
                ],
            }
            for name, weight, duration in zip(MODULE.CLASSES, (4, 3, 2, 1), durations, strict=True)
        ],
    }


class StoragePolicyBenchmarkTests(unittest.TestCase):
    def test_parse_and_weighted_summary(self):
        control = result(MODULE.CONTROL, (100, 100, 100, 100))
        candidate = result("private-untracked", (95, 95, 95, 95))
        parsed = MODULE.parse_result(json.dumps(candidate), "private-untracked")
        self.assertEqual(MODULE.weighted_sustained_gbps(parsed), 100 / 95)
        summary = MODULE.summarize_pairs([(control, candidate)] * 3)
        self.assertTrue(summary["passes_three_pair_screen"])
        self.assertEqual(summary["paired_wins"], 3)

    def test_rejects_malformed_results_and_class_regression(self):
        with self.assertRaises(ValueError):
            MODULE.parse_result("not-json", MODULE.CONTROL)
        control = result(MODULE.CONTROL, (100, 100, 100, 100))
        candidate = result("private-untracked", (90, 90, 90, 110))
        summary = MODULE.summarize_pairs([(control, candidate)] * 3)
        self.assertFalse(summary["passes_three_pair_screen"])
        self.assertLess(summary["worst_class_regression_percent"], -2)


if __name__ == "__main__":
    unittest.main()

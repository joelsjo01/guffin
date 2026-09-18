import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SpeculationValidationStatisticsTests(unittest.TestCase):
    @staticmethod
    def _identity(value):
        return f"{value:064x}"

    def _observations(self):
        from training.speculation.manifest import WORKLOADS
        from training.speculation.validation_stats import ValidationObservation

        seeds = (11, 22, 33, 44, 55)
        observations = []

        def append(workload, cluster, conversation, seed, loss, deterministic, common):
            observations.append(ValidationObservation(
                workload=workload,
                source_cluster=self._identity(cluster),
                conversation=self._identity(conversation),
                generation_seed=seed,
                loss=loss,
                deterministic_prefix=deterministic,
                common_draw_prefix=common,
            ))

        first = "instruction_chat"
        append(first, 11, 111, 11, 0.0, 0, 1)
        append(first, 11, 111, 11, 10.0, 2, 3)
        for seed in seeds[1:]:
            append(first, 11, 111, seed, 0.0, 0, 0)
        for seed in seeds:
            append(first, 11, 112, seed, 3.0, 2, 1)
            append(first, 12, 121, seed, 10.0, 3, 2)

        for workload, code in WORKLOADS.items():
            if workload == first:
                continue
            for seed in seeds:
                append(workload, 100 + code, 1000 + code, seed, float(code), 1, 2)
        return seeds, observations

    def test_hierarchical_aggregation_equalizes_every_frozen_level(self):
        from training.speculation.manifest import WORKLOADS
        from training.speculation.validation_stats import (
            VALIDATION_AGGREGATION,
            hierarchical_validation_summary,
        )

        seeds, observations = self._observations()
        weights = {workload: 0.125 for workload in WORKLOADS}
        summary = hierarchical_validation_summary(observations, weights, seeds, 3)

        self.assertEqual(summary["aggregation"], VALIDATION_AGGREGATION)
        self.assertEqual(summary["origin_count"], 51)
        self.assertEqual(summary["conversation_seed_count"], 50)
        self.assertEqual(summary["conversation_count"], 10)
        self.assertEqual(summary["source_cluster_count"], 9)
        self.assertEqual(summary["weighted_metrics"], {
            "loss": 5.125,
            "deterministic_prefix": 1.13125,
            "common_draw_prefix": 1.91875,
        })

        first = summary["workloads"]["instruction_chat"]
        self.assertEqual(first["origin_count"], 16)
        self.assertEqual(first["conversation_seed_count"], 15)
        self.assertEqual(first["conversation_count"], 3)
        self.assertEqual(first["source_cluster_count"], 2)
        self.assertEqual(first["metrics"], {
            "loss": 6.0,
            "deterministic_prefix": 2.05,
            "common_draw_prefix": 1.35,
        })
        self.assertEqual(first["seeds"]["11"]["metrics"], {
            "loss": 7.0,
            "deterministic_prefix": 2.25,
            "common_draw_prefix": 1.75,
        })
        self.assertEqual(first["seeds"]["22"]["metrics"], {
            "loss": 5.75,
            "deterministic_prefix": 2.0,
            "common_draw_prefix": 1.25,
        })
        self.assertEqual(first["clusters"][self._identity(11)]["metrics"], {
            "loss": 2.0,
            "deterministic_prefix": 1.1,
            "common_draw_prefix": 0.7,
        })
        self.assertEqual(
            summary,
            hierarchical_validation_summary(reversed(observations), weights, seeds, 3),
        )

    def test_hierarchical_aggregation_rejects_incomplete_or_ambiguous_groups(self):
        import dataclasses
        from training.speculation.manifest import WORKLOADS
        from training.speculation.validation_stats import (
            ValidationStatisticsError,
            hierarchical_validation_summary,
        )

        seeds, observations = self._observations()
        weights = {workload: 0.125 for workload in WORKLOADS}

        missing_seed = [
            row for row in observations
            if not (
                row.conversation == self._identity(111)
                and row.generation_seed == seeds[-1]
            )
        ]
        with self.assertRaisesRegex(ValidationStatisticsError, "complete generation-seed"):
            hierarchical_validation_summary(missing_seed, weights, seeds, 3)

        changed_conversation = list(observations)
        changed_conversation[-1] = dataclasses.replace(
            changed_conversation[-1], conversation=observations[0].conversation
        )
        with self.assertRaisesRegex(ValidationStatisticsError, "changes workload/cluster"):
            hierarchical_validation_summary(changed_conversation, weights, seeds, 3)

        # A deduplication cluster may legitimately contain conversations that
        # were assigned different workloads. It remains a separate cluster
        # unit inside each workload, while the global unique count is honest.
        crossed_cluster = [
            dataclasses.replace(row, source_cluster=observations[0].source_cluster)
            if row.workload == "repeated_retrieval" else row
            for row in observations
        ]
        crossed_summary = hierarchical_validation_summary(
            crossed_cluster, weights, seeds, 3
        )
        self.assertEqual(crossed_summary["source_cluster_count"], 8)
        self.assertIn(
            observations[0].source_cluster,
            crossed_summary["workloads"]["repeated_retrieval"]["clusters"],
        )

    def test_hierarchical_aggregation_rejects_bad_measurements_and_coverage(self):
        import dataclasses
        from training.speculation.manifest import WORKLOADS
        from training.speculation.validation_stats import (
            ValidationStatisticsError,
            hierarchical_validation_summary,
        )

        seeds, observations = self._observations()
        weights = {workload: 0.125 for workload in WORKLOADS}
        bad_rows = (
            dataclasses.replace(observations[0], loss=float("nan")),
            dataclasses.replace(observations[0], deterministic_prefix=4),
            dataclasses.replace(observations[0], common_draw_prefix=-1),
            dataclasses.replace(observations[0], source_cluster="ABC"),
            dataclasses.replace(observations[0], generation_seed=999),
        )
        for bad in bad_rows:
            with self.subTest(bad=bad):
                changed = list(observations)
                changed[0] = bad
                with self.assertRaisesRegex(ValidationStatisticsError, "observation is malformed"):
                    hierarchical_validation_summary(changed, weights, seeds, 3)

        missing_workload = [
            row for row in observations if row.workload != "rare_non_english"
        ]
        with self.assertRaisesRegex(ValidationStatisticsError, "every frozen workload"):
            hierarchical_validation_summary(missing_workload, weights, seeds, 3)

        with self.assertRaisesRegex(ValidationStatisticsError, "workload weights are invalid"):
            hierarchical_validation_summary(
                observations,
                {workload: 0.25 for workload in WORKLOADS},
                seeds,
                3,
            )


if __name__ == "__main__":
    unittest.main()

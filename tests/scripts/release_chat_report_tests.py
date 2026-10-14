import copy
import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_chat_report", ROOT / "scripts" / "release_chat_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fixture():
    configs = [{"id": "guffin-default", "engine": "guffin", "mode": "default", "model_sha256": "a" * 64},
               {"id": "llama-default", "engine": "llama", "mode": "default", "model_sha256": "a" * 64}]
    case = {"id": "code<&>", "category": "coding", "suite": "core", "turns": ["Write it.", "Explain it."]}
    manifest = {"schema_version": 1, "campaign_id": "release", "configurations": configs, "cases": [case],
                "jobs": [{"configuration_id": config["id"], "case_id": case["id"], "run_id": config["id"] + "__code"} for config in configs],
                "timing_conventions": {"generation": "first selection to last selection"}}
    receipts = []
    for config in configs:
        turns = []
        for index, (steps, duration, prompt_tokens) in enumerate(((10, 1_000_000_000, 20), (30, 9_000_000_000, 10))):
            turns.append({"index": index, "prompt": case["turns"][index], "answer": "An answer.",
                          "metrics": {"prompt_tokens": prompt_tokens, "cached_tokens": index * 50,
                                      "selected_tokens": steps + 1, "generation_steps": steps,
                                      "prompt_ns": prompt_tokens * 10_000_000, "internal_ttft_ns": (2 - index) * 100_000_000,
                                      "generation_ns": duration, "generation_tps": 999, "native_generation_tps": 999,
                                      "engine_turn_ns": duration + 200_000_000, "wall_turn_ns": duration + 250_000_000},
                          "memory": {"peak_physical_footprint_bytes": 2**30}})
        receipts.append({"schema_version": 1, "campaign_id": "release", "run_id": config["id"] + "__code",
                         "configuration": config, "case": case, "status": "complete", "turns": turns,
                         "memory": {"peak_physical_footprint_bytes": 2 * 2**30, "peak_rss_bytes": 2**30}})
    return manifest, receipts


class ReleaseChatReportTests(unittest.TestCase):
    def test_weighted_rates_use_normalized_steps_and_exact_clocks(self):
        manifest, receipts = fixture()
        summary = REPORT.summarize_run(manifest, receipts)
        config = summary["configurations"][0]
        self.assertEqual(config["metrics"]["generation_tps"], 4)
        self.assertEqual(config["metrics"]["selected_tokens"], 42)
        self.assertEqual(config["metrics"]["prompt_tps"], 100)
        self.assertEqual(config["metrics"]["median_internal_ttft_ns"], 150_000_000)
        self.assertEqual(config["turn_positions"][0]["generation_tps"], 10)
        self.assertAlmostEqual(config["turn_positions"][1]["generation_tps"], 30 / 9)
        self.assertEqual(config["categories"][0]["generation_tps"], 4)
        self.assertEqual(summary["turns"][0]["native_generation_tps"], 999)
        self.assertEqual(summary["comparisons"][0]["generation_rate_ratio"], 1)
        self.assertEqual(config["peak_physical_footprint_bytes"], 2 * 2**30)
        pair = summary["paired_suites"][0]
        self.assertEqual(pair["complete_case_ids"], [manifest["cases"][0]["id"]])
        self.assertEqual(pair["candidate_metrics"]["generation_tps"], config["metrics"]["generation_tps"])
        self.assertEqual(pair["generation_rate_ratio"], 1)

    def test_partial_failure_preserves_turns_but_cannot_inflate_headline(self):
        manifest, receipts = fixture()
        receipts[0]["status"] = "failed"
        receipts[0]["turns"] = receipts[0]["turns"][:1]
        receipts[0]["error"] = "follow-up failed"
        summary = REPORT.summarize_run(manifest, receipts)
        self.assertIsNone(summary["configurations"][0]["metrics"]["generation_tps"])
        self.assertEqual(summary["conversations"][0]["metrics"]["generation_tps"], 10)
        self.assertEqual(summary["failures"][0]["recorded_turns"], 1)
        self.assertEqual(summary["turns"][0]["conversation_status"], "failed")
        self.assertEqual(summary["comparisons"], [])
        self.assertEqual(summary["paired_suites"][0]["complete_conversations"], 0)
        self.assertIsNone(summary["paired_suites"][0]["baseline_metrics"]["generation_tps"])

    def test_capped_reply_is_distinct_from_natural_completion(self):
        manifest, receipts = fixture()
        receipts[0]["turns"][1]["metrics"]["stop_reason"] = "token_limit"
        summary = REPORT.summarize_run(manifest, receipts)
        self.assertEqual(summary["conversations"][0]["status"], "capped")
        self.assertEqual(summary["configurations"][0]["complete_conversations"], 0)
        self.assertEqual(summary["turns"][1]["turn_status"], "capped")

    def test_known_early_stops_remain_visible_but_are_excluded_from_complete_workloads(self):
        for reason in ("interrupted", "callback_stop", "reverse_stop"):
            with self.subTest(reason=reason):
                manifest, receipts = fixture()
                receipts[0]["turns"][1]["metrics"]["stop_reason"] = reason
                summary = REPORT.summarize_run(manifest, receipts)
                self.assertEqual(summary["conversations"][0]["status"], "stopped_short")
                self.assertEqual(summary["turns"][1]["stop_reason"], reason)
                self.assertEqual(summary["conversations"][0]["metrics"]["generation_tps"], 4)
                self.assertIsNone(summary["configurations"][0]["metrics"]["generation_tps"])
                self.assertEqual(summary["paired_suites"][0]["complete_conversations"], 0)
                self.assertEqual(summary["comparisons"], [])
                self.assertIn("known limited, interrupted", REPORT.render_markdown(summary))

    def test_default_and_speculative_or_different_model_are_not_compared(self):
        manifest, receipts = fixture()
        for config in (manifest["configurations"][1], receipts[1]["configuration"]):
            config["mode"] = "speculative"
        self.assertEqual(REPORT.summarize_run(manifest, receipts)["comparisons"], [])
        manifest, receipts = fixture()
        for config in (manifest["configurations"][1], receipts[1]["configuration"]):
            config["model_sha256"] = "b" * 64
        self.assertEqual(REPORT.summarize_run(manifest, receipts)["comparisons"], [])

    def test_duplicate_stale_and_unplanned_receipts_are_rejected(self):
        manifest, receipts = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate receipt"):
            REPORT.summarize_run(manifest, receipts + [receipts[0]])
        for mutate in (lambda receipt: receipt.update(run_id="previous-run"),
                       lambda receipt: receipt.update(campaign_id="previous-campaign"),
                       lambda receipt: receipt["case"].update(id="other-case"),
                       lambda receipt: receipt["configuration"].update(model_sha256="b" * 64)):
            manifest, receipts = fixture()
            receipt = copy.deepcopy(receipts[0])
            mutate(receipt)
            with self.assertRaises(ValueError):
                REPORT.summarize_run(manifest, [receipt])

    def test_changed_execution_or_submitted_text_is_rejected(self):
        changes = (
            lambda receipt: receipt["configuration"].update(command=["different-executable"]),
            lambda receipt: receipt["case"].update(category="changed"),
            lambda receipt: receipt["turns"][0].update(prompt="Changed user message."),
        )
        for change in changes:
            manifest, receipts = fixture()
            receipts = copy.deepcopy(receipts)
            change(receipts[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                REPORT.summarize_run(manifest, receipts)

    def test_answer_hash_detects_changed_output(self):
        manifest, receipts = fixture()
        turn = receipts[0]["turns"][0]
        turn["answer_sha256"] = hashlib.sha256(turn["answer"].encode()).hexdigest()
        REPORT.summarize_run(manifest, receipts)
        turn["answer"] = "Changed answer."
        with self.assertRaisesRegex(ValueError, "answer hash"):
            REPORT.summarize_run(manifest, receipts)

    def test_fractional_counts_and_inconsistent_generation_steps_are_rejected(self):
        for field, value in (("selected_tokens", 11.5), ("prompt_ns", 1.5),
                             ("generation_steps", 11)):
            manifest, receipts = fixture()
            receipts[0]["turns"][0]["metrics"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                REPORT.summarize_run(manifest, receipts)

    def test_unattempted_jobs_are_visible_and_not_fake_empty_replies(self):
        manifest, receipts = fixture()
        summary = REPORT.summarize_run(manifest, receipts[:1])
        self.assertEqual(summary["failures"][0]["status"], "unattempted")
        self.assertEqual(summary["configurations"][1]["unattempted_conversations"], 1)
        self.assertIsNone(summary["configurations"][1]["metrics"]["selected_tokens"])
        self.assertEqual(len(summary["turns"]), 2)
        self.assertIn("unattempted", REPORT.render_markdown(summary))

    def test_incomplete_counts_or_nonfinite_metrics_do_not_create_rates(self):
        manifest, receipts = fixture()
        del receipts[0]["turns"][0]["metrics"]["generation_steps"]
        self.assertIsNone(REPORT.summarize_run(manifest, receipts)["configurations"][0]["metrics"]["generation_tps"])
        for invalid in (float("nan"), float("inf"), -1, True):
            manifest, receipts = fixture()
            receipts[0]["turns"][0]["metrics"]["generation_ns"] = invalid
            with self.assertRaises(ValueError):
                REPORT.summarize_run(manifest, receipts)
        manifest, receipts = fixture()
        receipts[0]["turns"].pop()
        with self.assertRaisesRegex(ValueError, "missing turns"):
            REPORT.summarize_run(manifest, receipts)

    def test_unknown_termination_stays_unknown_and_configuration_is_preserved(self):
        manifest, receipts = fixture()
        manifest["configurations"][0]["native_cli_output_cap"] = 4096
        receipts[0]["turns"][0]["metrics"]["stop_reason"] = "stop_token"
        summary = REPORT.summarize_run(manifest, receipts)
        self.assertEqual(summary["configurations"][0]["metrics"]["termination_counts"],
                         {"stop_token": 1, "unknown": 1})
        self.assertEqual(summary["configurations"][0]["configuration"]["native_cli_output_cap"], 4096)
        self.assertIsNone(summary["turns"][1]["stop_reason"])
        self.assertIn("Unknown stop reasons are not assumed to be EOS", REPORT.render_markdown(summary))

    def test_core_and_supplement_rates_cannot_blend_into_one_headline(self):
        manifest, receipts = fixture()
        extra_case = {"id": "long-supplement", "category": "summarization", "suite": "supplemental", "turns": ["Summarize."]}
        manifest["cases"].append(extra_case)
        manifest["jobs"].append({"configuration_id": "guffin-default", "case_id": extra_case["id"], "run_id": "guffin__extra"})
        extra = copy.deepcopy(receipts[0])
        extra.update(case=extra_case, run_id="guffin__extra", turns=extra["turns"][:1])
        extra["turns"][0]["prompt"] = extra_case["turns"][0]
        extra["turns"][0]["metrics"].update(generation_steps=1000, selected_tokens=1001, generation_ns=200_000_000)
        receipts.append(extra)
        summary = REPORT.summarize_run(manifest, receipts)
        suites = {suite["suite"]: suite for suite in summary["configurations"][0]["suites"]}
        self.assertEqual(suites["core"]["generation_tps"], 4)
        self.assertEqual(suites["supplemental"]["generation_tps"], 5000)
        markdown = REPORT.render_markdown(summary)
        self.assertIn("| core | default | guffin-default / llama-default | 1 / 1 | 4.00 | 4.00 | 1.00× |", markdown)
        self.assertIn("| supplemental | 1 / 1 | 5000.00 |", markdown)
        self.assertEqual(len(summary["paired_suites"]), 1)
        self.assertIn("Independent completed workload", markdown)

    def test_asymmetric_failure_keeps_headline_and_charts_on_jointly_complete_cases(self):
        manifest, receipts = fixture()
        extra_case = {"id": "unpaired-fast-case", "category": "coding", "suite": "core", "turns": ["Write more."]}
        manifest["cases"].append(extra_case)
        for config in manifest["configurations"]:
            job_id = config["id"] + "__extra"
            manifest["jobs"].append({"configuration_id": config["id"], "case_id": extra_case["id"], "run_id": job_id})
            extra = copy.deepcopy(receipts[0])
            extra.update(configuration=config, case=extra_case, run_id=job_id, turns=extra["turns"][:1])
            extra["turns"][0]["prompt"] = extra_case["turns"][0]
            extra["turns"][0]["metrics"].update(generation_steps=1000, selected_tokens=1001, generation_ns=1_000_000_000,
                                                internal_ttft_ns=99_000_000_000)
            extra["memory"]["peak_physical_footprint_bytes"] = 99 * 2**30
            if config["engine"] == "llama":
                extra.update(status="failed", error="case failed")
            receipts.append(extra)
        summary = REPORT.summarize_run(manifest, receipts)
        self.assertAlmostEqual(summary["configurations"][0]["metrics"]["generation_tps"], 1040 / 11)
        self.assertEqual(summary["configurations"][1]["metrics"]["generation_tps"], 4)
        pair = summary["paired_suites"][0]
        self.assertEqual(pair["complete_case_ids"], [manifest["cases"][0]["id"]])
        self.assertEqual(pair["planned_conversations"], 2)
        self.assertEqual(pair["complete_conversations"], 1)
        self.assertEqual(pair["candidate_metrics"]["generation_tps"], 4)
        self.assertEqual(pair["baseline_metrics"]["generation_tps"], 4)
        self.assertEqual(pair["candidate_metrics"]["peak_physical_footprint_bytes"], 2 * 2**30)
        self.assertEqual(pair["candidate_metrics"]["median_internal_ttft_ns"], 150_000_000)
        self.assertEqual(pair["generation_rate_ratio"], 1)
        self.assertEqual(len(summary["comparisons"]), 1)
        markdown = REPORT.render_markdown(summary)
        self.assertIn("| core | default | guffin-default / llama-default | 1 / 2 | 4.00 | 4.00 | 1.00× |", markdown)
        with tempfile.TemporaryDirectory() as temporary:
            REPORT.write_report(summary, temporary)
            for filename in ("generation.svg", "ttft.svg", "memory.svg"):
                chart = (pathlib.Path(temporary) / filename).read_text()
                self.assertNotIn(extra_case["id"], chart)
                self.assertIn("jointly complete", chart)

    def test_native_memory_high_water_and_collection_errors_are_explicit(self):
        manifest, receipts = fixture()
        receipts[0]["memory"].update(footprint_basis="macOS lifetime maximum physical footprint", error="final observation unavailable")
        summary = REPORT.summarize_run(manifest, receipts)
        self.assertEqual(summary["conversations"][0]["memory"]["footprint_basis"], receipts[0]["memory"]["footprint_basis"])
        self.assertEqual(summary["memory_errors"][0]["error"], "final observation unavailable")
        markdown = REPORT.render_markdown(summary)
        self.assertIn("native lifetime maximum", markdown)
        self.assertIn("Memory collection gaps", markdown)
        self.assertNotIn("between-sample peak", markdown)

    def test_offline_cli_writes_shareable_reports_and_escaped_svg(self):
        manifest, receipts = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            (directory / "receipts").mkdir()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            for index, receipt in enumerate(receipts):
                (directory / "receipts" / f"{index}.json").write_text(json.dumps(receipt))
            with redirect_stdout(StringIO()):
                self.assertEqual(REPORT.main([str(directory)]), 0)
            output = directory / "report"
            self.assertEqual({path.name for path in output.iterdir()},
                             {"summary.json", "summary.csv", "summary.md", "generation.svg", "ttft.svg", "memory.svg"})
            for filename in ("generation.svg", "ttft.svg", "memory.svg"):
                root = ET.parse(output / filename).getroot()
                self.assertTrue(root.tag.endswith("svg"))
                self.assertIn("code&lt;&amp;&gt;", (output / filename).read_text())
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved["timing_conventions"], manifest["timing_conventions"])
            self.assertIn("no repetitions", (output / "summary.md").read_text())
            stale = copy.deepcopy(receipts[0])
            stale["run_id"] = "stale"
            (directory / "receipts" / "stale.json").write_text(json.dumps(stale))
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(REPORT.main([str(directory)]), 1)


if __name__ == "__main__":
    unittest.main()

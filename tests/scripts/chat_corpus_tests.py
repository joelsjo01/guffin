import collections
import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "benchmarks" / "chat"
CORPUS = json.loads((CORPUS_ROOT / "corpus.json").read_text(encoding="utf-8"))
SOURCES = {source["id"]: source for source in CORPUS["sources"]}
SELECTED_IDS = [
    81, 82, 87, 90, 95, 96, 97, 100,
    101, 105, 106, 109, 111, 114, 116, 119,
    121, 123, 124, 126, 131, 133, 134, 140,
    141, 143, 145, 149, 151, 154, 155, 157,
]


def canonical_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ChatCorpusTests(unittest.TestCase):
    def test_ids_sources_and_nonempty_turns(self):
        self.assertEqual(CORPUS["schema_version"], 1)
        self.assertEqual(CORPUS["default_suite"], "all")
        self.assertEqual(len(SOURCES), len(CORPUS["sources"]))
        conversations = CORPUS["conversations"]
        self.assertEqual(len({case["id"] for case in conversations}), len(conversations))
        for case in conversations:
            with self.subTest(case=case["id"]):
                self.assertRegex(case["id"], r"^[a-z0-9][a-z0-9-]*$")
                self.assertIn(case["source_id"], SOURCES)
                self.assertIn(case["suite"], ("core", "supplementary"))
                self.assertIsInstance(case["category"], str)
                self.assertTrue(case["category"].strip())
                self.assertIsInstance(case["turns"], list)
                self.assertGreaterEqual(len(case["turns"]), 2)
                for turn in case["turns"]:
                    self.assertIsInstance(turn, str)
                    self.assertTrue(turn.strip())

    def test_core_is_balanced_and_preserves_selected_order(self):
        core = [case for case in CORPUS["conversations"] if case["suite"] == "core"]
        self.assertEqual(len(core), 32)
        self.assertEqual([case["source_question_id"] for case in core], SELECTED_IDS)
        self.assertEqual(SOURCES["mt-bench"]["selected_question_ids"], SELECTED_IDS)
        self.assertEqual(collections.Counter(case["category"] for case in core), {
            "writing": 4, "roleplay": 4, "reasoning": 4, "math": 4,
            "coding": 4, "extraction": 4, "stem": 4, "humanities": 4,
        })
        self.assertTrue(all(len(case["turns"]) == 2 for case in core))

    def test_both_core_turns_are_verbatim_upstream(self):
        source = SOURCES["mt-bench"]
        snapshot = CORPUS_ROOT / source["snapshot_path"]
        questions = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([question["question_id"] for question in questions], SELECTED_IDS)
        by_id = {question["question_id"]: question for question in questions}
        for case in CORPUS["conversations"]:
            if case["suite"] != "core":
                continue
            upstream = by_id[case["source_question_id"]]
            with self.subTest(case=case["id"]):
                self.assertEqual(case["source_id"], "mt-bench")
                self.assertEqual(case["id"], f"mt-bench-{upstream['question_id']}")
                self.assertEqual(case["category"], upstream["category"])
                self.assertEqual(case["turns"], upstream["turns"])

    def test_pinned_upstream_identity(self):
        source = SOURCES["mt-bench"]
        self.assertEqual(source["revision"], "b494d0c6b4e7935f1764f8439e75da3e66beccc7")
        self.assertEqual(source["sha256"], "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7")
        self.assertEqual(source["snapshot_sha256"], "866d4a0088280d53adc6448fc5a0d0b65d5f5df2dbb6edef933e23ef07779df1")
        self.assertEqual(source["source_question_count"], 80)
        self.assertEqual(source["license"], "Apache-2.0")
        self.assertIn(source["revision"], source["url"])
        self.assertIn(source["revision"], source["license_url"])

    def test_snapshot_and_license_bytes_match_source_hashes(self):
        for source in CORPUS["sources"]:
            for field, hash_field in (
                ("snapshot_path", "snapshot_sha256"),
                ("license_path", "license_sha256"),
            ):
                if field not in source:
                    continue
                with self.subTest(source=source["id"], field=field):
                    path = (CORPUS_ROOT / source[field]).resolve()
                    self.assertTrue(path.is_relative_to(CORPUS_ROOT.resolve()))
                    self.assertRegex(source[hash_field], r"^[0-9a-f]{64}$")
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source[hash_field])

    def test_turn_identity_matches_utf8_content(self):
        for case in CORPUS["conversations"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(canonical_digest(case["turns"]), case["turns_sha256"])
        original = [{"id": case["id"], "turns": case["turns"]}
                    for case in CORPUS["conversations"] if case["suite"] == "supplementary"]
        self.assertEqual(canonical_digest(original), SOURCES["guffin-original"]["sha256"])

    def test_contexts_match_complete_sources_or_declared_function_excerpt(self):
        contexts = []
        for case in CORPUS["conversations"]:
            for source_id in case.get("context_source_ids", []):
                source = SOURCES[source_id]
                self.assertEqual(source["kind"], "repository")
                self.assertEqual(source["revision"], "644312e18197379cb5e6d3ef25f1e90dd3ed5553")
                self.assertEqual(source["sha256"], source["snapshot_sha256"])
                snapshot = (CORPUS_ROOT / source["snapshot_path"]).read_text(encoding="utf-8")
                excerpts = {part["source_id"]: part for part in case.get("context_excerpts", [])}
                if source_id in excerpts:
                    excerpt = excerpts[source_id]
                    self.assertEqual((excerpt["start_line"], excerpt["end_line"]), (1, 130))
                    snapshot = "".join(snapshot.splitlines(keepends=True)[
                        excerpt["start_line"] - 1:excerpt["end_line"]])
                    self.assertEqual(hashlib.sha256(snapshot.encode("utf-8")).hexdigest(), excerpt["sha256"])
                    self.assertEqual(excerpt["sha256"], "91c580340d7cec10d9d3f8eca5282af9b9fcf5adc3528f7aa10dacc730895532")
                    self.assertIn("void write_help(", snapshot)
                    self.assertNotIn("[[nodiscard]] bool parse_command(", snapshot)
                self.assertIn(snapshot, case["turns"][0])
                contexts.append(source_id)
        self.assertCountEqual(contexts, [
            "guffin-readme", "guffin-principles",
            "guffin-cli-options", "guffin-cli-options-header",
        ])

    def test_supplementary_purposes_and_total_counts(self):
        cases = CORPUS["conversations"]
        self.assertEqual(len(cases), 38)
        self.assertEqual(sum(len(case["turns"]) for case in cases), 80)
        supplementary = [case for case in cases if case["suite"] == "supplementary"]
        self.assertEqual(len(supplementary), 6)
        groups = collections.Counter(group for case in supplementary for group in case["groups"])
        self.assertEqual(groups["short"], 1)
        self.assertEqual(groups["multilingual"], 2)
        self.assertEqual(groups["long-context"], 2)
        sustained = [case for case in supplementary if "sustained" in case["groups"]]
        self.assertEqual(len(sustained), 1)
        self.assertEqual(len(sustained[0]["turns"]), 6)


if __name__ == "__main__":
    unittest.main()

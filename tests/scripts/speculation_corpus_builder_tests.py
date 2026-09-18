import gzip
import hashlib
import io
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculation.corpus_builder import (
    CorpusBuildError,
    REQUIRED_SOURCE_IDS,
    SOURCE_SCHEMA,
    SourceDefinition,
    acquire_sources,
    prepare_public_corpus,
)
from training.speculation.manifest import WORKLOADS, load_source_jsonl


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SpeculationCorpusBuilderTests(unittest.TestCase):
    def _write_fixture_sources(self, root: pathlib.Path):
        cache = root / "cache"
        cache.mkdir()
        alpaca = [
            {"instruction": "Translate the following sentence into French.",
             "input": "The library closes at six.", "output": "unused"},
            {"instruction": "Translate this question into Spanish.",
             "input": "Where is the station?", "output": "unused"},
        ]
        code = [
            {"instruction": "Write a Python function that reverses a list.",
             "input": "Preserve the input list.", "output": "unused"},
            {"instruction": "Explain the ownership bug in this Rust function.",
             "input": "fn f(x: String) { drop(x); println!(\"{}\", x); }", "output": "unused"},
        ]
        dolly = [
            {"instruction": "Summarize this report.", "context": "A sufficiently long report body.",
             "response": "unused", "category": "summarization"},
            {"instruction": "Summarize the meeting notes.", "context": "Several decisions were made.",
             "response": "unused", "category": "summarization"},
        ]
        gsm = [
            {"question": "If Ana has two apples and buys three, how many does she have?",
             "answer": "unused"},
            {"question": "A box has twelve pens shared by four people. How many each?",
             "answer": "unused"},
        ]
        hermes_function = [
            {"id": "function-1", "conversations": [
                {"from": "system", "value": "Use the supplied weather function schema."},
                {"from": "human", "value": "Find the weather in Madrid tomorrow."},
                {"from": "gpt", "value": "unused"},
            ], "category": "weather", "subcategory": "forecast"},
            {"id": "function-2", "conversations": [
                {"from": "system", "value": "Use the supplied calendar function schema."},
                {"from": "human", "value": "Schedule a meeting at noon tomorrow."},
                {"from": "gpt", "value": "unused"},
            ], "category": "calendar", "subcategory": "events"},
        ]
        hermes_json = [
            {"id": "json-1", "conversations": [
                {"from": "system", "value": "Return JSON matching the supplied person schema."},
                {"from": "human", "value": "Encode Ana, age thirty, as JSON."},
                {"from": "gpt", "value": "unused"},
            ], "category": "json", "subcategory": "person"},
            {"id": "json-2", "conversations": [
                {"from": "system", "value": "Return JSON matching the supplied order schema."},
                {"from": "human", "value": "Encode order seven with two items."},
                {"from": "gpt", "value": "unused"},
            ], "category": "json", "subcategory": "order"},
        ]
        oasst = [
            {"message_id": "en-root-a", "parent_id": None, "text": "Explain how rainbows form in simple terms.",
             "role": "prompter", "lang": "en", "deleted": False, "review_result": True, "rank": None},
            {"message_id": "en-answer-a", "parent_id": "en-root-a", "text": "unused assistant",
             "role": "assistant", "lang": "en", "deleted": False, "review_result": True, "rank": 0},
            {"message_id": "en-follow-a", "parent_id": "en-answer-a", "text": "How does a double rainbow differ?",
             "role": "prompter", "lang": "en", "deleted": False, "review_result": True, "rank": None},
            {"message_id": "en-root-b", "parent_id": None, "text": "Suggest a practical weekend gardening plan.",
             "role": "prompter", "lang": "en", "deleted": False, "review_result": True, "rank": None},
            {"message_id": "es-root", "parent_id": None, "text": "Explica por qué el cielo parece azul durante el día.",
             "role": "prompter", "lang": "es", "deleted": False, "review_result": True, "rank": None},
            {"message_id": "fr-root", "parent_id": None, "text": "Explique comment préparer une soupe de légumes.",
             "role": "prompter", "lang": "fr", "deleted": False, "review_result": True, "rank": None},
        ]
        squad = {"data": [{"title": "Fixture article", "paragraphs": [
            {"context": "The first reference paragraph contains enough information for two questions.",
             "qas": [
                 {"id": "q1", "question": "Which paragraph is this?", "is_impossible": False},
                 {"id": "q2", "question": "How many questions use it?", "is_impossible": False},
             ]},
            {"context": "The second reference paragraph discusses a separate factual subject.",
             "qas": [
                 {"id": "q3", "question": "Which subject is discussed?", "is_impossible": False},
                 {"id": "q4", "question": "Is it a separate subject?", "is_impossible": False},
             ]},
        ]}]}
        payloads = {
            "alpaca_data.json": json.dumps(alpaca).encode(),
            "code_alpaca_20k.json": json.dumps(code).encode(),
            "dolly_15k.jsonl": "".join(json.dumps(row) + "\n" for row in dolly).encode(),
            "gsm8k_train.jsonl": "".join(json.dumps(row) + "\n" for row in gsm).encode(),
            "hermes_function_calling_singleturn.json": json.dumps(hermes_function).encode(),
            "hermes_json_mode_singleturn.json": json.dumps(hermes_json).encode(),
            "squad_train_v2.json": json.dumps(squad).encode(),
        }
        oasst_bytes = io.BytesIO()
        with gzip.GzipFile(fileobj=oasst_bytes, mode="wb", mtime=0) as output:
            output.write("".join(json.dumps(row) + "\n" for row in oasst).encode())
        payloads["oasst2_ready.messages.jsonl.gz"] = oasst_bytes.getvalue()
        source_ids = {
            "alpaca_data.json": "stanford-alpaca",
            "code_alpaca_20k.json": "code-alpaca-20k",
            "dolly_15k.jsonl": "databricks-dolly-15k",
            "gsm8k_train.jsonl": "gsm8k-train",
            "hermes_function_calling_singleturn.json": "hermes-function-calling-singleturn",
            "hermes_json_mode_singleturn.json": "hermes-json-mode-singleturn",
            "oasst2_ready.messages.jsonl.gz": "openassistant-oasst2-ready-messages",
            "squad_train_v2.json": "squad-v2-train",
        }
        definitions = []
        for filename, payload in payloads.items():
            (cache / filename).write_bytes(payload)
            definitions.append({
                "id": source_ids[filename], "revision": "fixture-v1",
                "url": f"https://example.invalid/{filename}", "filename": filename,
                "byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            })
        registry = root / "sources.json"
        registry.write_text(canonical({"schema": SOURCE_SCHEMA, "sources": definitions}))
        self.assertEqual({item["id"] for item in definitions}, REQUIRED_SOURCE_IDS)
        return registry, cache

    def test_builds_balanced_authenticated_corpus_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            registry, cache = self._write_fixture_sources(root)
            output = root / "sources-v1.jsonl"
            first = prepare_public_corpus(registry, cache, output, 2)
            second = prepare_public_corpus(registry, cache, output, 2)
            self.assertEqual(first, second)
            self.assertEqual(first["conversation_count"], 2 * len(WORKLOADS))
            self.assertEqual(first["workload_counts"], {key: 2 for key in sorted(WORKLOADS)})
            self.assertEqual(
                first["selected_stratum_counts"]["structured_tool"],
                {"function_call": 1, "json_mode": 1},
            )
            self.assertEqual(
                first["selected_stratum_counts"]["summarization_translation"],
                {"summarization": 1, "translation": 1},
            )
            self.assertEqual(first["source_jsonl_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            conversations = load_source_jsonl(output)
            self.assertEqual(len(conversations), 16)
            self.assertEqual({item.workload for item in conversations}, set(WORKLOADS))
            self.assertTrue(any(len(item.turns) > 1 for item in conversations))
            squad = [item for item in conversations if item.workload in {
                "factual_grounded_qa", "repeated_retrieval"
            }]
            self.assertEqual(len({item.root_source for item in squad}), 1)

    def test_rejects_tampered_cached_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            registry, cache = self._write_fixture_sources(root)
            (cache / "gsm8k_train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(CorpusBuildError, "byte count mismatch|SHA-256 mismatch"):
                prepare_public_corpus(registry, cache, root / "output.jsonl", 1)

    def test_download_is_atomic_and_checks_pinned_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            payload = b"exact source bytes"
            definition = SourceDefinition(
                "fixture", "revision", "https://example.invalid/source", "source.bin",
                len(payload), hashlib.sha256(payload).hexdigest(),
            )
            paths = acquire_sources((definition,), root, lambda *_args, **_kwargs: Response(payload))
            self.assertEqual(paths["fixture"].read_bytes(), payload)
            self.assertFalse(any(root.glob("*.partial")))
            bad = SourceDefinition(
                "bad", "revision", "https://example.invalid/bad", "bad.bin",
                len(payload), "0" * 64,
            )
            with self.assertRaisesRegex(CorpusBuildError, "identity mismatch"):
                acquire_sources((bad,), root, lambda *_args, **_kwargs: Response(payload))
            self.assertFalse((root / "bad.bin").exists())

    def test_rejects_incomplete_source_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "sources.json"
            path.write_text(canonical({"schema": SOURCE_SCHEMA, "sources": []}))
            with self.assertRaisesRegex(CorpusBuildError, "coverage differs"):
                prepare_public_corpus(path, path.parent / "cache", path.parent / "out", 1)


if __name__ == "__main__":
    unittest.main()

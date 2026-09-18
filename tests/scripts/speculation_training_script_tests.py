import copy
import hashlib
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.speculation.manifest import (  # noqa: E402
    DEFAULT_GENERATION_SEEDS,
    ManifestError,
    load_split_receipt,
    publish_manifests,
)
from training.speculation.corpus import validate_trace_corpus  # noqa: E402
from training.speculation.gguf import GGUFFile  # noqa: E402
from training.speculation.registry import (  # noqa: E402
    REQUIRED_FAMILIES,
    RegistryError,
    load_registry,
)
from training.speculation.run_contract import (  # noqa: E402
    RunContractError,
    load_run_contract,
)
from training.speculation.trace import TraceError, TraceFile  # noqa: E402


def digest(seed):
    return bytes((seed + index) & 0xFF for index in range(32))


def trace_fixture():
    feature = b"\x00\x00"
    payload = bytearray()
    payload += b"LGTRC5\0\0" + struct.pack("<I", 2) + bytes(32)
    payload += struct.pack("<IQI", 1, 1, 1) + digest(1) + digest(2) + digest(3)
    payload += struct.pack("<IIQ", 1, 1, 1) + digest(4)
    payload += digest(5) + digest(6) + digest(7) + digest(8) + digest(9)
    payload += struct.pack("<fQfffQQ", 0.0, 1, 1.0, 0.0, 1.0, 42, 0)
    payload += struct.pack("<Q", 1)
    payload += struct.pack("<QIBBQIIIBIQ", 1, 0, 0, 1, 2**64 - 1, 0, 1, 0, 0, 0, 0)
    payload += digest(10) + digest(11) + digest(12) + digest(13)
    payload += struct.pack("<BBIIQIBBIBH", 1, 1, 0, 1, 42, 0, 0, 0, 0, 0, 1)
    payload += struct.pack("<QQIIBHIIIIBI", 1, 0, 1, 1, 2, 0, 0, 1, 1, 1, 1, len(feature))
    payload += hashlib.sha256(feature).digest() + feature
    envelope = hashlib.sha256(payload).digest()
    payload[12:44] = envelope
    return bytes(payload)


def gguf_fixture():
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    payload += struct.pack("<Q", 1) + b"x"
    payload += struct.pack("<IQIQ", 1, 1, 1, 0)
    payload += bytes((-len(payload)) % 32)
    payload += b"\x00\x00"
    return bytes(payload)


class SpeculationTrainingScriptTests(unittest.TestCase):
    def _mutated_registry(self, mutation):
        raw = json.loads((ROOT / "training/speculation/registry.json").read_text())
        mutation(raw)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = pathlib.Path(temporary.name) / "registry.json"
        path.write_text(json.dumps(raw, allow_nan=True))
        return path

    def test_gguf_digest_uses_the_same_open_mapping_as_tensor_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "model.gguf"
            original = gguf_fixture()
            path.write_bytes(original)
            with GGUFFile(path) as gguf:
                replacement = directory / "replacement.gguf"
                replacement.write_bytes(b"replacement")
                replacement.replace(path)
                self.assertEqual(
                    gguf.sha256(), hashlib.sha256(original).hexdigest()
                )

    def test_registry_expands_all_families_and_three_seeds(self):
        registry = load_registry(ROOT / "training/speculation/registry.json")
        self.assertEqual(
            registry.digest,
            hashlib.sha256((ROOT / "training/speculation/registry.json").read_bytes()).hexdigest(),
        )
        cells = list(registry.cells())
        self.assertEqual({cell.family for cell in cells}, REQUIRED_FAMILIES - {"bundle"})
        self.assertEqual({cell.seed for cell in cells}, set(registry.training_seeds))
        self.assertEqual(len({cell.identifier for cell in cells}), len(cells))
        stream_cells = [dict(cell.parameters) for cell in cells if cell.family == "S"]
        self.assertTrue(stream_cells)
        self.assertFalse(any(
            cell["gamma"] > 4 and cell["fanout"] != 1 for cell in stream_cells
        ))
        tree_units = {unit["id"]: unit for unit in registry.raw["training_units"]
                      if unit["family"] == "T"}
        self.assertNotIn(
            "treeflash_immediate_predecessor",
            tree_units["T-constructor-reuse-suite"]["constructors"],
        )
        self.assertEqual(tree_units["T-treeflash-gemma-transfer"]["status"], "train")

    def test_registry_rejects_malformed_objective_contracts(self):
        cases = (
            ("boolean weight", lambda raw: raw["transfer_objective"].__setitem__(
                "forward_kl_weight", True
            )),
            ("nonfinite weight", lambda raw: raw["transfer_objective"].__setitem__(
                "hard_ce_weight", math.nan
            )),
            ("negative weight", lambda raw: raw["transfer_objective"].__setitem__(
                "confidence_weight", -0.1
            )),
            ("integer auxiliary zero", lambda raw: raw["transfer_objective"].__setitem__(
                "future_state_auxiliary_weight", 0
            )),
            ("negative alpha", lambda raw: raw["transfer_objective"].__setitem__(
                "dpace_alpha", -0.01
            )),
            ("alpha above one", lambda raw: raw["transfer_objective"].__setitem__(
                "dpace_alpha", 1.01
            )),
            ("boolean alpha", lambda raw: raw["transfer_objective"].__setitem__(
                "dpace_alpha", False
            )),
            ("unsupported probability compute", lambda raw: raw["transfer_objective"].__setitem__(
                "target_probability_compute", "f16_approximation"
            )),
            ("empty label", lambda raw: raw["transfer_objective"].__setitem__(
                "label", ""
            )),
            ("ambiguous label", lambda raw: raw["transfer_objective"].__setitem__(
                "label", "source faithful"
            )),
        )
        for description, mutation in cases:
            with self.subTest(description=description), self.assertRaises(RegistryError):
                load_registry(self._mutated_registry(mutation))

        def clear_losses(raw):
            objective = raw["transfer_objective"]
            for name in (
                "forward_kl_weight", "hard_ce_weight", "common_draw_overlap_weight",
                "confidence_weight",
            ):
                objective[name] = 0.0

        with self.assertRaisesRegex(RegistryError, "objective"):
            load_registry(self._mutated_registry(clear_losses))

    def test_registry_rejects_nonfinite_or_wrongly_typed_training_numbers(self):
        cases = (
            lambda raw: raw["training_seeds"].__setitem__(0, True),
            lambda raw: raw["target_generation_seeds"].__setitem__(0, True),
            lambda raw: raw["workload_weights"].__setitem__("code", math.inf),
            lambda raw: raw["scales"].__setitem__(0, 100000.0),
            lambda raw: raw["target_generation_sampler"].__setitem__(
                "repetition_penalty", True
            ),
            lambda raw: raw["development_cluster_validation"].__setitem__(
                "validation_residue", False
            ),
            lambda raw: raw["default_transfer_optimizer"].__setitem__(
                "learning_rate", math.nan
            ),
            lambda raw: raw["default_transfer_optimizer"].__setitem__(
                "betas", [0.9, 1.0]
            ),
            lambda raw: raw["default_transfer_optimizer"].__setitem__(
                "warmup_fraction", 1.0
            ),
            lambda raw: raw["default_transfer_optimizer"].__setitem__(
                "maximum_epochs", True
            ),
            lambda raw: raw["default_transfer_optimizer"].__setitem__(
                "gradient_accumulation", 0
            ),
            lambda raw: raw["training_units"][0].__setitem__("implementation", " "),
            lambda raw: raw["training_units"][0]["axes"]["width"].__setitem__(0, -1),
            lambda raw: raw["source_exact_blockers"]["redrafter"].__setitem__(0, ""),
        )
        for index, mutation in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(RegistryError):
                load_registry(self._mutated_registry(mutation))

    def test_registry_json_rejects_duplicate_keys_and_nonstandard_constants(self):
        payload = (ROOT / "training/speculation/registry.json").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            duplicate = pathlib.Path(temporary) / "duplicate.json"
            duplicate.write_text(payload.replace(
                "{\n", "{\n  \"schema\": \"duplicate\",\n", 1
            ))
            with self.assertRaisesRegex(RegistryError, "duplicate key"):
                load_registry(duplicate)
            nonfinite = pathlib.Path(temporary) / "nonfinite.json"
            nonfinite.write_text(payload.replace('"learning_rate": 0.0002',
                                                 '"learning_rate": NaN'))
            with self.assertRaisesRegex(RegistryError, "unsupported constant"):
                load_registry(nonfinite)

    def test_h3_run_contract_binds_policy_data_order_and_validation(self):
        registry = load_registry(ROOT / "training/speculation/registry.json")
        path = ROOT / "training/speculation/phase5_h3_control_run.json"
        contract = load_run_contract(path, registry)
        self.assertEqual(contract.digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(contract.registry_sha256, registry.digest)
        self.assertEqual(contract.data_order_seed, 10723151780598845931)
        self.assertEqual(
            contract.data_order_derivation,
            "seed_xor_zero_based_epoch/v1",
        )
        self.assertEqual(contract.target_architecture, "gemma2")
        self.assertEqual(contract.target_embedding_width, 3584)
        self.assertEqual(contract.target_vocabulary, 256000)
        self.assertEqual(contract.target_output_head, "tied_token_embedding_f16")
        self.assertEqual(
            contract.checkpoint_metric,
            "hierarchical_workload_weighted_exact_accepted_prefix/v1",
        )
        self.assertEqual(
            contract.validation_loss,
            "teacher_prefix_uniform_kl_ce_all_edges",
        )
        self.assertEqual(
            contract.validation_aggregation,
            "origin_mean_within_conversation_seed/equal_seeds_within_conversation/"
            "equal_conversations_within_source_cluster/equal_clusters_within_workload/"
            "frozen_workload_weights/v1",
        )
        self.assertEqual(
            contract.validation_cadence,
            "fixed_optimizer_interval_plus_final_update/v1",
        )
        self.assertEqual(
            contract.minimum_validation_source_clusters_per_workload, 10
        )
        self.assertEqual(
            contract.root_only_interleave_algorithm,
            "sampler-stratified-source-cluster-round-robin/v1",
        )
        self.assertEqual(contract.root_only_interleave_clusters_per_sampler, 16)
        self.assertEqual(contract.root_only_origin_shuffle_buffer, 4096)
        self.assertEqual(contract.root_only_validation_physical_batch_size, 16)
        self.assertEqual(len(contract.cells), 3)
        self.assertEqual({cell.proposal_policy for cell in contract.cells}, {"common_draw"})
        self.assertEqual({cell.dpace_alpha for cell in contract.cells}, {0.5})

    def test_run_contract_rejects_ambiguous_or_malformed_choices(self):
        registry = load_registry(ROOT / "training/speculation/registry.json")
        source = json.loads(
            (ROOT / "training/speculation/phase5_h3_control_run.json").read_text()
        )
        cases = (
            ("wrong registry", lambda raw: raw.__setitem__("registry_sha256", "0" * 64)),
            ("boolean seed", lambda raw: raw.__setitem__("data_order_seed", True)),
            ("unknown data order", lambda raw: raw.__setitem__(
                "data_order_derivation", "seed_plus_epoch"
            )),
            ("wrong target architecture", lambda raw: raw.__setitem__(
                "target_architecture", "llama"
            )),
            ("boolean target width", lambda raw: raw.__setitem__(
                "target_embedding_width", True
            )),
            ("wrong target vocabulary", lambda raw: raw.__setitem__(
                "target_vocabulary", 32000
            )),
            ("untied target head", lambda raw: raw.__setitem__(
                "target_output_head", "output_weight_f16"
            )),
            ("unknown metric", lambda raw: raw.__setitem__("checkpoint_metric", "loss")),
            ("moving validation", lambda raw: raw.__setitem__(
                "validation_loss", "deployment_masked"
            )),
            ("flat validation", lambda raw: raw.__setitem__(
                "validation_aggregation", "origin_mean"
            )),
            ("irregular validation cadence", lambda raw: raw.__setitem__(
                "validation_cadence", "interval_plus_epoch_end"
            )),
            ("boolean validation clusters", lambda raw: raw.__setitem__(
                "minimum_validation_source_clusters_per_workload", True
            )),
            ("too few validation clusters", lambda raw: raw.__setitem__(
                "minimum_validation_source_clusters_per_workload", 1
            )),
            ("unknown root interleave", lambda raw: raw.__setitem__(
                "root_only_interleave_algorithm", "shuffle"
            )),
            ("boolean root interleave width", lambda raw: raw.__setitem__(
                "root_only_interleave_clusters_per_sampler", True
            )),
            ("wrong root interleave width", lambda raw: raw.__setitem__(
                "root_only_interleave_clusters_per_sampler", 8
            )),
            ("wrong root shuffle buffer", lambda raw: raw.__setitem__(
                "root_only_origin_shuffle_buffer", 2048
            )),
            ("boolean validation physical batch", lambda raw: raw.__setitem__(
                "root_only_validation_physical_batch_size", True
            )),
            ("wrong validation physical batch", lambda raw: raw.__setitem__(
                "root_only_validation_physical_batch_size", 8
            )),
            ("unknown policy", lambda raw: raw["cells"][0].__setitem__(
                "proposal_policy", "sample_independently"
            )),
            ("missing D-PACE alpha", lambda raw: raw["cells"][0].pop(
                "dpace_alpha"
            )),
            ("nonfinite D-PACE alpha", lambda raw: raw["cells"][0].__setitem__(
                "dpace_alpha", math.nan
            )),
            ("boolean D-PACE alpha", lambda raw: raw["cells"][0].__setitem__(
                "dpace_alpha", True
            )),
            ("non-source D-PACE alpha", lambda raw: raw["cells"][0].__setitem__(
                "dpace_alpha", 0.1
            )),
            ("duplicate cell", lambda raw: raw["cells"].append(dict(raw["cells"][0]))),
        )
        for description, mutation in cases:
            with self.subTest(description=description), tempfile.TemporaryDirectory() as temporary:
                raw = copy.deepcopy(source)
                mutation(raw)
                path = pathlib.Path(temporary) / "run.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(RunContractError):
                    load_run_contract(path, registry)

        non_dpace_cell = next(
            cell for cell in registry.cells()
            if dict(cell.parameters).get("position_weighting") != "dpace"
            and dict(cell.parameters).get("loss") != "dpace"
        )
        with tempfile.TemporaryDirectory() as temporary:
            raw = copy.deepcopy(source)
            raw["cells"] = [{
                "cell_id": non_dpace_cell.identifier,
                "proposal_policy": "deterministic",
                "dpace_alpha": 0.5,
            }]
            path = pathlib.Path(temporary) / "run.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RunContractError, "non-D-PACE"):
                load_run_contract(path, registry)

        with tempfile.TemporaryDirectory() as temporary:
            duplicate = pathlib.Path(temporary) / "duplicate.json"
            payload = (ROOT / "training/speculation/phase5_h3_control_run.json").read_text()
            duplicate.write_text(payload.replace(
                "{\n", "{\n  \"schema\": \"duplicate\",\n", 1
            ))
            with self.assertRaisesRegex(RunContractError, "duplicate key"):
                load_run_contract(duplicate, registry)
            nonfinite = pathlib.Path(temporary) / "nonfinite.json"
            nonfinite.write_text(payload.replace(
                "10723151780598845931", "NaN"
            ))
            with self.assertRaisesRegex(RunContractError, "unsupported constant"):
                load_run_contract(nonfinite, registry)

    def test_manifest_clusters_before_split_and_preserves_caps_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "sources.jsonl"
            rows = [
                {
                    "source_group": "shared", "root_source": "document-a",
                    "prompt_family": "chat", "topic": "topic-a",
                    "conversation_id": "conversation-a", "workload": "instruction_chat",
                    "turns": ["hello", {"prompt": "write more", "maximum_new_tokens": 17}],
                },
                {
                    "source_group": "shared", "root_source": "document-a",
                    "prompt_family": "qa", "topic": "topic-b",
                    "conversation_id": "conversation-b", "workload": "factual_grounded_qa",
                    "turns": ["what happened?"],
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "published"
            receipt = publish_manifests(
                source, output, ROOT / "training/speculation/registry.json", "fixture-salt"
            )
            self.assertEqual(receipt["conversation_count"], 2)
            self.assertEqual(receipt["source_cluster_count"], 1)
            manifest_rows = []
            for path in output.glob("*.lgsmanifest"):
                lines = path.read_text().splitlines()
                magic, registry_digest = lines[0].split("\t")
                self.assertEqual(magic, "LGSOURCE4")
                self.assertEqual(registry_digest, receipt["registry_sha256"])
                manifest_rows.extend(line.split("\t") for line in lines[1:])
            self.assertEqual(len(manifest_rows), 3 * len(DEFAULT_GENERATION_SEEDS))
            self.assertTrue(all(len(row) == 11 for row in manifest_rows))
            self.assertEqual({int(row[8]) for row in manifest_rows}, set(DEFAULT_GENERATION_SEEDS))
            self.assertIn(17, {int(row[9]) for row in manifest_rows})
            self.assertEqual(len({row[3] for row in manifest_rows}), 1)
            loaded = load_split_receipt(
                output / "split_receipt.json", receipt["registry_sha256"]
            )
            self.assertEqual(loaded["receipt_sha256"], receipt["receipt_sha256"])
            with self.assertRaises(FileExistsError):
                publish_manifests(
                    source, output, ROOT / "training/speculation/registry.json", "fixture-salt"
                )

            tampered = json.loads((output / "split_receipt.json").read_text())
            tampered["conversation_count"] += 1
            (output / "split_receipt.json").write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ManifestError, "digest is invalid"):
                load_split_receipt(
                    output / "split_receipt.json", receipt["registry_sha256"]
                )

    def test_manifest_rejects_malformed_turn_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "sources.jsonl"
            source.write_text(json.dumps({
                "source_group": "x", "root_source": "x", "prompt_family": "x",
                "topic": "x", "conversation_id": "x", "workload": "instruction_chat",
                "turns": [{"prompt": "x", "maximum_new_tokens": -1}],
            }) + "\n")
            with self.assertRaises(ManifestError):
                publish_manifests(
                    source, root / "out", ROOT / "training/speculation/registry.json", "salt"
                )

    def test_manifest_near_duplicate_clustering_has_no_lsh_false_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "sources.jsonl"
            words = [f"word{index}" for index in range(100)]
            changed = list(words)
            changed[50] = "replacement"
            rows = []
            for index, prompt in enumerate((" ".join(words), " ".join(changed))):
                rows.append({
                    "source_group": f"group-{index}",
                    "root_source": f"root-{index}",
                    "prompt_family": f"family-{index}",
                    "topic": f"topic-{index}",
                    "conversation_id": f"conversation-{index}",
                    "workload": "instruction_chat",
                    "turns": [prompt],
                })
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            receipt = publish_manifests(
                source, root / "published", ROOT / "training/speculation/registry.json",
                "near-duplicate-salt", near_duplicate_threshold=0.82,
            )
            self.assertEqual(receipt["source_cluster_count"], 1)

    def test_independent_trace_reader_authenticates_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "fixture.lgtrace"
            path.write_bytes(trace_fixture())
            with TraceFile(path) as trace:
                records = list(trace.records())
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0].terminal)
                self.assertEqual(records[0].context_generation, 0)
                self.assertEqual(records[0].features[0].payload, b"\x00\x00")
                envelope = trace.envelope_sha256
                fingerprint = (
                    trace.file_size, trace.mtime_ns, trace.ctime_ns,
                    trace.device, trace.inode,
                )
            with TraceFile(
                path,
                verify_envelope=False,
                expected_envelope_sha256=envelope,
                expected_size=fingerprint[0],
                expected_mtime_ns=fingerprint[1],
                expected_ctime_ns=fingerprint[2],
                expected_device=fingerprint[3],
                expected_inode=fingerprint[4],
            ) as trace:
                records = list(trace.records(frozenset()))
                self.assertEqual(records[0].features, ())
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            with self.assertRaisesRegex(TraceError, "inode changed"):
                TraceFile(
                    path,
                    verify_envelope=False,
                    expected_envelope_sha256=envelope,
                    expected_size=fingerprint[0],
                    expected_mtime_ns=path.stat().st_mtime_ns,
                    expected_device=fingerprint[3],
                    expected_inode=fingerprint[4],
                )
            corrupted = bytearray(path.read_bytes())
            corrupted[-1] ^= 1
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(TraceError, "envelope SHA-256 mismatch"):
                TraceFile(path)

    def test_trace_corpus_counts_anchor_plus_h_future_selections(self):
        registry = load_registry(ROOT / "training/speculation/registry.json")
        sampler = registry.raw["target_generation_sampler"]
        f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
        generation_seed = registry.raw["target_generation_seeds"][0]
        identity = SimpleNamespace(
            registry_sha256=bytes.fromhex(registry.digest),
            model_sha256=digest(20),
            canonical_metadata_sha256=digest(30),
            source_tensors_sha256=digest(31),
            source_manifest_sha256=digest(21),
            resident_sha256=digest(22),
            feature_contract_sha256=digest(23),
            tokenizer_sha256=digest(24),
            chat_template_sha256=digest(25),
            model_bytes=1,
            source_tensor_count=1,
            resident_allocation_count=1,
            resident_bytes=1,
            sampler=(
                f32(sampler["temperature"]), sampler["top_k"],
                f32(sampler["top_p"]), f32(sampler["min_p"]),
                f32(sampler["repetition_penalty"]), generation_seed,
                sampler["repetition_window"],
            ),
        )
        cluster = b"\x01" + bytes(31)  # development-training residue
        records = []
        token = 9
        for position in range(5):
            label = token + 1
            records.append(SimpleNamespace(
                split=1, generation_seed=generation_seed, selection=object(),
                target_label=label, features=(), source_cluster=cluster,
                source_group=digest(26), conversation=digest(27),
                prompt_derivation=digest(32), sequence_id=1,
                turn_index=0, workload=1, context_generation=0,
                target_position=position, token_id=token,
            ))
            token = label
        records.append(SimpleNamespace(
            split=1, generation_seed=generation_seed, selection=None,
            target_label=None, features=(), source_cluster=cluster,
            source_group=digest(26), conversation=digest(27),
            prompt_derivation=digest(32), sequence_id=1,
            turn_index=0, workload=1, context_generation=0,
            target_position=5, token_id=token,
        ))

        class FakeTrace:
            envelope_sha256 = digest(28)
            file_size = 1
            mtime_ns = 1
            ctime_ns = 1
            device = 1
            inode = 1

            def __init__(self):
                self.identity = identity

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def records(self):
                return iter(records)

        receipt = {
            "manifests": {
                "training_development": {"sha256": identity.source_manifest_sha256.hex()}
            },
            "receipt_sha256": digest(29).hex(),
        }
        manifest_turn = SimpleNamespace(
            turn_index=0, source_group=digest(26), source_cluster=cluster,
            conversation=digest(27), prompt_derivation=digest(32), split=1,
            workload=1, generation_seed=generation_seed,
        )
        with mock.patch(
            "training.speculation.corpus.discover_traces",
            return_value=[pathlib.Path("00000000000000000001.lgtrace")],
        ), mock.patch(
            "training.speculation.corpus.TraceFile", side_effect=lambda _: FakeTrace()
        ), mock.patch(
            "training.speculation.corpus.load_split_receipt", return_value=receipt
        ), mock.patch(
            "training.speculation.corpus._load_development_manifest",
            return_value=(
                {1: manifest_turn}, (1,), ((digest(27), generation_seed, 1),),
            ),
        ):
            result = validate_trace_corpus(
                pathlib.Path("unused"), registry, None, None, pathlib.Path("receipt.json")
            )
            manifest_turn.workload = 2
            with self.assertRaisesRegex(
                    RuntimeError, "source provenance differs from manifest"):
                validate_trace_corpus(
                    pathlib.Path("unused"), registry, None, None,
                    pathlib.Path("receipt.json"),
                )

        # Five consecutive selected rows contain two F_t+x_t -> H=3 windows
        # and one H=4 window.  The old H-row count would incorrectly report 3/2.
        self.assertEqual(result["eligible_origins_by_horizon"]["3"], 2)
        self.assertEqual(result["eligible_origins_by_horizon"]["4"], 1)
        self.assertEqual(result["eligible_origins_by_horizon"]["7"], 0)
        self.assertEqual(result["training_origins_by_horizon"]["3"], 2)
        self.assertEqual(result["validation_origins_by_horizon"]["3"], 0)


if __name__ == "__main__":
    unittest.main()

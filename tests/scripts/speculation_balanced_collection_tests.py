import hashlib
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.speculation.corpus import (  # noqa: E402
    BALANCED_ALGORITHM,
    CorpusError,
    IndexedTrace,
    _ManifestTurn,
    _load_balanced_collection_receipt,
    _validate_balanced_collection_schedule,
    validate_trace_corpus,
)


def identity(value: int) -> bytes:
    return value.to_bytes(8, "little") + bytes(24)


def indexed(sequence: int, turn: _ManifestTurn, validation: bool) -> IndexedTrace:
    return IndexedTrace(
        path=pathlib.Path(f"{sequence:020d}.lgtrace"),
        envelope_sha256=identity(sequence + 10_000),
        file_size=1,
        mtime_ns=1,
        ctime_ns=1,
        device=1,
        inode=sequence,
        sequence_id=sequence,
        turn_index=turn.turn_index,
        workload=turn.workload,
        generation_seed=turn.generation_seed,
        source_group=turn.source_group,
        source_cluster=turn.source_cluster,
        conversation=turn.conversation,
        prompt_derivation=turn.prompt_derivation,
        validation=validation,
        sampler=(),
        eligible_origins=((3, (0,)),),
        available_taps=("final",),
    )


class SpeculationBalancedCollectionTests(unittest.TestCase):
    def test_corpus_rejects_an_interrupted_trace_before_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            (directory / "00000000000000000001.lgtrace.body.partial").write_bytes(b"x")
            with self.assertRaisesRegex(CorpusError, "unpublished partial state"):
                validate_trace_corpus(
                    directory, None, None, None, pathlib.Path("unused")  # type: ignore[arg-type]
                )

    def test_receipt_authenticates_every_field_and_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            prefix = (
                "LGBALANCED1\n"
                f"source_manifest_sha256\t{identity(1).hex()}\n"
                f"registry_sha256\t{identity(2).hex()}\n"
                f"feature_contract_sha256\t{identity(3).hex()}\n"
                f"algorithm\t{BALANCED_ALGORITHM}\n"
                "horizon\t3\n"
                "training_origins_per_workload_seed\t2500\n"
                "minimum_validation_clusters_per_workload\t10\n"
                "sequence_count\t2\n"
                "sequence\t1\n"
                "sequence\t7\n"
            )
            payload = prefix + f"sha256\t{hashlib.sha256(prefix.encode()).hexdigest()}\n"
            path = directory / "balanced_collection.lgsreceipt"
            path.write_text(payload, encoding="ascii")
            receipt = _load_balanced_collection_receipt(directory)
            self.assertEqual(receipt["sequence_ids"], (1, 7))
            self.assertEqual(receipt["horizon"], 3)

            path.write_text(payload.replace("sequence\t7", "sequence\t8"), encoding="ascii")
            with self.assertRaisesRegex(CorpusError, "digest is invalid"):
                _load_balanced_collection_receipt(directory)

    def test_schedule_is_whole_product_canonical_and_reaches_every_goal(self):
        seeds = {11, 12, 13, 14, 15}
        manifest: dict[int, _ManifestTurn] = {}
        order = []
        selected = []
        sequence = 1

        def conversation(workload: int, kind: int, choose: bool) -> None:
            nonlocal sequence
            # Multiples of ten are validation clusters under the frozen
            # little-endian u64 modulo-ten rule; +1 values are training.
            cluster_value = workload * 100 + (0 if kind == 2 else kind * 10 + 1)
            cluster = identity(cluster_value)
            conversation_id = identity(workload * 1000 + kind)
            for seed in sorted(seeds):
                turn = _ManifestTurn(
                    sequence_id=sequence,
                    turn_index=0,
                    source_group=identity(workload * 10),
                    source_cluster=cluster,
                    conversation=conversation_id,
                    prompt_derivation=identity(sequence + 20_000),
                    split=1,
                    workload=workload,
                    generation_seed=seed,
                )
                manifest[sequence] = turn
                order.append(sequence)
                if choose:
                    selected.append(indexed(sequence, turn, kind == 2))
                sequence += 1

        for workload in range(1, 9):
            conversation(workload, 1, True)   # fills the one-origin train quota
            conversation(workload, 3, False)  # canonical skip after quota
            conversation(workload, 2, True)   # first unique validation cluster

        receipt = {
            "sequence_ids": tuple(entry.sequence_id for entry in selected),
            "horizon": 3,
            "training_origins_per_workload_seed": 1,
            "minimum_validation_clusters_per_workload": 1,
        }
        _validate_balanced_collection_schedule(
            receipt, manifest, tuple(order), selected, seeds
        )

        skipped = order[5]
        tampered = sorted(
            selected + [indexed(skipped, manifest[skipped], False)],
            key=lambda entry: entry.sequence_id,
        )
        receipt["sequence_ids"] = tuple(entry.sequence_id for entry in tampered)
        with self.assertRaisesRegex(CorpusError, "canonical balanced"):
            _validate_balanced_collection_schedule(
                receipt, manifest, tuple(order), tampered, seeds
            )

        missing = [entry for entry in selected if entry.sequence_id != 1]
        receipt["sequence_ids"] = tuple(entry.sequence_id for entry in missing)
        with self.assertRaisesRegex(CorpusError, "canonical balanced|whole conversation"):
            _validate_balanced_collection_schedule(
                receipt, manifest, tuple(order), missing, seeds
            )


if __name__ == "__main__":
    unittest.main()

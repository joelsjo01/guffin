import pathlib
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # The offline training environment is optional for repository CI.
    torch = None


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class SpeculationTrainingTorchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_frozen_gemma_head_matches_independent_equation_and_gradient(self):
        from training.speculation.objectives import GemmaTargetHead

        table = torch.randn(19, 8, dtype=torch.float16)
        norm = torch.randn(8)
        states = torch.randn(2, 3, 8, requires_grad=True)
        head = GemmaTargetHead(table, norm, 1e-6, 30.0, chunk_size=5)
        actual = head(states)
        reference_states = states.float()
        reference_states = reference_states * torch.rsqrt(
            reference_states.square().mean(dim=-1, keepdim=True) + 1e-6
        ) * norm
        expected = 30.0 * torch.tanh((reference_states @ table.float().T) / 30.0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        upstream = torch.randn_like(actual)
        actual.backward(upstream, retain_graph=True)
        actual_gradient = states.grad.detach().clone()
        states.grad = None
        expected.backward(upstream)
        torch.testing.assert_close(actual_gradient, states.grad, rtol=1e-5, atol=1e-6)

    def test_deterministic_gather_matches_equation_and_duplicate_gradient(self):
        from training.speculation.objectives import deterministic_last_dim_gather

        values = torch.randn(2, 3, 11, requires_grad=True)
        reference = values.detach().clone().requires_grad_(True)
        indices = torch.tensor([
            [[1, 7, 1, 4], [0, 10, 3, 3], [8, 2, 6, 8]],
            [[5, 5, 9, 0], [4, 1, 10, 2], [7, 6, 7, 6]],
        ])
        upstream = torch.randn(2, 3, 4)
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = deterministic_last_dim_gather(values, indices)
            actual.backward(upstream)
        finally:
            torch.use_deterministic_algorithms(previous)
        expected = torch.gather(reference, -1, indices)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(values.grad, reference.grad, rtol=0.0, atol=0.0)

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS training device is unavailable",
    )
    def test_mps_full_vocabulary_objective_has_deterministic_backward(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import (
            GemmaTargetHead,
            ObjectiveWeights,
            TargetSupport,
            standard_objective,
        )

        device = torch.device("mps")
        table = torch.randn(19, 8, dtype=torch.float16, device=device)
        norm = torch.randn(8, dtype=torch.float32, device=device)
        base_states = torch.randn(2, 3, 8, dtype=torch.float32, device=device)
        base_confidence = torch.zeros(2, 3, dtype=torch.float32, device=device)
        labels = torch.zeros(2, 3, dtype=torch.int64, device=device)
        support_tokens = torch.tensor([0, 1, 2], device=device).expand(2, 3, -1)
        support_probabilities = torch.tensor(
            [0.6, 0.3, 0.1], dtype=torch.float32, device=device
        ).expand(2, 3, -1)
        support = TargetSupport(
            token_ids=support_tokens,
            probabilities=support_probabilities,
            mask=torch.ones(2, 3, 3, dtype=torch.bool, device=device),
        )
        target_unit_draws = torch.full((2, 3), 0.5, dtype=torch.float64)
        weights = ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1)
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            results = []
            for _ in range(2):
                states = base_states.detach().clone().requires_grad_(True)
                confidence = base_confidence.detach().clone().requires_grad_(True)
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    loss, _, _ = standard_objective(
                        DraftOutput(states, confidence=confidence),
                        GemmaTargetHead(table, norm, 1e-6, 30.0, 5),
                        labels, support, weights, (0.8, 4, 0.95, 0.0, 1.0, 17, 64),
                        "dpace", proposal_policy="common_draw",
                        target_unit_draws=target_unit_draws,
                    )
                loss.backward()
                results.append((loss.detach().cpu(), states.grad.cpu(), confidence.grad.cpu()))
        finally:
            torch.use_deterministic_algorithms(previous)
        self.assertTrue(torch.isfinite(results[0][0]).item())
        self.assertTrue(torch.isfinite(results[0][1]).all().item())
        self.assertTrue(torch.isfinite(results[0][2]).all().item())
        for first, second in zip(results[0], results[1]):
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_accelerator_boundary_keeps_authoritative_float64_draws_on_cpu(self):
        from training.speculation.dataset import TrainingExample
        from training.speculation.objectives import TargetSupport
        from training.speculation.train import _to_device

        example = TrainingExample(
            context_taps=torch.zeros(1, 1, 4, dtype=torch.float16),
            context_positions=torch.zeros(1, dtype=torch.int64),
            anchor_indices=torch.zeros(1, dtype=torch.int64),
            rollout_tokens=torch.zeros(1, 3, dtype=torch.int64),
            labels=torch.zeros(1, 3, dtype=torch.int64),
            target_support=TargetSupport(
                token_ids=torch.zeros(1, 3, 1, dtype=torch.int64),
                probabilities=torch.ones(1, 3, 1, dtype=torch.float64),
                mask=torch.ones(1, 3, 1, dtype=torch.bool),
            ),
            target_selected_probabilities=torch.ones(1, 3, dtype=torch.float64),
            target_unit_draws=torch.full((1, 3), 0.5, dtype=torch.float64),
            source_clusters=("cluster",), conversations=("conversation",),
            sequence_ids=(1,),
            workloads=(1,), generation_seeds=(17,),
            target_positions=torch.zeros(1, dtype=torch.int64),
            sampler=(0.8, 40, 0.95, 0.0, 1.0, 17, 64),
        )
        moved = _to_device(example, torch.device("meta"))
        self.assertEqual(moved.target_support.probabilities.dtype, torch.float32)
        self.assertEqual(moved.target_selected_probabilities.dtype, torch.float32)
        self.assertEqual(moved.target_unit_draws.dtype, torch.float64)
        self.assertEqual(moved.target_unit_draws.device.type, "cpu")

    def test_owned_tensor_copy_does_not_warn_for_read_only_numpy_input(self):
        import numpy
        from training.speculation.train import _owned_tensor

        source = numpy.arange(8, dtype=numpy.float16)
        source.flags.writeable = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copied = _owned_tensor(source, torch.device("cpu"))
        self.assertEqual(copied.dtype, torch.float16)
        self.assertEqual(caught, [])

    def test_active_gemma_head_contract_rejects_wrong_or_nonfinite_weights(self):
        import copy
        import math
        import numpy
        from training.speculation.train import _validated_active_gemma_head

        contract = SimpleNamespace(
            target_architecture="gemma2",
            target_embedding_width=4,
            target_vocabulary=5,
            target_output_head="tied_token_embedding_f16",
        )

        class FakeGGUF:
            def __init__(self):
                self.metadata = {
                    "general.architecture": "gemma2",
                    "gemma2.embedding_length": 4,
                    "gemma2.attention.layer_norm_rms_epsilon": 1e-6,
                    "gemma2.final_logit_softcapping": 30.0,
                }
                self.tensors = {"token_embd.weight": object(),
                                "output_norm.weight": object()}
                self.table = numpy.zeros((5, 4), dtype=numpy.float16)
                self.norm = numpy.ones(4, dtype=numpy.float32)

            def f16_matrix(self, name):
                self.assert_name(name, "token_embd.weight")
                return self.table

            def floating_vector(self, name):
                self.assert_name(name, "output_norm.weight")
                return self.norm

            @staticmethod
            def assert_name(actual, expected):
                if actual != expected:
                    raise AssertionError((actual, expected))

        valid = FakeGGUF()
        table, norm, epsilon, softcap = _validated_active_gemma_head(
            valid, contract
        )
        self.assertIs(table, valid.table)
        self.assertIs(norm, valid.norm)
        self.assertEqual((epsilon, softcap), (1e-6, 30.0))

        cases = []
        wrong_architecture = copy.deepcopy(valid)
        wrong_architecture.metadata["general.architecture"] = "llama"
        cases.append(wrong_architecture)
        wrong_width = copy.deepcopy(valid)
        wrong_width.metadata["gemma2.embedding_length"] = True
        cases.append(wrong_width)
        untied = copy.deepcopy(valid)
        untied.tensors["output.weight"] = object()
        cases.append(untied)
        bad_epsilon = copy.deepcopy(valid)
        bad_epsilon.metadata["gemma2.attention.layer_norm_rms_epsilon"] = 0.0
        cases.append(bad_epsilon)
        bad_softcap = copy.deepcopy(valid)
        bad_softcap.metadata["gemma2.final_logit_softcapping"] = math.nan
        cases.append(bad_softcap)
        bad_table = copy.deepcopy(valid)
        bad_table.table[2, 1] = numpy.inf
        cases.append(bad_table)
        bad_norm = copy.deepcopy(valid)
        bad_norm.norm[1] = -numpy.inf
        cases.append(bad_norm)
        for malformed in cases:
            with self.subTest(metadata=malformed.metadata,
                              tensors=sorted(malformed.tensors)), self.assertRaises(
                                  RuntimeError
                              ):
                _validated_active_gemma_head(malformed, contract)

    def test_mechanism_distinct_models_have_exact_horizon_shapes_and_backward(self):
        from training.speculation.models import (
            BlockDiffusionDrafter,
            DBlastDrafter,
            DSparkDrafter,
            DirectRecurrentDrafter,
            Eagle3Drafter,
            JetSpecLatentDrafter,
            ReDrafter,
            TreeFlashDrafter,
        )

        width, vocabulary, context_count, anchors, horizon, taps = 16, 31, 7, 2, 3, 3
        table = torch.randn(vocabulary, width, dtype=torch.float16)
        context = torch.randn(context_count, taps, width)
        positions = torch.arange(context_count)
        anchor_indices = torch.tensor([2, 5])
        tokens = torch.randint(0, vocabulary, (anchors, horizon))
        models = (
            DirectRecurrentDrafter(width, vocabulary, taps, 12, horizon,
                                   "gated_residual", table),
            ReDrafter(width, vocabulary, taps, 12, horizon, table),
            Eagle3Drafter(width, vocabulary, 12, horizon, table),
            BlockDiffusionDrafter(width, vocabulary, taps, width, 2,
                                  horizon + 1, table, False),
            BlockDiffusionDrafter(width, vocabulary, taps, width, 2,
                                  horizon + 1, table, True),
            DSparkDrafter(width, vocabulary, taps, width, horizon, 4,
                          "source_rnn", table),
            DBlastDrafter(width, vocabulary, taps, width, horizon, 2, table),
            JetSpecLatentDrafter(width, vocabulary, taps, width, 2, horizon, table),
            TreeFlashDrafter(width, vocabulary, taps, width, 2, horizon, 32,
                             "predecessor_then_hidden", table),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                output = model(context, positions, anchor_indices, tokens)
                expected = ((anchors, horizon, 2, width)
                            if isinstance(model, DBlastDrafter)
                            else (anchors, horizon, width))
                self.assertEqual(tuple(output.states.shape), expected)
                output.states.float().sum().backward()
                self.assertTrue(any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ))

    def test_full_context_attention_cannot_read_past_each_anchor(self):
        from training.speculation.models import BlockDiffusionDrafter

        width, vocabulary, context_count, horizon, taps = 16, 23, 8, 3, 2
        table = torch.randn(vocabulary, width, dtype=torch.float16)
        context = torch.randn(context_count, taps, width)
        positions = torch.arange(context_count)
        anchor_indices = torch.tensor([2, 5])
        tokens = torch.randint(0, vocabulary, (2, horizon))
        poisoned = context.clone()
        poisoned[3:] += 100.0
        for flare in (False, True):
            with self.subTest(flare=flare):
                model = BlockDiffusionDrafter(
                    width, vocabulary, taps, width, 2, horizon + 1, table, flare
                )
                clean = model(context, positions, anchor_indices, tokens).states
                changed = model(poisoned, positions, anchor_indices, tokens).states
                torch.testing.assert_close(clean[0], changed[0], rtol=0.0, atol=0.0)
                self.assertGreater(float((clean[1] - changed[1]).abs().max().detach()), 0.0)

    def test_training_windows_reject_context_generation_boundaries(self):
        from training.speculation.dataset import _eligible_window, count_eligible_origins

        def row(position, generation, token, label):
            return SimpleNamespace(
                selection=object(), target_label=label, context_generation=generation,
                sequence_id=9, turn_index=0, source_cluster=b"c" * 32,
                conversation=b"v" * 32, split=1, workload=1,
                target_position=position, token_id=token,
            )

        valid = [
            row(7, 0, 3, 4), row(8, 0, 4, 5),
            row(9, 0, 5, 6), row(10, 0, 6, 7),
        ]
        self.assertFalse(_eligible_window(valid[:3], 0, 3))
        self.assertTrue(_eligible_window(valid, 0, 3))
        crossed = [
            valid[0], row(0, 1, 4, 5),
            row(1, 1, 5, 6), row(2, 1, 6, 7),
        ]
        self.assertFalse(_eligible_window(crossed, 0, 3))
        wrong_token = [valid[0], row(8, 0, 99, 5), valid[2], valid[3]]
        self.assertFalse(_eligible_window(wrong_token, 0, 3))

        five_selected = valid + [row(11, 0, 7, 8)]
        registry_digest = (b"r" * 32).hex()

        class FakeTrace:
            identity = SimpleNamespace(registry_sha256=b"r" * 32)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def records(self):
                return iter(five_selected)

        with mock.patch(
            "training.speculation.dataset.discover_traces",
            return_value=[pathlib.Path("one.lgtrace")],
        ), mock.patch(
            "training.speculation.dataset.TraceFile", side_effect=lambda _: FakeTrace()
        ):
            self.assertEqual(
                count_eligible_origins(
                    pathlib.Path("unused"), 3, registry_digest, "train"
                ),
                2,
            )

    def test_training_example_uses_selected_anchor_token_and_future_labels(self):
        import numpy
        from training.speculation.dataset import _Row, _example

        def row(position, token, label, draw):
            return _Row(
                features=numpy.asarray([[position, position + 0.5]], dtype=numpy.float16),
                token=token,
                label=label,
                selection=SimpleNamespace(
                    prepared=((label, 0.0, 0.75, 0.75), (99, -1.0, 0.25, 0.25)),
                    unit_draw=draw,
                ),
                target_position=position,
                source_cluster=b"c" * 32,
                conversation=b"v" * 32,
                sequence_id=9,
                context_generation=0,
                workload=1,
                generation_seed=17,
            )

        # F_7 produced x0=10.  The incoming row token 3 is x_-1 and must not
        # be supplied as the first draft token or used as a target.
        rows = [
            row(7, 3, 10, 0.1), row(8, 10, 11, 0.2),
            row(9, 11, 12, 0.3), row(10, 12, 13, 0.4),
        ]
        example = _example(
            [], rows, [0], 3, (0.8, 40, 0.95, 0.05, 1.0, 64, 17), "root_only"
        )

        torch.testing.assert_close(
            example.context_taps,
            torch.tensor([[[7.0, 7.5]]], dtype=torch.float16),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(example.rollout_tokens, torch.tensor([[10, 11, 12]]))
        torch.testing.assert_close(example.labels, torch.tensor([[11, 12, 13]]))
        torch.testing.assert_close(
            example.target_support.token_ids[:, :, 0], torch.tensor([[11, 12, 13]])
        )
        torch.testing.assert_close(
            example.target_selected_probabilities,
            torch.tensor([[0.75, 0.75, 0.75]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            example.target_unit_draws,
            torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64),
        )
        torch.testing.assert_close(example.target_positions, torch.tensor([7]))

    def test_root_only_collator_preserves_identity_and_pads_target_support(self):
        import dataclasses
        from training.speculation.dataset import (
            DatasetError,
            TrainingExample,
            _collate_root_only,
        )
        from training.speculation.objectives import TargetSupport

        sampler = (0.8, 40, 0.95, 0.05, 1.0, 17, 64)

        def one(index, support_width):
            support_tokens = torch.arange(
                3 * support_width, dtype=torch.int64
            ).reshape(1, 3, support_width) + index * 100
            return TrainingExample(
                context_taps=torch.full(
                    (1, 2, 4), float(index), dtype=torch.float16
                ),
                context_positions=torch.tensor([index], dtype=torch.int64),
                anchor_indices=torch.zeros(1, dtype=torch.int64),
                rollout_tokens=torch.full((1, 3), index, dtype=torch.int64),
                labels=torch.full((1, 3), index + 1, dtype=torch.int64),
                target_support=TargetSupport(
                    token_ids=support_tokens,
                    probabilities=torch.full(
                        (1, 3, support_width), 1.0 / support_width,
                        dtype=torch.float64,
                    ),
                    mask=torch.ones(
                        1, 3, support_width, dtype=torch.bool
                    ),
                ),
                target_selected_probabilities=torch.full(
                    (1, 3), 0.5, dtype=torch.float64
                ),
                target_unit_draws=torch.full(
                    (1, 3), index / 10.0, dtype=torch.float64
                ),
                source_clusters=(f"cluster-{index}",),
                conversations=(f"conversation-{index}",),
                sequence_ids=(index,), workloads=(index + 1,),
                generation_seeds=(17,),
                target_positions=torch.tensor([index * 10], dtype=torch.int64),
                sampler=sampler,
            )

        first, second = one(1, 2), one(2, 4)
        packed = _collate_root_only([first, second])
        self.assertEqual(packed.origin_count, 2)
        torch.testing.assert_close(packed.anchor_indices, torch.tensor([0, 1]))
        self.assertEqual(packed.source_clusters, ("cluster-1", "cluster-2"))
        self.assertEqual(
            packed.conversations, ("conversation-1", "conversation-2")
        )
        self.assertEqual(packed.sequence_ids, (1, 2))
        self.assertEqual(tuple(packed.target_support.token_ids.shape), (2, 3, 4))
        torch.testing.assert_close(
            packed.target_support.token_ids[0, :, :2],
            first.target_support.token_ids[0],
        )
        self.assertFalse(packed.target_support.mask[0, :, 2:].any().item())
        self.assertTrue(packed.target_support.mask[1].all().item())
        self.assertEqual(packed.target_support.probabilities.dtype, torch.float64)
        self.assertEqual(packed.target_unit_draws.dtype, torch.float64)

        with self.assertRaisesRegex(DatasetError, "sampler identity"):
            _collate_root_only([
                first,
                dataclasses.replace(
                    second,
                    sampler=(0.8, 40, 0.95, 0.05, 1.0, 29, 64),
                    generation_seeds=(29,),
                ),
            ])
        with self.assertRaisesRegex(DatasetError, "sampler identity"):
            _collate_root_only([
                first, dataclasses.replace(second, generation_seeds=(29,))
            ])
        with self.assertRaisesRegex(DatasetError, "input shape"):
            _collate_root_only([
                first, dataclasses.replace(second, conversations=())
            ])

    def test_origin_membership_is_nested_and_balanced_across_workloads_and_seeds(self):
        import dataclasses
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import (
            DatasetError,
            TraceWindowStream,
            build_origin_membership,
            validate_minimum_workload_clusters,
        )
        from training.speculation.manifest import WORKLOADS
        from training.speculation.train import _membership_report

        seeds = (11, 22, 33, 44, 55)
        entries = []
        sequence = 1
        for workload in sorted(WORKLOADS.values()):
            for seed in seeds:
                entries.append(IndexedTrace(
                    path=pathlib.Path(f"{sequence:020d}.lgtrace"),
                    envelope_sha256=bytes([sequence & 0xFF]) * 32,
                    file_size=1,
                    mtime_ns=1,
                    ctime_ns=1,
                    device=1,
                    inode=sequence,
                    sequence_id=sequence,
                    turn_index=0,
                    workload=workload,
                    generation_seed=seed,
                    source_group=bytes([0x40 + workload]) * 32,
                    source_cluster=bytes([workload]) * 32,
                    conversation=workload.to_bytes(8, "little") + bytes(24),
                    prompt_derivation=bytes([seed & 0xFF]) * 32,
                    validation=False,
                    sampler=(0.8, 40, 0.95, 0.05, 1.0, seed, 64),
                    eligible_origins=((3, tuple(range(4))),),
                    available_taps=("final",),
                ))
                sequence += 1
        catalog = TraceCorpusIndex(
            "ab" * 32, "cd" * 32, tuple(entries),
            tuple(sorted(
                (entry.conversation, entry.generation_seed, 1) for entry in entries
            )),
        )
        weights = {name: 0.125 for name in WORKLOADS}
        small = build_origin_membership(
            catalog, 3, "train", 40, weights, seeds, ("final",)
        )
        large = build_origin_membership(
            catalog, 3, "train", 80, weights, seeds, ("final",)
        )
        with self.assertRaisesRegex(DatasetError, "workload/seed coverage"):
            build_origin_membership(
                catalog, 3, "train", None, weights, seeds, ("layer:1",)
            )

        self.assertEqual(dict(small.workload_counts), {
            workload: 5 for workload in WORKLOADS.values()
        })
        self.assertEqual(dict(small.generation_seed_counts), {seed: 8 for seed in seeds})
        self.assertEqual(
            {(workload, seed): count for workload, seed, count in small.workload_seed_counts},
            {(workload, seed): 1 for workload in WORKLOADS.values() for seed in seeds},
        )
        self.assertEqual(small.source_group_count, 8)
        self.assertEqual(small.source_cluster_count, 8)
        self.assertEqual(small.conversation_count, 8)
        self.assertEqual(dict(small.workload_cluster_counts), {
            workload: 1 for workload in WORKLOADS.values()
        })
        report = _membership_report(small)
        self.assertEqual(report["source_group_count"], 8)
        self.assertEqual(report["source_cluster_count"], 8)
        self.assertEqual(report["conversation_count"], 8)
        self.assertEqual(
            report["workload_source_cluster_counts"],
            {name: 1 for name in WORKLOADS},
        )
        self.assertEqual(
            report["workload_target_generation_seed_counts"]["instruction_chat"],
            {str(seed): 1 for seed in seeds},
        )
        self.assertEqual(
            {(workload, seed): count for workload, seed, count in large.workload_seed_counts},
            {(workload, seed): 2 for workload in WORKLOADS.values() for seed in seeds},
        )
        self.assertEqual(set(small.references) - set(large.references), set())
        self.assertNotEqual(small.sha256, large.sha256)
        self.assertEqual(
            small,
            build_origin_membership(
                catalog, 3, "train", 40, weights, seeds, ("final",)
            ),
        )
        with self.assertRaisesRegex(DatasetError, "workload contract"):
            build_origin_membership(
                catalog, 3, "train", None,
                {name: 0.25 for name in WORKLOADS}, seeds, ("final",),
            )
        with self.assertRaisesRegex(DatasetError, "expected conversation products"):
            build_origin_membership(
                dataclasses.replace(
                    catalog,
                    expected_conversation_seed_turn_counts=tuple(
                        reversed(catalog.expected_conversation_seed_turn_counts)
                    ),
                ),
                3, "train", 40, weights, seeds, ("final",),
            )
        with self.assertRaisesRegex(DatasetError, "receipt is malformed"):
            TraceWindowStream(
                pathlib.Path("unused"), ("final",), 3, catalog.registry_sha256,
                1, 40, partition="train", catalog=catalog,
                membership=dataclasses.replace(small, sha256="00" * 32),
            )
        receipt_tampers = (
            {"workload_seed_counts": small.workload_seed_counts[:-1]},
            {"source_group_count": small.source_group_count + 1},
            {"source_cluster_count": small.source_cluster_count + 1},
            {"conversation_count": small.conversation_count + 1},
            {"workload_cluster_counts": small.workload_cluster_counts[:-1]},
        )
        for tamper in receipt_tampers:
            with self.subTest(tamper=next(iter(tamper))):
                with self.assertRaisesRegex(DatasetError, "receipt is malformed"):
                    TraceWindowStream(
                        pathlib.Path("unused"), ("final",), 3,
                        catalog.registry_sha256, 1, 40, partition="train",
                        catalog=catalog, membership=dataclasses.replace(small, **tamper),
                    )

        validation_receipt = dataclasses.replace(small, partition="validation")
        self.assertEqual(
            validate_minimum_workload_clusters(validation_receipt, 1),
            {workload: 1 for workload in WORKLOADS.values()},
        )
        with self.assertRaisesRegex(DatasetError, "fewer than 2 independent clusters"):
            validate_minimum_workload_clusters(validation_receipt, 2)

    def test_origin_membership_excludes_incomplete_conversation_seed_products(self):
        import dataclasses
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import build_origin_membership
        from training.speculation.manifest import WORKLOADS

        seeds = (11, 22, 33, 44, 55)
        entries = []
        sequence = 1

        def append(workload, seed, conversation, origins):
            nonlocal sequence
            entries.append(IndexedTrace(
                path=pathlib.Path(f"{sequence:020d}.lgtrace"),
                envelope_sha256=bytes([sequence & 0xFF]) * 32,
                file_size=1, mtime_ns=1, ctime_ns=1, device=1, inode=sequence,
                sequence_id=sequence, turn_index=0, workload=workload,
                generation_seed=seed, source_group=bytes([0x40 + workload]) * 32,
                source_cluster=bytes([workload]) * 32,
                conversation=conversation,
                prompt_derivation=bytes([seed & 0xFF]) * 32,
                validation=False,
                sampler=(0.8, 40, 0.95, 0.05, 1.0, seed, 64),
                eligible_origins=((3, origins),),
                available_taps=("final",),
            ))
            sequence += 1

        for workload in sorted(WORKLOADS.values()):
            conversation = workload.to_bytes(8, "little") + bytes(24)
            for seed in seeds:
                append(workload, seed, conversation, (0, 1))
        complete_entry_count = len(entries)
        incomplete_conversation = b"incomplete" + bytes(22)
        for seed in seeds[:2]:
            append(1, seed, incomplete_conversation, tuple(range(100)))
        incomplete_turns = b"missing-turns" + bytes(19)
        for seed in seeds:
            append(1, seed, incomplete_turns, tuple(range(100)))

        expected_products = [
            (entry.conversation, entry.generation_seed, 1)
            for entry in entries[:complete_entry_count]
        ]
        expected_products.extend(
            (incomplete_conversation, seed, 1) for seed in seeds
        )
        expected_products.extend((incomplete_turns, seed, 3) for seed in seeds)
        catalog = TraceCorpusIndex(
            "ab" * 32, "cd" * 32, tuple(entries), tuple(sorted(expected_products))
        )
        membership = build_origin_membership(
            catalog, 3, "train", None,
            {name: 0.125 for name in WORKLOADS}, seeds, ("final",),
        )
        self.assertEqual(len(membership.references), complete_entry_count * 2)
        self.assertTrue(all(
            reference.entry_index < complete_entry_count
            for reference in membership.references
        ))
        selected_conversations = {
            catalog.entries[reference.entry_index].conversation
            for reference in membership.references
        }
        self.assertNotIn(incomplete_conversation, selected_conversations)
        self.assertNotIn(incomplete_turns, selected_conversations)

        validation_entries = [
            dataclasses.replace(entry, validation=True)
            for entry in entries[:complete_entry_count]
        ]
        short_response_conversation = b"short-response" + bytes(18)
        for seed_index, seed in enumerate(seeds):
            template = entries[seed_index]
            sequence_id = len(validation_entries) + 1
            validation_entries.append(dataclasses.replace(
                template,
                path=pathlib.Path(f"{sequence_id:020d}.lgtrace"),
                envelope_sha256=bytes([sequence_id]) * 32,
                inode=sequence_id,
                sequence_id=sequence_id,
                generation_seed=seed,
                conversation=short_response_conversation,
                prompt_derivation=bytes([0x80 + seed_index]) * 32,
                validation=True,
                sampler=(0.8, 40, 0.95, 0.05, 1.0, seed, 64),
                eligible_origins=((3, () if seed == seeds[-1] else (0, 1)),),
            ))
        validation_expected = [
            (entry.conversation, entry.generation_seed, 1)
            for entry in validation_entries[:complete_entry_count]
        ]
        validation_expected.extend(
            (short_response_conversation, seed, 1) for seed in seeds
        )
        validation_catalog = TraceCorpusIndex(
            "ab" * 32, "ef" * 32, tuple(validation_entries),
            tuple(sorted(validation_expected)),
        )
        validation_membership = build_origin_membership(
            validation_catalog, 3, "validation", None,
            {name: 0.125 for name in WORKLOADS}, seeds, ("final",),
        )
        self.assertNotIn(
            short_response_conversation,
            {
                validation_catalog.entries[reference.entry_index].conversation
                for reference in validation_membership.references
            },
        )
        self.assertEqual(
            len(validation_membership.references), complete_entry_count * 2
        )

    def test_indexed_stream_never_batches_across_context_generations(self):
        import random
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import TraceWindowStream

        registry_digest = "ab" * 32
        sampler = (0.8, 40, 0.95, 0.05, 1.0, 17, 64)
        conversation = b"v" * 32
        entry = IndexedTrace(
            path=pathlib.Path("00000000000000000001.lgtrace"),
            envelope_sha256=b"e" * 32, file_size=1, mtime_ns=1,
            ctime_ns=1, device=1, inode=1, sequence_id=1, turn_index=0,
            workload=1, generation_seed=17, source_group=b"g" * 32,
            source_cluster=b"c" * 32, conversation=conversation,
            prompt_derivation=b"p" * 32, validation=False, sampler=sampler,
            eligible_origins=((3, (0, 1, 4, 5)),),
            available_taps=("final",),
        )
        catalog = TraceCorpusIndex(
            registry_digest, "cd" * 32, (entry,), ((conversation, 17, 1),)
        )
        records = [SimpleNamespace(
            context_generation=0 if index < 4 else 1, sequence_id=1,
            turn_index=0, workload=1, generation_seed=17,
            source_group=entry.source_group, source_cluster=entry.source_cluster,
            conversation=conversation, prompt_derivation=entry.prompt_derivation,
        ) for index in range(8)]

        class FakeTrace:
            identity = SimpleNamespace(
                registry_sha256=bytes.fromhex(registry_digest), sampler=sampler
            )

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def records(self, _selected):
                return iter(records)

        stream = TraceWindowStream(
            pathlib.Path("unused"), ("final",), 3, registry_digest, 1, None,
            partition="train", anchor_batch_size=4, catalog=catalog,
        )
        with mock.patch.object(stream, "_open_indexed", return_value=FakeTrace()), \
                mock.patch("training.speculation.dataset._select_row", side_effect=lambda row, _: row), \
                mock.patch("training.speculation.dataset._eligible_window", return_value=True), \
                mock.patch(
                    "training.speculation.dataset._example",
                    side_effect=lambda _context, rows, origins, *_: tuple(
                        rows[origin].context_generation for origin in origins
                    ),
                ):
            batches = list(stream._yield_batches(
                random.Random(7), [((conversation, 17), [(0, 0)])],
                {0: [0, 1, 4, 5]},
            ))
        self.assertEqual(sum(len(batch) for batch in batches), 4)
        self.assertTrue(all(len(set(batch)) == 1 for batch in batches))

    def test_root_only_interleave_is_diverse_replayable_and_width_invariant(self):
        from collections import Counter
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import TrainingExample, TraceWindowStream
        from training.speculation.objectives import TargetSupport

        registry_digest = "ab" * 32
        seeds = (17, 29)
        origins = tuple(range(4))
        entries = []
        expected_products = []
        sequence = 1
        for cluster_number in range(20):
            conversation = cluster_number.to_bytes(8, "little") + b"v" * 24
            source_cluster = (
                ((cluster_number + 1) * 10).to_bytes(8, "little") + b"c" * 24
            )
            for generation_seed in seeds:
                sampler = (
                    0.8, 40, 0.95, 0.05, 1.0, generation_seed, 64
                )
                entries.append(IndexedTrace(
                    path=pathlib.Path(f"{sequence:020d}.lgtrace"),
                    envelope_sha256=sequence.to_bytes(8, "little") * 4,
                    file_size=1, mtime_ns=1, ctime_ns=1, device=1,
                    inode=sequence, sequence_id=sequence, turn_index=0,
                    workload=1, generation_seed=generation_seed,
                    source_group=source_cluster, source_cluster=source_cluster,
                    conversation=conversation,
                    prompt_derivation=sequence.to_bytes(8, "little") * 4,
                    validation=True, sampler=sampler,
                    eligible_origins=((3, origins),),
                    available_taps=("final",),
                ))
                expected_products.append((conversation, generation_seed, 1))
                sequence += 1
        catalog = TraceCorpusIndex(
            registry_digest, "cd" * 32, tuple(entries),
            tuple(sorted(expected_products)),
        )
        expected_membership = {
            (entry.sequence_id, origin)
            for entry in entries for origin in origins
        }

        def run(width, order_seed=0x1234, model_seed=7, capacity=16):
            torch.manual_seed(model_seed)
            stream = TraceWindowStream(
                pathlib.Path("unused"), ("final",), 3, registry_digest,
                order_seed, len(expected_membership), partition="validation",
                anchor_batch_size=width, context_mode="root_only",
                catalog=catalog,
                root_only_interleave_clusters_per_sampler=capacity,
            )
            opened = Counter()

            def indexed(entry_index, selected_origins, _selected_features):
                opened[entry_index] += 1
                entry = catalog.entries[entry_index]
                rows = [SimpleNamespace(entry_index=entry_index)]
                return (
                    entry, rows, entry.sampler,
                    {origin: 0 for origin in selected_origins},
                )

            def example(_context, rows, selected_origins, horizon, sampler, mode):
                self.assertEqual(mode, "root_only")
                self.assertEqual(horizon, 3)
                self.assertEqual(len(selected_origins), 1)
                entry = catalog.entries[rows[0].entry_index]
                origin = selected_origins[0]
                support_width = origin % 3 + 1
                return TrainingExample(
                    context_taps=torch.full(
                        (1, 1, 2), float(entry.sequence_id),
                        dtype=torch.float16,
                    ),
                    context_positions=torch.tensor([origin], dtype=torch.int64),
                    anchor_indices=torch.zeros(1, dtype=torch.int64),
                    rollout_tokens=torch.full(
                        (1, 3), origin, dtype=torch.int64
                    ),
                    labels=torch.full(
                        (1, 3), entry.sequence_id, dtype=torch.int64
                    ),
                    target_support=TargetSupport(
                        token_ids=torch.full(
                            (1, 3, support_width), entry.sequence_id,
                            dtype=torch.int64,
                        ),
                        probabilities=torch.full(
                            (1, 3, support_width), 1.0 / support_width,
                            dtype=torch.float64,
                        ),
                        mask=torch.ones(
                            1, 3, support_width, dtype=torch.bool
                        ),
                    ),
                    target_selected_probabilities=torch.full(
                        (1, 3), 0.5, dtype=torch.float64
                    ),
                    target_unit_draws=torch.full(
                        (1, 3), 0.25, dtype=torch.float64
                    ),
                    source_clusters=(entry.source_cluster.hex(),),
                    conversations=(entry.conversation.hex(),),
                    sequence_ids=(entry.sequence_id,), workloads=(entry.workload,),
                    generation_seeds=(entry.generation_seed,),
                    target_positions=torch.tensor([origin], dtype=torch.int64),
                    sampler=sampler,
                )

            with mock.patch.object(
                    stream, "_indexed_root_entry", side_effect=indexed
            ), mock.patch(
                    "training.speculation.dataset._example", side_effect=example
            ):
                batches = list(stream)
            flattened = [
                (sequence_id, int(position))
                for batch in batches
                for sequence_id, position in zip(
                    batch.sequence_ids, batch.target_positions
                )
            ]
            return batches, flattened, opened

        logical, logical_order, logical_opens = run(1)
        odd, odd_order, odd_opens = run(7)
        packed, packed_order, packed_opens = run(16)
        replay, replay_order, _ = run(16, model_seed=999)
        changed, changed_order, _ = run(16, order_seed=0x5678)

        self.assertEqual(logical_order, odd_order)
        self.assertEqual(logical_order, packed_order)
        self.assertEqual(packed_order, replay_order)
        self.assertNotEqual(packed_order, changed_order)
        self.assertEqual(set(packed_order), expected_membership)
        self.assertEqual(len(packed_order), len(expected_membership))
        self.assertEqual(len(set(packed_order)), len(packed_order))
        self.assertEqual(logical_opens, Counter({index: 1 for index in range(40)}))
        self.assertEqual(odd_opens, logical_opens)
        self.assertEqual(packed_opens, logical_opens)
        for batches in (logical, odd, packed, replay, changed):
            for batch in batches:
                self.assertEqual(
                    len(set(batch.source_clusters)), batch.origin_count
                )
                self.assertEqual(
                    set(batch.generation_seeds), {batch.sampler[5]}
                )
        self.assertTrue(any(batch.origin_count == 16 for batch in packed))
        self.assertTrue(all(batch.origin_count <= 7 for batch in odd))

    def test_root_only_interleave_capacity_is_bounded_and_validated(self):
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import (
            DatasetError,
            TrainingExample,
            TraceWindowStream,
        )
        from training.speculation.objectives import TargetSupport

        registry_digest = "ab" * 32
        sampler = (0.8, 40, 0.95, 0.05, 1.0, 17, 64)
        entries = []
        products = []
        for index in range(5):
            conversation = index.to_bytes(8, "little") + b"v" * 24
            cluster = (index + 1).to_bytes(8, "little") + b"c" * 24
            entries.append(IndexedTrace(
                path=pathlib.Path(f"{index + 1:020d}.lgtrace"),
                envelope_sha256=bytes([index + 1]) * 32,
                file_size=1, mtime_ns=1, ctime_ns=1, device=1,
                inode=index + 1, sequence_id=index + 1, turn_index=0,
                workload=1, generation_seed=17, source_group=cluster,
                source_cluster=cluster, conversation=conversation,
                prompt_derivation=bytes([index + 2]) * 32,
                validation=False, sampler=sampler,
                eligible_origins=((3, (0, 1)),),
                available_taps=("final",),
            ))
            products.append((conversation, 17, 1))
        catalog = TraceCorpusIndex(
            registry_digest, "cd" * 32, tuple(entries), tuple(sorted(products))
        )
        stream = TraceWindowStream(
            pathlib.Path("unused"), ("final",), 3, registry_digest, 91, 10,
            partition="train", anchor_batch_size=16, context_mode="root_only",
            catalog=catalog, root_only_interleave_clusters_per_sampler=3,
        )
        opened = []

        def indexed(entry_index, selected_origins, _selected_features):
            opened.append(entry_index)
            entry = catalog.entries[entry_index]
            return (
                entry, [SimpleNamespace(entry_index=entry_index)], sampler,
                {origin: 0 for origin in selected_origins},
            )

        def example(_context, rows, selected_origins, *_):
            entry = catalog.entries[rows[0].entry_index]
            origin = selected_origins[0]
            return TrainingExample(
                context_taps=torch.zeros(1, 1, 2, dtype=torch.float16),
                context_positions=torch.tensor([origin]),
                anchor_indices=torch.zeros(1, dtype=torch.int64),
                rollout_tokens=torch.zeros(1, 3, dtype=torch.int64),
                labels=torch.zeros(1, 3, dtype=torch.int64),
                target_support=TargetSupport(
                    token_ids=torch.zeros(1, 3, 1, dtype=torch.int64),
                    probabilities=torch.ones(1, 3, 1, dtype=torch.float64),
                    mask=torch.ones(1, 3, 1, dtype=torch.bool),
                ),
                target_selected_probabilities=torch.ones(
                    1, 3, dtype=torch.float64
                ),
                target_unit_draws=torch.zeros(1, 3, dtype=torch.float64),
                source_clusters=(entry.source_cluster.hex(),),
                conversations=(entry.conversation.hex(),),
                sequence_ids=(entry.sequence_id,), workloads=(1,),
                generation_seeds=(17,),
                target_positions=torch.tensor([origin]), sampler=sampler,
            )

        with mock.patch.object(
                stream, "_indexed_root_entry", side_effect=indexed
        ), mock.patch(
                "training.speculation.dataset._example", side_effect=example
        ):
            iterator = iter(stream)
            first = next(iterator)
            self.assertEqual(first.origin_count, 3)
            self.assertEqual(len(opened), 3)
            remaining = list(iterator)
        self.assertEqual(len(opened), 5)
        self.assertEqual(sum(batch.origin_count for batch in [first] + remaining), 10)
        self.assertTrue(all(
            batch.origin_count <= 3 for batch in [first] + remaining
        ))

        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(DatasetError, "shape is invalid"):
                    TraceWindowStream(
                        pathlib.Path("unused"), ("final",), 3,
                        registry_digest, 91, 10, partition="train",
                        context_mode="root_only", catalog=catalog,
                        root_only_interleave_clusters_per_sampler=invalid,
                    )

    def test_root_only_interleave_rejects_cross_cluster_conversation_turn(self):
        from training.speculation.corpus import IndexedTrace, TraceCorpusIndex
        from training.speculation.dataset import DatasetError, TraceWindowStream

        registry_digest = "ab" * 32
        sampler = (0.8, 40, 0.95, 0.05, 1.0, 17, 64)
        conversation = b"v" * 32

        def entry(sequence_id, turn_index, cluster, origins):
            return IndexedTrace(
                path=pathlib.Path(f"{sequence_id:020d}.lgtrace"),
                envelope_sha256=bytes([sequence_id]) * 32,
                file_size=1, mtime_ns=1, ctime_ns=1, device=1,
                inode=sequence_id, sequence_id=sequence_id,
                turn_index=turn_index, workload=1, generation_seed=17,
                source_group=b"g" * 32, source_cluster=cluster,
                conversation=conversation,
                prompt_derivation=bytes([sequence_id + 1]) * 32,
                validation=False, sampler=sampler,
                eligible_origins=((3, origins),),
                available_taps=("final",),
            )

        entries = (
            entry(1, 0, b"a" * 32, (0,)),
            entry(2, 1, b"b" * 32, ()),
        )
        catalog = TraceCorpusIndex(
            registry_digest, "cd" * 32, entries, ((conversation, 17, 2),)
        )
        stream = TraceWindowStream(
            pathlib.Path("unused"), ("final",), 3, registry_digest, 1, 1,
            partition="train", context_mode="root_only", catalog=catalog,
        )
        with self.assertRaisesRegex(DatasetError, "crossed an authenticated identity"):
            next(iter(stream))


if __name__ == "__main__":
    unittest.main()

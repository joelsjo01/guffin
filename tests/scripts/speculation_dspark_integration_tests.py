from __future__ import annotations

from dataclasses import asdict, replace
import pathlib
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import numpy
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class DSparkIntegrationTests(unittest.TestCase):
    def setUp(self):
        from training.speculation.models import DraftOutput, LowRankLogitCorrection
        from training.speculation.objectives import GemmaTargetHead, draft_logits
        torch.manual_seed(617)
        self.DraftOutput = DraftOutput
        self.Correction = LowRankLogitCorrection
        self.project = draft_logits
        self.table = torch.randn(11, 8, dtype=torch.float16)
        # Non-unit target norm makes an accidental second RMSNorm observable.
        self.head = GemmaTargetHead(self.table, torch.linspace(0.3, 1.9, 8),
                                    epsilon=1e-6, softcap=3.0, chunk_size=3)

    def _p_model_and_example(self, correction="markov"):
        from training.speculation.dspark import DSparkConfig
        from training.speculation.models import DSparkDrafter
        config = DSparkConfig(
            target_width=8, vocabulary=11, tap_count=2, width=8,
            layers=2, horizon=3, rank=4, correction=correction,
            attention_heads=2, kv_heads=1, head_dim=4,
            intermediate_width=20, mask_token_id=0, rms_epsilon=1e-6,
            rope_theta=10000.0, embedding_scale=8 ** 0.5, initializer_std=0.02,
        )
        model = DSparkDrafter(config, self.table)
        example = SimpleNamespace(
            context_taps=torch.randn(7, 2, 8), context_positions=torch.arange(10, 17),
            anchor_indices=torch.tensor([2, 5]),
            rollout_tokens=torch.tensor([[1, 3, 4], [2, 5, 6]]),
            labels=torch.tensor([[3, 4, 5], [5, 6, 7]]),
            target_unit_draws=torch.tensor([[0.23, 0.41, 0.89], [0.71, 0.13, 0.52]],
                                           dtype=torch.float64),
            sampler=(0.8, 5, 0.95, 0.05, 1.0, 42, 64),
        )
        return model, example

    def test_logit_correction_and_all_factor_gradients_match_dense_equations(self):
        states = torch.randn(2, 3, 8, requires_grad=True)
        w1 = torch.randn(11, 4, requires_grad=True)
        w2 = torch.randn(11, 4, requires_grad=True)
        predecessors = torch.tensor([[1, 2, 1], [2, 3, 4]])
        output = self.DraftOutput(states, head_pre_normalized=True,
                                 logit_correction=self.Correction(w1[predecessors], w2))
        actual = self.project(output, self.head)
        dense_states = states.detach().clone().requires_grad_(True)
        dense_w1 = w1.detach().clone().requires_grad_(True)
        dense_w2 = w2.detach().clone().requires_grad_(True)
        raw = dense_states @ self.table.float().T + dense_w1[predecessors] @ dense_w2.T
        expected = 3.0 * torch.tanh(raw / 3.0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for actual_gradient, expected_gradient in (
                (states.grad, dense_states.grad), (w1.grad, dense_w1.grad),
                (w2.grad, dense_w2.grad)):
            torch.testing.assert_close(actual_gradient, expected_gradient,
                                       rtol=1e-5, atol=2e-6)
        capped_before_correction = self.head(states.detach(), apply_norm=False) \
            + w1.detach()[predecessors] @ w2.detach().T
        self.assertGreater(float((actual.detach() - capped_before_correction).abs().max()), 0.1)
        with_second_norm = self.head(states.detach(), apply_softcap=False) \
            + w1.detach()[predecessors] @ w2.detach().T
        self.assertGreater(float((actual.detach() - 3.0 * torch.tanh(with_second_norm / 3.0)).abs().max()), 0.1)

    def test_uncorrected_R_head_is_bitwise_unchanged_at_every_depth(self):
        states = torch.randn(2, 3, 8, requires_grad=True)
        for normalized in (False, True):
            output = self.DraftOutput(states, head_pre_normalized=normalized)
            torch.testing.assert_close(self.project(output, self.head),
                                       self.head(states, apply_norm=not normalized), rtol=0.0, atol=0.0)
            for depth in range(3):
                torch.testing.assert_close(
                    self.project(output, self.head, depth),
                    self.head(states[:, depth], apply_norm=not normalized), rtol=0.0, atol=0.0,
                )

    def test_cached_base_logits_and_depth_slices_preserve_value_and_gradients(self):
        states = torch.randn(2, 3, 8, requires_grad=True)
        factors = torch.randn(2, 3, 4, requires_grad=True)
        weight = torch.randn(11, 4, requires_grad=True)
        output = self.DraftOutput(states, head_pre_normalized=True,
                                 logit_correction=self.Correction(factors, weight))
        ordinary = self.project(output, self.head)
        base = self.head(states, apply_norm=False, apply_softcap=False)
        cached = replace(output, semantic_logits=base)
        with mock.patch.object(self.head, "forward", wraps=self.head.forward) as projection:
            actual = self.project(cached, self.head)
            for depth in range(3):
                torch.testing.assert_close(self.project(cached, self.head, depth),
                                           actual[:, depth], rtol=0.0, atol=0.0)
            projection.assert_not_called()
        torch.testing.assert_close(actual, ordinary, rtol=0.0, atol=0.0)
        wanted_gradients = torch.autograd.grad(ordinary.sum(), (states, factors, weight), retain_graph=True)
        actual_gradients = torch.autograd.grad(actual.sum(), (states, factors, weight))
        for first, second in zip(actual_gradients, wanted_gradients):
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        zero_correction = replace(cached, logit_correction=None)
        torch.testing.assert_close(self.project(zero_correction, self.head),
                                   3.0 * torch.tanh(base / 3.0), rtol=0.0, atol=0.0)

    def test_malformed_or_nonfinite_corrections_and_caches_fail_closed(self):
        states, features, weight = torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(11, 4)
        valid = self.DraftOutput(states, head_pre_normalized=True,
                                logit_correction=self.Correction(features, weight))
        invalid = [
            replace(valid, states=states[:, :0]),
            replace(valid, head_pre_normalized=False),
            replace(valid, logit_correction=self.Correction(features[:, :2], weight)),
            replace(valid, logit_correction=self.Correction(features, weight[:10])),
            replace(valid, logit_correction=self.Correction(features.long(), weight)),
            replace(valid, logit_correction=self.Correction(features, weight.long())),
            replace(valid, logit_correction=self.Correction(features * float("nan"), weight)),
            replace(valid, logit_correction=self.Correction(features, weight * float("inf"))),
            replace(valid, semantic_logits=torch.zeros(2, 3, 10)),
            replace(valid, semantic_logits=torch.zeros(2, 3, 11, dtype=torch.float16)),
            replace(valid, semantic_logits=torch.full((2, 3, 11), float("inf"))),
        ]
        for output in invalid:
            with self.assertRaises(ValueError):
                self.project(output, self.head)
        for depth in (-1, 3, True, 1.0):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.project(valid, self.head, depth)

    def test_zero_correction_P_refuses_overflow_before_softcap_can_hide_it(self):
        from training.speculation.objectives import GemmaTargetHead
        states = torch.full((1, 3, 8), 1e38)
        head = GemmaTargetHead(torch.ones(11, 8, dtype=torch.float16),
                               torch.ones(8), epsilon=1e-6, softcap=3.0)
        self.assertTrue(bool(torch.isfinite(states).all()))
        self.assertTrue(bool(torch.isfinite(head(states, apply_norm=False)).all()))
        output = self.DraftOutput(states, head_pre_normalized=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.project(output, head)

    def test_P_self_rollout_runs_one_backbone_and_one_full_vocabulary_base_projection(self):
        from training.speculation.train import _rollout
        for policy in ("deterministic", "common_draw"):
            with self.subTest(policy=policy):
                model, example = self._p_model_and_example()
                with mock.patch.object(model, "prepare_backbone", wraps=model.prepare_backbone) as backbone, \
                        mock.patch.object(self.head, "forward", wraps=self.head.forward) as head:
                    actual, reached = _rollout(model, self.head, example, "deployment_prefix", policy)
                    logits = self.project(actual, self.head)
                    self.assertEqual(backbone.call_count, 1)
                    self.assertEqual(head.call_count, 1)
                self.assertEqual(reached.shape, example.labels.shape)
                self.assertTrue(bool(reached[:, 0].all()))
                # Only a preceding proposal may supply the next correction.
                own = torch.zeros_like(example.rollout_tokens)
                own[:, 0] = example.rollout_tokens[:, 0]
                expected_reached = torch.ones_like(reached)
                from training.speculation.objectives import select_exact_proposal_tokens
                with torch.no_grad():
                    for depth in range(2):
                        preview = model(example.context_taps, example.context_positions,
                                        example.anchor_indices, own)
                        edge = self.project(preview, self.head, depth)
                        proposed = (torch.argmax(edge, dim=-1) if policy == "deterministic" else
                                    select_exact_proposal_tokens(edge, example.target_unit_draws[:, depth], example.sampler))
                        own[:, depth + 1] = proposed
                        expected_reached[:, depth + 1] = expected_reached[:, depth] & (proposed == example.labels[:, depth])
                    reference = model(example.context_taps, example.context_positions,
                                      example.anchor_indices, own)
                    reference_logits = self.project(reference, self.head)
                torch.testing.assert_close(logits, reference_logits, rtol=0.0, atol=0.0)
                self.assertTrue(torch.equal(reached, expected_reached))
                logits.square().mean().backward()
                self.assertGreater(float(model.core.fusion.weight.grad.abs().sum()), 0.0)
                self.assertGreater(float(model.core.token_rank.weight.grad.abs().sum()), 0.0)
                self.assertGreater(float(model.core.correction_output.weight.grad.abs().sum()), 0.0)

    def test_self_rollout_never_uses_future_teacher_tokens(self):
        from training.speculation.train import _rollout
        model, example = self._p_model_and_example("source_rnn")
        first, _ = _rollout(model, self.head, example, "deployment_prefix", "common_draw")
        poisoned_tokens = example.rollout_tokens.clone()
        poisoned_tokens[:, 1:] = (poisoned_tokens[:, 1:] + 3) % 11
        poisoned = SimpleNamespace(**{**vars(example), "rollout_tokens": poisoned_tokens})
        second, _ = _rollout(model, self.head, poisoned, "deployment_prefix", "common_draw")
        torch.testing.assert_close(self.project(first, self.head), self.project(second, self.head),
                                   rtol=0.0, atol=0.0)

    def test_factory_requires_explicit_P_geometry_and_preserves_all_layers(self):
        from training.speculation.models import make_model
        model, _ = self._p_model_and_example()
        parameters = asdict(model.core.config)
        for key in ("target_width", "vocabulary", "tap_count"):
            del parameters[key]
        parameters.update(p_model_contract="gemma2-dspark-logit-correction/v1",
                          tap_set=["layer:1", "layer:2"], layers=5,
                          context_mode="full_context")
        actual = make_model("dspark", parameters, 8, 11, self.table)
        self.assertEqual(len(actual.core.layers), 5)
        self.assertEqual(actual.core.correction_output.weight.shape, (11, 4))
        for key in ("p_model_contract", "layers", "attention_heads", "mask_token_id"):
            malformed = dict(parameters)
            del malformed[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model("dspark", malformed, 8, 11, self.table)

    def test_legacy_sparse_trainer_refuses_P_before_device_or_trace_work(self):
        from training.speculation.registry import TrainingCell
        from training.speculation.train import TrainingError, _validate_training_implementation, train_cells
        p = TrainingCell("P-test", "P", "dspark", (("horizon", 7),), 1)
        _validate_training_implementation(replace(p, family="R", implementation="direct_recurrent"))
        with self.assertRaisesRegex(TrainingError, "full-teacher"):
            _validate_training_implementation(p)
        registry = SimpleNamespace(raw={"scales": [100000], "training_units": [{"id": "P-test"}]},
                                   cells=lambda: iter([p]))
        contract = SimpleNamespace(cells=[SimpleNamespace(cell_id=p.identifier)])
        with mock.patch("training.speculation.train._validated_optimizer_contract"), \
                mock.patch("training.speculation.train.load_run_contract", return_value=contract), \
                mock.patch("training.speculation.train._device") as device, \
                mock.patch("training.speculation.train.validate_trace_corpus_with_index") as traces:
            with self.assertRaisesRegex(TrainingError, "No P training has started"):
                train_cells(ROOT, registry, ROOT, ROOT, ROOT, 100000, {"P"}, set(), set(),
                            "mps", False, ROOT, None, ROOT)
            device.assert_not_called()
            traces.assert_not_called()

    def test_full_context_capacity_preserves_boundary_and_refuses_truncation(self):
        from training.speculation.dataset import DatasetError, _example
        feature = numpy.ones((1, 2), dtype=numpy.float16)
        def row(position):
            label = (position + 1) % 7
            return SimpleNamespace(
                context_generation=0, target_position=position, features=feature,
                token=position % 7, label=label,
                selection=SimpleNamespace(prepared=((label, 0.0, 1.0, 1.0),), unit_draw=0.25),
                source_cluster=b"s" * 32, conversation=b"c" * 32,
                sequence_id=1, workload=1, generation_seed=42,
            )
        context = [row(position) for position in range(8191)]
        allowed = _example(context, [row(8191), row(8192)], [0], 1, (), "full_context")
        self.assertEqual(allowed.context_taps.shape[0], 8192)
        self.assertEqual(int(allowed.context_positions[0]), 0)
        self.assertEqual(int(allowed.anchor_indices[0]), 8191)
        with self.assertRaisesRegex(DatasetError, "8192-row capacity"):
            _example(context + [row(8191)], [row(8192), row(8193)], [0], 1, (), "full_context")

    def test_turn_context_keeps_current_generation_intact_or_refuses_overflow(self):
        from training.speculation.dataset import DatasetError, _retained_turn_context
        rows = [SimpleNamespace(context_generation=0, target_position=position)
                for position in range(8193)]
        first_turn = _retained_turn_context([], rows[:4000], "full_context")
        boundary = _retained_turn_context(first_turn, rows[4000:8192], "full_context")
        self.assertEqual(boundary, rows[:8192])
        self.assertIs(boundary[0], rows[0])
        with self.assertRaisesRegex(DatasetError, "8192"):
            _retained_turn_context(boundary, rows[8192:], "full_context")
        new_generation = [SimpleNamespace(context_generation=1, target_position=position)
                          for position in range(2)]
        self.assertEqual(_retained_turn_context(rows, new_generation, "full_context"), new_generation)
        next_turn = SimpleNamespace(context_generation=1, target_position=2)
        self.assertEqual(_retained_turn_context(new_generation, [next_turn], "full_context"),
                         new_generation + [next_turn])
        self.assertEqual(_retained_turn_context(rows[:8192], rows[8192:], "root_only"), rows[1:])

    def test_real_mps_logit_correction_and_backward_match_dense_cpu(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable; corrected-head parity requires an actual device")
        from training.speculation.objectives import GemmaTargetHead
        states = torch.randn(2, 3, 8, requires_grad=True)
        features = torch.randn(2, 3, 4, requires_grad=True)
        weight = torch.randn(11, 4, requires_grad=True)
        expected = 3.0 * torch.tanh((states @ self.table.float().T + features @ weight.T) / 3.0)
        expected.sum().backward()
        gpu_states, gpu_features, gpu_weight = [
            value.detach().to("mps").requires_grad_(True) for value in (states, features, weight)
        ]
        head = GemmaTargetHead(self.table.to("mps"), torch.ones(8, device="mps"),
                               epsilon=1e-6, softcap=3.0, chunk_size=3)
        actual = self.project(self.DraftOutput(gpu_states, head_pre_normalized=True,
                              logit_correction=self.Correction(gpu_features, gpu_weight)), head)
        actual.sum().backward()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=2e-6)
        for gpu, cpu in ((gpu_states, states), (gpu_features, features), (gpu_weight, weight)):
            torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=2e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()

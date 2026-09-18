from __future__ import annotations

import math
import pathlib
import struct
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Offline training dependencies are optional for repository CI.
    torch = None
    F = None


def _golden_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _next_f32(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def _production_support_golden(
    logits: list[float], top_k: int, top_p: float, min_p: float,
) -> list[int]:
    """Independent scalar transcription of prepare_token_distribution support."""
    ordered = sorted(
        ((token, _golden_f32(logit)) for token, logit in enumerate(logits)),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    maximum = ordered[0][1]
    pre_weights = []
    total = 0.0
    for _, logit in ordered:
        weight = math.exp(_golden_f32(logit - maximum))
        pre_weights.append(weight)
        total += weight
    retained_weight = 0.0
    retained = 0
    while True:
        retained_weight += pre_weights[retained]
        retained += 1
        if retained == len(ordered) or retained_weight / total >= _golden_f32(top_p):
            break
    threshold = _golden_f32(min_p)
    if threshold > 0.0:
        min_p_count = 1
        while min_p_count < retained and pre_weights[min_p_count] >= threshold:
            min_p_count += 1
        retained = min_p_count
    return [token for token, _ in ordered[:retained]]


def _production_selection_golden(
    logits: list[float], top_k: int, top_p: float, min_p: float,
    temperature: float, unit_draw: float,
) -> int:
    support = _production_support_golden(logits, top_k, top_p, min_p)
    rounded = [_golden_f32(value) for value in logits]
    maximum = rounded[support[0]]
    weights = []
    total = 0.0
    for token in support:
        weight = math.exp(
            _golden_f32(rounded[token] - maximum) / _golden_f32(temperature)
        )
        weights.append(weight)
        total += weight
    mapped = unit_draw * total
    cumulative = 0.0
    for token, weight in zip(support, weights):
        cumulative += weight
        if mapped < cumulative:
            return token
    return support[-1]


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class SpeculationObjectiveMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(917)

    @staticmethod
    def _identity_head(vocabulary: int):
        from training.speculation.objectives import GemmaTargetHead

        return GemmaTargetHead(
            torch.eye(vocabulary, dtype=torch.float16),
            torch.ones(vocabulary, dtype=torch.float32),
            epsilon=1e-6,
            softcap=0.0,
            chunk_size=2,
        )

    @staticmethod
    def _support(probabilities: torch.Tensor):
        from training.speculation.objectives import TargetSupport

        batch, horizon, support_width = probabilities.shape
        token_ids = torch.arange(support_width, dtype=torch.int64).view(1, 1, -1)
        return TargetSupport(
            token_ids=token_ids.expand(batch, horizon, -1).clone(),
            probabilities=probabilities,
            mask=torch.ones_like(probabilities, dtype=torch.bool),
        )

    @staticmethod
    def _weights(
        forward_kl: float = 0.0,
        hard_ce: float = 0.0,
        overlap: float = 0.0,
        confidence: float = 0.0,
        alpha: float = 0.1,
    ):
        from training.speculation.objectives import ObjectiveWeights

        return ObjectiveWeights(
            forward_kl=forward_kl,
            hard_ce=hard_ce,
            common_draw_overlap=overlap,
            confidence=confidence,
            future_state_auxiliary=0.0,
            dpace_alpha=alpha,
        )

    def test_direct_recurrent_scales_gemma_embeddings_and_confidence_is_causal(self):
        from training.speculation.models import DirectRecurrentDrafter

        target_width, draft_width, vocabulary, horizon = 4, 5, 7, 3
        token_table = torch.arange(
            vocabulary * target_width, dtype=torch.float16
        ).reshape(vocabulary, target_width) / 8.0
        model = DirectRecurrentDrafter(
            target_width, vocabulary, 1, draft_width, horizon,
            "gated_residual", token_table,
        )
        context = torch.randn(2, 1, target_width)
        positions = torch.arange(2)
        anchors = torch.tensor([1])
        tokens = torch.tensor([[1, 3, 5]])

        projected_inputs = []
        recurrent_states = []
        input_hook = model.token_projection.register_forward_pre_hook(
            lambda _module, arguments: projected_inputs.append(arguments[0].detach().clone())
        )
        state_hook = model.cell.register_forward_hook(
            lambda _module, _arguments, result: recurrent_states.append(result.detach().clone())
        )
        try:
            output = model(context, positions, anchors, tokens)
        finally:
            input_hook.remove()
            state_hook.remove()

        self.assertEqual(output.states.shape, (1, horizon, target_width))
        self.assertEqual(output.confidence.shape, (1, horizon))
        self.assertEqual(model.embedding_scale, math.sqrt(target_width))
        for depth in range(horizon):
            expected_embedding = token_table[tokens[:, depth]].float() * math.sqrt(target_width)
            torch.testing.assert_close(projected_inputs[depth], expected_embedding)
            expected_confidence = model.confidence(recurrent_states[depth]).squeeze(-1)
            torch.testing.assert_close(output.confidence[:, depth], expected_confidence)

        changed_future = tokens.clone()
        changed_future[:, 2] = 6
        changed = model(context, positions, anchors, changed_future)
        torch.testing.assert_close(output.states[:, :2], changed.states[:, :2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            output.confidence[:, :2], changed.confidence[:, :2], rtol=0.0, atol=0.0
        )

    def test_dpace_weights_match_equation_and_ignore_masked_suffix_poison(self):
        from training.speculation.objectives import dpace_weights

        probabilities = torch.tensor(
            [[0.50, 0.25, float("nan")], [0.80, float("nan"), float("nan")]],
            requires_grad=True,
        )
        reached = torch.tensor([[True, True, False], [True, False, False]])
        alpha = 0.2
        actual = dpace_weights(probabilities, alpha, reached)

        # Independent scalar construction of w_j=sum_(m=j)^R prod_(i=1)^m q_tilde_i.
        expected_rows = []
        for batch in range(probabilities.shape[0]):
            reached_count = int(reached[batch].sum())
            running = 1.0
            prefixes = []
            for depth in range(reached_count):
                q = float(probabilities.detach()[batch, depth])
                running *= (1.0 - alpha) * q + alpha
                prefixes.append(running)
            expected_rows.append([
                sum(prefixes[depth:]) if depth < reached_count else 0.0
                for depth in range(probabilities.shape[1])
            ])
        expected = torch.tensor(expected_rows)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.isfinite(actual).all())

        finite_poison = probabilities.detach().clone()
        finite_poison[~reached] = -1.0e20
        torch.testing.assert_close(
            dpace_weights(finite_poison, alpha, reached), actual, rtol=0.0, atol=0.0
        )
        with self.assertRaisesRegex(ValueError, "prefix-closed"):
            dpace_weights(torch.full((1, 3), 0.5), alpha,
                          torch.tensor([[True, False, True]]))

    def test_draft_sampler_support_matches_production_ties_and_exact_boundaries(self):
        from training.speculation.objectives import _draft_sampler_distribution

        # Two tie groups straddle the top-k cutoff.  The lower token ID must
        # win every equal-logit rank regardless of backend sort behavior.
        tie_logits = torch.tensor([3.0, 4.0, 3.0, 3.0, 4.0])
        tie_tokens, _ = _draft_sampler_distribution(tie_logits, 4, 1.0, 0.0, 0.8)
        expected_ties = _production_support_golden(tie_logits.tolist(), 4, 1.0, 0.0)
        self.assertEqual(tie_tokens.tolist(), expected_ties)
        self.assertEqual(expected_ties, [1, 4, 0, 2])

        # Four equal pre-weights make 0.5 an exact F64 CDF boundary.  The
        # production >= stopping rule retains two; the next F32 value retains
        # three.  A device-side approximate cumsum must not decide this edge.
        equal_logits = torch.zeros(4)
        at_boundary, _ = _draft_sampler_distribution(
            equal_logits, 4, 0.5, 0.0, 1.0
        )
        above_boundary, _ = _draft_sampler_distribution(
            equal_logits, 4, _next_f32(0.5), 0.0, 1.0
        )
        self.assertEqual(at_boundary.tolist(), [0, 1])
        self.assertEqual(above_boundary.tolist(), [0, 1, 2])

        # This boundary separates the ordered F64 production reduction from a
        # tempting device-F32 cumsum.  Production's first-prefix ratio is
        # 0.8419992143..., below the F32 threshold, while the F32 reduction
        # rounds the ratio up to 0.8419992924... and would stop one token early.
        reduction_logits = torch.tensor([
            -3.0513253211975098, -4.724507808685303, -17.312715530395508,
        ])
        reduction_top_p = _golden_f32(0.8419992327690125)
        reduction_tokens, _ = _draft_sampler_distribution(
            reduction_logits, 3, reduction_top_p, 0.0, 1.0
        )
        expected_reduction = _production_support_golden(
            reduction_logits.tolist(), 3, reduction_top_p, 0.0
        )
        self.assertEqual(reduction_tokens.tolist(), expected_reduction)
        self.assertEqual(expected_reduction, [0, 1])

        # min-p compares the pre-temperature weight with >=.  Tied maxima are
        # exactly one, so a threshold of one retains both equality cases and
        # stops at the first lower logit.
        min_p_logits = torch.tensor([5.0, 4.0, 5.0, 3.0])
        min_p_tokens, _ = _draft_sampler_distribution(
            min_p_logits, 4, 1.0, 1.0, 0.7
        )
        expected_min_p = _production_support_golden(
            min_p_logits.tolist(), 4, 1.0, 1.0
        )
        self.assertEqual(min_p_tokens.tolist(), expected_min_p)
        self.assertEqual(expected_min_p, [0, 2])

        # Here the exact F64 exponential is just below the F32 min-p value,
        # while a device-F32 exp rounds to equality and would incorrectly keep
        # the second token.
        min_p_reduction_logits = torch.tensor([
            -8.868972778320312, -10.566412925720215,
        ])
        min_p_boundary = _golden_f32(0.18315176665782928)
        min_p_reduction, _ = _draft_sampler_distribution(
            min_p_reduction_logits, 2, 1.0, min_p_boundary, 1.0
        )
        expected_min_p_reduction = _production_support_golden(
            min_p_reduction_logits.tolist(), 2, 1.0, min_p_boundary
        )
        self.assertEqual(min_p_reduction.tolist(), expected_min_p_reduction)
        self.assertEqual(expected_min_p_reduction, [0])

    def test_draft_sampler_probabilities_and_gradients_use_frozen_production_support(self):
        from training.speculation.objectives import _draft_sampler_distribution

        temperature = _golden_f32(0.73)
        top_p = _golden_f32(0.82)
        min_p = _golden_f32(0.11)
        coefficients = torch.tensor([0.2, -0.7, 1.1, -0.3])
        logits = torch.tensor(
            [0.5, -0.2, 0.5, 1.0, -1.0, 0.1], requires_grad=True
        )
        expected_ids = _production_support_golden(
            logits.detach().tolist(), 6, top_p, min_p
        )
        tokens, probabilities = _draft_sampler_distribution(
            logits, 6, top_p, min_p, temperature
        )
        self.assertEqual(tokens.tolist(), expected_ids)
        self.assertEqual(len(expected_ids), len(coefficients))
        actual_scalar = (probabilities * coefficients).sum()
        actual_scalar.backward()
        actual_gradient = logits.grad.detach().clone()

        independent_logits = logits.detach().clone().requires_grad_(True)
        independent_tokens = torch.tensor(expected_ids, dtype=torch.int64)
        retained_logits = independent_logits.index_select(0, independent_tokens)
        independent_probabilities = torch.exp(
            (retained_logits - retained_logits[0]) / temperature
        )
        independent_probabilities = independent_probabilities / independent_probabilities.sum()
        independent_scalar = (independent_probabilities * coefficients).sum()
        independent_scalar.backward()

        torch.testing.assert_close(probabilities.detach(), independent_probabilities.detach())
        torch.testing.assert_close(actual_gradient, independent_logits.grad)
        excluded = set(range(logits.numel())) - set(expected_ids)
        self.assertTrue(all(float(actual_gradient[token]) == 0.0 for token in excluded))

    def test_batched_sampler_materialization_matches_independent_support_and_gradient(self):
        from training.speculation.objectives import (
            _freeze_draft_sampler_decisions,
            _materialize_draft_sampler_distributions,
        )

        temperature = _golden_f32(0.73)
        top_p = _golden_f32(0.82)
        min_p = _golden_f32(0.11)
        logits = torch.tensor([
            [0.5, -0.2, 0.5, 1.0, -1.0, 0.1],
            [8.0, -7.0, 2.0, 0.0, -3.0, 1.0],
            [-0.3, 1.2, 0.4, 1.2, -0.8, 0.7],
        ], requires_grad=True)
        active_rows = (0, 2)
        decisions = _freeze_draft_sampler_decisions(
            logits.detach()[list(active_rows)], 6, top_p, min_p, temperature
        )
        materialized = _materialize_draft_sampler_distributions(
            logits, [decisions[0], None, decisions[1]], temperature
        )

        expected_supports = [
            _production_support_golden(
                logits.detach()[row].tolist(), 6, top_p, min_p
            )
            for row in active_rows
        ]
        actual_loss = logits.new_zeros(())
        independent_logits = logits.detach().clone().requires_grad_(True)
        expected_loss = independent_logits.new_zeros(())
        for row, distribution, expected_ids in zip(
            active_rows, (materialized.rows[0], materialized.rows[2]), expected_supports
        ):
            self.assertIsNotNone(distribution)
            self.assertEqual(distribution.token_ids, tuple(expected_ids))
            coefficients = torch.linspace(-0.9, 0.7, len(expected_ids))
            actual_loss = actual_loss + (
                distribution.differentiable_probabilities * coefficients
            ).sum()

            expected_tokens = torch.tensor(expected_ids, dtype=torch.int64)
            expected_values = independent_logits[row].index_select(0, expected_tokens)
            expected_weights = torch.exp(
                (expected_values - expected_values[0]) / temperature
            )
            expected_probabilities = expected_weights / expected_weights.sum()
            torch.testing.assert_close(
                distribution.differentiable_probabilities.detach(),
                expected_probabilities.detach(), rtol=1e-6, atol=1e-7,
            )
            expected_loss = expected_loss + (expected_probabilities * coefficients).sum()
        self.assertIsNone(materialized.rows[1])

        actual_loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(logits.grad, independent_logits.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            logits.grad[1], torch.zeros_like(logits.grad[1]), rtol=0.0, atol=0.0
        )

    def test_exact_proposal_selector_uses_f64_draw_and_strict_cdf_boundary(self):
        import training.speculation.objectives as objectives
        from training.speculation.objectives import select_exact_proposal_tokens

        sampler = (1.0, 2, 1.0, 0.0, 1.0, (1 << 64) - 1, 64)
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        draws = torch.tensor(
            [0.5, math.nextafter(0.5, 0.0)], dtype=torch.float64
        )
        with mock.patch.object(
            objectives, "_freeze_draft_sampler_decisions",
            wraps=objectives._freeze_draft_sampler_decisions,
        ) as freeze_batch:
            selected = select_exact_proposal_tokens(logits, draws, sampler)
        self.assertEqual(freeze_batch.call_count, 1)
        expected = [
            _production_selection_golden(row, 2, 1.0, 0.0, 1.0, float(draw))
            for row, draw in zip(logits.detach().tolist(), draws)
        ]
        self.assertEqual(selected.tolist(), expected)
        self.assertEqual(expected, [1, 0])
        self.assertEqual(selected.dtype, torch.int64)
        self.assertFalse(selected.requires_grad)
        self.assertIsNone(logits.grad)

        malformed_draws = [
            torch.tensor([0.5, 0.2], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.5, float("nan")], dtype=torch.float64),
            torch.tensor([0.5, 1.0], dtype=torch.float64),
            torch.tensor([0.5, -math.ulp(0.0)], dtype=torch.float64),
        ]
        for malformed in malformed_draws:
            with self.subTest(draws=malformed):
                with self.assertRaises(ValueError):
                    select_exact_proposal_tokens(logits, malformed, sampler)

    def test_common_draw_confidence_uses_realized_coupled_proposal(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import standard_objective

        states = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
        confidence = torch.tensor([[0.4]], requires_grad=True)
        labels = torch.tensor([[1]])
        support = self._support(torch.tensor([[[0.5, 0.5]]]))
        sampler = (1.0, 2, 1.0, 0.0, 1.0, 17, 64)
        boundary_draw = torch.tensor([[0.5]], dtype=torch.float64)
        output = DraftOutput(
            states, confidence=confidence, head_pre_normalized=True
        )

        common_loss, _, _ = standard_objective(
            output, self._identity_head(2), labels, support,
            self._weights(confidence=1.0), sampler, "uniform",
            proposal_policy="common_draw", target_unit_draws=boundary_draw,
            materialize_metrics=False,
        )
        deterministic_loss, _, _ = standard_objective(
            output, self._identity_head(2), labels, support,
            self._weights(confidence=1.0), sampler, "uniform",
            proposal_policy="deterministic", materialize_metrics=False,
        )
        expected_common = F.binary_cross_entropy_with_logits(
            confidence, torch.ones_like(confidence)
        )
        expected_deterministic = F.binary_cross_entropy_with_logits(
            confidence, torch.zeros_like(confidence)
        )
        torch.testing.assert_close(common_loss, expected_common)
        torch.testing.assert_close(deterministic_loss, expected_deterministic)
        self.assertNotEqual(float(common_loss.detach()), float(deterministic_loss.detach()))
        common_loss.backward()
        torch.testing.assert_close(states.grad, torch.zeros_like(states.grad))
        self.assertGreater(float(confidence.grad.abs().sum()), 0.0)

        invalid_policy_calls = [
            {"proposal_policy": "unknown"},
            {"proposal_policy": "common_draw"},
            {"proposal_policy": "common_draw",
             "target_unit_draws": boundary_draw.float()},
            {"proposal_policy": "deterministic",
             "target_unit_draws": boundary_draw},
        ]
        for keywords in invalid_policy_calls:
            with self.subTest(keywords=keywords):
                with self.assertRaises(ValueError):
                    standard_objective(
                        DraftOutput(states.detach(), confidence=confidence.detach(),
                                    head_pre_normalized=True),
                        self._identity_head(2), labels, support,
                        self._weights(confidence=1.0), sampler, "uniform",
                        materialize_metrics=False, **keywords,
                    )

    def test_batched_common_draw_overlap_matches_independent_scalar_equation(self):
        from training.speculation.objectives import (
            TargetSupport,
            _common_draw_overlap_and_proposals,
        )

        logits = torch.tensor([
            [[1.0, 0.4, 1.0, -0.2, 0.7, -1.0],
             [0.2, 1.4, -0.3, 0.5, 0.9, 0.1],
             [-0.4, 0.8, 0.8, 1.2, -0.1, 0.3]],
            [[0.5, 0.5, -0.7, 1.1, 0.2, -0.4],
             [1.3, -0.5, 0.6, 0.0, 0.6, -0.8],
             [9.0, -9.0, 2.0, 1.0, 0.0, -1.0]],
        ], requires_grad=True)
        target_ids = torch.tensor([
            [[2, 4, 0, 5], [1, 4, 3, 0], [3, 2, 1, 5]],
            [[3, 0, 1, 4], [4, 0, 2, 1], [0, 1, 2, 3]],
        ])
        target_probabilities = torch.tensor([
            [[0.40, 0.30, 0.20, 0.10], [0.55, 0.25, 0.20, 0.00],
             [0.50, 0.20, 0.20, 0.10]],
            [[0.45, 0.35, 0.20, 0.00], [0.40, 0.30, 0.20, 0.10],
             [0.70, 0.20, 0.10, 0.00]],
        ])
        target_mask = torch.tensor([
            [[True, True, True, True], [True, True, True, False],
             [True, True, True, True]],
            [[True, True, True, False], [True, True, True, True],
             [True, True, True, False]],
        ])
        support = TargetSupport(target_ids, target_probabilities, target_mask)
        reached = torch.tensor([[True, True, True], [True, True, False]])
        draws = torch.tensor([
            [0.0, 0.5, math.nextafter(1.0, 0.0)],
            [0.37, 0.81, 0.23],
        ], dtype=torch.float64)
        top_k = 6
        top_p = _golden_f32(0.82)
        min_p = _golden_f32(0.11)
        temperature = _golden_f32(0.73)

        actual, proposals = _common_draw_overlap_and_proposals(
            logits, support, top_k, top_p, min_p, temperature, reached, draws
        )
        reference_logits = logits.detach().clone().requires_grad_(True)
        expected_origins = []
        expected_proposals = torch.zeros_like(reached, dtype=torch.int64)
        for batch in range(logits.shape[0]):
            edge_losses = []
            for depth in range(logits.shape[1]):
                if not bool(reached[batch, depth]):
                    continue
                scalar_logits = reference_logits[batch, depth]
                scalar_values = scalar_logits.detach().tolist()
                draft_ids = _production_support_golden(
                    scalar_values, top_k, top_p, min_p
                )
                draft_values = scalar_logits.index_select(
                    0, torch.tensor(draft_ids, dtype=torch.int64)
                )
                draft_weights = torch.exp(
                    (draft_values - draft_values[0]) / temperature
                )
                draft_probabilities = draft_weights / draft_weights.sum()
                draft_high = torch.cumsum(draft_probabilities, dim=0)
                draft_low = draft_high - draft_probabilities

                selected_target_ids = target_ids[batch, depth][target_mask[batch, depth]]
                selected_target_probabilities = target_probabilities[batch, depth][
                    target_mask[batch, depth]
                ]
                target_high = torch.cumsum(selected_target_probabilities, dim=0)
                target_low = target_high - selected_target_probabilities
                collision = reference_logits.new_zeros(())
                for target_index, token in enumerate(selected_target_ids.tolist()):
                    if token in draft_ids:
                        draft_index = draft_ids.index(token)
                        collision = collision + torch.relu(
                            torch.minimum(target_high[target_index], draft_high[draft_index])
                            - torch.maximum(target_low[target_index], draft_low[draft_index])
                        )
                edge_losses.append(1.0 - collision)
                expected_proposals[batch, depth] = _production_selection_golden(
                    scalar_values, top_k, top_p, min_p, temperature,
                    float(draws[batch, depth]),
                )
            expected_origins.append(torch.stack(edge_losses).mean())
        expected = torch.stack(expected_origins).mean()

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertIsNotNone(proposals)
        torch.testing.assert_close(proposals.cpu(), expected_proposals, rtol=0.0, atol=0.0)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(
            logits.grad, reference_logits.grad, rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            logits.grad[1, 2], torch.zeros_like(logits.grad[1, 2]),
            rtol=0.0, atol=0.0,
        )

    def test_zero_discrete_weights_skip_sampler_but_preserve_contract_validation(self):
        import training.speculation.objectives as objectives
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import standard_objective

        states = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        confidence = torch.tensor([[0.2]], requires_grad=True)
        labels = torch.tensor([[0]])
        support = self._support(torch.tensor([[[0.7, 0.3]]]))
        sampler = (0.8, 2, 1.0, 0.0, 1.0, 17, 64)
        draws = torch.tensor([[0.4]], dtype=torch.float64)
        output = DraftOutput(
            states, confidence=confidence, head_pre_normalized=True
        )

        with mock.patch(
            "training.speculation.objectives._freeze_draft_sampler_decisions",
            side_effect=AssertionError("discrete sampler should be skipped"),
        ) as frozen:
            loss, metrics, _ = standard_objective(
                output, self._identity_head(2), labels, support,
                self._weights(forward_kl=1.0), sampler, "uniform",
                proposal_policy="common_draw", target_unit_draws=draws,
            )
        self.assertEqual(frozen.call_count, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["common_draw_overlap"], 0.0)
        self.assertEqual(metrics["confidence"], 0.0)

        # Skipping execution does not weaken the typed draw boundary.
        with self.assertRaises(ValueError):
            standard_objective(
                output, self._identity_head(2), labels, support,
                self._weights(forward_kl=1.0), sampler, "uniform",
                proposal_policy="common_draw", target_unit_draws=draws.float(),
                materialize_metrics=False,
            )
        with self.assertRaisesRegex(ValueError, "logits must be finite"):
            standard_objective(
                DraftOutput(
                    torch.tensor([[[float("nan"), 0.0]]]),
                    confidence=confidence.detach(), head_pre_normalized=True,
                ),
                self._identity_head(2), labels, support,
                self._weights(forward_kl=1.0), sampler, "uniform",
                proposal_policy="common_draw", target_unit_draws=draws,
                materialize_metrics=False,
            )

        # Either nonzero discrete term must still enter the frozen exact path.
        for objective_weights in (
            self._weights(overlap=0.1), self._weights(confidence=0.2),
        ):
            with self.subTest(weights=objective_weights):
                with mock.patch.object(
                    objectives, "_freeze_draft_sampler_decisions",
                    wraps=objectives._freeze_draft_sampler_decisions,
                ) as frozen:
                    standard_objective(
                        output, self._identity_head(2), labels, support,
                        objective_weights, sampler, "uniform",
                        proposal_policy="common_draw", target_unit_draws=draws,
                        materialize_metrics=False,
                    )
                self.assertGreater(frozen.call_count, 0)

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS training device is unavailable",
    )
    def test_mps_draft_sampler_support_and_gradient_match_cpu(self):
        from training.speculation.objectives import (
            _draft_sampler_distribution,
            select_exact_proposal_tokens,
        )

        cases = [
            ([3.0, 4.0, 3.0, 3.0, 4.0], 4, 1.0, 0.0, 0.8),
            ([-3.0513253211975098, -4.724507808685303, -17.312715530395508],
             3, _golden_f32(0.8419992327690125), 0.0, 1.0),
            ([-8.868972778320312, -10.566412925720215],
             2, 1.0, _golden_f32(0.18315176665782928), 1.0),
        ]
        for values, top_k, top_p, min_p, temperature in cases:
            with self.subTest(values=values):
                cpu_logits = torch.tensor(values, requires_grad=True)
                mps_logits = cpu_logits.detach().to("mps").requires_grad_(True)
                cpu_tokens, cpu_probabilities = _draft_sampler_distribution(
                    cpu_logits, top_k, top_p, min_p, temperature
                )
                mps_tokens, mps_probabilities = _draft_sampler_distribution(
                    mps_logits, top_k, top_p, min_p, temperature
                )
                self.assertEqual(mps_tokens.cpu().tolist(), cpu_tokens.tolist())
                torch.testing.assert_close(
                    mps_probabilities.cpu(), cpu_probabilities.detach(), rtol=2e-6, atol=2e-7
                )
                coefficients = torch.linspace(-0.7, 0.9, len(cpu_tokens))
                (cpu_probabilities * coefficients).sum().backward()
                (mps_probabilities * coefficients.to("mps")).sum().backward()
                torch.testing.assert_close(
                    mps_logits.grad.cpu(), cpu_logits.grad, rtol=2e-5, atol=2e-6
                )
                draw = torch.tensor([0.5], dtype=torch.float64)
                sampler = (temperature, top_k, top_p, min_p, 1.0, 17, 64)
                cpu_selected = select_exact_proposal_tokens(
                    cpu_logits.detach().unsqueeze(0), draw, sampler
                )
                mps_selected = select_exact_proposal_tokens(
                    mps_logits.detach().unsqueeze(0), draw, sampler
                )
                self.assertEqual(mps_selected.cpu().tolist(), cpu_selected.tolist())

    def test_draft_sampler_rejects_malformed_or_nonfinite_inputs(self):
        from training.speculation.objectives import _draft_sampler_distribution

        valid = torch.tensor([1.0, 0.0])
        cases = [
            (torch.empty(0), 1, 1.0, 0.0, 1.0),
            (torch.ones(1, 2), 1, 1.0, 0.0, 1.0),
            (torch.tensor([1, 0]), 1, 1.0, 0.0, 1.0),
            (torch.tensor([1.0, float("nan")]), 1, 1.0, 0.0, 1.0),
            (torch.tensor([1.0, float("inf")]), 1, 1.0, 0.0, 1.0),
            (torch.tensor([1.0, 1.0e100], dtype=torch.float64), 1, 1.0, 0.0, 1.0),
            (valid, True, 1.0, 0.0, 1.0),
            (valid, 0, 1.0, 0.0, 1.0),
            (valid, 3, 1.0, 0.0, 1.0),
            (valid, 1, 0.0, 0.0, 1.0),
            (valid, 1, float("nan"), 0.0, 1.0),
            (valid, 1, 1.0, -0.1, 1.0),
            (valid, 1, 1.0, 1.1, 1.0),
            (valid, 1, 1.0, 0.0, 0.0),
            (valid, 1, 1.0, 0.0, float("inf")),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    _draft_sampler_distribution(*arguments)

    def test_standard_common_draw_rejects_sampler_fields_it_cannot_reproduce(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import standard_objective

        states = torch.tensor([[[1.0, 0.0]]])
        labels = torch.tensor([[0]])
        support = self._support(torch.tensor([[[0.7, 0.3]]]))
        valid = (0.8, 2, 1.0, 0.0, 1.0, 17, 64)
        invalid = [
            (True, *valid[1:]),
            (*valid[:4], 1.1, *valid[5:]),
            (*valid[:5], True, valid[6]),
            (*valid[:5], -1, valid[6]),
            (*valid[:6], True),
            (*valid[:6], 65),
        ]
        for sampler in invalid:
            with self.subTest(sampler=sampler):
                with self.assertRaises(ValueError):
                    standard_objective(
                        DraftOutput(states, head_pre_normalized=True),
                        self._identity_head(2), labels, support,
                        self._weights(overlap=1.0), sampler, "uniform",
                        proposal_policy="deterministic", materialize_metrics=False,
                    )

    def test_temperature_forward_kl_and_raw_hard_ce_match_independent_equations(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import standard_objective

        logits = torch.tensor([[[2.0, 0.4, -0.7, 1.1]]], requires_grad=True)
        labels = torch.tensor([[3]])
        target_probabilities = torch.tensor([[[0.50, 0.25, 0.15, 0.10]]])
        support = self._support(target_probabilities)
        sampler = (0.5, 4, 1.0, 0.0, 1.0, 17, 64)
        output = DraftOutput(logits, head_pre_normalized=True)

        kl_loss, metrics, returned_logits = standard_objective(
            output, self._identity_head(4), labels, support,
            self._weights(forward_kl=1.0), sampler, "uniform",
            proposal_policy="deterministic", materialize_metrics=False,
        )
        independent_log_q = torch.log_softmax(logits.detach()[0, 0] / sampler[0], dim=-1)
        expected_kl = torch.sum(
            target_probabilities[0, 0]
            * (torch.log(target_probabilities[0, 0]) - independent_log_q)
        )
        torch.testing.assert_close(kl_loss, expected_kl, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(returned_logits, logits)
        self.assertEqual(metrics, {})

        hard_loss, _, _ = standard_objective(
            output, self._identity_head(4), labels, support,
            self._weights(hard_ce=1.0), sampler, "uniform",
            proposal_policy="deterministic", materialize_metrics=False,
        )
        expected_hard = F.cross_entropy(logits.detach().reshape(1, 4), labels.reshape(1))
        temperature_hard = F.cross_entropy(
            (logits.detach() / sampler[0]).reshape(1, 4), labels.reshape(1)
        )
        torch.testing.assert_close(hard_loss, expected_hard, rtol=1e-6, atol=1e-7)
        self.assertGreater(abs(float((hard_loss - temperature_hard).detach())), 0.1)

    def test_dpace_compound_loss_uses_reached_count_not_weight_sum(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import dpace_weights, standard_objective

        logits = torch.tensor([[[1.5, 0.2, -0.4], [0.1, 1.0, -0.2], [8.0, -8.0, 3.0]]])
        labels = torch.tensor([[0, 2, 1]])
        target_probabilities = torch.tensor([
            [[0.60, 0.25, 0.15], [0.20, 0.50, 0.30], [0.10, 0.20, 0.70]]
        ])
        support = self._support(target_probabilities)
        reached = torch.tensor([[True, True, False]])
        sampler = (0.8, 3, 1.0, 0.0, 1.0, 4, 64)
        alpha = 0.1
        hard_weight = 0.3

        actual, _, _ = standard_objective(
            DraftOutput(logits, head_pre_normalized=True), self._identity_head(3),
            labels, support,
            self._weights(forward_kl=1.0, hard_ce=hard_weight, alpha=alpha),
            sampler, "dpace", reached, proposal_policy="deterministic",
            materialize_metrics=False,
        )
        raw_log_q = torch.log_softmax(logits, dim=-1)
        hard_rows = -raw_log_q.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        position = dpace_weights(torch.exp(-hard_rows), alpha, reached)
        deployed_log_q = torch.log_softmax(logits / sampler[0], dim=-1)
        kl_rows = torch.sum(
            target_probabilities * (torch.log(target_probabilities) - deployed_log_q), dim=-1
        )
        expected = (
            position * (kl_rows + hard_weight * hard_rows)
        ).sum() / reached.sum()
        self_normalized_wrong_result = (
            position * (kl_rows + hard_weight * hard_rows)
        ).sum() / position.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertGreater(abs(float(actual - self_normalized_wrong_result)), 0.1)

    def test_confidence_uses_actual_match_mask_and_absence_fails_closed(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import standard_objective

        states = torch.tensor([
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 4.0]],
            [[0.0, 5.0, -1.0], [8.0, -8.0, 1.0], [-7.0, 9.0, 2.0]],
        ], requires_grad=True)
        labels = torch.tensor([[0, 2, 2], [0, 2, 0]])
        reached = torch.tensor([[True, True, True], [True, False, False]])
        confidence_logits = torch.tensor(
            [[0.7, -0.4, 1.2], [-0.3, 20.0, -20.0]], requires_grad=True
        )
        target_probabilities = torch.tensor(
            [[[0.5, 0.3, 0.2]] * 3, [[0.2, 0.7, 0.1]] * 3]
        )
        support = self._support(target_probabilities)
        sampler = (0.8, 3, 1.0, 0.0, 1.0, 9, 64)
        objective_weights = self._weights(confidence=1.0)

        actual, _, _ = standard_objective(
            DraftOutput(states, confidence=confidence_logits, head_pre_normalized=True),
            self._identity_head(3), labels, support, objective_weights,
            sampler, "uniform", reached, proposal_policy="deterministic",
            materialize_metrics=False,
        )
        expected_targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        rows = F.binary_cross_entropy_with_logits(
            confidence_logits, expected_targets, reduction="none"
        )
        expected = torch.stack((rows[0].mean(), rows[1, 0])).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual.backward()
        torch.testing.assert_close(states.grad, torch.zeros_like(states.grad), rtol=0.0, atol=0.0)
        self.assertGreater(float(confidence_logits.grad.abs().sum()), 0.0)

        with self.assertRaisesRegex(ValueError, "requires causal confidence"):
            standard_objective(
                DraftOutput(states.detach(), head_pre_normalized=True), self._identity_head(3),
                labels, support, objective_weights, sampler, "uniform", reached,
                proposal_policy="deterministic", materialize_metrics=False,
            )

    def test_vectorized_packing_equals_mean_of_microbatch_one_losses_and_gradients(self):
        from training.speculation.models import DraftOutput
        from training.speculation.objectives import TargetSupport, standard_objective

        states = torch.tensor([
            [[1.4, 0.3, -0.5, 0.1, -0.7],
             [0.2, 1.1, -0.2, 0.4, -0.8],
             [-0.4, 0.2, 1.3, 0.5, -0.1]],
            [[0.1, -0.4, 0.7, 1.2, -0.5],
             [1.7, -0.8, 0.4, 0.2, -0.3],
             [-0.7, 0.5, 0.1, 1.6, -0.2]],
        ], requires_grad=True)
        confidence = torch.tensor(
            [[0.3, -0.8, 0.1], [-0.5, 0.7, -0.2]], requires_grad=True
        )
        labels = torch.tensor([[0, 2, 2], [1, 4, 0]])
        support = TargetSupport(
            token_ids=torch.tensor([0, 1, 2]).view(1, 1, 3).expand(2, 3, 3).clone(),
            probabilities=torch.tensor([
                [[0.60, 0.25, 0.15], [0.20, 0.50, 0.30], [0.10, 0.25, 0.65]],
                [[0.35, 0.45, 0.20], [0.70, 0.20, 0.10], [0.15, 0.30, 0.55]],
            ]),
            mask=torch.ones(2, 3, 3, dtype=torch.bool),
        )
        reached = torch.tensor([[True, True, True], [True, False, False]])
        sampler = (0.8, 5, 1.0, 0.0, 1.0, 11, 64)
        objective_weights = self._weights(
            forward_kl=1.0, hard_ce=0.1, overlap=0.2, confidence=0.3, alpha=0.15
        )

        import training.speculation.objectives as objectives
        with mock.patch.object(
            objectives, "_freeze_draft_sampler_decisions",
            wraps=objectives._freeze_draft_sampler_decisions,
        ) as freeze_batch:
            packed_loss, _, packed_logits = standard_objective(
                DraftOutput(states, confidence=confidence, head_pre_normalized=True),
                self._identity_head(5), labels, support, objective_weights,
                sampler, "dpace", reached, proposal_policy="deterministic",
                materialize_metrics=False,
            )
        self.assertEqual(freeze_batch.call_count, 1)
        packed_loss.backward()
        packed_state_gradient = states.grad.detach().clone()
        packed_confidence_gradient = confidence.grad.detach().clone()
        torch.testing.assert_close(packed_logits, states.detach())

        separate_losses = []
        separate_state_gradients = []
        separate_confidence_gradients = []
        for batch in range(2):
            one_state = states.detach()[batch:batch + 1].clone().requires_grad_(True)
            one_confidence = confidence.detach()[batch:batch + 1].clone().requires_grad_(True)
            one_support = TargetSupport(
                token_ids=support.token_ids[batch:batch + 1],
                probabilities=support.probabilities[batch:batch + 1],
                mask=support.mask[batch:batch + 1],
            )
            one_loss, _, _ = standard_objective(
                DraftOutput(one_state, confidence=one_confidence, head_pre_normalized=True),
                self._identity_head(5), labels[batch:batch + 1], one_support,
                objective_weights, sampler, "dpace", reached[batch:batch + 1],
                proposal_policy="deterministic", materialize_metrics=False,
            )
            one_loss.backward()
            separate_losses.append(one_loss.detach())
            separate_state_gradients.append(one_state.grad.detach())
            separate_confidence_gradients.append(one_confidence.grad.detach())

        expected_loss = torch.stack(separate_losses).mean()
        origin_losses, _, _ = standard_objective(
            DraftOutput(
                states.detach(), confidence=confidence.detach(),
                head_pre_normalized=True,
            ),
            self._identity_head(5), labels, support, objective_weights,
            sampler, "dpace", reached, proposal_policy="deterministic",
            materialize_metrics=False, reduction="none",
        )
        self.assertEqual(tuple(origin_losses.shape), (2,))
        torch.testing.assert_close(
            origin_losses, torch.stack(separate_losses), rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            origin_losses.mean(), packed_loss.detach(), rtol=2e-6, atol=2e-7
        )
        expected_state_gradient = torch.cat(separate_state_gradients) / 2.0
        expected_confidence_gradient = torch.cat(separate_confidence_gradients) / 2.0
        torch.testing.assert_close(packed_loss, expected_loss, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            packed_state_gradient, expected_state_gradient, rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            packed_confidence_gradient, expected_confidence_gradient,
            rtol=2e-6, atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _softmax(values):
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def _scalar_golden(logits, labels, teacher, confidence, mask, gamma):
    horizon = len(labels[0])
    weights = [[float(mask[b][d]) * (math.exp(-d / gamma) if gamma else 1.0)
                for d in range(horizon)] for b in range(len(labels))]
    denominator = math.fsum(weight for row in weights for weight in row) + 1e-6
    terms = [0.0, 0.0, 0.0]
    logit_gradients, confidence_gradients = [], []
    for b, block in enumerate(logits):
        block_gradients, block_confidence_gradients = [], []
        for d, row in enumerate(block):
            p, q = _softmax(row), _softmax(teacher[b][d])
            l1 = math.fsum(abs(a - c) for a, c in zip(p, q))
            accepted = min(1.0, max(0.0, 1.0 - 0.5 * l1))
            c = confidence[b][d]
            weight = weights[b][d] / denominator
            terms[0] += -math.log(p[labels[b][d]]) * weight
            terms[1] += l1 * weight
            terms[2] += (max(c, 0.0) - c * accepted
                         + math.log1p(math.exp(-abs(c)))) * weight
            signs = [1.0 if a > z else -1.0 if a < z else 0.0
                     for a, z in zip(p, q)]
            expected_sign = math.fsum(a * s for a, s in zip(p, signs))
            block_gradients.append([
                weight * (0.1 * (a - float(v == labels[b][d]))
                          + 0.9 * a * (signs[v] - expected_sign))
                for v, a in enumerate(p)
            ])
            block_confidence_gradients.append(
                weight * (1.0 / (1.0 + math.exp(-c)) - accepted)
            )
        logit_gradients.append(block_gradients)
        confidence_gradients.append(block_confidence_gradients)
    return terms, logit_gradients, confidence_gradients


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class DSparkSourceObjectiveTests(unittest.TestCase):
    def setUp(self):
        from training.speculation.dspark_objective import dspark_source_loss
        self.objective = dspark_source_loss
        self.logits = torch.tensor([
            [[0.4, -0.7, 1.1, 0.2], [1.0, 0.1, -0.5, 0.7], [-0.3, 0.8, 0.4, 0.2]],
            [[-0.2, 0.6, 0.9, 0.1], [0.2, -0.4, 0.9, 0.5], [0.7, 0.3, -0.1, 0.8]],
        ], requires_grad=True)
        self.labels = torch.tensor([[2, 0, 1], [1, 3, 0]])
        self.teacher = torch.tensor([
            [[0.1, 0.3, 0.8, -0.4], [0.4, -0.6, 0.3, 0.8], [0.8, 0.1, -0.2, 0.4]],
            [[0.5, -0.4, 0.2, 0.7], [0.6, 0.3, -0.7, 1.2], [0.3, 0.6, 0.4, -0.2]],
        ], requires_grad=True)
        self.confidence = torch.tensor([[0.5, -0.7, 1.2], [0.0, 0.9, -0.3]], requires_grad=True)
        self.mask = torch.tensor([[True, True, False], [True, False, True]])

    def _call(self, **changes):
        inputs = dict(
            draft_logits=self.logits, target_ids=self.labels,
            aligned_target_logits=self.teacher, confidence_logits=self.confidence,
            eval_mask=self.mask, loss_decay_gamma=4.0,
        )
        inputs.update(changes)
        return self.objective(**inputs)

    def test_full_vocabulary_loss_and_all_gradients_match_scalar_equations(self):
        expected, logit_gradient, confidence_gradient = _scalar_golden(
            self.logits.tolist(), self.labels.tolist(), self.teacher.tolist(),
            self.confidence.tolist(), self.mask.tolist(), 4.0,
        )
        result = self._call()
        actual_terms = [result.cross_entropy, result.l1_distance, result.confidence]
        for actual, wanted in zip(actual_terms, expected):
            self.assertAlmostEqual(float(actual.detach()), wanted, delta=2e-7)
        self.assertAlmostEqual(float(result.loss.detach()),
                               0.1 * expected[0] + 0.9 * expected[1] + expected[2], delta=3e-7)
        result.loss.backward()
        torch.testing.assert_close(self.logits.grad, torch.tensor(logit_gradient), rtol=2e-6, atol=2e-8)
        torch.testing.assert_close(self.confidence.grad, torch.tensor(confidence_gradient), rtol=2e-6, atol=2e-8)
        self.assertIsNone(self.teacher.grad)
        self.assertFalse(result.confidence_targets.requires_grad)
        self.assertTrue(torch.equal(self.logits.grad[~self.mask], torch.zeros_like(self.logits.grad[~self.mask])))

    def test_confidence_loss_does_not_backpropagate_through_its_soft_target(self):
        result = self._call()
        result.confidence.backward()
        self.assertIsNone(self.logits.grad)
        self.assertIsNone(self.teacher.grad)
        self.assertGreater(float(self.confidence.grad.abs().sum()), 0.0)

    def test_l1_not_half_l1_and_confidence_is_soft_overlap(self):
        result = self.objective(
            torch.log(torch.tensor([[[0.9, 0.1]]])), torch.tensor([[0]]),
            torch.log(torch.tensor([[[0.1, 0.9]]])), torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.bool), loss_decay_gamma=7.0,
        )
        self.assertAlmostEqual(float(result.l1_distance), 1.6 / 1.000001, delta=2e-7)
        self.assertAlmostEqual(float(result.confidence_targets[0, 0]), 0.2, delta=1e-7)

    def test_release_decay_and_paper_decay_are_explicitly_different(self):
        release, paper_decay = self._call(), self._call(loss_decay_gamma=7.0)
        self.assertNotEqual(float(release.loss.detach()), float(paper_decay.loss.detach()))
        for gamma in (None, 7.0):
            expected, _, _ = _scalar_golden(
                self.logits.tolist(), self.labels.tolist(), self.teacher.tolist(),
                self.confidence.tolist(), self.mask.tolist(), gamma,
            )
            actual = self._call(loss_decay_gamma=gamma)
            self.assertAlmostEqual(float(actual.loss.detach()),
                                   0.1 * expected[0] + 0.9 * expected[1] + expected[2], delta=3e-7)

    def test_source_batch_block_shape_is_same_weighted_reduction(self):
        flat = self._call()
        grouped = self._call(
            draft_logits=self.logits.unsqueeze(0), target_ids=self.labels.unsqueeze(0),
            aligned_target_logits=self.teacher.unsqueeze(0),
            confidence_logits=self.confidence.unsqueeze(0), eval_mask=self.mask.unsqueeze(0),
        )
        torch.testing.assert_close(flat.loss, grouped.loss, rtol=0.0, atol=0.0)

    def test_masks_do_not_renormalize_each_block_independently(self):
        actual = self._call()
        changed = self.logits.detach().clone()
        changed[~self.mask] = torch.tensor([50.0, -50.0, 0.0, -1.0])
        torch.testing.assert_close(actual.loss, self._call(draft_logits=changed).loss, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(actual.weight_sum),
                               2.0 + math.exp(-0.25) + math.exp(-0.5), delta=3e-7)

    def test_invalid_shapes_values_and_sparse_target_evidence_fail_closed(self):
        invalid = [
            dict(aligned_target_logits=self.teacher[..., :2]),
            dict(aligned_target_logits={"token_ids": [0, 1], "probabilities": [0.5, 0.5]}),
            dict(aligned_target_logits=torch.full_like(self.teacher, float("nan"))),
            dict(draft_logits=torch.full_like(self.logits, float("inf"))),
            dict(confidence_logits=torch.full_like(self.confidence, float("nan"))),
            dict(confidence_logits=self.confidence[:, :2]),
            dict(target_ids=self.labels.float()),
            dict(target_ids=torch.full_like(self.labels, 4)),
            dict(target_ids=torch.full_like(self.labels, -1)),
            dict(eval_mask=self.mask.float()),
            dict(eval_mask=torch.zeros_like(self.mask)),
            dict(eval_mask=self.mask[:, :2]),
        ] + [dict(loss_decay_gamma=value) for value in (0.0, -1, float("inf"), float("nan"), True, "4")]
        for change in invalid:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    self._call(**change)

    def test_real_mps_forward_backward_matches_cpu_with_deterministic_algorithms(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable; offline diagnostic requires an actual device")
        cpu = self._call()
        cpu.loss.backward()
        mps_logits = self.logits.detach().to("mps").requires_grad_(True)
        mps_confidence = self.confidence.detach().to("mps").requires_grad_(True)
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            mps = self._call(
                draft_logits=mps_logits, target_ids=self.labels.to("mps"),
                aligned_target_logits=self.teacher.detach().to("mps"),
                confidence_logits=mps_confidence, eval_mask=self.mask.to("mps"),
            )
            mps.loss.backward()
        finally:
            torch.use_deterministic_algorithms(deterministic)
        torch.testing.assert_close(mps.loss.cpu(), cpu.loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(mps_logits.grad.cpu(), self.logits.grad, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(mps_confidence.grad.cpu(), self.confidence.grad, rtol=2e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

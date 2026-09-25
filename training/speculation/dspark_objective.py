"""Pinned DeepSpec DSpark loss equations, separate from the Gemma transfer loss.

Source: deepseek-ai/DeepSpec, commit
005e03b81cec38b7da6399833d609ee89a2587f2,
deepspec/modeling/dspark/loss.py. This is its single-process reduction, not a
distributed trainer or a same-seed acceptance estimator. The source's ``l1``
term is the full L1 distance, despite the paper naming it total variation.

The released Qwen configuration uses decay gamma 4; the paper uses block
length gamma (7 in its named control). Callers must choose the decay explicitly.
The paper's unnormalized equation and this code's weighted-mean reduction are
not represented as one source identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class DSparkSourceLoss:
    loss: Tensor
    cross_entropy: Tensor
    l1_distance: Tensor
    confidence: Tensor
    confidence_targets: Tensor
    weight_sum: Tensor


def _finite_tensor(name: str, value: Tensor) -> None:
    if (not isinstance(value, Tensor) or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())):
        raise ValueError(f"DSpark {name} must be a finite floating tensor")


def dspark_source_loss(
    draft_logits: Tensor,
    target_ids: Tensor,
    aligned_target_logits: Tensor,
    confidence_logits: Tensor,
    eval_mask: Tensor,
    *,
    loss_decay_gamma: float | None,
) -> DSparkSourceLoss:
    """Compute the released CE/L1/confidence objective with 0.1/0.9/1 weights.

    Inputs are [batch, horizon, vocabulary] or the source's
    [batch, blocks, horizon, vocabulary], with labels/confidence/mask matching
    every leading dimension. Teacher logits must cover the same *complete*
    vocabulary as the draft. Sparse prepared sampler supports are deliberately
    not accepted: these equations use raw, untruncated teacher probabilities.

    ``None`` disables positional decay, as in the upstream helper. A numeric
    decay must be finite and positive. The teacher is frozen and confidence
    labels are detached from both distributions. Masked rows receive zero
    gradient; an entirely empty loss is rejected before optimization.
    """
    _finite_tensor("draft logits", draft_logits)
    if (draft_logits.ndim not in (3, 4)
            or any(dimension <= 0 for dimension in draft_logits.shape)
            or draft_logits.shape[-1] < 2):
        raise ValueError("DSpark draft logits must have full-vocabulary block shape")
    _finite_tensor("full teacher logits", aligned_target_logits)
    _finite_tensor("confidence logits", confidence_logits)
    edge_shape = draft_logits.shape[:-1]
    if (aligned_target_logits.shape != draft_logits.shape
            or aligned_target_logits.device != draft_logits.device):
        raise ValueError("DSpark requires aligned full-vocabulary teacher logits")
    if (confidence_logits.shape != edge_shape
            or confidence_logits.device != draft_logits.device):
        raise ValueError("DSpark confidence shape/device differs from draft edges")
    if (not isinstance(target_ids, Tensor) or target_ids.shape != edge_shape
            or target_ids.dtype != torch.int64
            or target_ids.device != draft_logits.device
            or bool(((target_ids < 0) | (target_ids >= draft_logits.shape[-1])).any().item())):
        raise ValueError("DSpark target IDs must be aligned int64 vocabulary indices")
    if (not isinstance(eval_mask, Tensor) or eval_mask.shape != edge_shape
            or eval_mask.dtype != torch.bool
            or eval_mask.device != draft_logits.device
            or not bool(eval_mask.any().item())):
        raise ValueError("DSpark evaluation mask must select at least one aligned edge")
    if loss_decay_gamma is not None and (
            isinstance(loss_decay_gamma, bool)
            or not isinstance(loss_decay_gamma, (int, float))
            or not math.isfinite(float(loss_decay_gamma))
            or float(loss_decay_gamma) <= 0.0):
        raise ValueError("DSpark decay gamma must be finite and positive or None")

    weights = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None:
        positions = torch.arange(
            draft_logits.shape[-2], device=draft_logits.device, dtype=torch.float32
        )
        weights = weights * torch.exp(-positions / float(loss_decay_gamma))
    weight_sum = weights.sum()
    # The epsilon is part of the pinned released reduction, not an arbitrary
    # stabilizer that may be dropped when comparing source goldens.
    denominator = weight_sum + 1e-6
    cross_entropy_rows = F.cross_entropy(
        draft_logits.reshape(-1, draft_logits.shape[-1]),
        target_ids.reshape(-1), reduction="none",
    ).reshape(edge_shape)
    draft_probability = torch.softmax(draft_logits.float(), dim=-1)
    teacher_probability = torch.softmax(aligned_target_logits.detach().float(), dim=-1)
    l1_rows = (draft_probability - teacher_probability).abs().sum(dim=-1)
    confidence_targets = (1.0 - 0.5 * l1_rows).clamp(0.0, 1.0).detach()
    confidence_rows = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), confidence_targets, reduction="none"
    )
    cross_entropy = (cross_entropy_rows * weights).sum() / denominator
    l1_distance = (l1_rows * weights).sum() / denominator
    confidence = (confidence_rows * weights).sum() / denominator
    loss = 0.1 * cross_entropy + 0.9 * l1_distance + confidence
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ValueError("DSpark objective produced nonfinite arithmetic")
    return DSparkSourceLoss(
        loss=loss, cross_entropy=cross_entropy, l1_distance=l1_distance,
        confidence=confidence, confidence_targets=confidence_targets,
        weight_sum=weight_sum,
    )

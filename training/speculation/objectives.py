"""Full-vocabulary and prefix-aware Phase-5 training objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import struct

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import DraftOutput


@dataclass(frozen=True)
class ObjectiveWeights:
    forward_kl: float
    hard_ce: float
    common_draw_overlap: float
    confidence: float
    future_state_auxiliary: float
    dpace_alpha: float


@dataclass
class TargetSupport:
    token_ids: Tensor       # [batch,horizon,support]
    probabilities: Tensor   # [batch,horizon,support]
    mask: Tensor            # bool [batch,horizon,support]


@dataclass
class _FrozenDraftDecision:
    device_token_ids: Tensor
    token_ids: tuple[int, ...]
    sampling_weights: tuple[float, ...]
    total_sampling_weight: float


@dataclass
class _FrozenDraftDistribution(_FrozenDraftDecision):
    differentiable_probabilities: Tensor


@dataclass
class _FrozenDraftBatchDistribution:
    rows: tuple[_FrozenDraftDistribution | None, ...]
    device_token_ids: Tensor
    differentiable_probabilities: Tensor
    mask: Tensor


class _DeterministicLastDimGather(torch.autograd.Function):
    """Gather with a fixed-order backward that does not use atomic scatter."""

    @staticmethod
    def forward(ctx, values: Tensor, indices: Tensor) -> Tensor:
        if (values.ndim == 0 or indices.ndim != values.ndim
                or values.shape[:-1] != indices.shape[:-1]
                or indices.dtype != torch.int64):
            raise ValueError("deterministic gather shape/dtype is invalid")
        ctx.save_for_backward(indices)
        ctx.input_shape = values.shape
        return torch.gather(values, -1, indices)

    @staticmethod
    def backward(ctx, output_gradient: Tensor) -> tuple[Tensor, None]:
        (indices,) = ctx.saved_tensors
        vocabulary = ctx.input_shape[-1]
        input_gradient = torch.empty(
            ctx.input_shape, device=output_gradient.device, dtype=output_gradient.dtype
        )
        # Each vocabulary column is formed by a fixed-order reduction over the
        # gathered support. This is the exact gather derivative, including
        # repeated indices, without MPS's nondeterministic atomic scatter.
        chunk_size = 4096
        for start in range(0, vocabulary, chunk_size):
            end = min(vocabulary, start + chunk_size)
            vocabulary_ids = torch.arange(
                start, end, device=indices.device, dtype=indices.dtype
            )
            matches = indices.unsqueeze(-1) == vocabulary_ids
            input_gradient[..., start:end] = (
                matches.to(output_gradient.dtype) * output_gradient.unsqueeze(-1)
            ).sum(dim=-2)
        return input_gradient, None


def deterministic_last_dim_gather(values: Tensor, indices: Tensor) -> Tensor:
    return _DeterministicLastDimGather.apply(values, indices)


class _FrozenVocabularyMatmul(torch.autograd.Function):
    """F32-accumulating frozen F16 head without a persistent F32 table copy."""

    @staticmethod
    def forward(ctx, states: Tensor, table: Tensor, chunk_size: int) -> Tensor:
        if states.shape[-1] != table.shape[1] or chunk_size <= 0:
            raise ValueError("frozen vocabulary matmul shape/chunk is invalid")
        flat = states.reshape(-1, states.shape[-1]).float()
        output = torch.empty(
            flat.shape[0], table.shape[0], device=states.device, dtype=torch.float32
        )
        with torch.autocast(device_type=states.device.type, enabled=False):
            for start in range(0, table.shape[0], chunk_size):
                end = min(table.shape[0], start + chunk_size)
                output[:, start:end] = flat @ table[start:end].float().transpose(0, 1)
        ctx.save_for_backward(table)
        ctx.input_shape = states.shape
        ctx.input_dtype = states.dtype
        ctx.chunk_size = chunk_size
        return output.reshape(*states.shape[:-1], table.shape[0])

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None, None]:
        (table,) = ctx.saved_tensors
        flat_gradient = gradient.reshape(-1, gradient.shape[-1]).float()
        input_gradient = torch.zeros(
            flat_gradient.shape[0], table.shape[1],
            device=gradient.device, dtype=torch.float32,
        )
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            for start in range(0, table.shape[0], ctx.chunk_size):
                end = min(table.shape[0], start + ctx.chunk_size)
                input_gradient.add_(flat_gradient[:, start:end] @ table[start:end].float())
        return input_gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None, None


class GemmaTargetHead(nn.Module):
    """Frozen tied target table with Gemma multiplicative RMSNorm/softcap."""

    def __init__(self, token_table: Tensor, norm_weight: Tensor,
                 epsilon: float, softcap: float, chunk_size: int = 4096) -> None:
        super().__init__()
        if token_table.ndim != 2 or norm_weight.shape != (token_table.shape[1],):
            raise ValueError("Gemma target head tensors have incompatible shapes")
        self.register_buffer("token_table", token_table, persistent=False)
        self.register_buffer("norm_weight", norm_weight, persistent=False)
        self.epsilon = epsilon
        self.softcap = softcap
        self.chunk_size = chunk_size

    def forward(self, states: Tensor, apply_norm: bool = True) -> Tensor:
        normalized = states.float()
        if apply_norm:
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + self.epsilon
            )
            # GGUF stores Gemma's resolved multiplicative RMSNorm weight. The
            # engine consumes it directly; adding one here would train against
            # a different target head.
            normalized = normalized * self.norm_weight.float()
        logits = _FrozenVocabularyMatmul.apply(
            normalized, self.token_table, self.chunk_size
        )
        if self.softcap > 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


_SAMPLER_MAX_TOP_K = 256


def _f32(value: float) -> float:
    """Round through IEEE binary32, matching a C++ ``float`` operation."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _sampler_f32(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"draft sampler {name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"draft sampler {name} must be finite")
    try:
        rounded = _f32(scalar)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"draft sampler {name} is outside F32 range") from error
    if not math.isfinite(rounded):
        raise ValueError(f"draft sampler {name} is outside F32 range")
    return rounded


def _validated_proposal_sampler(
    sampler: tuple[float, int, float, float, float, int, int],
    vocabulary: int,
) -> tuple[float, int, float, float]:
    if not isinstance(sampler, tuple) or len(sampler) != 7:
        raise ValueError("draft sampler contract must contain exactly seven fields")
    temperature, top_k, top_p, min_p, repetition_penalty, seed, repetition_window = sampler
    temperature = _sampler_f32("temperature", temperature)
    top_p = _sampler_f32("top_p", top_p)
    min_p = _sampler_f32("min_p", min_p)
    repetition_penalty = _sampler_f32("repetition_penalty", repetition_penalty)
    if temperature <= 0.0:
        raise ValueError("draft sampler temperature must be positive")
    if (isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral)
            or top_k <= 0 or top_k > _SAMPLER_MAX_TOP_K or top_k > vocabulary):
        raise ValueError("draft sampler top_k must be in [1,min(vocabulary,256)]")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("draft sampler top_p must be in (0,1]")
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("draft sampler min_p must be in [0,1]")
    if repetition_penalty != 1.0:
        raise ValueError(
            "common-draw proposal requires repetition_penalty=1 without sampler history"
        )
    if (isinstance(repetition_window, bool)
            or not isinstance(repetition_window, numbers.Integral)
            or repetition_window < 0 or repetition_window > 64):
        raise ValueError("draft sampler repetition_window must be in [0,64]")
    if (isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
            or seed < 0 or seed > (1 << 64) - 1):
        raise ValueError("draft sampler seed must be a uint64")
    return temperature, int(top_k), top_p, min_p


def _freeze_draft_sampler_decisions(
    logits: Tensor, top_k: int, top_p: float, min_p: float, temperature: float,
) -> list[_FrozenDraftDecision]:
    """Freeze exact production decisions for a physical batch in one sort/readback."""
    if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0
            or not torch.is_floating_point(logits)):
        raise ValueError("draft sampler logits must be nonempty floating-point rows")
    if (isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral)
            or top_k <= 0 or top_k > _SAMPLER_MAX_TOP_K
            or top_k > logits.shape[1]):
        raise ValueError("draft sampler top_k must be in [1,min(vocabulary,256)]")
    top_k = int(top_k)
    top_p = _sampler_f32("top_p", top_p)
    min_p = _sampler_f32("min_p", min_p)
    temperature = _sampler_f32("temperature", temperature)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("draft sampler top_p must be in (0,1]")
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("draft sampler min_p must be in [0,1]")
    if temperature <= 0.0:
        raise ValueError("draft sampler temperature must be positive")

    # The target head produces F32 logits. Explicit conversion makes the
    # contract equally precise for diagnostic callers with another float dtype.
    device_logits = logits.float()
    detached_logits = device_logits.detach()
    if not bool(torch.isfinite(detached_logits).all().item()):
        raise ValueError("draft sampler logits must be finite after F32 conversion")
    # Vocabulary columns are in ascending token-ID order, so stable descending
    # sorting gives the runtime's lower-token-ID tie break. Sort the whole
    # physical row batch once, retain only bounded top-k storage, and perform
    # two bounded host transfers for all rows. This avoids a synchronization
    # per row without moving a full vocabulary row to the host.
    sorted_tokens = torch.argsort(
        detached_logits, dim=-1, descending=True, stable=True
    )
    top_tokens = sorted_tokens[:, :top_k].clone()
    del sorted_tokens
    top_values = torch.gather(detached_logits, 1, top_tokens)
    token_rows = top_tokens.to(device="cpu").tolist()
    decision_rows = top_values.to(device="cpu").tolist()

    decisions: list[_FrozenDraftDecision] = []
    for row_index, (token_ids, decision_values) in enumerate(zip(
        token_rows, decision_rows
    )):
        ordered = list(zip(token_ids, decision_values))
        maximum = ordered[0][1]
        pre_weights: list[float] = []
        full_total = 0.0
        for _, value in ordered:
            # C++ evaluates ``float - float`` before converting the operand to
            # double for std::exp.
            weight = math.exp(_f32(value - maximum))
            pre_weights.append(weight)
            full_total += weight
        if not math.isfinite(full_total) or full_total <= 0.0:
            raise ValueError("draft sampler pre-temperature normalization is invalid")

        retained_weight = 0.0
        retained = 0
        while True:
            retained_weight += pre_weights[retained]
            retained += 1
            if retained >= len(ordered) or retained_weight / full_total >= top_p:
                break
        if min_p > 0.0:
            min_p_count = 1
            while min_p_count < retained and pre_weights[min_p_count] >= min_p:
                min_p_count += 1
            retained = min_p_count

        sampling_weights: list[float] = []
        total_sampling_weight = 0.0
        for _, value in ordered[:retained]:
            weight = math.exp(_f32(value - maximum) / temperature)
            sampling_weights.append(weight)
            total_sampling_weight += weight
        if not math.isfinite(total_sampling_weight) or total_sampling_weight <= 0.0:
            raise ValueError("draft sampler exact normalization is invalid")
        decisions.append(_FrozenDraftDecision(
            device_token_ids=top_tokens[row_index, :retained],
            token_ids=tuple(int(token) for token in token_ids[:retained]),
            sampling_weights=tuple(sampling_weights),
            total_sampling_weight=total_sampling_weight,
        ))
    return decisions


def _materialize_draft_sampler_distributions(
    logits: Tensor,
    decisions: list[_FrozenDraftDecision | None],
    temperature: float,
) -> _FrozenDraftBatchDistribution:
    """Materialize retained probabilities with one deterministic batch gather.

    ``None`` rows are benign padding for masked speculative edges. They remain
    outside the returned objective but let the gather operate directly on the
    original flattened logits, avoiding an indexed copy and its scatter-style
    backward on MPS.
    """
    if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0
            or not torch.is_floating_point(logits)
            or len(decisions) != logits.shape[0]
            or not any(decision is not None for decision in decisions)):
        raise ValueError("draft sampler materialization batch is invalid")
    temperature = _sampler_f32("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("draft sampler temperature must be positive")

    retained_counts = [
        len(decision.token_ids) if decision is not None else 1
        for decision in decisions
    ]
    maximum_retained = max(retained_counts)
    padded_token_rows: list[list[int]] = []
    normalization_mask_rows: list[list[bool]] = []
    objective_mask_rows: list[list[bool]] = []
    for decision, retained in zip(decisions, retained_counts):
        token_ids = decision.token_ids if decision is not None else (0,)
        if (not token_ids
                or any(token < 0 or token >= logits.shape[1] for token in token_ids)):
            raise ValueError("draft sampler frozen token ID is outside vocabulary")
        padded_token_rows.append(
            list(token_ids) + [token_ids[0]] * (maximum_retained - retained)
        )
        normalization_mask_rows.append(
            [True] * retained + [False] * (maximum_retained - retained)
        )
        objective_mask_rows.append(
            ([True] * retained if decision is not None else [False] * retained)
            + [False] * (maximum_retained - retained)
        )

    padded_token_ids = torch.tensor(
        padded_token_rows, dtype=torch.int64, device=logits.device
    )
    normalization_mask = torch.tensor(
        normalization_mask_rows, dtype=torch.bool, device=logits.device
    )
    objective_mask = torch.tensor(
        objective_mask_rows, dtype=torch.bool, device=logits.device
    )
    device_logits = logits.float()
    values = deterministic_last_dim_gather(
        device_logits, padded_token_ids
    )
    weights = torch.where(
        normalization_mask,
        torch.exp((values - values[:, :1]) / temperature),
        torch.zeros_like(values),
    )
    normalization = weights.sum(dim=-1, keepdim=True)
    # The retained maximum has exponent exactly one. With validated finite F32
    # logits and a positive temperature, normalization is finite and at least
    # one without another device synchronization.
    probabilities = weights / normalization
    materialized: list[_FrozenDraftDistribution | None] = []
    for row_index, (decision, retained) in enumerate(zip(decisions, retained_counts)):
        if decision is None:
            materialized.append(None)
            continue
        materialized.append(_FrozenDraftDistribution(
            device_token_ids=padded_token_ids[row_index, :retained],
            token_ids=decision.token_ids,
            sampling_weights=decision.sampling_weights,
            total_sampling_weight=decision.total_sampling_weight,
            differentiable_probabilities=probabilities[row_index, :retained],
        ))
    return _FrozenDraftBatchDistribution(
        rows=tuple(materialized),
        device_token_ids=padded_token_ids,
        differentiable_probabilities=probabilities,
        mask=objective_mask,
    )


def _materialize_draft_sampler_distribution(
    logits: Tensor, decision: _FrozenDraftDecision, temperature: float,
) -> _FrozenDraftDistribution:
    if logits.ndim != 1 or logits.numel() == 0 or not torch.is_floating_point(logits):
        raise ValueError("draft sampler logits must be a nonempty floating-point row")
    materialized = _materialize_draft_sampler_distributions(
        logits.unsqueeze(0), [decision], temperature
    ).rows[0]
    if materialized is None:  # The single-row contract never supplies padding.
        raise RuntimeError("draft sampler distribution was not materialized")
    return materialized


def _freeze_draft_sampler_distribution(
    logits: Tensor, top_k: int, top_p: float, min_p: float, temperature: float,
) -> _FrozenDraftDistribution:
    """Freeze one exact production distribution and differentiable relaxation."""
    if logits.ndim != 1 or logits.numel() == 0 or not torch.is_floating_point(logits):
        raise ValueError("draft sampler logits must be a nonempty floating-point row")
    decision = _freeze_draft_sampler_decisions(
        logits.unsqueeze(0), top_k, top_p, min_p, temperature
    )[0]
    return _materialize_draft_sampler_distribution(logits, decision, temperature)


def _draft_sampler_distribution(
    logits: Tensor, top_k: int, top_p: float, min_p: float, temperature: float,
) -> tuple[Tensor, Tensor]:
    """Apply exact production support with differentiable F32 probabilities."""
    frozen = _freeze_draft_sampler_distribution(
        logits, top_k, top_p, min_p, temperature
    )
    return frozen.device_token_ids, frozen.differentiable_probabilities


def _validated_unit_draws(unit_draws: Tensor, expected_shape: torch.Size) -> Tensor:
    if (not isinstance(unit_draws, Tensor) or unit_draws.shape != expected_shape
            or unit_draws.dtype != torch.float64 or unit_draws.device.type != "cpu"):
        raise ValueError("proposal unit draws must be exact CPU F64 with the logits row shape")
    flat = unit_draws.contiguous().view(-1)
    if (flat.numel() == 0 or not bool(torch.isfinite(flat).all().item())
            or not bool(((flat >= 0.0) & (flat < 1.0)).all().item())):
        raise ValueError("proposal unit draws must be finite values in [0,1)")
    return flat


def _select_from_frozen_distribution(
    distribution: _FrozenDraftDecision, unit_draw: float,
) -> int:
    mapped_draw = unit_draw * distribution.total_sampling_weight
    cumulative = 0.0
    for token, weight in zip(distribution.token_ids, distribution.sampling_weights):
        cumulative += weight
        # The runtime comparison is strict: an exact boundary belongs to the
        # following token's interval.
        if mapped_draw < cumulative:
            return token
    return distribution.token_ids[-1]


def select_exact_proposal_tokens(
    logits: Tensor,
    unit_draws: Tensor,
    sampler: tuple[float, int, float, float, float, int, int],
) -> Tensor:
    """Select coupled proposal tokens with the target runtime's exact CDF law.

    ``unit_draws`` are authoritative trace tickets and intentionally remain
    CPU F64.  The returned token IDs are detached and live beside ``logits``;
    proposal sampling never becomes a gradient path.
    """
    if (not isinstance(logits, Tensor) or logits.ndim < 2
            or logits.shape[-1] == 0 or logits.numel() == 0
            or not torch.is_floating_point(logits)):
        raise ValueError("proposal logits must have nonempty [...,vocabulary] floating shape")
    temperature, top_k, top_p, min_p = _validated_proposal_sampler(
        sampler, logits.shape[-1]
    )
    flat_draws = _validated_unit_draws(unit_draws, logits.shape[:-1])
    rows = logits.reshape(-1, logits.shape[-1])
    decisions = _freeze_draft_sampler_decisions(
        rows, top_k, top_p, min_p, temperature
    )
    selected = [
        _select_from_frozen_distribution(decision, float(flat_draws[row_index]))
        for row_index, decision in enumerate(decisions)
    ]
    return torch.tensor(
        selected, dtype=torch.int64, device=logits.device
    ).reshape(logits.shape[:-1]).detach()


def _common_draw_overlap_and_proposals(
    logits: Tensor,
    support: TargetSupport,
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    reached_mask: Tensor | None = None,
    unit_draws: Tensor | None = None,
    reduction: str = "mean",
) -> tuple[Tensor, Tensor | None]:
    if reduction not in {"mean", "none"}:
        raise ValueError("common-draw reduction must be mean or none")
    flat_draws = None
    if unit_draws is not None:
        flat_draws = _validated_unit_draws(unit_draws, logits.shape[:-1])

    if reached_mask is None:
        reached_rows = [
            [True] * logits.shape[1] for _ in range(logits.shape[0])
        ]
    else:
        reached_rows = reached_mask.detach().to(device="cpu").tolist()
    locations = [
        (batch, depth)
        for batch, row in enumerate(reached_rows)
        for depth, is_reached in enumerate(row)
        if is_reached
    ]
    if (not locations
            or any(not any(row) for row in reached_rows)):
        raise ValueError("common-draw objective has no reached edge for an origin")

    # The discrete production decisions have no gradient path. Pack only
    # reached rows into one physical sort/readback so support freezing has a
    # constant number of device synchronizations per objective, not one per
    # speculative edge.
    if len(locations) == logits.shape[0] * logits.shape[1]:
        decision_logits = logits.detach().reshape(-1, logits.shape[-1])
    else:
        decision_logits = torch.stack([
            logits[batch, depth].detach() for batch, depth in locations
        ])
    decisions = _freeze_draft_sampler_decisions(
        decision_logits, top_k, top_p, min_p, temperature
    )
    decisions_by_flat_row: list[_FrozenDraftDecision | None] = [
        None
    ] * (logits.shape[0] * logits.shape[1])
    for (batch, depth), decision in zip(locations, decisions):
        decisions_by_flat_row[batch * logits.shape[1] + depth] = decision
    distribution_batch = _materialize_draft_sampler_distributions(
        logits.reshape(-1, logits.shape[-1]), decisions_by_flat_row, temperature
    )

    batch_size, horizon = logits.shape[:2]
    draft_width = distribution_batch.device_token_ids.shape[-1]
    draft_tokens = distribution_batch.device_token_ids.reshape(
        batch_size, horizon, draft_width
    )
    draft_probabilities = distribution_batch.differentiable_probabilities.reshape(
        batch_size, horizon, draft_width
    )
    draft_mask = distribution_batch.mask.reshape(batch_size, horizon, draft_width)
    target_probabilities = torch.where(
        support.mask, support.probabilities, torch.zeros_like(support.probabilities)
    )
    target_high = torch.cumsum(target_probabilities, dim=-1)
    target_low = target_high - target_probabilities
    draft_high = torch.cumsum(draft_probabilities, dim=-1)
    draft_low = draft_high - draft_probabilities
    interval_overlap = torch.relu(
        torch.minimum(target_high.unsqueeze(-1), draft_high.unsqueeze(-2))
        - torch.maximum(target_low.unsqueeze(-1), draft_low.unsqueeze(-2))
    )
    token_match = (
        support.token_ids.unsqueeze(-1) == draft_tokens.unsqueeze(-2)
    ) & support.mask.unsqueeze(-1) & draft_mask.unsqueeze(-2)
    collision_by_target = torch.where(
        token_match, interval_overlap, torch.zeros_like(interval_overlap)
    ).sum(dim=-1)
    # Preserve the previous target-rank addition order exactly while doing
    # each rank for every physical row in one device operation.
    collision = logits.new_zeros((batch_size, horizon))
    for target_index in range(collision_by_target.shape[-1]):
        collision = collision + collision_by_target[:, :, target_index]

    proposal_rows: list[list[int]] | None = [] if flat_draws is not None else None
    for batch in range(logits.shape[0]):
        row_proposals = [0] * logits.shape[1]
        for depth in range(logits.shape[1]):
            if not reached_rows[batch][depth]:
                continue
            distribution = distribution_batch.rows[batch * logits.shape[1] + depth]
            if distribution is None:
                raise RuntimeError("reached draft distribution was not materialized")
            if flat_draws is not None:
                draw_index = batch * logits.shape[1] + depth
                row_proposals[depth] = _select_from_frozen_distribution(
                    distribution, float(flat_draws[draw_index])
                )
        if proposal_rows is not None:
            proposal_rows.append(row_proposals)
    # Preserve the established per-origin/per-edge reduction tree; only the
    # support matching above is vectorized across physical rows.
    origin_losses = []
    for batch in range(batch_size):
        edge_losses = [
            1.0 - collision[batch, depth]
            for depth in range(horizon) if reached_rows[batch][depth]
        ]
        origin_losses.append(torch.stack(edge_losses).mean())
    proposals = None
    if proposal_rows is not None:
        proposals = torch.tensor(
            proposal_rows, dtype=torch.int64, device=logits.device
        ).detach()
    origin_loss = torch.stack(origin_losses)
    return (origin_loss.mean() if reduction == "mean" else origin_loss), proposals


def common_draw_overlap_loss(
    logits: Tensor,
    support: TargetSupport,
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    reached_mask: Tensor | None = None,
) -> Tensor:
    loss, _ = _common_draw_overlap_and_proposals(
        logits, support, top_k, top_p, min_p, temperature, reached_mask
    )
    return loss


def dpace_weights(
    selected_probabilities: Tensor,
    alpha: float,
    reached_mask: Tensor | None = None,
) -> Tensor:
    """Detached D-PACE weights with a prefix-closed deployment mask.

    This is only the paper-described weighting equation. ``standard_objective``
    deliberately applies it to this project's compound distribution/CE loss;
    that broader use is a local interpretation, not a source-exact claim.
    """
    if selected_probabilities.ndim != 2:
        raise ValueError("D-PACE probabilities must have [batch,horizon] shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("D-PACE alpha must be in [0,1]")
    if reached_mask is None:
        reached = torch.ones_like(selected_probabilities, dtype=torch.bool)
    else:
        if reached_mask.shape != selected_probabilities.shape or reached_mask.dtype != torch.bool:
            raise ValueError("D-PACE reached mask shape/dtype changed")
        reached = reached_mask
    if not bool(reached.any(dim=1).all().item()):
        raise ValueError("D-PACE reached mask removed every edge from an origin")
    if reached.shape[1] > 1 and bool((~reached[:, :-1] & reached[:, 1:]).any().item()):
        raise ValueError("D-PACE reached mask is not prefix-closed")

    smoothed = (1.0 - alpha) * selected_probabilities.detach() + alpha
    # A masked suffix is neither a failed token nor an extra surviving token.
    # Give it the multiplicative identity, then exclude its prefix term from
    # the reverse sum so poisoned/unavailable suffix probabilities cannot
    # affect any earlier weight.
    smoothed = torch.where(reached, smoothed, torch.ones_like(smoothed))
    prefix = torch.cumprod(smoothed, dim=1) * reached.to(smoothed.dtype)
    # w_j=sum_(m=j)^H product_(i=1)^m q_tilde_i
    return (
        torch.flip(torch.cumsum(torch.flip(prefix, dims=(1,)), dim=1), dims=(1,))
        * reached.to(prefix.dtype)
    ).detach()


def standard_objective(
    output: DraftOutput,
    head: GemmaTargetHead,
    labels: Tensor,
    target_support: TargetSupport,
    weights: ObjectiveWeights,
    sampler: tuple[float, int, float, float, float, int, int],
    position_weighting: str,
    reached_mask: Tensor | None = None,
    *,
    proposal_policy: str,
    target_unit_draws: Tensor | None = None,
    materialize_metrics: bool = True,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, float], Tensor]:
    if reduction not in {"mean", "none"}:
        raise ValueError("standard objective reduction must be mean or none")
    if output.states.ndim != 3:
        raise ValueError("standard objective requires one state per edge")
    batch_horizon = output.states.shape[:2]
    if labels.shape != batch_horizon or labels.dtype != torch.int64:
        raise ValueError("standard objective label shape/dtype changed")
    if (target_support.token_ids.ndim != 3
            or target_support.token_ids.shape[:2] != batch_horizon
            or target_support.probabilities.shape != target_support.token_ids.shape
            or target_support.mask.shape != target_support.token_ids.shape
            or target_support.token_ids.dtype != torch.int64
            or target_support.mask.dtype != torch.bool):
        raise ValueError("standard objective target-support shape/dtype changed")
    if weights.future_state_auxiliary != 0.0:
        raise ValueError("future-state auxiliary has no supplied training target")
    if output.confidence is None:
        if weights.confidence > 0.0:
            raise ValueError("positive confidence weight requires causal confidence logits")
    elif output.confidence.shape != batch_horizon:
        raise ValueError("confidence logit shape changed")
    if proposal_policy not in {"deterministic", "common_draw"}:
        raise ValueError("proposal policy must be deterministic or common_draw")
    if proposal_policy == "deterministic":
        if target_unit_draws is not None:
            raise ValueError("deterministic proposal policy must not receive unit draws")
    elif target_unit_draws is None:
        raise ValueError("common_draw proposal policy requires exact unit draws")
    else:
        _validated_unit_draws(target_unit_draws, batch_horizon)

    if reached_mask is None:
        reached = torch.ones(batch_horizon, dtype=torch.bool, device=output.states.device)
    else:
        if reached_mask.shape != batch_horizon or reached_mask.dtype != torch.bool:
            raise ValueError("deployment reached-mask shape/dtype changed")
        reached = reached_mask
    if not bool(reached.any(dim=1).all().item()):
        raise ValueError("deployment reached mask removed every edge from an origin")
    if reached.shape[1] > 1 and bool((~reached[:, :-1] & reached[:, 1:]).any().item()):
        raise ValueError("deployment reached mask is not prefix-closed")

    temperature, top_k, top_p, min_p = _validated_proposal_sampler(
        sampler, head.token_table.shape[0]
    )

    logits = head(output.states, apply_norm=not output.head_pre_normalized)
    raw_log_normalizer = torch.logsumexp(logits, dim=-1)
    selected_raw_logit = deterministic_last_dim_gather(
        logits, labels.unsqueeze(-1)
    ).squeeze(-1)
    # Classification and D-PACE q_i intentionally stay in the raw-logit
    # training space. Temperature belongs to the deployed distribution term,
    # not to ordinary hard classification.
    hard_rows = raw_log_normalizer - selected_raw_logit
    selected_probability = torch.exp(-hard_rows)

    # The differentiable forward-KL control is the full-vocabulary relaxation
    # of the deployed temperature transform. Discrete top-k/top-p/min-p
    # truncation remains exclusive to the separate common-draw term.
    temperature_logits = logits / temperature
    temperature_log_normalizer = torch.logsumexp(temperature_logits, dim=-1)
    gathered_temperature_log_probability = deterministic_last_dim_gather(
        temperature_logits, target_support.token_ids
    ) - temperature_log_normalizer.unsqueeze(-1)
    safe_target_probability = torch.where(
        target_support.mask,
        target_support.probabilities,
        torch.ones_like(target_support.probabilities),
    )
    target_log = torch.where(
        target_support.mask,
        torch.log(safe_target_probability),
        torch.zeros_like(target_support.probabilities),
    )
    kl_rows = torch.where(
        target_support.mask,
        target_support.probabilities
        * (target_log - gathered_temperature_log_probability),
        torch.zeros_like(gathered_temperature_log_probability),
    ).sum(dim=-1)

    if position_weighting == "uniform":
        position = torch.ones_like(hard_rows)
    elif position_weighting == "fixed_gamma7":
        depth = torch.arange(hard_rows.shape[1], device=hard_rows.device, dtype=hard_rows.dtype)
        position = torch.exp(-depth / 7.0).unsqueeze(0).expand_as(hard_rows)
    elif position_weighting.startswith("gamma:"):
        gamma = float(position_weighting.split(":", 1)[1])
        if gamma <= 0.0:
            raise ValueError("position gamma must be positive")
        depth = torch.arange(hard_rows.shape[1], device=hard_rows.device, dtype=hard_rows.dtype)
        position = torch.exp(-depth / gamma).unsqueeze(0).expand_as(hard_rows)
    elif position_weighting == "dpace":
        position = dpace_weights(selected_probability, weights.dpace_alpha, reached)
    else:
        raise ValueError(f"unknown position weighting {position_weighting}")
    position = position * reached.to(position.dtype)
    reached_count = reached.sum(dim=1).to(position.dtype)
    if position_weighting == "dpace":
        # The paper equation is an ordinary reached-position average of
        # weighted terms, not a self-normalized weighted mean. Applying the
        # weights to both local compound terms is an explicit project choice.
        denominator = reached_count
    else:
        denominator = position.sum(dim=1)
    kl_by_origin = (kl_rows * position).sum(dim=1) / denominator
    hard_by_origin = (hard_rows * position).sum(dim=1) / denominator

    confidence_by_origin = logits.new_zeros(batch_horizon[0])
    overlap_by_origin = logits.new_zeros(batch_horizon[0])
    needs_discrete_sampler = (
        weights.common_draw_overlap != 0.0 or weights.confidence != 0.0
    )
    if not needs_discrete_sampler:
        # The frozen path normally enforces this input invariant row by row.
        # Retain the fail-closed check without paying for support construction.
        if not bool(torch.isfinite(logits.detach().float()).all().item()):
            raise ValueError("draft sampler logits must be finite after F32 conversion")
    else:
        if proposal_policy == "common_draw":
            overlap_by_origin, proposal_tokens = _common_draw_overlap_and_proposals(
                logits, target_support, top_k, top_p, min_p, temperature,
                reached, target_unit_draws, reduction="none",
            )
            if proposal_tokens is None:
                raise RuntimeError("common-draw proposal selection was not materialized")
        else:
            overlap_by_origin, _ = _common_draw_overlap_and_proposals(
                logits, target_support, top_k, top_p, min_p, temperature,
                reached, reduction="none",
            )
            # argmax returns the first vocabulary index at a tie, which is the
            # deterministic sampler's lower-token-ID rule.
            proposal_tokens = torch.argmax(logits.detach(), dim=-1)
        if output.confidence is not None:
            proposal_match = (
                proposal_tokens == labels
            ).to(logits.dtype).detach()
            confidence_rows = F.binary_cross_entropy_with_logits(
                output.confidence.float(), proposal_match, reduction="none"
            )
            confidence_by_origin = (
                (confidence_rows * reached.to(confidence_rows.dtype)).sum(dim=1)
                / reached_count.to(confidence_rows.dtype)
            )
    total_by_origin = (
        weights.forward_kl * kl_by_origin
        + weights.hard_ce * hard_by_origin
        + weights.common_draw_overlap * overlap_by_origin
        + weights.confidence * confidence_by_origin
    )
    kl = kl_by_origin.mean()
    hard = hard_by_origin.mean()
    overlap = overlap_by_origin.mean()
    confidence = confidence_by_origin.mean()
    total = (
        weights.forward_kl * kl
        + weights.hard_ce * hard
        + weights.common_draw_overlap * overlap
        + weights.confidence * confidence
        if reduction == "mean" else total_by_origin
    )
    metrics = {}
    if materialize_metrics:
        # Converting device scalars to Python numbers synchronizes MPS. Keep it
        # an explicit caller choice so steady-state training need not do this
        # on every optimization step.
        metrics = {
            "loss": float(
                (total if reduction == "mean" else total_by_origin.mean()).detach()
            ),
            "forward_kl": float(kl.detach()),
            "hard_ce": float(hard.detach()),
            "common_draw_overlap": float(overlap.detach()),
            "confidence": float(confidence.detach()),
        }
    return total, metrics, logits


def dblast_objective(
    output: DraftOutput,
    head: GemmaTargetHead,
    labels: Tensor,
    target_selected_probabilities: Tensor,
    objective: str,
    tau: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    if output.states.ndim != 4 or output.category_logits is None:
        raise ValueError("DBlast objective requires [batch,H,K,D] states and a category prior")
    logits = head(output.states, apply_norm=not output.head_pre_normalized)
    category_log_prob = F.log_softmax(output.category_logits, dim=-1)
    token_log_prob = F.log_softmax(logits, dim=-1)
    labels_expanded = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, logits.shape[2], 1)
    selected = deterministic_last_dim_gather(
        token_log_prob, labels_expanded
    ).squeeze(-1)
    log_prefix_by_category = category_log_prob.unsqueeze(1) + torch.cumsum(selected, dim=1)
    log_q_prefix = torch.logsumexp(log_prefix_by_category, dim=-1)
    nll = -log_q_prefix[:, -1].mean()
    al_terms = []
    log_p_prefix = torch.cumsum(torch.log(target_selected_probabilities), dim=1)
    log_ratio = log_q_prefix - log_p_prefix
    previous_log_q = torch.zeros_like(log_q_prefix[:, :1])
    conditional_log_q = log_q_prefix - torch.cat((previous_log_q, log_q_prefix[:, :-1]), dim=1)
    accept = torch.minimum(
        torch.ones_like(conditional_log_q),
        torch.exp(torch.log(target_selected_probabilities) - conditional_log_q),
    )
    survival_sum = torch.cumsum(torch.cumprod(accept, dim=1), dim=1)
    for batch in range(labels.shape[0]):
        below = torch.nonzero(log_ratio[batch].detach() < math_log(tau), as_tuple=False)
        last = int(below[0, 0].item()) if below.numel() else labels.shape[1] - 1
        al_terms.append(-(log_ratio[batch, :last + 1] + torch.log(survival_sum[batch, :last + 1])).sum())
    al = torch.stack(al_terms).mean()
    if objective == "nll":
        loss = nll
    elif objective == "acceptance_length":
        loss = al
    elif objective == "nll_plus_acceptance_length":
        loss = nll + al
    else:
        raise ValueError(f"unknown DBlast objective {objective}")
    return loss, {"loss": float(loss.detach()), "nll": float(nll.detach()), "acceptance_length": float(al.detach())}


def math_log(value: float) -> float:
    # Keeps the scalar outside the graph; no numerical floor is inserted.
    import math
    return math.log(value)

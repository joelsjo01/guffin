"""Mechanism-distinct PyTorch modules for Phase-5 Gemma transfer training.

These are offline training modules. Production execution and token selection
remain in the C++/Metal engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DraftOutput:
    states: Tensor
    confidence: Tensor | None = None
    category_logits: Tensor | None = None
    head_pre_normalized: bool = False


def _origin_taps(context_taps: Tensor, anchor_indices: Tensor) -> Tensor:
    if context_taps.ndim != 3 or anchor_indices.ndim != 1:
        raise ValueError("causal feature sequence/anchor index shape is invalid")
    return context_taps.index_select(0, anchor_indices)


def _neox_rope(value: Tensor, positions: Tensor, theta: float = 10_000.0) -> Tensor:
    """Gemma/Qwen NeoX half-split RoPE with explicit absolute positions."""
    if value.ndim != 4 or positions.shape != (value.shape[0], value.shape[2]):
        raise ValueError("RoPE value/position shape is invalid")
    width = value.shape[-1]
    if width % 2:
        raise ValueError("RoPE head width must be even")
    half = width // 2
    frequency = torch.pow(
        torch.tensor(theta, device=value.device, dtype=torch.float32),
        -torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width,
    )
    angle = positions.float().unsqueeze(1).unsqueeze(-1) * frequency.view(1, 1, 1, half)
    cosine, sine = torch.cos(angle).to(value.dtype), torch.sin(angle).to(value.dtype)
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(width)) if affine else None

    def forward(self, value: Tensor) -> Tensor:
        normalized = value * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.epsilon)
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized.to(value.dtype)


class TapFusion(nn.Module):
    def __init__(self, target_width: int, tap_count: int, output_width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(target_width * tap_count, output_width, bias=False)
        self.norm = RMSNorm(output_width)

    def forward(self, taps: Tensor) -> Tensor:
        return self.norm(self.projection(taps.flatten(-2)))


class TargetSpaceProjection(nn.Module):
    def __init__(self, input_width: int, target_width: int) -> None:
        super().__init__()
        self.projection = (nn.Identity() if input_width == target_width
                           else nn.Linear(input_width, target_width, bias=False))

    def forward(self, value: Tensor) -> Tensor:
        return self.projection(value)


class GatedResidualCell(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width * 2, width)
        self.candidate = nn.Linear(width * 2, width)

    def forward(self, token: Tensor, state: Tensor) -> Tensor:
        joined = torch.cat((state, token), dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        candidate = F.silu(self.candidate(joined))
        return gate * state + (1.0 - gate) * candidate


class SemanticTwoLayerCell(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(width * 2, width * 2, bias=False)
        self.second = nn.Linear(width * 2, width)

    def forward(self, token: Tensor, state: Tensor) -> Tensor:
        return state + self.second(F.silu(self.first(torch.cat((state, token), dim=-1))))


class DirectRecurrentDrafter(nn.Module):
    """Direct-token recurrent transfer; future target features are never read."""

    def __init__(
        self, target_width: int, vocabulary: int, tap_count: int, width: int,
        horizon: int, cell: str, token_table: Tensor,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        # Gemma scales its tied input embedding by sqrt(hidden_width) before
        # the first learned projection.  Keeping that same contract here avoids
        # making the recurrent token path roughly sixty times smaller than the
        # RMS-normalized target-state path at Gemma-2 9B width.
        self.embedding_scale = math.sqrt(target_width)
        self.fusion = TapFusion(target_width, tap_count, width)
        self.token_projection = nn.Linear(target_width, width, bias=False)
        if cell == "gated_residual":
            self.cell = GatedResidualCell(width)
        elif cell == "gru":
            self.cell = nn.GRUCell(width, width)
        elif cell == "semantic_two_layer":
            self.cell = SemanticTwoLayerCell(width)
        else:
            raise ValueError(f"unknown direct recurrent cell {cell}")
        self.to_target = TargetSpaceProjection(width, target_width)
        self.confidence = nn.Linear(width, 1)
        self.register_buffer("token_table", token_table, persistent=False)
        if token_table.shape != (vocabulary, target_width):
            raise ValueError("token table shape does not match recurrent drafter")

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        del context_positions
        state = self.fusion(_origin_taps(context_taps, anchor_indices))
        outputs, confidences = [], []
        for depth in range(self.horizon):
            token = self.token_projection(
                self.token_table[rollout_tokens[:, depth]].to(state.dtype)
                * self.embedding_scale
            )
            state = self.cell(token, state)
            outputs.append(self.to_target(state))
            confidences.append(self.confidence(state).squeeze(-1))
        return DraftOutput(
            torch.stack(outputs, dim=1),
            confidence=torch.stack(confidences, dim=1),
        )


class ReDrafter(nn.Module):
    """Gemma transfer of the published bias/recurrent/residual equations."""

    def __init__(self, target_width: int, vocabulary: int, tap_count: int, width: int,
                 horizon: int, token_table: Tensor) -> None:
        super().__init__()
        self.horizon = horizon
        self.origin = TapFusion(target_width, tap_count, width)
        self.token_projection = nn.Linear(target_width, width, bias=False)
        self.w_no_bias = nn.Linear(width, width, bias=False)
        self.u_with_bias = nn.Linear(width, width, bias=True)
        self.residuals = nn.ModuleList(nn.Linear(width * 2, width * 2, bias=True) for _ in range(2))
        self.to_target = TargetSpaceProjection(width * 2, target_width)
        self.register_buffer("token_table", token_table, persistent=False)
        if token_table.shape != (vocabulary, target_width):
            raise ValueError("token table shape does not match ReDrafter")

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        del context_positions
        origin = self.origin(_origin_taps(context_taps, anchor_indices))
        state = torch.zeros_like(origin)
        outputs = []
        for depth in range(self.horizon):
            token = self.token_projection(
                self.token_table[rollout_tokens[:, depth]].to(origin.dtype)
            )
            state = F.silu(self.w_no_bias(token) + self.u_with_bias(state))
            exit_state = torch.cat((state, origin), dim=-1)
            for residual in self.residuals:
                exit_state = exit_state + F.silu(residual(exit_state))
            outputs.append(self.to_target(exit_state))
        return DraftOutput(torch.stack(outputs, dim=1))


class Eagle3Drafter(nn.Module):
    """Three-tap fusion and mandatory training-time self-rollout decoder."""

    def __init__(self, target_width: int, vocabulary: int, width: int,
                 horizon: int, token_table: Tensor) -> None:
        super().__init__()
        self.horizon = horizon
        self.fusion = nn.Linear(target_width * 3, width, bias=False)
        self.token_projection = nn.Linear(target_width, width, bias=False)
        self.token_norm = RMSNorm(width)
        self.feature_norm = RMSNorm(width)
        self.query = nn.Linear(width * 2, width, bias=False)
        self.key = nn.Linear(width * 2, width, bias=False)
        self.value = nn.Linear(width * 2, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width)
        self.gate = nn.Linear(width, width * 4, bias=False)
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)
        self.to_target = TargetSpaceProjection(width, target_width)
        self.register_buffer("token_table", token_table, persistent=False)
        if token_table.shape != (vocabulary, target_width):
            raise ValueError("token table shape does not match EAGLE-3")

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        del context_positions
        feature = self.fusion(_origin_taps(context_taps, anchor_indices).flatten(-2))
        keys, values, outputs = [], [], []
        scale = self.query.out_features ** -0.5
        for depth in range(self.horizon):
            token = self.token_norm(self.token_projection(
                self.token_table[rollout_tokens[:, depth]].to(feature.dtype)
            ))
            joined = torch.cat((token, self.feature_norm(feature)), dim=-1)
            query, key, value = self.query(joined), self.key(joined), self.value(joined)
            keys.append(key)
            values.append(value)
            key_cache = torch.stack(keys, dim=1)
            value_cache = torch.stack(values, dim=1)
            weights = torch.softmax(torch.einsum("bd,btd->bt", query, key_cache) * scale, dim=-1)
            state = feature + self.attention_output(torch.einsum("bt,btd->bd", weights, value_cache))
            normalized = self.ffn_norm(state)
            feature = state + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
            outputs.append(self.to_target(feature))
        return DraftOutput(torch.stack(outputs, dim=1))


class InjectedAttentionLayer(nn.Module):
    def __init__(
        self, width: int, context_width: int, separate_context_kv: bool,
        block_causal: bool,
    ) -> None:
        super().__init__()
        heads = max(1, width // 128)
        if width % heads:
            raise ValueError("draft width must divide its attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.block_causal = block_causal
        self.norm = RMSNorm(width)
        # Both registered fusion paths already apply their source RMSNorm
        # before the layer-specific K/V projection.
        self.context_norm = nn.Identity()
        self.query = nn.Linear(width, width, bias=False)
        self.block_key = nn.Linear(width, width, bias=False)
        self.block_value = nn.Linear(width, width, bias=False)
        self.context_key = (nn.Linear(context_width, width, bias=False)
                            if separate_context_kv else self.block_key)
        self.context_value = (nn.Linear(context_width, width, bias=False)
                              if separate_context_kv else self.block_value)
        self.output = nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width)
        self.gate = nn.Linear(width, width * 4, bias=False)
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(*value.shape[:-1], self.heads, self.head_width).transpose(-3, -2)

    def forward(
        self,
        block: Tensor,
        context: Tensor,
        block_positions: Tensor,
        context_positions: Tensor,
        anchor_indices: Tensor,
    ) -> Tensor:
        if (block.ndim != 3 or context.ndim != 2 or context_positions.ndim != 1
                or context.shape[0] != context_positions.numel()
                or anchor_indices.shape != (block.shape[0],)):
            raise ValueError("injected-attention causal shapes are invalid")
        normalized_block = self.norm(block)
        normalized_context = self.context_norm(context)
        query = _neox_rope(self._heads(self.query(normalized_block)), block_positions)
        block_keys = _neox_rope(self._heads(self.block_key(normalized_block)), block_positions)
        context_keys = self.context_key(normalized_context).view(
            context.shape[0], self.heads, self.head_width
        ).permute(1, 0, 2).unsqueeze(0).expand(block.shape[0], -1, -1, -1)
        context_keys = _neox_rope(
            context_keys,
            context_positions.unsqueeze(0).expand(block.shape[0], -1),
        )
        block_values = self._heads(self.block_value(normalized_block))
        context_values = self.context_value(normalized_context).view(
            context.shape[0], self.heads, self.head_width
        ).permute(1, 0, 2).unsqueeze(0).expand(block.shape[0], -1, -1, -1)
        keys = torch.cat((context_keys, block_keys), dim=2)
        values = torch.cat((context_values, block_values), dim=2)
        scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)) \
            * (self.head_width ** -0.5)
        context_indices = torch.arange(context.shape[0], device=block.device)
        context_visible = context_indices.unsqueeze(0) <= anchor_indices.unsqueeze(1)
        if self.block_causal:
            block_visible = torch.ones(
                block.shape[1], block.shape[1], device=block.device, dtype=torch.bool
            ).tril().unsqueeze(0).expand(block.shape[0], -1, -1)
        else:
            block_visible = torch.ones(
                block.shape[0], block.shape[1], block.shape[1],
                device=block.device, dtype=torch.bool,
            )
        visible = torch.cat((
            context_visible.unsqueeze(1).expand(-1, block.shape[1], -1),
            block_visible,
        ), dim=2).unsqueeze(1)
        probabilities = torch.softmax(scores.masked_fill(~visible, -torch.inf), dim=-1).to(values.dtype)
        attended = torch.matmul(probabilities, values).transpose(-3, -2).contiguous().view_as(block)
        block = block + self.output(attended)
        normalized = self.ffn_norm(block)
        return block + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class BlockDiffusionDrafter(nn.Module):
    """One-pass anchor-plus-mask DFlash/DFlare mechanism."""

    def __init__(
        self, target_width: int, vocabulary: int, tap_count: int, width: int,
        layers: int, block_size: int, token_table: Tensor, flare: bool,
        block_causal: bool = False,
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block diffusion requires anchor plus at least one mask")
        self.horizon = block_size - 1
        self.flare = flare
        self.token_projection = nn.Linear(target_width, width, bias=False)
        self.mask_embeddings = nn.Parameter(torch.empty(self.horizon, width))
        self.position_embeddings = nn.Parameter(torch.empty(block_size, width))
        nn.init.normal_(self.mask_embeddings, std=width ** -0.5)
        nn.init.normal_(self.position_embeddings, std=width ** -0.5)
        if flare:
            self.fusion_logits = nn.Parameter(torch.zeros(layers, tap_count))
            self.fusion_norms = nn.ModuleList(RMSNorm(target_width) for _ in range(layers))
            context_width = target_width
        else:
            self.shared_fusion = TapFusion(target_width, tap_count, width)
            context_width = width
        self.layers = nn.ModuleList(
            InjectedAttentionLayer(width, context_width, separate_context_kv=flare,
                                   block_causal=block_causal)
            for _ in range(layers)
        )
        self.to_target = TargetSpaceProjection(width, target_width)
        self.register_buffer("token_table", token_table, persistent=False)
        if token_table.shape != (vocabulary, target_width):
            raise ValueError("token table shape does not match block drafter")

    def _context(self, taps: Tensor, layer: int) -> Tensor:
        if not self.flare:
            return self.shared_fusion(taps)
        weights = torch.softmax(self.fusion_logits[layer], dim=0)
        return self.fusion_norms[layer]((taps * weights.view(1, -1, 1)).sum(dim=1))

    def block_states(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, anchor_tokens: Tensor,
    ) -> Tensor:
        anchor = self.token_projection(self.token_table[anchor_tokens].float()).unsqueeze(1)
        masks = self.mask_embeddings.unsqueeze(0).expand(anchor.shape[0], -1, -1)
        block = torch.cat((anchor, masks), dim=1) + self.position_embeddings
        block_positions = context_positions.index_select(0, anchor_indices).unsqueeze(1) \
            + torch.arange(block.shape[1], device=block.device).unsqueeze(0)
        for layer_index, layer in enumerate(self.layers):
            block = layer(
                block, self._context(context_taps, layer_index), block_positions,
                context_positions, anchor_indices,
            )
        return block

    def semantic_states(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, anchor_tokens: Tensor,
    ) -> Tensor:
        return self.block_states(
            context_taps, context_positions, anchor_indices, anchor_tokens
        )[:, 1:]

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        states = self.semantic_states(
            context_taps, context_positions, anchor_indices, rollout_tokens[:, 0]
        )
        return DraftOutput(self.to_target(states))


class DSparkDrafter(nn.Module):
    """Block semantic backbone plus source-distinct sequential corrections."""

    def __init__(
        self, target_width: int, vocabulary: int, tap_count: int, width: int,
        horizon: int, rank: int, correction: str, token_table: Tensor,
    ) -> None:
        super().__init__()
        # DSpark gamma rows are anchor plus gamma-1 masks; the anchor output is
        # proposal one, so a dedicated block backbone retains all gamma rows.
        self.horizon = horizon
        self.backbone = BlockDiffusionDrafter(
            target_width, vocabulary, tap_count, width, 3, horizon,
            token_table, flare=False,
        )
        self.token_rank = nn.Embedding(vocabulary, rank)
        self.correction = correction
        if correction == "markov":
            self.rank_to_target = nn.Linear(rank, target_width, bias=False)
        elif correction == "gated":
            self.gate = nn.Linear(rank + width, target_width)
            self.rank_to_target = nn.Linear(rank, target_width, bias=False)
        elif correction == "source_rnn":
            joined = rank + width + rank
            self.wg = nn.Linear(joined, rank)
            self.wc = nn.Linear(joined, rank)
            self.wo = nn.Linear(joined, rank)
            self.rank_to_target = nn.Linear(rank, target_width, bias=False)
        else:
            raise ValueError(f"unknown DSpark correction {correction}")
        self.base_to_target = TargetSpaceProjection(width, target_width)
        self.confidence = nn.Linear(width + rank, 1)

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        base = self.backbone.block_states(
            context_taps, context_positions, anchor_indices, rollout_tokens[:, 0]
        )
        state = torch.zeros(base.shape[0], self.token_rank.embedding_dim, device=base.device, dtype=base.dtype)
        outputs, confidences = [], []
        for depth in range(self.horizon):
            token = self.token_rank(rollout_tokens[:, depth])
            if self.correction == "markov":
                correction = self.rank_to_target(token)
            elif self.correction == "gated":
                correction = torch.sigmoid(self.gate(torch.cat((base[:, depth], token), dim=-1))) \
                    * self.rank_to_target(token)
            else:
                joined = torch.cat((state, token, base[:, depth]), dim=-1)
                gate = torch.sigmoid(self.wg(joined))
                candidate = torch.tanh(self.wc(joined))
                state = gate * state + (1.0 - gate) * candidate
                correction = self.rank_to_target(torch.tanh(self.wo(joined)))
            outputs.append(self.base_to_target(base[:, depth]) + correction)
            confidences.append(self.confidence(torch.cat((base[:, depth], token), dim=-1)).squeeze(-1))
        return DraftOutput(torch.stack(outputs, 1), torch.stack(confidences, 1))


class DBlastDrafter(nn.Module):
    """K-category latent mixture used by the DBlast NLL/AL control matrix."""

    def __init__(self, target_width: int, vocabulary: int, tap_count: int, width: int,
                 horizon: int, categories: int, token_table: Tensor) -> None:
        super().__init__()
        self.horizon = horizon
        self.categories = categories
        self.trunk = BlockDiffusionDrafter(
            target_width, vocabulary, tap_count, width, 5, horizon + 1,
            token_table, flare=False,
        )
        # Source layout: [K*d,d] + [K*d], then category-major reshape.
        self.expander = nn.Linear(width, categories * width, bias=True)
        self.prior = nn.Linear(width, categories, bias=True)
        self.to_target = TargetSpaceProjection(width, target_width)

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        block = self.trunk.block_states(
            context_taps, context_positions, anchor_indices, rollout_tokens[:, 0]
        )
        anchor, positions = block[:, 0], block[:, 1:]
        expanded = self.expander(positions).view(
            positions.shape[0], self.horizon, self.categories, positions.shape[-1]
        )
        category_states = positions.unsqueeze(2) + expanded
        return DraftOutput(
            states=self.to_target(category_states),
            category_logits=self.prior(anchor),
        )


class JetSpecLatentDrafter(nn.Module):
    """One-pass anchor-plus-mask causal latent-depth JetSpec transfer."""

    def __init__(
        self, target_width: int, vocabulary: int, tap_count: int, width: int,
        layers: int, horizon: int, token_table: Tensor,
    ) -> None:
        super().__init__()
        self.block = BlockDiffusionDrafter(
            target_width, vocabulary, tap_count, width, layers, horizon + 1,
            token_table, flare=False, block_causal=True,
        )

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        return self.block(context_taps, context_positions, anchor_indices, rollout_tokens)


class TreeFlashDrafter(nn.Module):
    """DFlash marginals plus the exact immediate-predecessor approximator."""

    def __init__(
        self, target_width: int, vocabulary: int, tap_count: int, width: int,
        layers: int, horizon: int, approximator_width: int, concat_order: str,
        token_table: Tensor,
    ) -> None:
        super().__init__()
        if width != target_width:
            raise ValueError("TreeFlash source equation requires draft and token widths to match")
        if concat_order not in {"predecessor_then_hidden", "hidden_then_predecessor"}:
            raise ValueError("TreeFlash concat order is invalid")
        self.concat_order = concat_order
        self.horizon = horizon
        self.trunk = BlockDiffusionDrafter(
            target_width, vocabulary, tap_count, width, layers, horizon + 1,
            token_table, flare=False,
        )
        self.draft_norm = RMSNorm(width)
        self.predecessor_norm = RMSNorm(width, epsilon=1e-6, affine=False)
        self.approximator_up = nn.Linear(width * 2, approximator_width * 2, bias=True)
        self.approximator_down = nn.Linear(approximator_width, width, bias=False)
        nn.init.zeros_(self.approximator_down.weight)
        self.register_buffer("token_table", token_table, persistent=False)

    def forward(
        self, context_taps: Tensor, context_positions: Tensor,
        anchor_indices: Tensor, rollout_tokens: Tensor,
    ) -> DraftOutput:
        raw = self.trunk.semantic_states(
            context_taps, context_positions, anchor_indices, rollout_tokens[:, 0]
        )
        predecessor = self.predecessor_norm(
            self.token_table[rollout_tokens[:, :self.horizon]].to(raw.dtype)
        )
        normalized = self.draft_norm(raw)
        joined = (torch.cat((predecessor, normalized), dim=-1)
                  if self.concat_order == "predecessor_then_hidden"
                  else torch.cat((normalized, predecessor), dim=-1))
        value, gate = self.approximator_up(joined).chunk(2, dim=-1)
        corrected = raw + self.approximator_down(F.silu(gate) * value)
        return DraftOutput(self.draft_norm(corrected), head_pre_normalized=True)


class SpeculativeStreamingAdapters(nn.Module):
    """Adapter/stream parameters; requires an exact differentiable target executor."""

    def __init__(self, target_width: int, layers: int, streams: int, rank: int = 8,
                 identity_initializer: bool = True) -> None:
        super().__init__()
        self.stream_ids = nn.Parameter(torch.zeros(streams, target_width))
        self.adapters_down = nn.Parameter(torch.empty(layers, streams, target_width, rank))
        self.adapters_up = nn.Parameter(torch.empty(layers, streams, rank, target_width))
        self.early_exit = nn.Linear(target_width, rank, bias=False)
        nn.init.normal_(self.adapters_down, std=target_width ** -0.5)
        if identity_initializer:
            nn.init.zeros_(self.adapters_up)
        else:
            nn.init.normal_(self.adapters_up, std=rank ** -0.5)

    def forward(self, *_: Tensor) -> DraftOutput:
        raise RuntimeError(
            "Arm S cannot be trained from frozen traces alone: bind the exact differentiable "
            "quantized Gemma target-block executor before invoking its adapters"
        )


def make_model(
    implementation: str,
    parameters: dict[str, object],
    target_width: int,
    vocabulary: int,
    token_table: Tensor,
) -> nn.Module:
    taps = len(parameters.get("tap_set", ["final"]))
    width = int(parameters.get("width", 1024))
    horizon = int(parameters.get("horizon", int(parameters.get("block", 4)) - 1))
    if implementation == "direct_recurrent":
        return DirectRecurrentDrafter(
            target_width, vocabulary, taps, width, horizon,
            str(parameters["cell"]), token_table,
        )
    if implementation == "redrafter":
        return ReDrafter(target_width, vocabulary, taps, width, horizon, token_table)
    if implementation == "eagle3":
        if taps != 3:
            raise ValueError("EAGLE-3 requires exactly three frozen pre-block taps")
        return Eagle3Drafter(target_width, vocabulary, width, horizon, token_table)
    if implementation in {"dflash", "dflare"}:
        return BlockDiffusionDrafter(
            target_width, vocabulary, taps, width, int(parameters["layers"]),
            int(parameters["block"]), token_table, implementation == "dflare",
        )
    if implementation == "dspark":
        return DSparkDrafter(
            target_width, vocabulary, taps, width, horizon, int(parameters["rank"]),
            str(parameters["correction"]), token_table,
        )
    if implementation == "dblast":
        return DBlastDrafter(
            target_width, vocabulary, taps, width, horizon,
            int(parameters["categories"]), token_table,
        )
    if implementation == "jetspec_latent":
        return JetSpecLatentDrafter(
            target_width, vocabulary, taps, width, int(parameters["layers"]),
            horizon, token_table,
        )
    if implementation == "treeflash":
        return TreeFlashDrafter(
            target_width, vocabulary, taps, width, int(parameters["layers"]),
            horizon, int(parameters["approximator_width"]),
            str(parameters["concat_order"]), token_table,
        )
    if implementation == "speculative_streaming":
        return SpeculativeStreamingAdapters(
            target_width, int(parameters["top_layers"]), int(parameters["gamma"]),
            identity_initializer=parameters["initializer"] == "identity",
        )
    raise ValueError(f"no faithful model implementation for {implementation}")

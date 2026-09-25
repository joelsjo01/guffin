"""Offline Arm-P equations; production inference remains engine-owned.

The source mechanism is DeepSpec commit
005e03b81cec38b7da6399833d609ee89a2587f2, modeling/dspark/{qwen3,markov_head}.
This is an independent implementation, not a dependency on that engine. Gemma
embedding scaling, compact-width projections, and GRU are explicit transfers;
none is a claim to reproduce the released Qwen/Gemma-4 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SOURCE_COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"


@dataclass(frozen=True)
class DSparkConfig:
    target_width: int
    vocabulary: int
    tap_count: int
    width: int
    layers: int
    horizon: int
    rank: int
    correction: str
    attention_heads: int
    kv_heads: int
    head_dim: int
    intermediate_width: int
    mask_token_id: int
    rms_epsilon: float
    rope_theta: float
    embedding_scale: float
    initializer_std: float

    def __post_init__(self) -> None:
        for name in ("target_width", "vocabulary", "tap_count", "width", "layers",
                     "horizon", "attention_heads", "kv_heads", "head_dim",
                     "intermediate_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"DSpark {name} must be a positive integer")
        if self.attention_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("DSpark GQA heads must divide and RoPE head width must be even")
        if self.horizon > 23 or self.layers > 8 or self.tap_count > 44:
            raise ValueError("DSpark geometry exceeds the declared horizon/layer/tap capacity")
        if self.correction not in {"none", "markov", "gated", "source_rnn", "gru"}:
            raise ValueError("DSpark correction identity is unsupported")
        if type(self.rank) is not int or not 0 <= self.rank <= 512 or (
                self.correction != "none" and self.rank == 0):
            raise ValueError("DSpark correction requires a positive rank")
        if type(self.mask_token_id) is not int or not 0 <= self.mask_token_id < self.vocabulary:
            raise ValueError("DSpark frozen mask token is outside the vocabulary")
        for name in ("rms_epsilon", "rope_theta", "embedding_scale", "initializer_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"DSpark {name} must be finite and positive")
        if self.rope_theta <= 1:
            raise ValueError("DSpark RoPE theta must exceed one")


@dataclass
class DSparkOutput:
    # The head must NOT normalize these states again. Correction is independent
    # learned vocabulary space, added to head logits rather than hidden states.
    base_states: Tensor
    correction_features: Tensor | None
    correction_weight: Tensor | None
    confidence: Tensor
    head_pre_normalized: bool = True


class DSparkRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        # Match the pinned source boundary: normalize in F32, round back to
        # activation precision, then multiply the learned norm weight.
        work = value.float()
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.epsilon)
        return self.weight * normalized.to(value.dtype)


def _rotary(value: Tensor, positions: Tensor, theta: float) -> Tensor:
    width = value.shape[-1]
    frequency = 1.0 / torch.pow(
        torch.tensor(theta, device=value.device, dtype=torch.float32),
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width,
    )
    angles = positions.float().unsqueeze(1).unsqueeze(-1) * frequency
    cosine, sine = angles.cos().to(value.dtype), angles.sin().to(value.dtype)
    first, second = value.chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine,
                      second * cosine + first * sine), dim=-1)


class DSparkLayer(nn.Module):
    def __init__(self, config: DSparkConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = DSparkRMSNorm(config.width, config.rms_epsilon)
        self.query = nn.Linear(config.width, config.attention_heads * config.head_dim, bias=False)
        self.key = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.value = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.query_norm = DSparkRMSNorm(config.head_dim, config.rms_epsilon)
        self.key_norm = DSparkRMSNorm(config.head_dim, config.rms_epsilon)
        self.output = nn.Linear(config.attention_heads * config.head_dim, config.width, bias=False)
        self.ffn_norm = DSparkRMSNorm(config.width, config.rms_epsilon)
        self.gate = nn.Linear(config.width, config.intermediate_width, bias=False)
        self.up = nn.Linear(config.width, config.intermediate_width, bias=False)
        self.down = nn.Linear(config.intermediate_width, config.width, bias=False)

    def forward(self, block: Tensor, context: Tensor, block_positions: Tensor,
                context_positions: Tensor, context_visible: Tensor) -> Tensor:
        cfg = self.config
        batch, horizon, _ = block.shape
        normalized = self.input_norm(block)
        query = self.query_norm(self.query(normalized).view(
            batch, horizon, cfg.attention_heads, cfg.head_dim)).transpose(1, 2)
        query = _rotary(query, block_positions, cfg.rope_theta)
        # Context K/V are projected once per layer, shared by all sampled origins
        # of this sequence. They do not receive the draft input norm a second time.
        keys_context = self.key_norm(self.key(context).view(
            -1, cfg.kv_heads, cfg.head_dim)).permute(1, 0, 2).unsqueeze(0)
        keys_context = _rotary(keys_context, context_positions.unsqueeze(0), cfg.rope_theta)
        keys_block = self.key_norm(self.key(normalized).view(
            batch, horizon, cfg.kv_heads, cfg.head_dim)).transpose(1, 2)
        keys_block = _rotary(keys_block, block_positions, cfg.rope_theta)
        values_context = self.value(context).view(
            -1, cfg.kv_heads, cfg.head_dim).permute(1, 0, 2).unsqueeze(0)
        values_block = self.value(normalized).view(
            batch, horizon, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        keys = torch.cat((keys_context.expand(batch, -1, -1, -1), keys_block), dim=2)
        values = torch.cat((values_context.expand(batch, -1, -1, -1), values_block), dim=2)
        groups = cfg.attention_heads // cfg.kv_heads
        if groups != 1:
            keys = keys.repeat_interleave(groups, dim=1)
            values = values.repeat_interleave(groups, dim=1)
        visible = torch.cat((context_visible, torch.ones(
            batch, horizon, device=block.device, dtype=torch.bool)), dim=1)[:, None, None, :]
        # Offline eager equation with source softmax-in-F32. Every draft slot
        # sees its own complete mask block; no future target feature is visible.
        scores = (query @ keys.transpose(-2, -1)) * (cfg.head_dim ** -0.5)
        probabilities = torch.softmax(scores.masked_fill(~visible, -torch.inf),
                                      dim=-1, dtype=torch.float32).to(query.dtype)
        attended = (probabilities @ values).transpose(1, 2).reshape(batch, horizon, -1)
        block = block + self.output(attended)
        normalized = self.ffn_norm(block)
        return block + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class DSparkDrafter(nn.Module):
    """One parallel semantic pass followed by source-distinct corrections."""

    def __init__(self, config: DSparkConfig, token_table: Tensor) -> None:
        super().__init__()
        if token_table.shape != (config.vocabulary, config.target_width):
            raise ValueError("DSpark frozen token table shape is incompatible")
        if not token_table.is_floating_point() or token_table.requires_grad:
            raise ValueError("DSpark token table must be floating and frozen")
        self.config = config
        self.horizon = config.horizon
        self.fusion = nn.Linear(config.tap_count * config.target_width, config.width, bias=False)
        self.fusion_norm = DSparkRMSNorm(config.width, config.rms_epsilon)
        self.token_projection = (nn.Identity() if config.width == config.target_width else
                                 nn.Linear(config.target_width, config.width, bias=False))
        self.layers = nn.ModuleList(DSparkLayer(config) for _ in range(config.layers))
        self.final_norm = DSparkRMSNorm(config.width, config.rms_epsilon)
        self.to_target = (nn.Identity() if config.width == config.target_width else
                          nn.Linear(config.width, config.target_width, bias=False))
        if config.rank:
            self.token_rank = nn.Embedding(config.vocabulary, config.rank)
        else:
            self.token_rank = None
        self.correction_output = (None if config.correction == "none" else
                                  nn.Linear(config.rank, config.vocabulary, bias=False))
        if config.correction == "gated":
            self.correction_gate = nn.Linear(config.width + config.rank, config.rank)
        elif config.correction == "source_rnn":
            self.correction_joint = nn.Linear(config.width + 2 * config.rank, 3 * config.rank)
        elif config.correction == "gru":
            # Local ablation: this is PyTorch's GRU equation, not DSpark's RNN.
            self.correction_gru = nn.GRUCell(config.width + config.rank, config.rank)
        self.confidence = nn.Linear(config.width + config.rank, 1)
        self.register_buffer("token_table", token_table, persistent=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        # DeepSpec calls post_init inherited from transformers 5.10.2: all
        # Linear/Embedding weights normal(initializer_range), biases zero,
        # learned RMS gains one. In particular W2 is NOT initialized to zero.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, DSparkRMSNorm):
            nn.init.ones_(module.weight)

    def _validate_tokens(self, tokens: Tensor, shape: tuple[int, ...]) -> None:
        if tokens.shape != shape or tokens.dtype != torch.int64:
            raise ValueError("DSpark predecessor token shape/dtype is invalid")
        if tokens.numel() and (bool((tokens < 0).any()) or
                              bool((tokens >= self.config.vocabulary).any())):
            raise ValueError("DSpark predecessor token is outside the vocabulary")

    def prepare_backbone(self, context_taps: Tensor, context_positions: Tensor,
                         anchor_indices: Tensor, anchor_tokens: Tensor) -> Tensor:
        cfg = self.config
        if (context_taps.ndim != 3 or context_taps.shape[1:] != (cfg.tap_count, cfg.target_width)
                or context_taps.shape[0] == 0 or not context_taps.is_floating_point()
                or context_positions.shape != (context_taps.shape[0],)
                or context_positions.dtype != torch.int64 or anchor_indices.ndim != 1
                or anchor_indices.dtype != torch.int64 or anchor_indices.numel() == 0):
            raise ValueError("DSpark feature context/anchor shapes are invalid")
        device = self.fusion.weight.device
        if any(value.device != device for value in
               (context_taps, context_positions, anchor_indices, anchor_tokens, self.token_table)):
            raise ValueError("DSpark inputs and weights must share one device")
        if (bool((anchor_indices < 0).any())
                or bool((anchor_indices >= context_taps.shape[0]).any())):
            raise ValueError("DSpark anchor lies outside feature context")
        if (bool((context_positions < 0).any()) or
                bool((context_positions[1:] != context_positions[:-1] + 1).any())):
            raise ValueError("DSpark full context positions must be contiguous and nonnegative")
        if bool((context_positions > 2**24 - cfg.horizon - 1).any()):
            raise ValueError("DSpark position exceeds exact F32 RoPE integer range")
        if not bool(torch.isfinite(context_taps).all()):
            raise ValueError("DSpark feature context contains nonfinite values")
        batch = anchor_indices.numel()
        self._validate_tokens(anchor_tokens, (batch,))
        # Trace F_t already describes token position t; its selected label is
        # the anchor x_(t+1). Source DSpark positions the anchor there and sees
        # target features strictly before it. Reusing position t duplicates RoPE.
        anchor_positions = context_positions.index_select(0, anchor_indices) + 1
        positions = anchor_positions[:, None] + torch.arange(cfg.horizon, device=device)[None, :]
        visible = context_positions[None, :] < anchor_positions[:, None]
        mask_ids = torch.full((batch, cfg.horizon - 1), cfg.mask_token_id,
                              dtype=torch.int64, device=device)
        ids = torch.cat((anchor_tokens[:, None], mask_ids), dim=1)
        block = self.token_projection(self.token_table[ids].to(self.fusion.weight.dtype)
                                      * cfg.embedding_scale)
        context = self.fusion_norm(self.fusion(context_taps.flatten(-2).to(self.fusion.weight.dtype)))
        for layer in self.layers:
            block = layer(block, context, positions, context_positions, visible)
        return self.final_norm(block)

    def correct(self, base: Tensor, previous_tokens: Tensor) -> DSparkOutput:
        cfg = self.config
        if (base.ndim != 3 or base.shape[0] == 0 or base.shape[1:] != (cfg.horizon, cfg.width)
                or not base.is_floating_point()):
            raise ValueError("DSpark semantic block shape is invalid")
        if base.device != self.fusion.weight.device or previous_tokens.device != base.device:
            raise ValueError("DSpark semantic block, tokens and weights must share one device")
        if not bool(torch.isfinite(base).all()):
            raise ValueError("DSpark semantic block contains nonfinite values")
        self._validate_tokens(previous_tokens, base.shape[:2])
        embeddings = (self.token_rank(previous_tokens) if self.token_rank is not None else
                      base.new_empty((*base.shape[:2], 0)))
        features = torch.cat((base, embeddings.to(base.dtype)), dim=-1)
        confidence = self.confidence(features).squeeze(-1)
        correction = None
        if cfg.correction == "markov":
            correction = embeddings
        elif cfg.correction == "gated":
            correction = torch.sigmoid(self.correction_gate(features)).to(embeddings.dtype) * embeddings
        elif cfg.correction in {"source_rnn", "gru"}:
            state = base.new_zeros((base.shape[0], cfg.rank))
            corrected = []
            for depth in range(cfg.horizon):
                if cfg.correction == "source_rnn":
                    joined = torch.cat((state, embeddings[:, depth], base[:, depth]), dim=-1)
                    gate_raw, candidate_raw, output_raw = self.correction_joint(joined).chunk(3, dim=-1)
                    gate = torch.sigmoid(gate_raw)
                    state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
                    corrected.append(torch.tanh(output_raw))
                else:
                    state = self.correction_gru(features[:, depth], state)
                    corrected.append(state)
            correction = torch.stack(corrected, dim=1)
        return DSparkOutput(
            self.project_base(base), correction,
            None if self.correction_output is None else self.correction_output.weight,
            confidence,
        )

    def project_base(self, base: Tensor) -> Tensor:
        """Project an already normalized semantic block to the frozen head width."""
        return self.to_target(base)

    def forward(self, context_taps: Tensor, context_positions: Tensor,
                anchor_indices: Tensor, rollout_tokens: Tensor) -> DSparkOutput:
        if rollout_tokens.ndim != 2 or rollout_tokens.shape[1] != self.horizon:
            raise ValueError("DSpark rollout shape is invalid")
        base = self.prepare_backbone(context_taps, context_positions, anchor_indices,
                                     rollout_tokens[:, 0])
        return self.correct(base, rollout_tokens)

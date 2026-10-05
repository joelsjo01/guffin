#!/usr/bin/env python3
"""Bounded independent CPU equations for the local Gemma 3 12B Q4_K_M.

Development oracle only. No engine kernels, tensor framework production path,
or entire-model F32 expansion. Quantized matrices are decoded in bounded output
row chunks; tied embeddings decode only requested rows, while the output head
still projects the entire vocabulary. CPU attention uses logical positions,
independent of native ring-cache ownership. The reference uses stored GGUF norm
gamma, F32 activations and accumulation, and explicit F32 or F16 K/V rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from reference_gguf import GGUFError, GGUFFile
from gemma3_reference_golden import SOURCES as DRAFT_SOURCES, gelu_tanh, rms

MODEL_SHA256 = "7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5"
WIDTH, FFN, HEAD_DIM, HEADS, KV_HEADS, LAYERS, VOCAB = 3840, 15360, 256, 16, 8, 48, 262144
EPSILON, WINDOW, CONTEXT = 1e-6, 1024, 131072
SHORT_IDS = [2, 9259, 12074, 108]
SOURCES = dict(DRAFT_SOURCES, **{
    "gguf_blocks": "https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-quants.c",
    "google_config": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py",
    "google_equations": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py",
})


def expected_shapes():
    result = {"token_embd.weight": (WIDTH, VOCAB), "output_norm.weight": (WIDTH,)}
    shapes = {
        "attn_q": (WIDTH, HEADS * HEAD_DIM), "attn_k": (WIDTH, KV_HEADS * HEAD_DIM),
        "attn_v": (WIDTH, KV_HEADS * HEAD_DIM), "attn_output": (HEADS * HEAD_DIM, WIDTH),
        "ffn_gate": (WIDTH, FFN), "ffn_up": (WIDTH, FFN), "ffn_down": (FFN, WIDTH),
        "attn_norm": (WIDTH,), "post_attention_norm": (WIDTH,), "ffn_norm": (WIDTH,),
        "post_ffw_norm": (WIDTH,), "attn_q_norm": (HEAD_DIM,), "attn_k_norm": (HEAD_DIM,),
    }
    for layer in range(LAYERS):
        for name, shape in shapes.items():
            result[f"blk.{layer}.{name}.weight"] = shape
    return result


def decode_k_quant(payload, dimensions, ggml_type):
    """Independent NumPy implementation of pinned GGML block equations.

    Q4_K has unsigned 6-bit per-32 scales and minima; Q6_K has signed per-16
    scales and signed 6-bit values. GGUF order is contiguous input first.
    """
    import numpy as np

    if len(dimensions) != 2 or min(dimensions) <= 0 or dimensions[0] % 256:
        raise ValueError("K-quant matrix dimensions require complete 256-element blocks")
    if ggml_type not in (12, 14):
        raise ValueError("K-quant reference supports Q4_K and Q6_K only")
    block_bytes = 144 if ggml_type == 12 else 210
    count = math.prod(dimensions) // 256
    if len(payload) != count * block_bytes:
        raise ValueError("K-quant payload length differs from matrix dimensions")
    blocks = np.frombuffer(payload, np.uint8).reshape(count, block_bytes)
    values = np.empty((count, 256), dtype=np.float32)
    if ggml_type == 12:
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        if not np.isfinite(d).all() or not np.isfinite(dmin).all():
            raise ValueError("Q4_K contains non-finite block scales")
        packed = blocks[:, 4:16]
        for group in range(8):
            if group < 4:
                scale, minimum = packed[:, group] & 63, packed[:, group + 4] & 63
            else:
                scale = (packed[:, group + 4] & 15) | ((packed[:, group - 4] >> 6) << 4)
                minimum = (packed[:, group + 4] >> 4) | ((packed[:, group] >> 6) << 4)
            quant = blocks[:, 16 + (group // 2) * 32:48 + (group // 2) * 32]
            quant = (quant & 15) if group % 2 == 0 else (quant >> 4)
            values[:, group * 32:(group + 1) * 32] = (d * scale[:, None]) * quant - dmin * minimum[:, None]
    else:
        d = blocks[:, 208:210].copy().view("<f2").astype(np.float32)
        if not np.isfinite(d).all():
            raise ValueError("Q6_K contains non-finite block scales")
        scales = blocks[:, 192:208].view(np.int8)
        for half in range(2):
            low = blocks[:, half * 64:(half + 1) * 64]
            high = blocks[:, 128 + half * 32:160 + half * 32]
            for quarter in range(4):
                nibble = low[:, (quarter % 2) * 32:(quarter % 2 + 1) * 32]
                nibble = (nibble & 15) if quarter < 2 else (nibble >> 4)
                quant = ((nibble | (((high >> (quarter * 2)) & 3) << 4)).astype(np.int16) - 32)
                offset = half * 128 + quarter * 32
                for subgroup in range(2):
                    first = subgroup * 16
                    scale = scales[:, half * 8 + quarter * 2 + subgroup]
                    values[:, offset + first:offset + first + 16] = (d * scale[:, None]) * quant[:, first:first + 16]
    if not np.isfinite(values).all():
        raise ValueError("K-quant decoded matrix contains non-finite values")
    return values.reshape(dimensions[1], dimensions[0])


def tensor_rows(gguf, name, first, count):
    """Return independent writable F32 rows; never alias the GGUF mmap."""
    import numpy as np
    import torch

    entry = gguf.tensors[name]
    if len(entry.dimensions) != 2 or first < 0 or count <= 0 or first + count > entry.dimensions[1]:
        raise GGUFError(f"invalid reference row range: {name}")
    width = entry.dimensions[0]
    if entry.ggml_type not in (12, 14) or width % 256:
        raise GGUFError(f"unsupported reference matrix type/layout: {name}")
    row_bytes = width // 256 * (144 if entry.ggml_type == 12 else 210)
    offset = gguf.data_offset + entry.relative_offset + first * row_bytes
    size = count * row_bytes
    if offset > len(gguf._map) or size > len(gguf._map) - offset:
        raise GGUFError(f"reference tensor exceeds GGUF: {name}")
    payload = memoryview(gguf._map)[offset:offset + size]
    try:
        values = decode_k_quant(payload, (width, count), entry.ggml_type)
    finally:
        payload.release()
    if not np.isfinite(values).all():
        raise GGUFError(f"reference tensor is non-finite: {name}")
    return torch.from_numpy(values)


def vector(gguf, name):
    import torch

    result = torch.from_numpy(gguf.floating_vector(name).copy()).float()
    if not torch.isfinite(result).all():
        raise GGUFError(f"reference norm is non-finite: {name}")
    return result


def project(gguf, name, x, chunk_rows=512):
    import torch

    if chunk_rows <= 0:
        raise ValueError("reference matrix chunk_rows must be positive")
    entry = gguf.tensors[name]
    if len(entry.dimensions) != 2 or x.ndim != 2 or x.shape[1] != entry.dimensions[0]:
        raise ValueError(f"reference matrix input shape differs: {name}")
    result = torch.empty((x.shape[0], entry.dimensions[1]), dtype=torch.float32)
    for first in range(0, entry.dimensions[1], chunk_rows):
        count = min(chunk_rows, entry.dimensions[1] - first)
        result[:, first:first + count] = x @ tensor_rows(gguf, name, first, count).T
    return result


def authenticate(gguf):
    if gguf.sha256() != MODEL_SHA256:
        raise ValueError("Gemma3 target reference checkpoint authentication failed")
    shapes = expected_shapes()
    if set(gguf.tensors) != set(shapes):
        raise ValueError("Gemma3 target reference requires the complete language checkpoint")
    for name, shape in shapes.items():
        entry = gguf.tensors[name]
        allowed = (0,) if len(shape) == 1 else ((14,) if name == "token_embd.weight" else (12, 14))
        if entry.dimensions != shape or entry.ggml_type not in allowed:
            raise ValueError(f"Gemma3 target reference tensor differs: {name}")
    expected = {
        "general.architecture": "gemma3", "gemma3.context_length": CONTEXT,
        "gemma3.embedding_length": WIDTH, "gemma3.block_count": LAYERS,
        "gemma3.feed_forward_length": FFN, "gemma3.attention.head_count": HEADS,
        "gemma3.attention.head_count_kv": KV_HEADS, "gemma3.attention.key_length": HEAD_DIM,
        "gemma3.attention.value_length": HEAD_DIM, "gemma3.attention.sliding_window": WINDOW,
        "gemma3.rope.freq_base": 1e6, "gemma3.rope.scaling.type": "linear", "gemma3.rope.scaling.factor": 8.0,
    }
    for key, value in expected.items():
        if gguf.metadata.get(key) != value:
            raise ValueError(f"Gemma3 target reference metadata differs: {key}")
    if not math.isclose(gguf.metadata.get("gemma3.attention.layer_norm_rms_epsilon", 0), EPSILON, rel_tol=1e-7):
        raise ValueError("Gemma3 target reference RMS epsilon differs")


def rope(x, positions, local):
    import torch

    half = x.shape[-1] // 2
    frequencies = (1e4 if local else 1e6) ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = (positions.float() * (1.0 if local else 0.125))[:, None, None] * frequencies[None, None, :]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * angles.cos() - second * angles.sin(), first * angles.sin() + second * angles.cos()), dim=-1)


def attention(q, k, v, positions, local, query_chunk=64):
    """GQA reference with bounded score tiles, without engine online softmax."""
    import torch

    if query_chunk <= 0 or q.shape[1] % k.shape[1] or k.shape != v.shape:
        raise ValueError("invalid reference attention geometry")
    group = q.shape[1] // k.shape[1]
    result = torch.empty_like(q)
    for first in range(0, len(positions), query_chunk):
        end = min(first + query_chunk, len(positions))
        # All queries in this tile attend only these logical keys. Per-query
        # causal/local masking then excludes extra keys at the tile boundaries.
        key_first = max(0, first - WINDOW + 1) if local else 0
        distance = positions[first:end, None] - positions[None, key_first:end]
        mask = (distance >= 0) & (distance < WINDOW) if local else distance >= 0
        for head in range(q.shape[1]):
            scores = (q[first:end, head] @ k[key_first:end, head // group].float().T) / math.sqrt(q.shape[-1])
            scores.masked_fill_(~mask, -float("inf"))
            result[first:end, head] = torch.softmax(scores, dim=-1) @ v[key_first:end, head // group].float()
    return result


def reference(gguf, token_ids, selected_positions, save_layer, *, cache_dtype="f32", chunk_rows=512):
    import torch

    if cache_dtype not in ("f32", "f16"):
        raise ValueError("reference cache dtype must be f32 or f16")
    if not token_ids or len(token_ids) > CONTEXT or any(not 0 <= token < VOCAB for token in token_ids):
        raise ValueError("reference token IDs/context are invalid")
    if not selected_positions or any(not 0 <= position < len(token_ids) for position in selected_positions):
        raise ValueError("reference selected positions are invalid")
    if chunk_rows <= 0:
        raise ValueError("reference matrix chunk_rows must be positive")
    cache_type = torch.float32 if cache_dtype == "f32" else torch.float16
    x = torch.cat([tensor_rows(gguf, "token_embd.weight", token, 1) for token in token_ids]) * torch.tensor(WIDTH, dtype=torch.float32).sqrt()
    positions = torch.arange(len(token_ids))
    for layer in range(LAYERS):
        prefix = f"blk.{layer}."
        def norm(name):
            return vector(gguf, prefix + name + ".weight")
        def linear(name, value):
            return project(gguf, prefix + name + ".weight", value, chunk_rows)
        local = (layer + 1) % 6 != 0
        u = rms(x, norm("attn_norm"))
        q = rope(rms(linear("attn_q", u).reshape(-1, HEADS, HEAD_DIM), norm("attn_q_norm")), positions, local)
        k = rope(rms(linear("attn_k", u).reshape(-1, KV_HEADS, HEAD_DIM), norm("attn_k_norm")), positions, local).to(cache_type)
        v = linear("attn_v", u).reshape(-1, KV_HEADS, HEAD_DIM).to(cache_type)
        attended = attention(q, k, v, positions, local).reshape(-1, HEADS * HEAD_DIM)
        x = x + rms(linear("attn_output", attended), norm("post_attention_norm"))
        u = rms(x, norm("ffn_norm"))
        activated = gelu_tanh(linear("ffn_gate", u)) * linear("ffn_up", u)
        x = x + rms(linear("ffn_down", activated), norm("post_ffw_norm"))
        if not torch.isfinite(x).all():
            raise ValueError(f"non-finite target reference state at layer {layer}")
        save_layer(layer, k, v, x)
        print(json.dumps({"event": "reference_layer", "layer": layer, "sequence_length": len(token_ids)}), flush=True)
    hidden = rms(x[selected_positions], vector(gguf, "output_norm.weight"))
    logits = project(gguf, "token_embd.weight", hidden, chunk_rows)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite target reference logits")
    return logits, hidden


def generate(model, output, *, token_ids=SHORT_IDS, cache_dtype="f32", chunk_rows=512):
    import numpy as np
    import torch

    if output.exists():
        raise ValueError("refusing to overwrite Gemma3 target reference output")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with GGUFFile(model) as gguf:
        authenticate(gguf)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gemma3-target-reference-", dir=output.parent) as directory:
            staging = Path(directory)
            files = {}
            def save(name, tensor, dtype="<f4"):
                values = tensor.numpy().astype(dtype)
                data = values.tobytes()
                (staging / name).write_bytes(data)
                files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "shape": list(values.shape), "dtype": dtype}
            def save_layer(layer, k, v, hidden):
                # Preserve all logical states for short goldens, with no ring
                # indexing or physical layout copied from the implementation.
                dtype = "<f4" if cache_dtype == "f32" else "<f2"
                save(f"layer-{layer:02d}-key.{cache_dtype}", k, dtype)
                save(f"layer-{layer:02d}-value.{cache_dtype}", v, dtype)
                save(f"layer-{layer:02d}-hidden.f32", hidden)
            with torch.inference_mode():
                logits, hidden = reference(gguf, token_ids, list(range(len(token_ids))), save_layer,
                    cache_dtype=cache_dtype, chunk_rows=chunk_rows)
            for position, values in enumerate(logits):
                save(f"position-{position:06d}-logits.f32", values)
            save("final-hidden.f32", hidden)
            identity = {
                "schema": "gemma3-12b-k-quant-independent-cpu-golden/v1", "model_sha256": MODEL_SHA256,
                "equation_sources": SOURCES, "torch": torch.__version__, "numpy": np.__version__,
                "device": "cpu", "torch_threads": 1, "matrix_chunk_rows": chunk_rows,
                "arithmetic": "F32 activations/accumulation; native Q4_K/Q6_K values; stored GGUF gamma; no softcaps",
                "cache_dtype": cache_dtype, "kv_layout": "logical [position,8,256]; K after head RMS/NeoX RoPE",
                "global_rope_frequency_scale": 0.125, "local_rope_frequency_scale": 1.0,
                "token_ids": token_ids, "files": files,
            }
            (staging / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            staging.rename(output)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-ids", type=lambda text: [int(piece) for piece in text.split(",")], default=SHORT_IDS)
    parser.add_argument("--cache-dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--chunk-rows", type=int, default=512)
    args = parser.parse_args()
    generate(args.model, args.output, token_ids=args.token_ids, cache_dtype=args.cache_dtype, chunk_rows=args.chunk_rows)


if __name__ == "__main__":
    main()

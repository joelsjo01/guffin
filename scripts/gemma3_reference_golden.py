#!/usr/bin/env python3
"""Offline CPU equations for the authenticated Gemma-3-270M-IT Q8_0 GGUF.

This is a development oracle, never an inference runtime. It uses the complete
checkpoint, full vocabulary, ordinary CPU PyTorch operations and an unbounded
logical K/V history, independently of the engine's Metal/ring-cache code.

Equation sources are pinned below. GGUF conversion folds 1 + HF's norm weight
into the stored gamma; adding one again is wrong. GGUF Q/K rows retain NEOX
(split-half) RoPE order. Execution matches the GGUF/F32 contract, not original
BF16 Transformers execution: notably sqrt(640) is F32, not BF16-rounded 25.25.
K/V remain F32 after key normalization/rotation. The optional F16-cache mode
reproduces the rejected diagnostic contract: correct CPU reduction variants
crossed half-rounding thresholds and exceeded parity bounds in later layers.
The 512-position local mask includes self, not 512 past positions plus self.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.speculation.gguf import GGUFError, GGUFFile

MODEL_SHA256 = "0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3"
LLAMA_COMMIT = "d48a56effbba99944a677938d4beb63a0065ecdf"
SOURCES = {
    "gguf_conversion": f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_COMMIT}/conversion/gemma.py",
    "gguf_execution": f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_COMMIT}/src/models/gemma3.cpp",
    "gguf_rope_type": f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_COMMIT}/src/llama-model.cpp",
    "hf_equations": "https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gemma3/modeling_gemma3.py",
    "hf_mask": "https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/masking_utils.py",
}
WIDTH, FFN, HEAD_DIM, HEADS, LAYERS, VOCAB = 640, 2048, 256, 4, 18, 262144
EPSILON, WINDOW, CONTEXT = 1e-6, 512, 32768
SHORT_IDS = [2, 9259, 12074, 108]
WRAP_IDS = [2] + [3 + (i * 499) % 255997 for i in range(1, 536)]


def expected_shapes():
    """GGUF order is innermost-first, unlike the returned CPU row matrices."""
    result = {"token_embd.weight": (WIDTH, VOCAB), "output_norm.weight": (WIDTH,)}
    layer_shapes = {
        "attn_q": (WIDTH, HEADS * HEAD_DIM), "attn_k": (WIDTH, HEAD_DIM),
        "attn_v": (WIDTH, HEAD_DIM), "attn_output": (HEADS * HEAD_DIM, WIDTH),
        "ffn_gate": (WIDTH, FFN), "ffn_up": (WIDTH, FFN), "ffn_down": (FFN, WIDTH),
        "attn_norm": (WIDTH,), "post_attention_norm": (WIDTH,),
        "ffn_norm": (WIDTH,), "post_ffw_norm": (WIDTH,),
        "attn_q_norm": (HEAD_DIM,), "attn_k_norm": (HEAD_DIM,),
    }
    for layer in range(LAYERS):
        for name, shape in layer_shapes.items():
            result[f"blk.{layer}.{name}.weight"] = shape
    return result


def decode_q8(payload, dimensions):
    import numpy as np

    if len(dimensions) != 2 or dimensions[0] % 32 or min(dimensions) <= 0:
        raise ValueError("Q8_0 matrix dimensions must contain complete 32-element blocks")
    count = math.prod(dimensions) // 32
    if len(payload) != count * 34:
        raise ValueError("Q8_0 payload length differs from its matrix dimensions")
    blocks = np.frombuffer(payload, dtype=np.dtype([("scale", "<f2"), ("values", "i1", (32,))]))
    scales = blocks["scale"].astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("Q8_0 matrix contains non-finite scales")
    values = blocks["values"].astype(np.float32) * scales[:, None]
    return values.reshape(dimensions[1], dimensions[0])


def load_tensor(gguf, name):
    import numpy as np
    import torch

    entry = gguf.tensors[name]
    if entry.ggml_type == 8:
        size = math.prod(entry.dimensions) // 32 * 34
    elif entry.ggml_type == 0 and len(entry.dimensions) == 1:
        size = math.prod(entry.dimensions) * 4
    else:
        raise GGUFError(f"unsupported golden tensor type/shape: {name}")
    offset = gguf.data_offset + entry.relative_offset
    if offset > len(gguf._map) or size > len(gguf._map) - offset:
        raise GGUFError(f"golden tensor exceeds GGUF: {name}")
    # Copy/dequantize into independent writable arrays. No torch tensor aliases
    # a read-only mapping, and closing the GGUF cannot invalidate tensor storage.
    payload = memoryview(gguf._map)[offset:offset + size]
    try:
        values = decode_q8(payload, entry.dimensions) if entry.ggml_type == 8 else np.frombuffer(payload, "<f4").copy()
    finally:
        payload.release()
    if not np.isfinite(values).all():
        raise GGUFError(f"non-finite golden tensor: {name}")
    return torch.from_numpy(values)


def authenticate(gguf):
    if gguf.sha256() != MODEL_SHA256:
        raise ValueError("Gemma3 golden checkpoint authentication failed")
    shapes = expected_shapes()
    if set(gguf.tensors) != set(shapes):
        raise ValueError("Gemma3 golden tensor set differs from the complete checkpoint")
    for name, shape in shapes.items():
        entry = gguf.tensors[name]
        if entry.dimensions != shape or entry.ggml_type != (8 if len(shape) == 2 else 0):
            raise ValueError(f"Gemma3 golden tensor contract differs: {name}")
    expected = {
        "general.architecture": "gemma3", "gemma3.context_length": CONTEXT,
        "gemma3.embedding_length": WIDTH, "gemma3.block_count": LAYERS,
        "gemma3.feed_forward_length": FFN, "gemma3.attention.head_count": HEADS,
        "gemma3.attention.head_count_kv": 1, "gemma3.attention.key_length": HEAD_DIM,
        "gemma3.attention.value_length": HEAD_DIM, "gemma3.attention.sliding_window": WINDOW,
        "gemma3.rope.freq_base": 1e6,
    }
    for key, value in expected.items():
        if gguf.metadata.get(key) != value:
            raise ValueError(f"Gemma3 golden metadata differs: {key}")
    if not math.isclose(gguf.metadata.get("gemma3.attention.layer_norm_rms_epsilon", 0), EPSILON, rel_tol=1e-7):
        raise ValueError("Gemma3 golden RMS epsilon differs")


def rms(x, gamma):
    import torch

    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPSILON) * gamma


def rope(x, positions, base):
    import torch

    half = x.shape[-1] // 2
    frequency = base ** (-torch.arange(half, dtype=torch.float32) / half)
    angle = positions.float()[:, None, None] * frequency[None, None, :]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * angle.cos() - second * angle.sin(), first * angle.sin() + second * angle.cos()), dim=-1)


def causal_mask(positions, local):
    distance = positions[:, None] - positions[None, :]
    return (distance >= 0) & (distance < WINDOW) if local else distance >= 0


def gelu_tanh(x):
    import torch

    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


def cache_values(values, cache_dtype):
    import torch

    if cache_dtype not in ("f32", "f16"):
        raise ValueError("reference cache dtype must be f32 or diagnostic f16")
    return values.to(torch.float32 if cache_dtype == "f32" else torch.float16)


def reference(gguf, token_ids, selected_positions, save_cache, *, cache_dtype="f32"):
    """Layerwise causal CPU equations; no engine cache or inference calls.

    The complete logical K/V arrays make masking independent of ring indexing.
    Only selected states need a final full-vocabulary projection; this does not
    remove any transformer work or truncate any projected vocabulary.
    """
    import torch

    if cache_dtype not in ("f32", "f16"):
        raise ValueError("reference cache dtype must be f32 or diagnostic f16")
    if not token_ids or len(token_ids) > CONTEXT or any(not 0 <= t < VOCAB for t in token_ids):
        raise ValueError("reference token IDs/context are invalid")
    if not selected_positions or any(not 0 <= p < len(token_ids) for p in selected_positions):
        raise ValueError("reference selected positions are invalid")
    embeddings = load_tensor(gguf, "token_embd.weight")
    positions = torch.arange(len(token_ids), dtype=torch.int64)
    x = embeddings[token_ids] * torch.tensor(WIDTH, dtype=torch.float32).sqrt()
    for layer in range(LAYERS):
        def weight(name):
            return load_tensor(gguf, f"blk.{layer}.{name}.weight")

        local = (layer + 1) % 6 != 0
        normalized = rms(x, weight("attn_norm"))
        q = rms((normalized @ weight("attn_q").T).reshape(-1, HEADS, HEAD_DIM), weight("attn_q_norm"))
        k = rms((normalized @ weight("attn_k").T).reshape(-1, 1, HEAD_DIM), weight("attn_k_norm"))
        v = cache_values(normalized @ weight("attn_v").T, cache_dtype)
        q = rope(q, positions, 1e4 if local else 1e6)
        k = cache_values(rope(k, positions, 1e4 if local else 1e6).squeeze(1), cache_dtype)
        save_cache(layer, k, v)
        scores = (q.transpose(0, 1) @ k.float().T) / math.sqrt(HEAD_DIM)
        scores.masked_fill_(~causal_mask(positions, local)[None, :, :], -float("inf"))
        attention = (torch.softmax(scores, dim=-1) @ v.float()).transpose(0, 1).reshape(-1, HEADS * HEAD_DIM)
        x = x + rms(attention @ weight("attn_output").T, weight("post_attention_norm"))
        normalized = rms(x, weight("ffn_norm"))
        activated = gelu_tanh(normalized @ weight("ffn_gate").T) * (normalized @ weight("ffn_up").T)
        x = x + rms(activated @ weight("ffn_down").T, weight("post_ffw_norm"))
        if not torch.isfinite(x).all():
            raise ValueError(f"non-finite CPU reference state at layer {layer}")
        print(json.dumps({"event": "reference_layer", "layer": layer, "sequence_length": len(token_ids)}), flush=True)
    result = rms(x[selected_positions], load_tensor(gguf, "output_norm.weight")) @ embeddings.T
    if result.shape != (len(selected_positions), VOCAB) or not torch.isfinite(result).all():
        raise ValueError("invalid complete-vocabulary reference logits")
    return result


def generate(model, output, cache_dtype="f32"):
    import numpy as np
    import torch

    if output.exists():
        raise ValueError("refusing to overwrite Gemma3 reference output")
    if cache_dtype not in ("f32", "f16"):
        raise ValueError("reference cache dtype must be f32 or diagnostic f16")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with GGUFFile(model) as gguf:
        authenticate(gguf)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gemma3-reference-", dir=output.parent) as directory:
            staging = Path(directory)
            files = {}

            def save(name, tensor, dtype):
                values = tensor.numpy().astype(dtype)
                data = values.tobytes()
                (staging / name).write_bytes(data)
                files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "shape": list(values.shape), "dtype": dtype}

            cases = (
                (SHORT_IDS, [("prefill", 2), ("decode", 3)], "decode"),
                (WRAP_IDS, [("wrap-before", 511), ("wrap-crossing", 512), ("wrap-after", 535)], "wrap-after"),
            )
            for tokens, checkpoints, cache_label in cases:
                def save_cache(layer, k, v):
                    first = max(0, len(tokens) - WINDOW) if (layer + 1) % 6 else 0
                    for name, values in (("key", k), ("value", v)):
                        save(f"{cache_label}-layer-{layer:02d}-{name}.{cache_dtype}", values[first:], "<f4" if cache_dtype == "f32" else "<f2")

                with torch.inference_mode():
                    logits = reference(gguf, tokens, [p for _, p in checkpoints], save_cache, cache_dtype=cache_dtype)
                for (label, _), values in zip(checkpoints, logits):
                    save(f"{label}-logits.f32", values, "<f4")
            identity = {
                "schema": f"gemma3-q8-{cache_dtype}-kv-independent-cpu-golden/v2", "model_sha256": MODEL_SHA256,
                "equation_sources": SOURCES, "torch": torch.__version__, "numpy": np.__version__,
                "device": "cpu", "torch_threads": 1, "arithmetic": "F32; Q8_0 dequantized into F32; stored GGUF gamma",
                "cache_dtype": cache_dtype, "diagnostic_only": cache_dtype != "f32",
                "kv": f"{cache_dtype.upper()} after K normalization/NEOX rotation, V projection; logical [position,256] order",
                "original_bf16_source_execution": False, "short_ids": SHORT_IDS, "wrap_ids": WRAP_IDS,
                "checkpoints": {label: position for _, points, _ in cases for label, position in points},
                "cache_first_positions": {"decode": {"local": 0, "global": 0}, "wrap-after": {"local": 24, "global": 0}},
                "files": files,
            }
            (staging / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            if output.exists():
                raise ValueError("refusing to overwrite Gemma3 reference output")
            staging.rename(output)
    print(json.dumps({"event": "reference_complete", "output": str(output), "file_count": len(files)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dtype", choices=("f32", "f16"), default="f32",
                        help="F32 production cache; F16 reproduces the rejected precision diagnostic only")
    args = parser.parse_args()
    try:
        generate(args.model, args.output, args.cache_dtype)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

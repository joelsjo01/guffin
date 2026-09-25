#!/usr/bin/env python3
"""Convert the pinned, complete double7/vicuna-68m checkpoint to native F16 GGUF.

This offline tool uses NumPy only for IEEE round-to-nearest F32 -> F16. It does
not execute inference. Source hashes bind tokenizer normalization and the known
model/tokenizer BOS discrepancy; runtime context conversion never inserts BOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile


SOURCE_REVISION = "f35c45e548302e8edd0a31db7490b42ea2ddd109"
SOURCE_HASHES = {
    "config.json": "5494c8542a876014e2df85a7f8a18d798848cad66d7eb5c5930b0b59063afe43",
    "model.safetensors": "049aa4dc91bd8bb0e3009b8408b59512d8e8eb6aced19346570cc82fa657113d",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "tokenizer_config.json": "09a8aec7f7914be90099d034878e26cb612d8ddf8a6061af75a62ed85c7553a5",
    "special_tokens_map.json": "2ea88203b955648951a454e73a73967ff77ba4e79ae8ceb4400d4ab2cb040777",
}


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {"model.embed_tokens.weight": (32000, 768),
              "lm_head.weight": (32000, 768), "model.norm.weight": (768,)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + name + ".weight"] = (768,)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[prefix + "self_attn." + name + ".weight"] = (768, 768)
        for name in ("gate_proj", "up_proj"):
            shapes[prefix + "mlp." + name + ".weight"] = (3072, 768)
        shapes[prefix + "mlp.down_proj.weight"] = (768, 3072)
    return shapes


def protobuf_fields(data: bytes):
    """Read the bounded primitive subset used by SentencePiece ModelProto."""
    offset = 0

    def varint():
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset == len(data):
                raise ValueError("truncated SentencePiece varint")
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("oversized SentencePiece varint")

    while offset < len(data):
        key = varint()
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError("invalid SentencePiece field")
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            length = varint() if wire == 2 else (8 if wire == 1 else 4)
            if length > len(data) - offset:
                raise ValueError("truncated SentencePiece field")
            value = data[offset:offset + length]
            offset += length
        else:
            raise ValueError("unsupported SentencePiece wire type")
        yield field, wire, value


def sentencepiece_metadata(data: bytes):
    pieces, scores, types = [], [], []
    trainer = normalizer = None
    for field, wire, value in protobuf_fields(data):
        if wire != 2:
            raise ValueError("invalid SentencePiece top-level field")
        if field == 1:
            values = {f: v for f, _, v in protobuf_fields(value)}
            pieces.append(values[1].decode("utf-8"))
            scores.append(struct.unpack("<f", values[2])[0])
            types.append(values.get(3, 1))
        elif field == 2:
            trainer = {f: v for f, _, v in protobuf_fields(value)}
        elif field == 3:
            normalizer = {f: v for f, _, v in protobuf_fields(value)}
        else:
            raise ValueError("unsupported SentencePiece extension")
    if (trainer is None or normalizer is None or trainer.get(3, 1) != 2
            or trainer.get(35, 0) != 1 or trainer.get(24, 0) != 0
            or normalizer.get(1) != b"identity" or normalizer.get(2, b"")
            or normalizer.get(3, 1) != 1 or normalizer.get(4, 1) != 0
            or normalizer.get(5, 1) != 1):
        raise ValueError("unsupported SentencePiece normalization/BPE profile")
    if (len(pieces) != 32000 or len(set(pieces)) != 32000
            or pieces[:3] != ["<unk>", "<s>", "</s>"]
            or types[:3] != [2, 3, 3] or not all(math.isfinite(x) for x in scores)
            or any(kind not in (1, 2, 3, 6) for kind in types)):
        raise ValueError("invalid Vicuna vocabulary")
    return pieces, scores, types


def string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_value(kind: int, value) -> bytes:
    if kind == 8:
        return string(value)
    if kind == 9:
        element, values = value
        return struct.pack("<IQ", element, len(values)) + b"".join(
            metadata_value(element, item) for item in values)
    return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], value)


def convert(source: Path, output: Path) -> str:
    if output.exists():
        raise ValueError("refusing to overwrite an existing draft artifact")
    sources = {}
    for name, digest in SOURCE_HASHES.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"source authentication failed: {name}")
        sources[name] = data
    import numpy as np

    pieces, scores, types = sentencepiece_metadata(sources["tokenizer.model"])
    checkpoint = sources["model.safetensors"]
    if len(checkpoint) < 8:
        raise ValueError("truncated safetensors")
    header_length, = struct.unpack_from("<Q", checkpoint)
    header = json.loads(checkpoint[8:8 + header_length])
    payload = memoryview(checkpoint)[8 + header_length:]
    shapes = expected_shapes()
    if set(header) - {"__metadata__"} != set(shapes):
        raise ValueError("missing or extra checkpoint tensors")
    tensors = []
    intervals = []
    for name, shape in sorted(shapes.items()):
        entry = header[name]
        begin, end = entry["data_offsets"]
        if (entry["dtype"] != "F32" or tuple(entry["shape"]) != shape
                or begin < 0 or end > len(payload)
                or end - begin != math.prod(shape) * 4):
            raise ValueError(f"invalid tensor layout: {name}")
        values = np.frombuffer(payload[begin:end], dtype="<f4")
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 65504:
            raise ValueError(f"non-finite/unrepresentable tensor: {name}")
        tensors.append((name, shape, values.astype("<f2").tobytes()))
        intervals.append((begin, end))
    end = 0
    for begin, finish in sorted(intervals):
        if begin != end:
            raise ValueError("overlapping or unclaimed safetensors payload")
        end = finish
    if end != len(payload):
        raise ValueError("unclaimed safetensors trailer")

    metadata = {
        "general.architecture": (8, "llama"),
        "general.name": (8, "double7/vicuna-68m"),
        "general.alignment": (4, 32),
        "standalone_draft.format": (8, "vicuna68m-f16-tli-v1"),
        "standalone_draft.source_revision": (8, SOURCE_REVISION),
        "llama.block_count": (4, 2), "llama.embedding_length": (4, 768),
        "llama.feed_forward_length": (4, 3072), "llama.context_length": (4, 2048),
        "llama.attention.head_count": (4, 12), "llama.attention.head_count_kv": (4, 12),
        "llama.attention.layer_norm_rms_epsilon": (6, 1e-6),
        "llama.rope.freq_base": (6, 10000.0),
        "tokenizer.ggml.model": (8, "llama"),
        "tokenizer.ggml.tokens": (9, (8, pieces)),
        "tokenizer.ggml.scores": (9, (6, scores)),
        "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.bos_token_id": (4, 1), "tokenizer.ggml.eos_token_id": (4, 2),
        "tokenizer.ggml.unknown_token_id": (4, 0),
        "tokenizer.ggml.add_bos_token": (7, True),
        "tokenizer.ggml.add_eos_token": (7, False),
        "tokenizer.ggml.add_space_prefix": (7, True),
        "tokenizer.ggml.normalizer": (8, "identity-no-collapse-byte-fallback"),
    }
    for name, digest in SOURCE_HASHES.items():
        metadata["standalone_draft.source_sha256." + name] = (8, digest)
    for name, _, data in tensors:
        metadata["standalone_draft.tensor_sha256." + name] = (8, hashlib.sha256(data).hexdigest())
    prefix = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, (kind, value) in sorted(metadata.items()):
        prefix += string(key) + struct.pack("<I", kind) + metadata_value(kind, value)
    offset = 0
    for name, shape, data in tensors:
        prefix += string(name) + struct.pack("<I", len(shape))
        prefix += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        prefix += struct.pack("<IQ", 1, offset)
        offset += len(data)
        offset = (offset + 31) & ~31
    prefix += bytes((-len(prefix)) % 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(prefix)
            digest.update(prefix)
            for _, _, data in tensors:
                padding = bytes((-len(data)) % 32)
                stream.write(data)
                stream.write(padding)
                digest.update(data)
                digest.update(padding)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication cannot replace another process's output.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        digest = convert(args.source, args.output)
    except (OSError, ValueError, KeyError, OverflowError) as error:
        parser.exit(1, f"Vicuna conversion: {error}\n")
    print(json.dumps({"artifact": str(args.output), "sha256": digest,
                      "source_revision": SOURCE_REVISION, "parameter_count": 68030208}))


if __name__ == "__main__":
    main()

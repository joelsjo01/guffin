"""Deterministic schema-3 weights export; never part of inference execution.

F32 master weights are retained exactly. The numeric contract explicitly
reproduces the trained BF16 recurrence and F32/F16 tied Gemma head. This is a
local direct-recurrent transfer, not a ReDrafter/EAGLE source reproduction.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile

from training.speculation.gguf import GGUFFile
from training.speculation.registry import load_registry
from training.speculation.run_contract import load_run_contract
from training.speculation.trace import TraceFile


TENSORS = (
    "fusion.projection.weight", "fusion.norm.weight", "token_projection.weight",
    "cell.gate.weight", "cell.gate.bias", "cell.candidate.weight",
    "cell.candidate.bias", "to_target.projection.weight", "confidence.weight",
    "confidence.bias",
)
PROVENANCE = (
    "checkpoint_sha256", "training_receipt", "registry_sha256", "run_contract_sha256",
    "training_code_sha256", "trace_corpus_sha256", "split_receipt_sha256",
    "environment", "feature_contract_sha256", "tokenizer_sha256",
    "chat_template_sha256", "exporter",
)


def digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def checked_json(path: Path, maximum: int = 32 << 20) -> tuple[bytes, dict]:
    with path.open("rb") as file:
        raw = file.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"oversized JSON: {path}")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result
    result = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return raw, result


def pack_weights(state, width: int, target: int) -> tuple[bytes, bytes]:
    import torch
    shapes = ((width,target),(width,),(width,target),(width,2*width),(width,),
              (width,2*width),(width,),(target,width),(1,width),(1,))
    if set(state) != set(TENSORS):
        raise ValueError("checkpoint tensor names do not match the complete recurrent mechanism")
    payload = bytearray()
    directory = bytearray()
    for name, shape in zip(TENSORS, shapes):
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid F32 weight shape/value: {name}")
        raw = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        rows, columns = shape if len(shape) == 2 else (1, shape[0])
        directory.extend(struct.pack("<IIQQ", rows, columns, len(payload), len(raw)))
        directory.extend(digest(raw))
        payload.extend(raw)
    return bytes(directory), bytes(payload)


def publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # link() is an atomic no-replace publication, unlike an exists/rename pair.
    handle, temporary = tempfile.mkstemp(prefix=".recurrent-export-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def export(checkpoint: Path, receipt: Path, registry_path: Path,
           contract_path: Path, model_path: Path, trace_path: Path, output: Path,
           split_receipt_path: Path) -> dict:
    import torch
    if output.exists():
        raise ValueError("refusing to overwrite exported weights")
    receipt_bytes, report = checked_json(receipt)
    registry = load_registry(registry_path)
    contract = load_run_contract(contract_path, registry)
    from training.speculation.manifest import load_split_receipt
    split_receipt = load_split_receipt(split_receipt_path, registry.digest)
    if report.get("schema") != "local-gguf-engine/training-result/v2":
        raise ValueError("a completed training receipt is required")
    if (report.get("registry_sha256") != registry.digest
            or report.get("run_contract_sha256") != contract.digest
            or report.get("proposal_policy") != "common_draw"):
        raise ValueError("training receipt registry/run/sampler mismatch")
    if split_receipt["manifests"]["training_development"]["sha256"] != report["source_manifest_sha256"]:
        raise ValueError("split receipt differs from the training manifest")
    cells = [c for c in registry.cells() if c.identifier == report.get("cell_id")]
    if len(cells) != 1:
        raise ValueError("training cell is not registered")
    cell = cells[0]
    run_cell = contract.cell(cell.identifier)
    if (run_cell.proposal_policy != report["proposal_policy"]
            or report.get("dpace_alpha") != run_cell.dpace_alpha):
        raise ValueError("training cell differs from its fixed run contract")
    parameters = dict(cell.parameters)
    if (report.get("implementation") != "direct_recurrent"
            or cell.implementation != "direct_recurrent"
            or report.get("parameters") != parameters
            or report.get("training_seed") != cell.seed
            or parameters.get("cell") != "gated_residual"
            or parameters.get("context_mode") != "root_only"
            or parameters.get("tap_set") != ["final"]
            or parameters.get("selector") != "full_vocabulary"
            or parameters.get("selected_head_precision") != "f16"):
        raise ValueError("exporter only implements the registered final-tap gated recurrent full-V cell")
    from training.speculation.train import _completion_cursor_fields
    optimizer = registry.raw["default_transfer_optimizer"]
    _completion_cursor_fields(report["terminal_epoch"], report["terminal_origins_in_epoch"],
                             report["update_step"], report["stop_requested"], report["scale"],
                             optimizer["gradient_accumulation"], optimizer["maximum_epochs"])
    with checkpoint.open("rb") as file:
        checkpoint_bytes = file.read((512 << 20) + 1)
    if len(checkpoint_bytes) > 512 << 20 or digest(checkpoint_bytes).hex() != report["checkpoint_sha256"]:
        raise ValueError("checkpoint digest/size differs from completed receipt")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if state.get("schema") != "local-gguf-engine/training-checkpoint/v2":
        raise ValueError("expected final training checkpoint, not progress state")
    for key in ("cell_id", "model_sha256", "registry_sha256", "run_contract_sha256",
                "training_code_sha256", "source_manifest_sha256", "trace_corpus_sha256",
                "training_membership_sha256", "validation_membership_sha256", "scale",
                "best_checkpoint_file", "best_checkpoint_sha256", "best_checkpoint_selection_value",
                "environment", "terminal_epoch", "terminal_origins_in_epoch", "update_step"):
        if state.get(key) != report.get(key):
            raise ValueError(f"checkpoint/receipt identity differs: {key}")
    with TraceFile(trace_path) as trace, GGUFFile(model_path) as model:
        identity = trace.identity
        if (model.sha256() != report["model_sha256"]
                or identity.model_sha256.hex() != report["model_sha256"]
                or identity.registry_sha256.hex() != registry.digest
                or identity.source_manifest_sha256.hex() != report["source_manifest_sha256"]):
            raise ValueError("GGUF/trace/training identity mismatch")
        meta = model.metadata
        if (meta.get("general.architecture") != "gemma2"
                or "output.weight" in model.tensors
                or model.tensors["token_embd.weight"].ggml_type != 1):
            raise ValueError("export requires the tied F16 Gemma head")
        target, vocab = model.tensors["token_embd.weight"].dimensions
        if target != report["target_embedding_width"] or vocab != report["target_vocabulary"]:
            raise ValueError("GGUF head dimensions disagree with training")
        fields = [meta["gemma2."+key] for key in (
            "context_length", "embedding_length", "block_count", "feed_forward_length",
            "attention.head_count", "attention.head_count_kv", "attention.key_length",
            "attention.value_length", "attention.sliding_window")]+[vocab]
        floats = [meta["gemma2."+key] for key in (
            "attention.layer_norm_rms_epsilon", "attn_logit_softcapping", "final_logit_softcapping")]
        target_header = struct.pack("<IQI",1,identity.model_bytes,identity.source_tensor_count)
        target_header += identity.model_sha256+identity.canonical_metadata_sha256+identity.source_tensors_sha256
        target_header += struct.pack("<IIQ",1,identity.resident_allocation_count,identity.resident_bytes)
        target_header += identity.resident_sha256+struct.pack("<10I3fII",*fields,*floats,1,2)
        provenance = dict(report)
        provenance.update(training_receipt=digest(receipt_bytes).hex(),
                          environment=digest(json.dumps(report["environment"],sort_keys=True,separators=(",",":")).encode()).hex(),
                          feature_contract_sha256=identity.feature_contract_sha256.hex(),
                          tokenizer_sha256=identity.tokenizer_sha256.hex(),
                          chat_template_sha256=identity.chat_template_sha256.hex(),
                          exporter=digest(Path(__file__).read_bytes()).hex())
        provenance["split_receipt_sha256"] = split_receipt["receipt_sha256"]
        provenance_bytes = b"".join(bytes.fromhex(provenance[k]) for k in PROVENANCE)
        if len(provenance_bytes) != 12*32 or any(provenance_bytes[i:i+32] == bytes(32) for i in range(0,384,32)):
            raise ValueError("missing artifact provenance")
        sampler = registry.raw["target_generation_sampler"]
        if sampler["repetition_penalty"] != 1.0:
            raise ValueError("recurrent sampler history implementation is unavailable")
        config = struct.pack("<6I3f",1,1,1,parameters["width"],parameters["horizon"],
                             sampler["top_k"],sampler["temperature"],sampler["top_p"],sampler["min_p"])
        directory, payload = pack_weights(state["model"], parameters["width"], target)
        suffix = target_header+config+provenance_bytes+directory
        header = struct.pack("<8sIIQQ32sQ",b"LGSPEC3\0",3,2,72+len(suffix),len(payload),
                             bytes(32),int.from_bytes(digest(checkpoint_bytes)[:8],"little"))+suffix
        encoded = bytearray(header+payload)
        encoded[32:64] = digest(encoded)
        publish(output, encoded)
        return {"artifact":str(output),"sha256":digest(encoded).hex(),"cell_id":cell.identifier,
                "role":"weights_only","weight_bytes":len(payload),"numeric_contract":"bf16_recurrence_f32_head/v1"}

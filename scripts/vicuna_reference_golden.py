#!/usr/bin/env python3
"""Offline pinned Transformers golden for the native Vicuna Metal graph.

Run with Transformers 4.50.0, PyTorch and NumPy in development tooling only.
Weights are rounded to F16 then executed in F32; cached K/V are explicitly
rounded to F16. This matches the native precision contract without using any
engine equations as the reference. No network access or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from convert_vicuna_draft import SOURCE_HASHES, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampler-only", action="store_true", help="generate full-vocabulary pinned proposal-transform reference")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite reference output")
    for name, expected in SOURCE_HASHES.items():
        with (args.source / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                parser.error(f"source authentication failed: {name}")

    import numpy as np
    import torch
    import transformers
    from transformers import LlamaForCausalLM
    from transformers.cache_utils import DynamicCache

    if transformers.__version__ != "4.50.0":
        parser.error("reference must use pinned Transformers 4.50.0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if args.sampler_only:
        from transformers.generation.logits_process import (
            LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
            SuppressTokensLogitsProcessor, TemperatureLogitsWarper,
            TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper,
        )
        # CPU torch computes and exports the exact input, so native tests do not
        # accidentally compare different host sin implementations/rounding.
        index = torch.arange(32000, dtype=torch.float32)
        raw = (torch.sin(index * .173) * 7 + (index % 31) * .03125).unsqueeze(0)
        history = torch.tensor([[1, 5, 5, 123, 31998]], dtype=torch.long)
        suppressed = sorted(set(range(0, 32000, 7)) | {31999, 255, 256})
        processors = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(1.1),
            SuppressTokensLogitsProcessor(suppressed),
            TemperatureLogitsWarper(.8), TopKLogitsWarper(40),
            TopPLogitsWarper(.95), MinPLogitsWarper(.05),
        ])
        filtered = processors(history, raw.clone())
        args.output.mkdir(parents=True)
        outputs = {}
        for name, values in (("raw-input", raw), ("filtered-logits", filtered)):
            data = values[0].numpy().astype("<f4").tobytes()
            (args.output / f"{name}.f32").write_bytes(data)
            outputs[name] = {"sha256": hashlib.sha256(data).hexdigest()}
        identity = {
            "transformers": transformers.__version__, "torch": torch.__version__,
            "source_revision": SOURCE_REVISION, "history": history[0].tolist(),
            "temperature": .8, "top_k": 40, "top_p": .95, "min_p": .05,
            "repetition_penalty": 1.1, "suppressed_ids": suppressed,
            "processor_order": [type(processor).__name__ for processor in processors],
            "retained_count": int(torch.isfinite(filtered).sum()), "outputs": outputs,
            "tie_order": "Transformers CPU torch.sort default unstable; no cross-device tied-boundary claim",
        }
        (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        print(json.dumps({key: value for key, value in identity.items() if key != "suppressed_ids"}, sort_keys=True))
        return

    class F16ResidentCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return super().update(key_states.half().float(), value_states.half().float(), layer_idx, cache_kwargs)

    model = LlamaForCausalLM.from_pretrained(
        args.source, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().half().float()
    cache = F16ResidentCache()
    with torch.inference_mode():
        prefill = model(input_ids=torch.tensor([[1, 15043, 29892]]), past_key_values=cache, use_cache=True)
        decoded = model(input_ids=torch.tensor([[3186]]), past_key_values=cache, use_cache=True)
    args.output.mkdir(parents=True)
    outputs = {}
    for name, result in (("prefill", prefill), ("decode", decoded)):
        logits = result.logits[0, -1].numpy().astype("<f4")
        if logits.shape != (32000,) or not np.isfinite(logits).all():
            raise ValueError("invalid full-vocabulary reference logits")
        data = logits.tobytes()
        (args.output / f"{name}-logits.f32").write_bytes(data)
        outputs[name] = {"sha256": hashlib.sha256(data).hexdigest(), "top_token": int(logits.argmax())}
    identity = {
        "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_HASHES,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "attention_implementation": "eager", "weight_storage": "F16-rounded/F32-execution",
        "kv_storage": "F16-rounded/F32-reference-attention", "prefill_ids": [1, 15043, 29892],
        "decode_ids": [3186], "outputs": outputs,
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()

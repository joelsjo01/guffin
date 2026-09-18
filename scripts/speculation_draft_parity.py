#!/usr/bin/env python3
"""Offline trained-checkpoint/native-Metal parity, not a chat benchmark.

Runs the genuine full-vocabulary dependent native chain, then independently
evaluates the training module on the same conditional prefixes. No framework
is linked into the engine. State differences are reported, not hidden by a
relaxed tolerance; any differing BF16 state or proposal fails this check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from training.speculation.gguf import GGUFFile
    from training.speculation.trace import TraceFile
    from training.speculation.models import DirectRecurrentDrafter
    from training.speculation.objectives import GemmaTargetHead, select_exact_proposal_tokens

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "weights", "checkpoint", "trace", "executable"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--origins", type=int, default=2)
    parser.add_argument("--device", choices=("equations", "cpu", "mps"), default="equations")
    args = parser.parse_args()
    if not 1 <= args.origins <= 1024:
        parser.error("origins must be in [1,1024]")
    inspected = json.loads(subprocess.check_output([str(args.executable), "inspect", str(args.weights)],text=True))
    with args.checkpoint.open("rb") as file:
        checkpoint_digest = hashlib.file_digest(file, "sha256").hexdigest()
    if inspected["checkpoint_sha256"] != checkpoint_digest:
        raise ValueError("reference checkpoint does not match the packed artifact")
    # Do not overlap framework residency with the real engine's target plan.
    run = subprocess.run([str(args.executable), "trace-metal", str(args.model),
                          str(args.weights), str(args.trace), str(args.origins), "states"],
                         text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    if len(rows) != args.origins or any(row["status"] for row in rows):
        raise RuntimeError("incomplete or numerically refused native parity run")
    if args.device == "equations":
        cpu = subprocess.run([str(args.executable), "trace-cpu", str(args.model),
                              str(args.weights), str(args.trace), str(args.origins), "states"],
                             text=True, capture_output=True, check=True)
        goldens = [json.loads(line) for line in cpu.stdout.splitlines()]
        if len(goldens) != len(rows):
            raise RuntimeError("incomplete independent CPU golden run")
        failed = False
        for row, golden in zip(rows, goldens):
            actual = np.array(row["states"], dtype=np.float32)
            expected = np.array(golden["states"], dtype=np.float32)
            mismatches = np.flatnonzero(actual != expected)
            tokens_match = row["tokens"] == golden["tokens"]
            confidence_match = row["confidence_logits"] == golden["confidence_logits"]
            failed |= bool(len(mismatches)) or not tokens_match or not confidence_match
            print(json.dumps({k:v for k,v in row.items() if k != "states"} | {
                "reference_device":"cpu_equations", "different_state_values":len(mismatches),
                "state_max_abs":float(np.max(np.abs(actual-expected))),
                "first_differences":[[int(i),float(actual[i]),float(expected[i])] for i in mismatches[:8]],
                "reference_tokens":golden["tokens"], "tokens_match":tokens_match,
                "confidence_match":confidence_match}), flush=True)
        return int(failed)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    with TraceFile(args.trace) as trace, GGUFFile(args.model) as gguf:
        records = list(trace.records())
        # Copy mmap-backed arrays before passing them to a mutable tensor API.
        table = torch.from_numpy(gguf.f16_matrix("token_embd.weight").copy()).to(device)
        norm = torch.from_numpy(gguf.floating_vector("output_norm.weight").copy()).to(device)
        width = checkpoint["model"]["fusion.projection.weight"].shape[0]
        h = len(rows[0]["tokens"])
        model = DirectRecurrentDrafter(table.shape[1], table.shape[0], 1, width,
                                      h, "gated_residual", table).to(device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        head = GemmaTargetHead(table, norm, gguf.metadata["gemma2.attention.layer_norm_rms_epsilon"],
                               gguf.metadata["gemma2.final_logit_softcapping"])
        failed = False
        for row in rows:
            origin = row["origin"]
            root = records[origin]
            feature = next(f for f in root.features if f.stage == 2)
            values = torch.from_numpy(np.frombuffer(feature.payload, dtype="<f2").copy()).to(device)
            prefix = torch.tensor([[root.target_label]+row["tokens"][:-1]], device=device)
            with torch.no_grad(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                result = model(values.reshape(1,1,-1), torch.tensor([root.target_position], device=device),
                               torch.tensor([0], device=device), prefix)
            expected = result.states.float().cpu().reshape(-1)
            actual = torch.tensor(row["states"], dtype=torch.float32)
            # The head is explicitly F32 in the trainer, outside autocast.
            with torch.no_grad():
                logits = head(result.states.float())
                draws = torch.tensor([[records[origin+i+1].selection.unit_draw for i in range(h)]],
                                     dtype=torch.float64)
                proposal = select_exact_proposal_tokens(logits, draws, trace.identity.sampler).cpu().reshape(-1)
            mismatch = int((actual != expected).sum())
            token_match = proposal.tolist() == row["tokens"]
            confidence_match = result.confidence.float().cpu().reshape(-1).tolist() == row["confidence_logits"]
            failed |= bool(mismatch) or not token_match or not confidence_match
            print(json.dumps({k:v for k,v in row.items() if k != "states"} | {
                "reference_device":args.device, "state_values":actual.numel(),
                "different_state_values":mismatch, "state_max_abs":float((actual-expected).abs().max()),
                "reference_tokens":proposal.tolist(), "tokens_match":token_match,
                "confidence_match":confidence_match}), flush=True)
        return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

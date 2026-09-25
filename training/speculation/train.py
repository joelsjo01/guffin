"""Deterministic, resumable Phase-5 training orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy
import torch

from .corpus import TraceCorpusIndex, file_sha256, validate_trace_corpus_with_index
from .dataset import (
    OriginMembership,
    TraceWindowStream,
    TrainingExample,
    build_origin_membership,
    validate_minimum_workload_clusters,
)
from .gguf import GGUFFile
from .manifest import WORKLOADS
from .models import DSparkDrafter, DraftOutput, make_model
from .objectives import (
    GemmaTargetHead,
    ObjectiveWeights,
    TargetSupport,
    dblast_objective,
    draft_logits,
    select_exact_proposal_tokens,
    standard_objective,
)
from .registry import Registry, TrainingCell
from .run_contract import (
    SOURCE_DPACE_ALPHA,
    RunCellContract,
    RunContract,
    load_run_contract,
)
from .validation_stats import (
    VALIDATION_AGGREGATION,
    ValidationObservation,
    ValidationStatisticsError,
    hierarchical_validation_summary,
)


class TrainingError(RuntimeError):
    pass


def _validate_training_implementation(cell: TrainingCell) -> None:
    if cell.implementation == "dspark":
        # The legacy trace objective sees the prepared top-k sampler support,
        # not the full raw teacher distribution required by DSpark's source
        # CE/L1/soft-confidence objective. Do not quietly train the new model
        # with the old R loss or accept its obsolete three-layer cell identity.
        raise TrainingError(
            "Arm P training requires its versioned full-teacher data and run contract; "
            "the legacy sparse-support R trainer cannot execute the DSpark source objective. "
            "No P training has started."
        )


VALIDATION_STREAM_SEED = 0xD1B54A32D192ED03
_WORKLOAD_BY_ID = {identifier: name for name, identifier in WORKLOADS.items()}
_SCHEDULER_SEMANTICS = "cosine_over_remaining_updates_with_fresh_warmup/v1"


def _progress(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def training_code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "training" / "speculation").glob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _derived_numpy_seed(seed: int) -> int:
    """Derive a stable uint32 NumPy seed without discarding uint64 seed bits."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**64:
        raise TrainingError("training seed is not an exact uint64")
    digest = hashlib.sha256(
        b"local-gguf-engine/numpy-training-seed/v1\0" + seed.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def _validated_optimizer_contract(registry: Registry) -> dict[str, object]:
    contract = registry.raw.get("default_transfer_optimizer")
    if not isinstance(contract, dict):
        raise TrainingError("default transfer optimizer contract is absent")
    if contract.get("algorithm") != "AdamW":
        raise TrainingError("this trainer implements only the declared AdamW optimizer")
    if contract.get("schedule") != "cosine":
        raise TrainingError("this trainer implements only the declared cosine schedule")
    if contract.get("precision") != "bf16":
        raise TrainingError("this trainer implements only the declared BF16 precision")

    def finite_number(name: str, *, minimum: float, maximum: float | None = None,
                      minimum_inclusive: bool = True) -> float:
        value = contract.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TrainingError(f"optimizer field {name} is not numeric")
        converted = float(value)
        valid_minimum = converted >= minimum if minimum_inclusive else converted > minimum
        if (not math.isfinite(converted) or not valid_minimum
                or (maximum is not None and converted > maximum)):
            raise TrainingError(f"optimizer field {name} is outside its supported range")
        return converted

    def positive_integer(name: str, *, allow_zero: bool = False) -> int:
        value = contract.get(name)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < (0 if allow_zero else 1)):
            raise TrainingError(f"optimizer field {name} is outside its supported range")
        return value

    finite_number("learning_rate", minimum=0.0, minimum_inclusive=False)
    betas = contract.get("betas")
    if (not isinstance(betas, list) or len(betas) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0
                   for value in betas)):
        raise TrainingError("optimizer betas must be two finite values in [0,1)")
    finite_number("epsilon", minimum=0.0, minimum_inclusive=False)
    finite_number("weight_decay", minimum=0.0)
    finite_number("global_norm_clip", minimum=0.0, minimum_inclusive=False)
    warmup_fraction = finite_number("warmup_fraction", minimum=0.0, maximum=1.0)
    if warmup_fraction >= 1.0:
        raise TrainingError("optimizer warmup fraction must be below one")
    maximum_epochs = positive_integer("maximum_epochs")
    minimum_epochs = positive_integer("minimum_epochs")
    if minimum_epochs > maximum_epochs:
        raise TrainingError("minimum epochs exceed maximum epochs")
    positive_integer("validation_interval_steps")
    positive_integer("patience_intervals")
    finite_number(
        "minimum_relative_improvement", minimum=0.0, maximum=1.0,
        minimum_inclusive=False,
    )
    positive_integer("maximum_restarts", allow_zero=True)
    finite_number("restart_learning_rate_factor", minimum=0.0, maximum=1.0,
                  minimum_inclusive=False)
    micro_batch = positive_integer("micro_batch")
    accumulation = positive_integer("gradient_accumulation")
    if micro_batch != 1:
        raise TrainingError("registry logical micro-batch contract must remain one")
    if accumulation % micro_batch:
        raise TrainingError("gradient accumulation is not divisible by the logical micro-batch")
    return contract


def _physical_training_batch_size(
    context_mode: str, optimizer_contract: dict[str, object]
) -> int:
    """Pack one logical accumulation window when origins own independent roots.

    ``micro_batch`` remains the registry's logical loss/data unit.  Root-only
    examples contain one independent target feature row per origin, and every
    registered standard-objective term first reduces within an origin and then
    takes the batch mean.  A physical batch spanning the accumulation window
    can therefore replace repeated forward/backward launches without changing
    its membership, ordering, weighting, optimizer-step boundary, or schedule.

    Full-context examples deliberately retain the logical batch size.  Their
    shared causal context is correct for multiple anchors, but its extent is
    determined by the latest anchor; coalescing distant anchors can materially
    increase attention memory/work and needs a separate capacity experiment.
    """
    logical = int(optimizer_contract["micro_batch"])
    accumulation = int(optimizer_contract["gradient_accumulation"])
    if context_mode == "root_only":
        return accumulation
    if context_mode == "full_context":
        return logical
    raise TrainingError(f"unknown feature context mode {context_mode}")


def _scheduler_plan(
    restart: int,
    global_start_update: int,
    total_updates: int,
    optimizer_contract: dict[str, object],
) -> dict[str, object]:
    if (not isinstance(restart, int) or isinstance(restart, bool) or restart < 0
            or not isinstance(global_start_update, int)
            or isinstance(global_start_update, bool)
            or not isinstance(total_updates, int) or isinstance(total_updates, bool)
            or not 0 <= global_start_update < total_updates):
        raise TrainingError("scheduler start is outside the frozen update horizon")
    span = total_updates - global_start_update
    warmup = max(1, int(span * float(optimizer_contract["warmup_fraction"])))
    base_rate = float(optimizer_contract["learning_rate"]) * (
        float(optimizer_contract["restart_learning_rate_factor"]) ** restart
    )
    return {
        "schema": "local-gguf-engine/cosine-scheduler-plan/v1",
        "semantics": _SCHEDULER_SEMANTICS,
        "restart": restart,
        "global_start_update": global_start_update,
        "span_updates": span,
        "warmup_updates": warmup,
        "base_learning_rate": base_rate,
    }


def _make_optimizer_and_scheduler(
    parameters,
    restart: int,
    global_start_update: int,
    total_updates: int,
    optimizer_contract: dict[str, object],
):
    plan = _scheduler_plan(
        restart, global_start_update, total_updates, optimizer_contract
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=float(plan["base_learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["betas"]),
        eps=float(optimizer_contract["epsilon"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    span = int(plan["span_updates"])
    warmup = int(plan["warmup_updates"])

    def learning_rate(local_step: int) -> float:
        if local_step < warmup:
            return (local_step + 1) / warmup
        progress = min(1.0, (local_step - warmup) / max(1, span - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    return optimizer, scheduler, plan


def _scheduler_start_from_learning_curve(
    restarts: int,
    learning_curve: list[dict[str, object]],
    total_updates: int,
) -> int:
    restart_updates: list[int] = []
    previous_update = -1
    for item in learning_curve:
        if not isinstance(item, dict):
            raise TrainingError("checkpoint learning curve is malformed")
        update = item.get("optimizer_updates")
        if (not isinstance(update, int) or isinstance(update, bool)
                or update < previous_update or update > total_updates):
            raise TrainingError("checkpoint learning curve update history is malformed")
        previous_update = update
        if item.get("restart_performed") is True:
            restart_updates.append(update)
        elif item.get("restart_performed") not in (False, None):
            raise TrainingError("checkpoint restart history is malformed")
    if len(restart_updates) != restarts:
        raise TrainingError("checkpoint restart count disagrees with its learning curve")
    if not restart_updates:
        return 0
    start = restart_updates[-1]
    if not 0 <= start < total_updates:
        raise TrainingError("checkpoint restart has no remaining scheduler horizon")
    return start


def _validate_scheduler_summary(
    value: dict[str, object],
    learning_curve: list[dict[str, object]],
    total_updates: int,
    optimizer_contract: dict[str, object],
) -> tuple[int, int, dict[str, object]]:
    restarts = value.get("restarts")
    update_step = value.get("update_step")
    if (not isinstance(restarts, int) or isinstance(restarts, bool)
            or not 0 <= restarts <= int(optimizer_contract["maximum_restarts"])
            or not isinstance(update_step, int) or isinstance(update_step, bool)
            or not 0 <= update_step <= total_updates):
        raise TrainingError("checkpoint scheduler summary is malformed")
    scheduler_start = _scheduler_start_from_learning_curve(
        restarts, learning_curve, total_updates
    )
    expected = _scheduler_plan(
        restarts, scheduler_start, total_updates, optimizer_contract
    )
    if update_step < scheduler_start or value.get("scheduler_plan") != expected:
        raise TrainingError("checkpoint scheduler plan is stale or inconsistent")
    return restarts, update_step, expected


def _validate_training_cursor(
    epoch: object,
    origins_in_epoch: object,
    update_step: object,
    scale: int,
    accumulation: int,
    maximum_epochs: int,
) -> tuple[int, int]:
    """Authenticate the logical origin cursor against completed optimizer work."""
    if (type(epoch) is not int or type(origins_in_epoch) is not int
            or type(update_step) is not int
            or not 0 <= epoch <= maximum_epochs
            or not 0 <= origins_in_epoch <= scale
            or epoch == maximum_epochs and origins_in_epoch != 0
            or origins_in_epoch % accumulation != 0 and origins_in_epoch != scale):
        raise TrainingError("checkpoint training cursor is malformed")
    updates_per_epoch = math.ceil(scale / accumulation)
    expected_updates = (
        epoch * updates_per_epoch
        + (origins_in_epoch + accumulation - 1) // accumulation
    )
    if update_step != expected_updates:
        raise TrainingError("checkpoint training cursor disagrees with optimizer updates")
    return epoch, origins_in_epoch


def _completion_cursor_fields(
    epoch: object,
    origins_in_epoch: object,
    update_step: object,
    stop_requested: object,
    scale: int,
    accumulation: int,
    maximum_epochs: int,
) -> dict[str, object]:
    epoch, origins_in_epoch = _validate_training_cursor(
        epoch, origins_in_epoch, update_step, scale, accumulation, maximum_epochs
    )
    if type(stop_requested) is not bool:
        raise TrainingError("completion stop provenance is malformed")
    if not stop_requested and (epoch, origins_in_epoch) != (maximum_epochs, 0):
        raise TrainingError("non-stopped training completion has a partial cursor")
    return {
        "terminal_epoch": epoch,
        "terminal_origins_in_epoch": origins_in_epoch,
        "stop_requested": stop_requested,
    }


def _advance_convergence(
    value: float,
    best: float,
    significant_reference: float,
    patience: int,
    minimum_relative_improvement: float,
    *,
    maximize: bool = False,
) -> tuple[float, float, int, bool, bool]:
    """Advance exact-best and significant-progress state independently."""
    if not math.isfinite(value):
        raise TrainingError("checkpoint selection value is nonfinite")
    if patience < 0 or minimum_relative_improvement <= 0.0:
        raise TrainingError("convergence state is invalid")
    new_best = value > best if maximize else value < best
    best = max(best, value) if maximize else min(best, value)
    if not math.isfinite(significant_reference):
        return best, value, 0, new_best, True
    improvement = (value - significant_reference) if maximize else (
        significant_reference - value
    )
    relative = improvement / max(abs(significant_reference), sys.float_info.epsilon)
    if relative >= minimum_relative_improvement:
        return best, value, 0, new_best, True
    return best, significant_reference, patience + 1, new_best, False


def _workload_name(value: object) -> str:
    if isinstance(value, str) and value in WORKLOADS:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in _WORKLOAD_BY_ID:
        return _WORKLOAD_BY_ID[value]
    raise TrainingError(f"validation example has unknown workload identity {value!r}")


def _membership_report(membership: OriginMembership) -> dict[str, object]:
    return {
        "sha256": membership.sha256,
        "tap_set": list(membership.tap_set),
        "origin_count": len(membership.references),
        "workload_counts": {
            _workload_name(workload): count
            for workload, count in membership.workload_counts
        },
        "target_generation_seed_counts": {
            str(seed): count for seed, count in membership.generation_seed_counts
        },
        "workload_target_generation_seed_counts": {
            _workload_name(workload): {
                str(seed): count
                for candidate_workload, seed, count in membership.workload_seed_counts
                if candidate_workload == workload
            }
            for workload, _count in membership.workload_counts
        },
        "source_group_count": membership.source_group_count,
        "source_cluster_count": membership.source_cluster_count,
        "conversation_count": membership.conversation_count,
        "workload_source_cluster_counts": {
            _workload_name(workload): count
            for workload, count in membership.workload_cluster_counts
        },
    }


def _rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state_all()
    else:
        state["accelerator"] = None
    return state


def _restore_rng_state(state: object, device: torch.device) -> None:
    if not isinstance(state, dict) or set(state) != {
        "python", "numpy", "torch_cpu", "accelerator"
    }:
        raise TrainingError("checkpoint RNG state is incomplete")
    try:
        random.setstate(state["python"])
        numpy.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if device.type == "mps":
            if state["accelerator"] is None:
                raise TrainingError("checkpoint lacks MPS RNG state")
            torch.mps.set_rng_state(state["accelerator"])
        elif device.type == "cuda":
            if not isinstance(state["accelerator"], list):
                raise TrainingError("checkpoint lacks CUDA RNG state")
            torch.cuda.set_rng_state_all(state["accelerator"])
        elif state["accelerator"] is not None:
            raise TrainingError("CPU checkpoint unexpectedly contains accelerator RNG state")
    except TrainingError:
        raise
    except Exception as error:
        raise TrainingError("checkpoint RNG state is malformed") from error


def _device(requested: str, allow_cpu_diagnostic: bool) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TrainingError("requested MPS training device is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError("requested CUDA training device is unavailable")
    if requested == "cpu" and not allow_cpu_diagnostic:
        raise TrainingError("CPU is diagnostic-only; pass --allow-cpu-diagnostic explicitly")
    return torch.device(requested)


def _horizon(cell: TrainingCell) -> int:
    parameters = dict(cell.parameters)
    if "horizon" in parameters:
        return int(parameters["horizon"])
    if "block" in parameters:
        return int(parameters["block"]) - 1
    raise TrainingError(f"cell has no horizon/block: {cell.identifier}")


def _objective_weights(
    registry: Registry, dpace_alpha_override: float | None = None,
) -> ObjectiveWeights:
    raw = registry.raw["transfer_objective"]
    if (dpace_alpha_override is not None
            and (not isinstance(dpace_alpha_override, (int, float))
                 or isinstance(dpace_alpha_override, bool)
                 or not math.isfinite(float(dpace_alpha_override))
                 or float(dpace_alpha_override) != SOURCE_DPACE_ALPHA)):
        raise TrainingError(
            f"paper-supported D-PACE control requires alpha={SOURCE_DPACE_ALPHA}"
        )
    dpace_alpha = (
        float(raw["dpace_alpha"])
        if dpace_alpha_override is None else float(dpace_alpha_override)
    )
    return ObjectiveWeights(
        forward_kl=float(raw["forward_kl_weight"]),
        hard_ce=float(raw["hard_ce_weight"]),
        common_draw_overlap=float(raw["common_draw_overlap_weight"]),
        confidence=float(raw["confidence_weight"]),
        future_state_auxiliary=float(raw["future_state_auxiliary_weight"]),
        dpace_alpha=dpace_alpha,
    )


def _owned_tensor(array: numpy.ndarray, device: torch.device) -> torch.Tensor:
    """Copy a read-only GGUF mmap view into storage owned by PyTorch."""
    return torch.tensor(array, device=device)


def _validated_active_gemma_head(
    gguf: GGUFFile, run_contract: RunContract,
) -> tuple[numpy.ndarray, numpy.ndarray, float, float]:
    """Load the exact tied-F16 head declared by the active run contract."""
    metadata = gguf.metadata
    if metadata.get("general.architecture") != run_contract.target_architecture:
        raise TrainingError("training GGUF architecture differs from the run contract")
    embedding_width = metadata.get("gemma2.embedding_length")
    if (not isinstance(embedding_width, int) or isinstance(embedding_width, bool)
            or embedding_width != run_contract.target_embedding_width):
        raise TrainingError("training GGUF embedding width differs from the run contract")
    if run_contract.target_output_head != "tied_token_embedding_f16":
        raise TrainingError("training run output-head contract is unsupported")
    if "output.weight" in gguf.tensors:
        raise TrainingError(
            "tied-head training refuses a GGUF with a distinct output.weight"
        )
    token_array = gguf.f16_matrix("token_embd.weight")
    norm_array = gguf.floating_vector("output_norm.weight")
    expected_table_shape = (
        run_contract.target_vocabulary, run_contract.target_embedding_width
    )
    if (token_array.shape != expected_table_shape
            or token_array.dtype != numpy.float16
            or norm_array.shape != (run_contract.target_embedding_width,)):
        raise TrainingError("training GGUF tied-head tensor shape/dtype is incompatible")
    epsilon_value = metadata.get("gemma2.attention.layer_norm_rms_epsilon")
    softcap_value = metadata.get("gemma2.final_logit_softcapping")
    if (not isinstance(epsilon_value, (int, float))
            or isinstance(epsilon_value, bool)
            or not math.isfinite(float(epsilon_value))
            or float(epsilon_value) <= 0.0):
        raise TrainingError("training GGUF RMS epsilon is not finite and positive")
    if (not isinstance(softcap_value, (int, float))
            or isinstance(softcap_value, bool)
            or not math.isfinite(float(softcap_value))
            or float(softcap_value) <= 0.0):
        raise TrainingError("training GGUF logit softcap is not finite and positive")
    # Scan in bounded row chunks before any accelerator allocation. In
    # particular, an infinite logit can be hidden by tanh softcapping and must
    # not silently enter a training run.
    for start in range(0, token_array.shape[0], 4096):
        if not bool(numpy.isfinite(token_array[start:start + 4096]).all()):
            raise TrainingError("training GGUF token table contains a nonfinite value")
    if not bool(numpy.isfinite(norm_array).all()):
        raise TrainingError("training GGUF output norm contains a nonfinite value")
    return (
        token_array, norm_array, float(epsilon_value), float(softcap_value)
    )


def _to_device(example: TrainingExample, device: torch.device) -> TrainingExample:
    feature_dtype = torch.float32 if device.type == "cpu" else torch.float16
    return replace(
        example,
        context_taps=example.context_taps.to(device=device, dtype=feature_dtype),
        context_positions=example.context_positions.to(device),
        anchor_indices=example.anchor_indices.to(device),
        rollout_tokens=example.rollout_tokens.to(device),
        labels=example.labels.to(device),
        target_support=TargetSupport(
            token_ids=example.target_support.token_ids.to(device),
            probabilities=example.target_support.probabilities.float().to(device),
            mask=example.target_support.mask.to(device),
        ),
        target_selected_probabilities=example.target_selected_probabilities.float().to(device),
        # The authoritative sampler ticket is an F64 boundary decision. MPS
        # cannot represent it, so retain the draws on CPU in their exact trace
        # dtype. Exact proposal selection and its independent evaluator both
        # make their discrete decisions from this CPU-F64 evidence.
        target_unit_draws=example.target_unit_draws.to(device="cpu", dtype=torch.float64),
        target_positions=example.target_positions.to(device),
    )


def _slice_example(example: TrainingExample, start: int, end: int) -> TrainingExample:
    if start < 0 or end <= start or end > example.origin_count:
        raise TrainingError("training-example slice is invalid")
    selected = slice(start, end)
    changes: dict[str, object] = {
        "anchor_indices": example.anchor_indices[selected],
        "rollout_tokens": example.rollout_tokens[selected],
        "labels": example.labels[selected],
        "target_support": TargetSupport(
            token_ids=example.target_support.token_ids[selected],
            probabilities=example.target_support.probabilities[selected],
            mask=example.target_support.mask[selected],
        ),
        "target_selected_probabilities": example.target_selected_probabilities[selected],
        "target_unit_draws": example.target_unit_draws[selected],
        "source_clusters": example.source_clusters[selected],
        "conversations": example.conversations[selected],
        "sequence_ids": example.sequence_ids[selected],
        "target_positions": example.target_positions[selected],
    }
    for field in ("workloads", "generation_seeds"):
        if hasattr(example, field):
            changes[field] = getattr(example, field)[selected]
    return replace(example, **changes)


def _autocast(device: torch.device, precision: str):
    if device.type == "cpu":
        return nullcontext()
    if precision != "bf16":
        raise TrainingError(f"unsupported mixed training precision {precision}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _rollout(
    model: torch.nn.Module,
    head: GemmaTargetHead,
    example: TrainingExample,
    curriculum: str,
    proposal_policy: str,
) -> tuple[DraftOutput, torch.Tensor | None]:
    if proposal_policy not in {"deterministic", "common_draw"}:
        raise TrainingError(f"unknown proposal policy {proposal_policy}")
    teacher = example.rollout_tokens
    if curriculum == "teacher_prefix":
        return model(
            example.context_taps, example.context_positions,
            example.anchor_indices, teacher,
        ), None
    if curriculum not in {"deployment_prefix", "training_time_self_rollout"}:
        raise TrainingError(f"unknown token-prefix curriculum {curriculum}")
    if (teacher.ndim != 2 or teacher.dtype != torch.int64
            or example.labels.shape != teacher.shape):
        raise TrainingError("self-rollout token/label shape is invalid")

    # Populate the prefix one proposal at a time.  A single teacher-forced
    # preview cannot do this: its proposal at depth d>0 has already consumed
    # the authoritative token at d, so shifting all preview predictions into
    # one final pass leaks the teacher path into later deployment states.
    # Initialize every unavailable suffix token independently of the teacher
    # so poisoning a future target cannot affect a pre-loss rollout.
    own = torch.zeros_like(teacher)
    own[:, 0] = teacher[:, 0]
    reached = torch.ones_like(example.labels, dtype=torch.bool)
    # P's defining division: one parallel backbone/head, followed only by
    # predecessor-conditioned corrections. Retain this graph for the final
    # loss; the proposal-only previews borrow detached copies, not new runs.
    p_base = p_logits = None
    if isinstance(model, DSparkDrafter):
        p_base = model.prepare_backbone(
            example.context_taps, example.context_positions,
            example.anchor_indices, teacher[:, 0],
        )
        p_logits = head(model.project_base(p_base), apply_norm=False,
                        apply_softcap=False)
    with torch.no_grad():
        for depth in range(teacher.shape[1] - 1):
            if p_base is not None:
                preview = model.correct(p_base.detach(), own)
                preview.semantic_logits = p_logits.detach()
            else:
                preview = model(
                    example.context_taps, example.context_positions,
                    example.anchor_indices, own,
                )
            preview_states = preview.states
            if (preview_states.ndim != 3
                    or preview_states.shape[:2] != teacher.shape):
                raise TrainingError("self-rollout state shape is invalid")
            logits = draft_logits(preview, head, depth)
            if proposal_policy == "common_draw":
                proposed = select_exact_proposal_tokens(
                    logits, example.target_unit_draws[:, depth], example.sampler
                )
            else:
                # argmax returns the first vocabulary index at a tie, matching
                # the deterministic lower-token-ID policy.
                proposed = torch.argmax(logits, dim=-1)
            own[:, depth + 1] = proposed
            reached[:, depth + 1] = (
                reached[:, depth] & (proposed == example.labels[:, depth])
            )
    if p_base is not None:
        output = model.correct(p_base, own)
        output.semantic_logits = p_logits
        return output, reached
    return model(
        example.context_taps, example.context_positions,
        example.anchor_indices, own,
    ), reached


def _position_weighting(parameters: dict[str, object], epoch: int) -> str:
    value = str(parameters.get("position_weighting", parameters.get("loss", "uniform")))
    if value == "paper_epoch_progressive":
        return f"gamma:{4.5 + epoch}"
    if value == "dpace":
        return "dpace"
    return value


def _loss(
    model: torch.nn.Module,
    head: GemmaTargetHead,
    example: TrainingExample,
    cell: TrainingCell,
    weights: ObjectiveWeights,
    epoch: int,
    *,
    proposal_policy: str,
    curriculum_override: str | None = None,
    position_weighting_override: str | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    if reduction not in {"mean", "none"}:
        raise TrainingError("loss reduction must be mean or none")
    parameters = dict(cell.parameters)
    curriculum = (curriculum_override if curriculum_override is not None
                  else str(parameters.get("curriculum", "teacher_prefix")))
    output, reached = _rollout(
        model, head, example, curriculum, proposal_policy
    )
    if cell.implementation == "dblast":
        if reduction != "mean":
            raise TrainingError("DBlast objective has no per-origin reduction contract")
        loss, metrics = dblast_objective(
            output, head, example.labels, example.target_selected_probabilities,
            str(parameters["objective"]),
        )
        category = torch.argmax(output.category_logits, dim=-1)
        chosen = output.states.gather(
            2,
            category[:, None, None, None].expand(
                -1, output.states.shape[1], 1, output.states.shape[-1]
            ),
        ).squeeze(2)
        logits = head(
            chosen, apply_norm=not output.head_pre_normalized
        )
        return loss, metrics, logits
    loss, metrics, logits = standard_objective(
        output, head, example.labels, example.target_support, weights,
        example.sampler,
        (position_weighting_override if position_weighting_override is not None
         else _position_weighting(parameters, epoch)),
        reached,
        proposal_policy=proposal_policy,
        target_unit_draws=(
            example.target_unit_draws if proposal_policy == "common_draw" else None
        ),
        materialize_metrics=False,
        reduction=reduction,
    )
    return loss, metrics, logits


def _raw_prefixes(
    logits: torch.Tensor, example: TrainingExample,
) -> list[tuple[int, int]]:
    if (logits.ndim != 3 or example.labels.shape != logits.shape[:2]
            or example.target_unit_draws.shape != logits.shape[:2]
            or example.target_unit_draws.device.type != "cpu"
            or example.target_unit_draws.dtype != torch.float64):
        raise TrainingError("raw proposal evaluation shape/dtype is invalid")
    _, _, _, _, repetition_penalty, _, _ = example.sampler
    if float(repetition_penalty) != 1.0:
        raise TrainingError(
            "raw proposal evaluation requires hypothetical sampler-history reconstruction"
        )
    deterministic = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    labels = example.labels.detach().cpu().numpy()
    try:
        common = select_exact_proposal_tokens(
            logits, example.target_unit_draws, example.sampler
        ).cpu().numpy()
    except (TypeError, ValueError) as error:
        raise TrainingError("raw proposal sampler contract is invalid") from error

    def prefix(candidate: numpy.ndarray, batch: int) -> int:
        for depth, matches in enumerate(candidate[batch] == labels[batch]):
            if not bool(matches):
                return depth
        return int(labels.shape[1])

    return [(prefix(deterministic, batch), prefix(common, batch))
            for batch in range(logits.shape[0])]


def _evaluate(
    model: torch.nn.Module,
    head: GemmaTargetHead,
    stream: TraceWindowStream,
    cell: TrainingCell,
    weights: ObjectiveWeights,
    device: torch.device,
    epoch: int,
    workload_weights: dict[str, object],
    proposal_policy: str,
    checkpoint_metric: str,
    validation_loss_contract: str,
    validation_aggregation: str,
    required_generation_seeds: tuple[int, ...],
    precision: str = "bf16",
) -> dict[str, object]:
    if proposal_policy not in {"deterministic", "common_draw"}:
        raise TrainingError("validation proposal policy is unsupported")
    if checkpoint_metric != (
            "hierarchical_workload_weighted_exact_accepted_prefix/v1"):
        raise TrainingError("validation checkpoint metric is unsupported")
    if validation_loss_contract != "teacher_prefix_uniform_kl_ce_all_edges":
        raise TrainingError("validation diagnostic contract is unsupported")
    if validation_aggregation != VALIDATION_AGGREGATION:
        raise TrainingError("validation aggregation contract is unsupported")
    if not isinstance(workload_weights, dict) or not workload_weights:
        raise TrainingError("validation workload weights are absent")
    validated_weights: dict[str, float] = {}
    for workload, raw_weight in workload_weights.items():
        if (not isinstance(workload, str) or workload not in WORKLOADS
                or not isinstance(raw_weight, (int, float))
                or isinstance(raw_weight, bool)
                or not math.isfinite(float(raw_weight))
                or float(raw_weight) <= 0.0):
            raise TrainingError("validation workload weights are malformed")
        validated_weights[workload] = float(raw_weight)
    if not math.isclose(
        math.fsum(validated_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise TrainingError("validation workload weights do not sum to one")

    model.eval()
    root_only = getattr(stream, "context_mode", None) == "root_only"
    observations: list[ValidationObservation] = []
    deterministic = Counter()
    common = Counter()
    validation_horizon: int | None = None
    # This fixed-population diagnostic deliberately excludes policy-overlap
    # and confidence terms. Both depend on discrete proposal decisions and are
    # already measured directly by the exact accepted-prefix checkpoint metric.
    diagnostic_weights = replace(
        weights, common_draw_overlap=0.0, confidence=0.0
    )
    with torch.no_grad():
        for raw in stream:
            example = _to_device(raw, device)
            with _autocast(device, precision):
                # The diagnostic must use a fixed population. A deployment-
                # masked loss changes which suffix edges enter the mean as the
                # drafter improves and can therefore prefer a worse model.
                loss, _, teacher_logits = _loss(
                    model, head, example, cell, diagnostic_weights, epoch,
                    proposal_policy=proposal_policy,
                    curriculum_override="teacher_prefix",
                    position_weighting_override="uniform",
                    reduction="none" if root_only else "mean",
                )
            batch_count = example.origin_count
            if root_only:
                if (loss.ndim != 1 or loss.shape != (batch_count,)
                        or batch_count <= 0
                        or batch_count > stream.anchor_batch_size):
                    raise TrainingError(
                        "root-only validation loss/batch shape is malformed"
                    )
                loss_values = loss.detach().cpu().tolist()
            else:
                if batch_count != 1 or loss.ndim != 0:
                    raise TrainingError(
                        "full-context validation violated its logical batch of one"
                    )
                loss_values = [float(loss)]
            if (len(loss_values) != batch_count
                    or any(not math.isfinite(float(value)) for value in loss_values)):
                raise TrainingError(
                    "validation loss is nonfinite or has the wrong origin count"
                )
            example_workloads = getattr(example, "workloads", None)
            example_conversations = getattr(example, "conversations", None)
            example_generation_seeds = getattr(example, "generation_seeds", None)
            if (not isinstance(example_workloads, tuple)
                    or len(example_workloads) != batch_count
                    or not isinstance(example.source_clusters, tuple)
                    or len(example.source_clusters) != batch_count
                    or not isinstance(example_conversations, tuple)
                    or len(example_conversations) != batch_count
                    or not isinstance(example_generation_seeds, tuple)
                    or len(example_generation_seeds) != batch_count):
                raise TrainingError(
                    "validation example lacks per-origin hierarchical identity"
                )
            if (example.labels.ndim != 2 or example.labels.shape[0] != batch_count
                    or example.labels.shape[1] <= 0):
                raise TrainingError("validation example label shape is malformed")
            row_horizon = int(example.labels.shape[1])
            if validation_horizon is None:
                validation_horizon = row_horizon
            elif validation_horizon != row_horizon:
                raise TrainingError("validation stream changes proposal horizon")
            # Exact accepted-prefix evaluation is teacher-forced without being
            # optimistic: every state before the first mismatch has consumed
            # exactly the same prefix in deployment. States after that mismatch
            # are excluded by definition, so a second self-rollout is needless.
            prefixes = _raw_prefixes(teacher_logits, example)
            if len(prefixes) != batch_count:
                raise TrainingError("validation prefix count differs from its origin batch")
            for index, (deterministic_prefix, common_prefix) in enumerate(prefixes):
                deterministic[deterministic_prefix] += 1
                common[common_prefix] += 1
                observations.append(ValidationObservation(
                    workload=_workload_name(example_workloads[index]),
                    source_cluster=example.source_clusters[index],
                    conversation=example_conversations[index],
                    generation_seed=example_generation_seeds[index],
                    loss=float(loss_values[index]),
                    deterministic_prefix=deterministic_prefix,
                    common_draw_prefix=common_prefix,
                ))
    model.train()
    if not observations or validation_horizon is None:
        raise TrainingError("source-cluster validation stream has no eligible windows")
    try:
        hierarchical = hierarchical_validation_summary(
            observations, validated_weights, required_generation_seeds,
            validation_horizon,
        )
    except ValidationStatisticsError as error:
        raise TrainingError(str(error)) from error
    weighted_metrics = hierarchical["weighted_metrics"]
    origin_metrics = hierarchical["origin_mean_metrics"]
    workload_reports = hierarchical["workloads"]
    if (not isinstance(weighted_metrics, dict)
            or not isinstance(origin_metrics, dict)
            or not isinstance(workload_reports, dict)):
        raise TrainingError("hierarchical validation summary is malformed")
    workload_means = {
        workload: report["metrics"]["loss"]
        for workload, report in workload_reports.items()
    }
    deterministic_workload_means = {
        workload: report["metrics"]["deterministic_prefix"]
        for workload, report in workload_reports.items()
    }
    common_workload_means = {
        workload: report["metrics"]["common_draw_prefix"]
        for workload, report in workload_reports.items()
    }
    source_cluster_mean_loss = {
        f"{workload}:{cluster}": cluster_report["metrics"]["loss"]
        for workload, report in workload_reports.items()
        for cluster, cluster_report in report["clusters"].items()
    }

    def prefix_mean(histogram: Counter[int]) -> float:
        count = sum(histogram.values())
        if count <= 0:
            raise TrainingError("validation prefix histogram is empty")
        return math.fsum(prefix * frequency for prefix, frequency in histogram.items()) / count

    deterministic_weighted_prefix = weighted_metrics["deterministic_prefix"]
    common_weighted_prefix = weighted_metrics["common_draw_prefix"]
    checkpoint_value = (
        common_weighted_prefix
        if proposal_policy == "common_draw"
        else deterministic_weighted_prefix
    )
    return {
        # Checkpoint selection follows the preregistered workload mixture,
        # rather than whichever workloads contribute more eligible origins.
        "checkpoint_selection_metric": checkpoint_metric,
        "checkpoint_selection_value": checkpoint_value,
        "proposal_policy": proposal_policy,
        "mean_loss": weighted_metrics["loss"],
        "workload_weighted_mean_loss": weighted_metrics["loss"],
        "origin_mean_loss": origin_metrics["loss"],
        "validation_loss_contract": validation_loss_contract,
        "validation_aggregation": validation_aggregation,
        "window_count": len(observations),
        "source_cluster_count": hierarchical["source_cluster_count"],
        "source_cluster_mean_loss": dict(sorted(source_cluster_mean_loss.items())),
        "workload_window_count": {
            workload: report["origin_count"]
            for workload, report in workload_reports.items()
        },
        "workload_mean_loss": workload_means,
        "workload_weights": {
            key: validated_weights[key] for key in sorted(validated_weights)
        },
        "deterministic_prefix_histogram": dict(sorted(deterministic.items())),
        "common_draw_prefix_histogram": dict(sorted(common.items())),
        "deterministic_mean_prefix": prefix_mean(deterministic),
        "common_draw_mean_prefix": prefix_mean(common),
        "deterministic_workload_mean_prefix": deterministic_workload_means,
        "common_draw_workload_mean_prefix": common_workload_means,
        "deterministic_workload_weighted_mean_prefix": deterministic_weighted_prefix,
        "common_draw_workload_weighted_mean_prefix": common_weighted_prefix,
        "hierarchical_validation": hierarchical,
    }


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial checkpoint must be inspected: {temporary}")
    with temporary.open("xb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _maximum_validation_events(
    scale: int, optimizer_contract: dict[str, object]
) -> int:
    accumulation = int(optimizer_contract["gradient_accumulation"])
    maximum_epochs = int(optimizer_contract["maximum_epochs"])
    validation_interval = int(optimizer_contract["validation_interval_steps"])
    updates_per_epoch = math.ceil(scale / accumulation)
    total_updates = maximum_epochs * updates_per_epoch
    periodic = total_updates // validation_interval
    maximum = periodic + (total_updates % validation_interval != 0)
    if maximum <= 0:
        raise TrainingError("optimizer contract cannot produce a validation checkpoint")
    return maximum


def _best_checkpoint_filename(
    cell_id: str, validation_index: int, maximum_validation_events: int
) -> str:
    if (not isinstance(validation_index, int) or isinstance(validation_index, bool)
            or validation_index < 0
            or validation_index >= maximum_validation_events):
        raise TrainingError("best checkpoint validation index exceeds the frozen schedule")
    return f"{cell_id}.best.{validation_index:012d}.pt"


def _resolve_best_checkpoint_path(
    output_directory: Path,
    cell_id: str,
    relative_filename: object,
    maximum_validation_events: int,
) -> tuple[Path, int]:
    if not isinstance(relative_filename, str) or Path(relative_filename).name != relative_filename:
        raise TrainingError("best checkpoint relative filename is malformed")
    prefix = f"{cell_id}.best."
    suffix = ".pt"
    if not relative_filename.startswith(prefix) or not relative_filename.endswith(suffix):
        raise TrainingError("best checkpoint relative filename is malformed")
    encoded_index = relative_filename[len(prefix):-len(suffix)]
    if len(encoded_index) != 12 or not encoded_index.isascii() or not encoded_index.isdigit():
        raise TrainingError("best checkpoint relative filename is malformed")
    validation_index = int(encoded_index)
    if relative_filename != _best_checkpoint_filename(
        cell_id, validation_index, maximum_validation_events
    ):
        raise TrainingError("best checkpoint relative filename is noncanonical")
    return output_directory / relative_filename, validation_index


def _validate_best_checkpoint_files(
    output_directory: Path,
    cell_id: str,
    maximum_validation_events: int,
    committed_validation_count: int,
    referenced_filename: str | None,
) -> None:
    if (not isinstance(committed_validation_count, int)
            or not 0 <= committed_validation_count <= maximum_validation_events):
        raise TrainingError("checkpoint validation history exceeds the frozen schedule")
    prefix = f"{cell_id}.best."
    paths: list[tuple[Path, int]] = []
    if output_directory.exists():
        for path in output_directory.iterdir():
            if not path.name.startswith(prefix):
                continue
            if path.name.endswith(".pt.partial"):
                published = path.with_suffix("")
                if (not path.is_file() or path.is_symlink()
                        or not published.is_file() or published.is_symlink()
                        or file_sha256(path) != file_sha256(published)):
                    raise TrainingError(
                        f"stale partial best checkpoint must be inspected: {path}"
                    )
                # The immutable link was created with the same bytes and only
                # cleanup was interrupted. Removing the redundant link cannot
                # invalidate either the old progress state or the new orphan.
                path.unlink()
                _fsync_directory(output_directory)
                continue
            if not path.name.endswith(".pt"):
                continue
            resolved, validation_index = _resolve_best_checkpoint_path(
                output_directory, cell_id, path.name, maximum_validation_events
            )
            if resolved != path or not path.is_file() or path.is_symlink():
                raise TrainingError(f"best checkpoint path is malformed: {path}")
            # Exactly one file at the next validation index may exist: it is
            # the crash window after immutable publication but before the
            # progress-state rename. Anything later cannot have been produced
            # by the committed deterministic execution.
            if validation_index > committed_validation_count:
                raise TrainingError(f"future best checkpoint is inconsistent: {path}")
            paths.append((path, validation_index))
    if len(paths) > maximum_validation_events:
        raise TrainingError("best checkpoint count exceeds the frozen validation schedule")
    if referenced_filename is None:
        if committed_validation_count != 0:
            raise TrainingError("checkpoint state does not bind its best checkpoint")
        return
    referenced_path, referenced_index = _resolve_best_checkpoint_path(
        output_directory, cell_id, referenced_filename, maximum_validation_events
    )
    if referenced_index >= committed_validation_count or not referenced_path.is_file():
        raise TrainingError("checkpoint state references an unavailable best checkpoint")


def _publish_best_checkpoint(path: Path, state: dict[str, object]) -> str:
    """Publish an immutable best candidate, or reuse one identical replay."""
    if state.get("best_checkpoint_file") != path.name:
        raise TrainingError("best checkpoint payload does not bind its relative filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        if (not path.is_file() or path.is_symlink()
                or not temporary.is_file() or temporary.is_symlink()
                or file_sha256(temporary) != file_sha256(path)):
            raise FileExistsError(f"stale partial checkpoint must be inspected: {temporary}")
        temporary.unlink()
        _fsync_directory(path.parent)
    with temporary.open("xb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    candidate_digest = file_sha256(temporary)
    try:
        try:
            # A hard link is an atomic create-without-replacement operation.
            # Unlike rename/replace, it can never destroy an older immutable
            # checkpoint if another writer or deterministic replay got here.
            os.link(temporary, path)
        except FileExistsError:
            if (not path.is_file() or path.is_symlink()
                    or file_sha256(path) != candidate_digest):
                raise TrainingError(
                    f"best checkpoint replay conflicts with published content: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return candidate_digest


def _load_best_checkpoint(
    path: Path,
    device: torch.device,
    expected_digest: str,
    expected_selection_value: object,
    expected_filename: str,
    validate_identity,
) -> dict[str, object]:
    if (not isinstance(expected_digest, str) or len(expected_digest) != 64
            or not path.is_file() or file_sha256(path) != expected_digest):
        raise TrainingError(f"best checkpoint digest mismatch: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as error:
        raise TrainingError(f"best checkpoint is unreadable: {path}") from error
    if not isinstance(checkpoint, dict):
        raise TrainingError(f"best checkpoint is malformed: {path}")
    validate_identity(checkpoint, "best checkpoint")
    selection_value = checkpoint.get("checkpoint_selection_value")
    if (checkpoint.get("schema") != "local-gguf-engine/training-best/v2"
            or path.name != expected_filename
            or checkpoint.get("best_checkpoint_file") != expected_filename
            or not isinstance(expected_selection_value, (int, float))
            or isinstance(expected_selection_value, bool)
            or not math.isfinite(float(expected_selection_value))
            or not isinstance(selection_value, (int, float))
            or isinstance(selection_value, bool)
            or not math.isfinite(float(selection_value))
            or float(selection_value) != float(expected_selection_value)
            or not isinstance(checkpoint.get("model"), dict)):
        raise TrainingError(
            f"best checkpoint identity/selection value is inconsistent: {path}"
        )
    return checkpoint


def train_cell(
    root: Path,
    registry: Registry,
    cell: TrainingCell,
    traces: Path,
    output_directory: Path,
    model_path: Path,
    model_digest: str,
    scale: int,
    device: torch.device,
    source_manifest_digest: str,
    trace_corpus_digest: str,
    catalog: TraceCorpusIndex,
    training_membership: OriginMembership,
    validation_membership: OriginMembership,
    run_contract: RunContract,
    cell_contract: RunCellContract,
) -> dict[str, object]:
    _validate_training_implementation(cell)
    if (run_contract.registry_sha256 != registry.digest
            or cell_contract.cell_id != cell.identifier
            or run_contract.cell(cell.identifier) != cell_contract):
        raise TrainingError("training cell run contract is stale or incompatible")
    if cell.implementation == "speculative_streaming":
        raise TrainingError(
            "Arm S requires an exact differentiable executor for the frozen quantized Gemma "
            "target blocks; trace-only adapter fitting is forbidden"
        )
    parameters = dict(cell.parameters)
    uses_dpace = str(parameters.get(
        "position_weighting", parameters.get("loss", "uniform")
    )) == "dpace"
    run_dpace_alpha = cell_contract.dpace_alpha
    if uses_dpace:
        if (not isinstance(run_dpace_alpha, (int, float))
                or isinstance(run_dpace_alpha, bool)
                or not math.isfinite(float(run_dpace_alpha))
                or float(run_dpace_alpha) != SOURCE_DPACE_ALPHA):
            raise TrainingError(
                f"D-PACE training cell requires run alpha={SOURCE_DPACE_ALPHA}"
            )
        run_dpace_alpha = float(run_dpace_alpha)
    elif run_dpace_alpha is not None:
        raise TrainingError("non-D-PACE training cell forbids a run D-PACE alpha")
    if (parameters.get("selector") != "full_vocabulary"
            or parameters.get("selected_head_precision") != "f16"):
        raise TrainingError(
            "this trainer currently executes only the explicitly registered F16 full-vocabulary "
            "quality-control head; learned/packed selector cells require their joint trainer"
        )
    tap_set = tuple(str(value) for value in parameters.get("tap_set", ("final",)))
    horizon = _horizon(cell)
    context_mode = str(parameters.get("context_mode", "full_context"))
    optimizer_contract = _validated_optimizer_contract(registry)
    logical_batch_size = int(optimizer_contract["micro_batch"])
    physical_training_batch_size = _physical_training_batch_size(
        context_mode, optimizer_contract
    )
    physical_validation_batch_size = (
        run_contract.root_only_validation_physical_batch_size
        if context_mode == "root_only" else logical_batch_size
    )
    if (not isinstance(physical_validation_batch_size, int)
            or isinstance(physical_validation_batch_size, bool)
            or physical_validation_batch_size <= 0):
        raise TrainingError("physical validation batch size is invalid")
    maximum_epochs = int(optimizer_contract["maximum_epochs"])
    accumulation = int(optimizer_contract["gradient_accumulation"])
    updates_per_epoch = math.ceil(scale / accumulation)
    validation_interval = int(optimizer_contract["validation_interval_steps"])
    maximum_validation_events = _maximum_validation_events(scale, optimizer_contract)
    if (catalog.registry_sha256 != registry.digest
            or catalog.trace_corpus_sha256 != trace_corpus_digest
            or training_membership.trace_corpus_sha256 != trace_corpus_digest
            or validation_membership.trace_corpus_sha256 != trace_corpus_digest
            or training_membership.horizon != horizon
            or validation_membership.horizon != horizon
            or training_membership.partition != "train"
            or validation_membership.partition != "validation"
            or training_membership.tap_set != tap_set
            or validation_membership.tap_set != tap_set
            or len(training_membership.references) != scale
            or not validation_membership.references):
        raise TrainingError("origin membership is stale, incomplete, or incompatible")
    if (not isinstance(model_digest, str) or len(model_digest) != 64
            or any(character not in "0123456789abcdef" for character in model_digest)):
        raise TrainingError("training model digest is malformed")
    membership_identity = {
        "training_membership_sha256": training_membership.sha256,
        "training_origin_count": len(training_membership.references),
        "validation_membership_sha256": validation_membership.sha256,
        "validation_origin_count": len(validation_membership.references),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / f"{cell.identifier}.pt"
    state_path = output_directory / f"{cell.identifier}.state.pt"
    receipt_path = output_directory / f"{cell.identifier}.training.json"
    code_digest = training_code_digest(root)
    run_identity = {
        "run_contract_sha256": run_contract.digest,
        "proposal_policy": cell_contract.proposal_policy,
        "dpace_alpha": run_dpace_alpha,
        "data_order_seed": run_contract.data_order_seed,
        "data_order_derivation": run_contract.data_order_derivation,
        "target_architecture": run_contract.target_architecture,
        "target_embedding_width": run_contract.target_embedding_width,
        "target_vocabulary": run_contract.target_vocabulary,
        "target_output_head": run_contract.target_output_head,
        "checkpoint_selection_metric": run_contract.checkpoint_metric,
        "validation_loss_contract": run_contract.validation_loss,
        "validation_aggregation": run_contract.validation_aggregation,
        "validation_cadence": run_contract.validation_cadence,
        "minimum_validation_source_clusters_per_workload": (
            run_contract.minimum_validation_source_clusters_per_workload
        ),
        "root_only_interleave_algorithm": (
            run_contract.root_only_interleave_algorithm
        ),
        "root_only_interleave_clusters_per_sampler": (
            run_contract.root_only_interleave_clusters_per_sampler
        ),
        "root_only_origin_shuffle_buffer": (
            run_contract.root_only_origin_shuffle_buffer
        ),
        "physical_validation_batch_size": physical_validation_batch_size,
    }

    def validate_identity(value: dict[str, object], description: str) -> None:
        expected = (
            registry.digest, cell.identifier, model_digest, code_digest, scale,
            source_manifest_digest, trace_corpus_digest,
            training_membership.sha256, len(training_membership.references),
            validation_membership.sha256, len(validation_membership.references),
            run_contract.digest, cell_contract.proposal_policy, run_dpace_alpha,
            run_contract.data_order_seed, run_contract.data_order_derivation,
            run_contract.target_architecture,
            run_contract.target_embedding_width,
            run_contract.target_vocabulary,
            run_contract.target_output_head,
            run_contract.checkpoint_metric, run_contract.validation_loss,
            run_contract.validation_aggregation,
            run_contract.validation_cadence,
            run_contract.minimum_validation_source_clusters_per_workload,
            run_contract.root_only_interleave_algorithm,
            run_contract.root_only_interleave_clusters_per_sampler,
            run_contract.root_only_origin_shuffle_buffer,
            physical_validation_batch_size,
        )
        try:
            actual = tuple(value[key] for key in (
                "registry_sha256", "cell_id", "model_sha256",
                "training_code_sha256", "scale", "source_manifest_sha256",
                "trace_corpus_sha256", "training_membership_sha256",
                "training_origin_count", "validation_membership_sha256",
                "validation_origin_count",
                "run_contract_sha256", "proposal_policy", "dpace_alpha",
                "data_order_seed",
                "data_order_derivation", "target_architecture",
                "target_embedding_width", "target_vocabulary",
                "target_output_head", "checkpoint_selection_metric",
                "validation_loss_contract", "validation_aggregation",
                "validation_cadence",
                "minimum_validation_source_clusters_per_workload",
                "root_only_interleave_algorithm",
                "root_only_interleave_clusters_per_sampler",
                "root_only_origin_shuffle_buffer",
                "physical_validation_batch_size",
            ))
        except KeyError as error:
            raise TrainingError(f"{description} identity is incomplete") from error
        if actual != expected:
            raise TrainingError(f"{description} identity is stale")

    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError(f"completed receipt is malformed: {receipt_path}") from error
        if not isinstance(receipt, dict):
            raise TrainingError(f"completed receipt is malformed: {receipt_path}")
        validate_identity(receipt, "completed receipt")
        learning_curve = receipt.get("learning_curve")
        best_filename = receipt.get("best_checkpoint_file")
        best_digest = receipt.get("best_checkpoint_sha256")
        if (receipt.get("schema") != "local-gguf-engine/training-result/v2"
                or receipt.get("parameters") != parameters
                or receipt.get("training_membership") != _membership_report(training_membership)
                or receipt.get("validation_membership") != _membership_report(validation_membership)
                or not isinstance(learning_curve, list)
                or not checkpoint_path.is_file()
                or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)):
            raise TrainingError(f"completed receipt/checkpoint integrity mismatch: {receipt_path}")
        _validate_scheduler_summary(
            receipt, learning_curve, maximum_epochs * updates_per_epoch,
            optimizer_contract,
        )
        _completion_cursor_fields(
            receipt.get("terminal_epoch"),
            receipt.get("terminal_origins_in_epoch"),
            receipt.get("update_step"), receipt.get("stop_requested"),
            scale, accumulation, maximum_epochs,
        )
        _validate_best_checkpoint_files(
            output_directory, cell.identifier, maximum_validation_events,
            len(learning_curve), best_filename,
        )
        best_path, _ = _resolve_best_checkpoint_path(
            output_directory, cell.identifier, best_filename, maximum_validation_events
        )
        _load_best_checkpoint(
            best_path, torch.device("cpu"), best_digest,
            receipt.get("best_checkpoint_selection_value"), best_filename,
            validate_identity,
        )
        return receipt

    # Publication is two atomic renames: the immutable final checkpoint first,
    # then its small receipt. Recover only this precisely identifiable crash
    # window; every other partial state remains closed for inspection.
    if checkpoint_path.exists():
        completed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(completed, dict):
            raise TrainingError(f"completed checkpoint recovery state is malformed: {checkpoint_path}")
        validate_identity(completed, "completed checkpoint")
        if (completed.get("schema") != "local-gguf-engine/training-checkpoint/v2"
                or not isinstance(completed.get("learning_curve"), list)
                or not isinstance(completed.get("environment"), dict)):
            raise TrainingError(f"completed checkpoint is not publication-ready: {checkpoint_path}")
        _validate_scheduler_summary(
            completed, completed["learning_curve"], maximum_epochs * updates_per_epoch,
            optimizer_contract,
        )
        completed_terminal_fields = _completion_cursor_fields(
            completed.get("terminal_epoch"),
            completed.get("terminal_origins_in_epoch"),
            completed.get("update_step"), completed.get("stop_requested"),
            scale, accumulation, maximum_epochs,
        )
        completed_best_filename = completed.get("best_checkpoint_file")
        completed_best_digest = completed.get("best_checkpoint_sha256")
        _validate_best_checkpoint_files(
            output_directory, cell.identifier, maximum_validation_events,
            len(completed["learning_curve"]), completed_best_filename,
        )
        completed_best_path, _ = _resolve_best_checkpoint_path(
            output_directory, cell.identifier, completed_best_filename,
            maximum_validation_events,
        )
        _load_best_checkpoint(
            completed_best_path, torch.device("cpu"), completed_best_digest,
            completed.get("best_checkpoint_selection_value"),
            completed_best_filename, validate_identity,
        )
        receipt = {
            "schema": "local-gguf-engine/training-result/v2",
            "cell_id": cell.identifier, "unit_id": cell.unit_id, "family": cell.family,
            "implementation": cell.implementation, "parameters": parameters,
            "training_seed": cell.seed, "scale": scale,
            "registry_sha256": registry.digest, "training_code_sha256": code_digest,
            "source_manifest_sha256": source_manifest_digest,
            "trace_corpus_sha256": trace_corpus_digest,
            **membership_identity,
            **run_identity,
            "model_sha256": model_digest, "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_checkpoint_file": completed_best_filename,
            "best_checkpoint_sha256": completed_best_digest,
            "restarts": completed["restarts"],
            "update_step": completed["update_step"],
            **completed_terminal_fields,
            "scheduler_plan": completed["scheduler_plan"],
            "best_checkpoint_selection_value": completed[
                "best_checkpoint_selection_value"
            ],
            "learning_curve": completed["learning_curve"],
            "environment": completed["environment"],
            "training_membership": _membership_report(training_membership),
            "validation_membership": _membership_report(validation_membership),
        }
        _atomic_json(receipt_path, receipt)
        return receipt

    _progress(
        "training_start", cell_id=cell.identifier, family=cell.family,
        scale=scale, horizon=horizon, device=str(device),
        run_contract_sha256=run_contract.digest,
        proposal_policy=cell_contract.proposal_policy,
        dpace_alpha=run_dpace_alpha,
        data_order_seed=run_contract.data_order_seed,
        checkpoint_selection_metric=run_contract.checkpoint_metric,
        logical_batch_size=logical_batch_size,
        physical_training_batch_size=physical_training_batch_size,
        physical_validation_batch_size=physical_validation_batch_size,
        training_membership_sha256=training_membership.sha256,
        training_origins=len(training_membership.references),
        validation_membership_sha256=validation_membership.sha256,
        validation_origins=len(validation_membership.references),
    )
    seed = cell.seed
    numpy_seed = _derived_numpy_seed(seed)
    random.seed(seed)
    numpy.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    weights = _objective_weights(registry, run_dpace_alpha)
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "numpy": numpy.__version__, "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": str(device),
        "python_seed": seed, "torch_seed": seed, "numpy_seed": numpy_seed,
        "data_order_seed": run_contract.data_order_seed,
        "data_order_derivation": run_contract.data_order_derivation,
        "target_architecture": run_contract.target_architecture,
        "target_embedding_width": run_contract.target_embedding_width,
        "target_vocabulary": run_contract.target_vocabulary,
        "target_output_head": run_contract.target_output_head,
        "proposal_policy": cell_contract.proposal_policy,
        "dpace_alpha": run_dpace_alpha,
        "validation_aggregation": run_contract.validation_aggregation,
        "validation_cadence": run_contract.validation_cadence,
        "root_only_interleave_algorithm": (
            run_contract.root_only_interleave_algorithm
        ),
        "root_only_interleave_clusters_per_sampler": (
            run_contract.root_only_interleave_clusters_per_sampler
        ),
        "root_only_origin_shuffle_buffer": (
            run_contract.root_only_origin_shuffle_buffer
        ),
        "scheduler_semantics": _SCHEDULER_SEMANTICS,
        "logical_batch_size": logical_batch_size,
        "physical_training_batch_size": physical_training_batch_size,
        "physical_validation_batch_size": physical_validation_batch_size,
        "gradient_accumulation_origins": accumulation,
    }

    with GGUFFile(model_path) as gguf:
        if gguf.sha256() != model_digest:
            raise TrainingError(
                "training GGUF mapping differs from the authenticated trace model"
            )
        token_array, norm_array, epsilon, softcap = _validated_active_gemma_head(
            gguf, run_contract
        )
        target_width = token_array.shape[1]
        vocabulary = token_array.shape[0]
        # GGUF arrays are read-only mmap views. torch.from_numpy would retain
        # that non-writable storage long enough to emit an undefined-behavior
        # warning before the device copy. torch.tensor owns the copy from the
        # outset and can transfer it directly to the selected device.
        token_table = _owned_tensor(token_array, device)
        norm_weight = _owned_tensor(norm_array, device)
        model = make_model(
            cell.implementation, parameters, target_width, vocabulary, token_table
        ).to(device)
        head = GemmaTargetHead(token_table, norm_weight, epsilon, softcap).to(device)
        total_updates = maximum_epochs * updates_per_epoch
        optimizer, scheduler, scheduler_plan = _make_optimizer_and_scheduler(
            model.parameters(), 0, 0, total_updates, optimizer_contract
        )
        start_epoch = origins_in_epoch = update_step = 0
        best = -math.inf
        significant_reference = -math.inf
        best_filename: str | None = None
        best_digest: str | None = None
        patience = restarts = 0
        stop_requested = False
        learning_curve: list[dict[str, object]] = []
        if state_path.exists():
            checkpoint = torch.load(state_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict):
                raise TrainingError(f"checkpoint state is malformed: {state_path}")
            validate_identity(checkpoint, "checkpoint state")
            if (checkpoint.get("schema") != "local-gguf-engine/training-state/v2"
                    or checkpoint.get("environment") != environment):
                raise TrainingError("checkpoint numerical environment changed")
            model.load_state_dict(checkpoint["model"])
            restarts = int(checkpoint["restarts"])
            start_epoch, origins_in_epoch = _validate_training_cursor(
                checkpoint.get("epoch"), checkpoint.get("origins_in_epoch"),
                checkpoint.get("update_step"), scale, accumulation,
                maximum_epochs,
            )
            update_step = checkpoint["update_step"]
            best = float(checkpoint["best_checkpoint_selection_value"])
            patience = int(checkpoint["patience"])
            significant_reference = float(
                checkpoint["significant_checkpoint_selection_value"]
            )
            best_filename = checkpoint.get("best_checkpoint_file")
            best_digest = checkpoint.get("best_checkpoint_sha256")
            learning_curve = checkpoint["learning_curve"]
            if (type(checkpoint.get("stop_requested")) is not bool
                    or type(checkpoint.get("training_complete")) is not bool):
                raise TrainingError("checkpoint stop/completion provenance is malformed")
            stop_requested = checkpoint["stop_requested"]
            if checkpoint["training_complete"]:
                raise TrainingError(f"completed progress state lacks final checkpoint: {state_path}")
            if (patience < 0
                    or not 0 <= restarts <= int(optimizer_contract["maximum_restarts"])
                    or not math.isfinite(best)
                    or not math.isfinite(significant_reference)
                    or not isinstance(learning_curve, list)):
                raise TrainingError(f"checkpoint progress state is invalid: {state_path}")
            _, _, expected_scheduler_plan = _validate_scheduler_summary(
                checkpoint, learning_curve, total_updates, optimizer_contract
            )
            scheduler_start = int(expected_scheduler_plan["global_start_update"])
            scheduler_state = checkpoint.get("scheduler")
            if (not isinstance(scheduler_state, dict)
                    or scheduler_state.get("last_epoch") != update_step - scheduler_start):
                raise TrainingError("checkpoint scheduler state is stale or inconsistent")
            optimizer, scheduler, scheduler_plan = _make_optimizer_and_scheduler(
                model.parameters(), restarts, scheduler_start,
                total_updates, optimizer_contract,
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(scheduler_state)
            _validate_best_checkpoint_files(
                output_directory, cell.identifier, maximum_validation_events,
                len(learning_curve), best_filename,
            )
            best_path, _ = _resolve_best_checkpoint_path(
                output_directory, cell.identifier, best_filename,
                maximum_validation_events,
            )
            _load_best_checkpoint(
                best_path, device, best_digest, best, best_filename, validate_identity
            )
            _restore_rng_state(checkpoint.get("rng_state"), device)
        else:
            _validate_best_checkpoint_files(
                output_directory, cell.identifier, maximum_validation_events, 0, None
            )
        terminal_epoch = start_epoch
        terminal_origins_in_epoch = origins_in_epoch
        terminal_stop_requested = stop_requested

        def checkpoint_state(
            epoch: int, origins: int, *, requested_stop: bool = False,
            training_complete: bool = False,
        ) -> dict[str, object]:
            return {
                "schema": "local-gguf-engine/training-state/v2",
                "registry_sha256": registry.digest, "cell_id": cell.identifier,
                "model_sha256": model_digest, "training_code_sha256": code_digest,
                "source_manifest_sha256": source_manifest_digest,
                "trace_corpus_sha256": trace_corpus_digest,
                **membership_identity,
                **run_identity,
                "scale": scale, "epoch": epoch, "origins_in_epoch": origins,
                "update_step": update_step,
                "best_checkpoint_selection_value": best,
                "patience": patience,
                "significant_checkpoint_selection_value": significant_reference,
                "best_checkpoint_file": best_filename,
                "best_checkpoint_sha256": best_digest,
                "restarts": restarts, "learning_curve": learning_curve,
                "scheduler_plan": scheduler_plan,
                "stop_requested": requested_stop,
                "training_complete": training_complete,
                "environment": environment,
                "rng_state": _rng_state(device),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

        def save_progress(
            epoch: int, origins: int, *, requested_stop: bool = False,
        ) -> None:
            nonlocal terminal_epoch, terminal_origins_in_epoch
            nonlocal terminal_stop_requested
            _validate_training_cursor(
                epoch, origins, update_step, scale, accumulation, maximum_epochs
            )
            if type(requested_stop) is not bool:
                raise TrainingError("checkpoint stop provenance is malformed")
            _save_checkpoint(
                state_path,
                checkpoint_state(epoch, origins, requested_stop=requested_stop),
            )
            terminal_epoch = epoch
            terminal_origins_in_epoch = origins
            terminal_stop_requested = requested_stop

        def validate(epoch: int, origins: int) -> bool:
            nonlocal best, significant_reference, best_filename, best_digest
            nonlocal patience, restarts, optimizer, scheduler, scheduler_plan
            validation_stream = TraceWindowStream(
                traces, tap_set, horizon, registry.digest,
                VALIDATION_STREAM_SEED, None,
                shuffle_buffer=run_contract.root_only_origin_shuffle_buffer,
                partition="validation",
                anchor_batch_size=physical_validation_batch_size,
                context_mode=context_mode,
                catalog=catalog, membership=validation_membership,
                root_only_interleave_clusters_per_sampler=(
                    run_contract.root_only_interleave_clusters_per_sampler
                ),
            )
            validation = _evaluate(
                model, head, validation_stream, cell, weights, device, epoch,
                workload_weights=registry.raw["workload_weights"],
                proposal_policy=cell_contract.proposal_policy,
                checkpoint_metric=run_contract.checkpoint_metric,
                validation_loss_contract=run_contract.validation_loss,
                validation_aggregation=run_contract.validation_aggregation,
                required_generation_seeds=tuple(
                    int(seed) for seed in registry.raw["target_generation_seeds"]
                ),
                precision=str(optimizer_contract["precision"]),
            )
            validation["epoch"] = epoch + 1
            validation["epoch_origin_count"] = origins
            validation["optimizer_updates"] = update_step
            value = float(validation["checkpoint_selection_value"])
            (best, significant_reference, patience, new_best,
             significant_improvement) = _advance_convergence(
                value, best, significant_reference, patience,
                float(optimizer_contract["minimum_relative_improvement"]),
                maximize=True,
            )
            if new_best:
                candidate_filename = _best_checkpoint_filename(
                    cell.identifier, len(learning_curve), maximum_validation_events
                )
                candidate_path = output_directory / candidate_filename
                candidate_digest = _publish_best_checkpoint(candidate_path, {
                    "schema": "local-gguf-engine/training-best/v2",
                    "registry_sha256": registry.digest, "cell_id": cell.identifier,
                    "model_sha256": model_digest, "training_code_sha256": code_digest,
                    "source_manifest_sha256": source_manifest_digest,
                    "trace_corpus_sha256": trace_corpus_digest,
                    **membership_identity,
                    **run_identity,
                    "scale": scale, "checkpoint_selection_value": value,
                    "validation_diagnostic_loss": validation["mean_loss"],
                    "best_checkpoint_file": candidate_filename,
                    "validation_index": len(learning_curve),
                    "model": model.state_dict(),
                })
                best_filename = candidate_filename
                best_digest = candidate_digest
            completed_epochs = epoch + (1 if origins == scale else 0)
            restart_performed = False
            stop = False
            if (completed_epochs >= int(optimizer_contract["minimum_epochs"])
                    and patience >= int(optimizer_contract["patience_intervals"])):
                if (restarts >= int(optimizer_contract["maximum_restarts"])
                        or update_step >= total_updates):
                    stop = True
                best_path, _ = _resolve_best_checkpoint_path(
                    output_directory, cell.identifier, best_filename,
                    maximum_validation_events,
                )
                best_checkpoint = _load_best_checkpoint(
                    best_path, device, best_digest, best, best_filename, validate_identity
                )
                model.load_state_dict(best_checkpoint["model"])
                if not stop:
                    restarts += 1
                    patience = 0
                    significant_reference = best
                    optimizer, scheduler, scheduler_plan = _make_optimizer_and_scheduler(
                        model.parameters(), restarts, update_step,
                        total_updates, optimizer_contract,
                    )
                    restart_performed = True
            validation["best_checkpoint_selection_value"] = best
            validation["significant_checkpoint_selection_value"] = significant_reference
            validation["significant_improvement"] = significant_improvement
            validation["patience"] = patience
            validation["restarts"] = restarts
            validation["restart_performed"] = restart_performed
            validation["stop_requested"] = stop
            validation["learning_rate"] = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            learning_curve.append(validation)
            _progress(
                "validation", cell_id=cell.identifier, epoch=epoch + 1,
                epoch_origins=origins, optimizer_updates=update_step,
                checkpoint_selection_metric=run_contract.checkpoint_metric,
                checkpoint_selection_value=value,
                best_checkpoint_selection_value=best,
                validation_diagnostic_loss=validation["mean_loss"],
                patience=patience, restarts=restarts, stop_requested=stop,
                restart_performed=restart_performed,
            )
            save_progress(epoch, origins, requested_stop=stop)
            return stop

        stopped = stop_requested
        for epoch in range(start_epoch, maximum_epochs):
            if stopped:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stream = TraceWindowStream(
                traces, tap_set, horizon, registry.digest,
                run_contract.data_order_seed ^ epoch, scale, partition="train",
                shuffle_buffer=run_contract.root_only_origin_shuffle_buffer,
                anchor_batch_size=physical_training_batch_size,
                context_mode=context_mode,
                catalog=catalog, membership=training_membership,
                root_only_interleave_clusters_per_sampler=(
                    run_contract.root_only_interleave_clusters_per_sampler
                ),
            )
            examples = 0
            pending_origins = 0
            for raw in stream:
                offset = 0
                while offset < raw.origin_count and examples < scale:
                    boundary_room = accumulation - (examples % accumulation)
                    count = min(boundary_room, raw.origin_count - offset, scale - examples)
                    if count <= 0:
                        raise TrainingError("training stream exceeded the frozen scale")
                    next_examples = examples + count
                    if examples < origins_in_epoch:
                        if next_examples > origins_in_epoch:
                            raise TrainingError("checkpoint resumes inside an accumulation slice")
                        examples = next_examples
                        offset += count
                        continue
                    example = _to_device(_slice_example(raw, offset, offset + count), device)
                    with _autocast(device, str(optimizer_contract["precision"])):
                        loss, _, _ = _loss(
                            model, head, example, cell, weights, epoch,
                            proposal_policy=cell_contract.proposal_policy,
                        )
                    (loss * count).backward()
                    examples = next_examples
                    offset += count
                    pending_origins += count
                    if pending_origins == accumulation or examples == scale:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(pending_origins)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(optimizer_contract["global_norm_clip"]),
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        update_step += 1
                        pending_origins = 0
                        if update_step % validation_interval == 0 and validate(epoch, examples):
                            stopped = True
                            break
                if stopped or examples == scale:
                    break
            if examples != scale and not stopped:
                raise TrainingError(f"training stream ended at {examples}; frozen scale is {scale}")
            if stopped:
                break
            # Patience advances only at the declared fixed update interval.
            # Validate once more at the final update when it is not itself an
            # interval boundary; ordinary epoch boundaries do not create an
            # extra, shorter patience interval.
            if (epoch + 1 == maximum_epochs
                    and (not learning_curve
                         or learning_curve[-1]["optimizer_updates"] != update_step)):
                if validate(epoch, examples):
                    break
            origins_in_epoch = 0
            save_progress(epoch + 1, 0)

        if best_filename is None or best_digest is None:
            raise TrainingError("training completed without a validation checkpoint")
        best_path, _ = _resolve_best_checkpoint_path(
            output_directory, cell.identifier, best_filename, maximum_validation_events
        )
        best_checkpoint = _load_best_checkpoint(
            best_path, device, best_digest, best, best_filename, validate_identity
        )
        model.load_state_dict(best_checkpoint["model"])
        terminal_fields = _completion_cursor_fields(
            terminal_epoch, terminal_origins_in_epoch, update_step,
            terminal_stop_requested, scale, accumulation, maximum_epochs,
        )
        _save_checkpoint(checkpoint_path, {
            "schema": "local-gguf-engine/training-checkpoint/v2",
            "registry_sha256": registry.digest, "cell_id": cell.identifier,
            "model_sha256": model_digest, "training_code_sha256": code_digest,
            "source_manifest_sha256": source_manifest_digest,
            "trace_corpus_sha256": trace_corpus_digest,
            **membership_identity,
            **run_identity,
            "scale": scale,
            "best_checkpoint_selection_value": best_checkpoint[
                "checkpoint_selection_value"
            ],
            "best_checkpoint_file": best_filename,
            "best_checkpoint_sha256": best_digest,
            "restarts": restarts,
            "update_step": update_step,
            **terminal_fields,
            "scheduler_plan": scheduler_plan,
            "learning_curve": learning_curve, "environment": environment,
            "model": model.state_dict(),
        })

        # Marks the final checkpoint as receipt-recoverable. This state rename
        # follows the final-checkpoint rename, so neither order can publish a
        # receipt for incomplete weights.
        _save_checkpoint(
            state_path,
            checkpoint_state(
                terminal_epoch, terminal_origins_in_epoch,
                requested_stop=terminal_stop_requested,
                training_complete=True,
            ),
        )

        receipt = {
            "schema": "local-gguf-engine/training-result/v2",
            "cell_id": cell.identifier, "unit_id": cell.unit_id, "family": cell.family,
            "implementation": cell.implementation, "parameters": parameters,
            "training_seed": cell.seed, "scale": scale,
            "registry_sha256": registry.digest, "training_code_sha256": code_digest,
            "source_manifest_sha256": source_manifest_digest,
            "trace_corpus_sha256": trace_corpus_digest,
            **membership_identity,
            **run_identity,
            "model_sha256": model_digest, "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_checkpoint_file": best_filename,
            "best_checkpoint_sha256": best_digest,
            "restarts": restarts,
            "update_step": update_step,
            **terminal_fields,
            "scheduler_plan": scheduler_plan,
            "best_checkpoint_selection_value": best,
            "learning_curve": learning_curve,
            "environment": environment,
            "training_membership": _membership_report(training_membership),
            "validation_membership": _membership_report(validation_membership),
        }
        _atomic_json(receipt_path, receipt)
        _progress(
            "training_complete", cell_id=cell.identifier,
            checkpoint_sha256=receipt["checkpoint_sha256"],
            checkpoint_selection_metric=run_contract.checkpoint_metric,
            best_checkpoint_selection_value=best,
        )
        return receipt


def train_cells(
    root: Path,
    registry: Registry,
    traces: Path,
    output: Path,
    model: Path,
    scale: int,
    requested_families: set[str],
    requested_units: set[str],
    requested_cells: set[str],
    device_name: str,
    allow_cpu_diagnostic: bool,
    split_receipt: Path,
    auditor: Path | None,
    run_contract_path: Path,
) -> list[dict[str, object]]:
    if scale not in registry.raw["scales"]:
        raise TrainingError("scale is not one of the frozen geometric origin counts")
    _validated_optimizer_contract(registry)
    known_units = {str(unit["id"]) for unit in registry.raw["training_units"]}
    unknown_units = requested_units - known_units
    if unknown_units:
        raise TrainingError(f"unknown training unit(s): {sorted(unknown_units)}")
    all_cells = list(registry.cells())
    run_contract = load_run_contract(run_contract_path, registry)
    cells_by_id = {cell.identifier: cell for cell in all_cells}
    unknown_contract_cells = {
        cell.cell_id for cell in run_contract.cells
    } - set(cells_by_id)
    if unknown_contract_cells:
        raise TrainingError(
            f"run contract contains unknown training cell(s): {sorted(unknown_contract_cells)}"
        )
    unknown = requested_cells - {cell.identifier for cell in all_cells}
    if unknown:
        raise TrainingError(f"unknown training cell(s): {sorted(unknown)}")
    selected_cells = [
        cell for cell in all_cells
        if (not requested_families or cell.family in requested_families)
        and (not requested_units or cell.unit_id in requested_units)
        and (not requested_cells or cell.identifier in requested_cells)
    ]
    if not selected_cells:
        raise TrainingError("training selection contains no registry cells")
    selected_ids = {cell.identifier for cell in selected_cells}
    contracted_ids = {cell.cell_id for cell in run_contract.cells}
    if selected_ids != contracted_ids:
        raise TrainingError(
            "training selection must exactly equal the run-contract cells; "
            f"missing={sorted(contracted_ids - selected_ids)}, "
            f"unexpected={sorted(selected_ids - contracted_ids)}"
        )
    # Contract order is itself authenticated and is the execution order.
    cells = [cells_by_id[cell.cell_id] for cell in run_contract.cells]
    for cell in cells:
        _validate_training_implementation(cell)
    device = _device(device_name, allow_cpu_diagnostic)
    corpus, catalog = validate_trace_corpus_with_index(
        traces, registry, model, auditor, split_receipt
    )
    corpus_model_digest = corpus.get("model_sha256")
    if (not isinstance(corpus_model_digest, str) or len(corpus_model_digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in corpus_model_digest)):
        raise TrainingError("authenticated trace corpus has no valid model digest")
    workload_weights = registry.raw["workload_weights"]
    target_generation_seeds = tuple(
        int(seed) for seed in registry.raw["target_generation_seeds"]
    )
    membership_keys = {
        (
            _horizon(cell),
            tuple(str(value) for value in dict(cell.parameters).get(
                "tap_set", ("final",)
            )),
        )
        for cell in cells
    }
    memberships: dict[
        tuple[int, tuple[str, ...]],
        tuple[OriginMembership, OriginMembership],
    ] = {}
    for horizon, tap_set in sorted(membership_keys):
        training_membership = build_origin_membership(
            catalog, horizon, "train", scale, workload_weights,
            target_generation_seeds, tap_set,
        )
        validation_membership = build_origin_membership(
            catalog, horizon, "validation", None, workload_weights,
            target_generation_seeds, tap_set,
        )
        validate_minimum_workload_clusters(
            validation_membership,
            run_contract.minimum_validation_source_clusters_per_workload,
        )
        memberships[horizon, tap_set] = (
            training_membership, validation_membership
        )
    results = []
    for cell in cells:
        cell_tap_set = tuple(str(value) for value in dict(cell.parameters).get(
            "tap_set", ("final",)
        ))
        training_membership, validation_membership = memberships[
            _horizon(cell), cell_tap_set
        ]
        results.append(train_cell(
            root, registry, cell, traces, output, model, corpus_model_digest,
            scale, device,
            str(corpus["source_manifest_sha256"]), str(corpus["trace_corpus_sha256"]),
            catalog, training_membership, validation_membership,
            run_contract, run_contract.cell(cell.identifier),
        ))
    return results

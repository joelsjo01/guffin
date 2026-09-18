"""Strict loading and deterministic expansion of the finite training registry."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_FAMILIES = frozenset({"R", "D", "P", "L", "T", "S", "bundle"})
OBJECTIVE_PROBABILITY_COMPUTE = "f32_from_authenticated_f64"
OBJECTIVE_INTERPRETATION_LABEL = (
    "Gemma local complete interpretation; not a source-exact optimizer/loss claim"
)


class RegistryError(ValueError):
    pass


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"registry JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _json_float(encoded: str) -> float:
    value = float(encoded)
    if not math.isfinite(value):
        raise RegistryError("registry JSON contains a nonfinite number")
    return value


def _json_constant(encoded: str) -> object:
    raise RegistryError(f"registry JSON contains unsupported constant {encoded}")


def _finite_number(
    value: object, *, minimum: float, maximum: float | None = None,
    minimum_inclusive: bool = True, maximum_inclusive: bool = True,
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    converted = float(value)
    if not math.isfinite(converted):
        return False
    if converted < minimum or (not minimum_inclusive and converted == minimum):
        return False
    if maximum is not None and (
            converted > maximum or (not maximum_inclusive and converted == maximum)):
        return False
    return True


def _integer(value: object, *, minimum: int, maximum: int | None = None) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum and (maximum is None or value <= maximum))


def _label(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and "\0" not in value)


@dataclass(frozen=True)
class TrainingCell:
    unit_id: str
    family: str
    implementation: str
    parameters: tuple[tuple[str, object], ...]
    seed: int

    @property
    def identifier(self) -> str:
        encoded = json.dumps(
            [self.unit_id, self.parameters, self.seed], separators=(",", ":"), ensure_ascii=True
        ).encode()
        return f"{self.unit_id}-{hashlib.sha256(encoded).hexdigest()[:20]}"

    def parameter(self, name: str, default: object = None) -> object:
        return dict(self.parameters).get(name, default)


@dataclass(frozen=True)
class Registry:
    raw: dict[str, object]
    digest: str
    training_seeds: tuple[int, ...]

    def cells(self) -> Iterator[TrainingCell]:
        for unit in self.raw["training_units"]:
            if unit["status"] != "train":
                continue
            if "cells" in unit:
                combinations = (
                    tuple(sorted(cell.items())) for cell in unit["cells"]
                )
            else:
                axes = dict(unit.get("axes", {}))
                if unit.get("tap_sets"):
                    axes["tap_set"] = unit["tap_sets"]
                names = tuple(sorted(axes))
                values = tuple(axes[name] for name in names)
                combinations = (
                    tuple(zip(names, combination))
                    for combination in itertools.product(*values)
                )
            for parameters in combinations:
                for seed in self.training_seeds:
                    yield TrainingCell(
                        unit_id=unit["id"],
                        family=unit["family"],
                        implementation=unit["implementation"],
                        parameters=parameters,
                        seed=seed,
                    )


def load_registry(path: Path) -> Registry:
    payload = path.read_bytes()
    try:
        raw = json.loads(
            payload, object_pairs_hook=_json_object,
            parse_float=_json_float, parse_constant=_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RegistryError(f"invalid registry JSON: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != "local-gguf-engine/phase5-training-registry/v1":
        raise RegistryError("unsupported registry schema")
    seeds = raw.get("training_seeds")
    if (not isinstance(seeds, list) or len(seeds) != 3
            or any(not _integer(seed, minimum=0, maximum=2**64 - 1) for seed in seeds)
            or len(set(seeds)) != 3):
        raise RegistryError("registry must contain exactly three distinct uint64 training seeds")
    target_seeds = raw.get("target_generation_seeds")
    if (not isinstance(target_seeds, list) or len(target_seeds) != 5
            or any(not _integer(seed, minimum=0, maximum=2**64 - 1)
                   for seed in target_seeds)
            or len(set(target_seeds)) != 5):
        raise RegistryError("registry must contain exactly five distinct target generation seeds")
    expected_sampler = {
        "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
        "repetition_penalty": 1.0, "repetition_window": 64,
    }
    sampler = raw.get("target_generation_sampler")
    if (not isinstance(sampler, dict) or set(sampler) != set(expected_sampler)
            or any(type(sampler[name]) is not type(expected)  # noqa: E721
                   or sampler[name] != expected
                   for name, expected in expected_sampler.items())):
        raise RegistryError("target generation sampler contract changed")
    workload_weights = raw.get("workload_weights")
    required_workloads = {
        "instruction_chat", "factual_grounded_qa", "code", "mathematics_reasoning",
        "summarization_translation", "structured_tool", "rare_non_english",
        "repeated_retrieval",
    }
    if (not isinstance(workload_weights, dict) or set(workload_weights) != required_workloads
            or any(not _finite_number(value, minimum=0.0, minimum_inclusive=False)
                   for value in workload_weights.values())
            or abs(sum(float(value) for value in workload_weights.values()) - 1.0) > 1e-12):
        raise RegistryError("workload weights are incomplete or do not sum to one")
    scales = raw.get("scales")
    expected_scales = [100000, 250000, 500000, 1000000, 2400000]
    if (scales != expected_scales or not isinstance(scales, list)
            or any(type(value) is not int for value in scales)):
        raise RegistryError("registry geometric scales changed")
    optimizer = raw.get("default_transfer_optimizer")
    required_optimizer = {
        "algorithm", "learning_rate", "betas", "epsilon", "weight_decay",
        "global_norm_clip", "warmup_fraction", "schedule", "maximum_epochs",
        "minimum_epochs", "validation_interval_steps", "patience_intervals",
        "minimum_relative_improvement", "maximum_restarts",
        "restart_learning_rate_factor", "precision", "micro_batch",
        "gradient_accumulation",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != required_optimizer:
        raise RegistryError("transfer optimizer/convergence contract is incomplete")
    if (optimizer["algorithm"] != "AdamW" or optimizer["schedule"] != "cosine"
            or optimizer["precision"] != "bf16"):
        raise RegistryError("transfer optimizer implementation contract changed")
    finite_optimizer_fields = (
        ("learning_rate", 0.0, None, False, True),
        ("epsilon", 0.0, None, False, True),
        ("weight_decay", 0.0, None, True, True),
        ("global_norm_clip", 0.0, None, False, True),
        ("warmup_fraction", 0.0, 1.0, True, False),
        ("minimum_relative_improvement", 0.0, 1.0, False, True),
        ("restart_learning_rate_factor", 0.0, 1.0, False, True),
    )
    if any(not _finite_number(
            optimizer[name], minimum=minimum, maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
    ) for name, minimum, maximum, minimum_inclusive, maximum_inclusive
            in finite_optimizer_fields):
        raise RegistryError("transfer optimizer numeric contract is invalid")
    betas = optimizer["betas"]
    if (not isinstance(betas, list) or len(betas) != 2
            or any(not _finite_number(value, minimum=0.0, maximum=1.0,
                                      maximum_inclusive=False) for value in betas)):
        raise RegistryError("transfer optimizer beta contract is invalid")
    positive_integer_fields = (
        "maximum_epochs", "minimum_epochs", "validation_interval_steps",
        "patience_intervals", "micro_batch", "gradient_accumulation",
    )
    if (any(not _integer(optimizer[name], minimum=1)
            for name in positive_integer_fields)
            or not _integer(optimizer["maximum_restarts"], minimum=0)
            or optimizer["minimum_epochs"] > optimizer["maximum_epochs"]):
        raise RegistryError("transfer optimizer integer contract is invalid")
    if (optimizer["micro_batch"] != 1
            or optimizer["gradient_accumulation"] % optimizer["micro_batch"] != 0):
        raise RegistryError("transfer batching contract is invalid")
    expected_validation = {
        "digest_u64_modulus": 10, "validation_residue": 0
    }
    validation = raw.get("development_cluster_validation")
    if (not isinstance(validation, dict) or set(validation) != set(expected_validation)
            or any(type(validation[name]) is not int or validation[name] != expected
                   for name, expected in expected_validation.items())):
        raise RegistryError("development source-cluster validation split changed")
    objective = raw.get("transfer_objective")
    objective_fields = {
        "forward_kl_weight", "hard_ce_weight", "common_draw_overlap_weight",
        "confidence_weight", "future_state_auxiliary_weight", "dpace_alpha",
        "target_probability_compute", "label",
    }
    if not isinstance(objective, dict) or set(objective) != objective_fields:
        raise RegistryError("transfer objective contract is incomplete")
    weight_fields = (
        "forward_kl_weight", "hard_ce_weight", "common_draw_overlap_weight",
        "confidence_weight", "future_state_auxiliary_weight",
    )
    if (any(not _finite_number(objective[name], minimum=0.0) for name in weight_fields)
            or not any(float(objective[name]) > 0.0 for name in weight_fields[:-1])
            or type(objective["future_state_auxiliary_weight"]) is not float
            or objective["future_state_auxiliary_weight"] != 0.0
            or not _finite_number(objective["dpace_alpha"], minimum=0.0, maximum=1.0)
            or objective["target_probability_compute"] != OBJECTIVE_PROBABILITY_COMPUTE
            or objective["label"] != OBJECTIVE_INTERPRETATION_LABEL):
        raise RegistryError("transfer objective or mandatory zero auxiliary arm changed")
    units = raw.get("training_units")
    if not isinstance(units, list) or not units:
        raise RegistryError("registry training_units are absent")
    identifiers: set[str] = set()
    families: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise RegistryError("training unit must be an object")
        identifier = unit.get("id")
        family = unit.get("family")
        implementation = unit.get("implementation")
        status = unit.get("status")
        if (not _label(identifier) or identifier in identifiers
                or not isinstance(family, str) or family not in REQUIRED_FAMILIES
                or not _label(implementation) or not isinstance(status, str)
                or status not in {"train", "reuse", "train_after_children"}):
            raise RegistryError("training unit identity/status is invalid")
        identifiers.add(identifier)
        families.add(family)
        axes = unit.get("axes", {})
        cells = unit.get("cells")
        if cells is not None and axes:
            raise RegistryError(f"training unit {identifier} mixes explicit cells and axes")
        if not isinstance(axes, dict) or any(
            not _label(name) or not isinstance(values, list) or not values
            for name, values in axes.items()
        ):
            raise RegistryError(f"training unit {identifier} has an invalid finite axis")
        if cells is not None and (not isinstance(cells, list) or not cells
                or any(not isinstance(cell, dict) or not cell for cell in cells)):
            raise RegistryError(f"training unit {identifier} has invalid explicit cells")
        tap_sets = unit.get("tap_sets")
        if tap_sets is not None and (
                not isinstance(tap_sets, list) or not tap_sets
                or any(not isinstance(tap_set, list) or not tap_set
                       or any(not _label(tap) for tap in tap_set)
                       or len(set(tap_set)) != len(tap_set) for tap_set in tap_sets)):
            raise RegistryError(f"training unit {identifier} has invalid target tap sets")
    if families != REQUIRED_FAMILIES:
        raise RegistryError(f"registry family coverage differs: {sorted(families)}")
    blockers = raw.get("source_exact_blockers")
    if (not isinstance(blockers, dict) or not blockers
            or any(not _label(name) or not isinstance(entries, list) or not entries
                   or any(not _label(entry) for entry in entries)
                   or len(set(entries)) != len(entries)
                   for name, entries in blockers.items())):
        raise RegistryError("source-exact blocker ledger is absent")
    registry = Registry(raw=raw, digest=hashlib.sha256(payload).hexdigest(), training_seeds=tuple(seeds))
    cells = list(registry.cells())
    if not cells or len({cell.identifier for cell in cells}) != len(cells):
        raise RegistryError("expanded training-cell identities collide or are empty")
    for cell in cells:
        parameters = dict(cell.parameters)
        for name, value in parameters.items():
            if not _label(name) or isinstance(value, bool):
                raise RegistryError(f"training cell {cell.identifier} has an invalid parameter")
            if isinstance(value, int) and value <= 0:
                raise RegistryError(f"training cell {cell.identifier} has a nonpositive axis")
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0.0):
                raise RegistryError(f"training cell {cell.identifier} has a nonpositive axis")
            if isinstance(value, str) and not _label(value):
                raise RegistryError(f"training cell {cell.identifier} has an empty label")
            if isinstance(value, list) and (
                    not value or any(not _label(item) for item in value)
                    or len(set(value)) != len(value)):
                raise RegistryError(f"training cell {cell.identifier} has an invalid finite list")
            if not isinstance(value, (int, float, str, list)):
                raise RegistryError(f"training cell {cell.identifier} has an unsupported value")
        selector = parameters.get("selector")
        precision = parameters.get("selected_head_precision")
        if (not isinstance(selector, str)
                or selector not in {"full_vocabulary", "target_integrated_full_vocabulary"}):
            raise RegistryError(f"training cell {cell.identifier} has no explicit selector contract")
        if (not isinstance(precision, str)
                or precision not in {"f16", "q8_0", "q6_k", "q4_k"}):
            raise RegistryError(f"training cell {cell.identifier} has no selected-head precision")
        if cell.family == "S":
            if selector != "target_integrated_full_vocabulary" or precision != "q8_0":
                raise RegistryError("Arm S must bind its target-integrated production-Q8 head")
        elif selector == "target_integrated_full_vocabulary":
            raise RegistryError("only Arm S may bind a target-integrated selector")
    return registry

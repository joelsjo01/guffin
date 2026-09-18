"""Strict, content-addressed contracts for concrete Phase-5 training runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data_order import (
    ROOT_ONLY_INTERLEAVE_ALGORITHM,
    ROOT_ONLY_INTERLEAVE_CLUSTERS_PER_SAMPLER,
)
from .registry import Registry, TrainingCell
from .validation_stats import VALIDATION_AGGREGATION


class RunContractError(ValueError):
    pass


SOURCE_DPACE_ALPHA = 0.5


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunContractError(f"training run contract has duplicate key {key!r}")
        result[key] = value
    return result


def _unsupported_constant(value: str) -> object:
    raise RunContractError(
        f"training run contract contains unsupported constant {value!r}"
    )


@dataclass(frozen=True)
class RunCellContract:
    cell_id: str
    proposal_policy: str
    dpace_alpha: float | None


@dataclass(frozen=True)
class RunContract:
    digest: str
    registry_sha256: str
    data_order_seed: int
    data_order_derivation: str
    target_architecture: str
    target_embedding_width: int
    target_vocabulary: int
    target_output_head: str
    checkpoint_metric: str
    validation_loss: str
    validation_aggregation: str
    validation_cadence: str
    minimum_validation_source_clusters_per_workload: int
    root_only_interleave_algorithm: str
    root_only_interleave_clusters_per_sampler: int
    root_only_origin_shuffle_buffer: int
    root_only_validation_physical_batch_size: int
    cells: tuple[RunCellContract, ...]

    def cell(self, identifier: str) -> RunCellContract:
        matches = [cell for cell in self.cells if cell.cell_id == identifier]
        if len(matches) != 1:
            raise RunContractError(
                f"training cell {identifier} has no unique run contract"
            )
        return matches[0]


def _uses_dpace(cell: TrainingCell) -> bool:
    parameters = dict(cell.parameters)
    return str(parameters.get(
        "position_weighting", parameters.get("loss", "uniform")
    )) == "dpace"


def load_run_contract(path: Path, registry: Registry) -> RunContract:
    try:
        payload = path.read_bytes()
        raw = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_unsupported_constant,
        )
    except RunContractError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RunContractError(f"training run contract is unreadable: {path}") from error
    required = {
        "schema", "registry_sha256", "data_order_seed", "data_order_derivation",
        "target_architecture", "target_embedding_width", "target_vocabulary",
        "target_output_head",
        "checkpoint_metric", "validation_loss", "validation_aggregation",
        "validation_cadence",
        "minimum_validation_source_clusters_per_workload",
        "root_only_interleave_algorithm",
        "root_only_interleave_clusters_per_sampler",
        "root_only_origin_shuffle_buffer",
        "root_only_validation_physical_batch_size", "cells",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise RunContractError("training run contract fields are incomplete")
    if raw["schema"] != "local-gguf-engine/phase5-run-contract/v1":
        raise RunContractError("unsupported training run contract schema")
    registry_sha256 = registry.digest
    if (not isinstance(registry_sha256, str) or len(registry_sha256) != 64
            or any(character not in "0123456789abcdef" for character in registry_sha256)):
        raise RunContractError("training registry digest is malformed")
    if raw["registry_sha256"] != registry_sha256:
        raise RunContractError("training run contract binds another registry")
    data_order_seed = raw["data_order_seed"]
    if (not isinstance(data_order_seed, int) or isinstance(data_order_seed, bool)
            or not 0 <= data_order_seed < 2**64):
        raise RunContractError("training data-order seed is not an exact uint64")
    data_order_derivation = raw["data_order_derivation"]
    if data_order_derivation != "seed_xor_zero_based_epoch/v1":
        raise RunContractError("training data-order derivation is unsupported")
    target_architecture = raw["target_architecture"]
    target_embedding_width = raw["target_embedding_width"]
    target_vocabulary = raw["target_vocabulary"]
    target_output_head = raw["target_output_head"]
    if target_architecture != "gemma2":
        raise RunContractError("training target architecture is unsupported")
    if (not isinstance(target_embedding_width, int)
            or isinstance(target_embedding_width, bool)
            or target_embedding_width != 3584):
        raise RunContractError("training target embedding width is unsupported")
    if (not isinstance(target_vocabulary, int)
            or isinstance(target_vocabulary, bool)
            or target_vocabulary != 256000):
        raise RunContractError("training target vocabulary is unsupported")
    if target_output_head != "tied_token_embedding_f16":
        raise RunContractError("training target output head is unsupported")
    checkpoint_metric = raw["checkpoint_metric"]
    if checkpoint_metric != (
            "hierarchical_workload_weighted_exact_accepted_prefix/v1"):
        raise RunContractError("training checkpoint metric is unsupported")
    validation_loss = raw["validation_loss"]
    if validation_loss != "teacher_prefix_uniform_kl_ce_all_edges":
        raise RunContractError("training validation diagnostic is unsupported")
    validation_aggregation = raw["validation_aggregation"]
    if validation_aggregation != VALIDATION_AGGREGATION:
        raise RunContractError("training validation aggregation is unsupported")
    validation_cadence = raw["validation_cadence"]
    if validation_cadence != "fixed_optimizer_interval_plus_final_update/v1":
        raise RunContractError("training validation cadence is unsupported")
    minimum_validation_clusters = raw[
        "minimum_validation_source_clusters_per_workload"
    ]
    if (not isinstance(minimum_validation_clusters, int)
            or isinstance(minimum_validation_clusters, bool)
            or minimum_validation_clusters != 10):
        raise RunContractError(
            "training validation requires exactly 10 source clusters per workload"
        )
    root_only_interleave_algorithm = raw["root_only_interleave_algorithm"]
    if root_only_interleave_algorithm != ROOT_ONLY_INTERLEAVE_ALGORITHM:
        raise RunContractError("training root-only interleave algorithm is unsupported")
    root_only_interleave_clusters = raw[
        "root_only_interleave_clusters_per_sampler"
    ]
    if (not isinstance(root_only_interleave_clusters, int)
            or isinstance(root_only_interleave_clusters, bool)
            or root_only_interleave_clusters
            != ROOT_ONLY_INTERLEAVE_CLUSTERS_PER_SAMPLER):
        raise RunContractError(
            "training root-only interleave cluster width is unsupported"
        )
    root_only_shuffle_buffer = raw["root_only_origin_shuffle_buffer"]
    if (not isinstance(root_only_shuffle_buffer, int)
            or isinstance(root_only_shuffle_buffer, bool)
            or root_only_shuffle_buffer != 4096):
        raise RunContractError("training root-only shuffle buffer is unsupported")
    root_only_validation_batch = raw[
        "root_only_validation_physical_batch_size"
    ]
    if (not isinstance(root_only_validation_batch, int)
            or isinstance(root_only_validation_batch, bool)
            or root_only_validation_batch != 16):
        raise RunContractError(
            "training root-only validation physical batch is unsupported"
        )
    raw_cells = raw["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise RunContractError("training run contract has no cells")
    cells = []
    identifiers: set[str] = set()
    registry_cells = {cell.identifier: cell for cell in registry.cells()}
    for raw_cell in raw_cells:
        if (not isinstance(raw_cell, dict)
                or set(raw_cell) not in (
                    {"cell_id", "proposal_policy"},
                    {"cell_id", "proposal_policy", "dpace_alpha"},
                )):
            raise RunContractError("training run cell contract is malformed")
        identifier = raw_cell["cell_id"]
        policy = raw_cell["proposal_policy"]
        if (not isinstance(identifier, str) or not identifier
                or identifier in identifiers
                or identifier not in registry_cells
                or policy not in {"deterministic", "common_draw"}):
            raise RunContractError("training run cell identity/policy is invalid")
        uses_dpace = _uses_dpace(registry_cells[identifier])
        has_dpace_alpha = "dpace_alpha" in raw_cell
        if uses_dpace != has_dpace_alpha:
            raise RunContractError(
                "D-PACE cells require dpace_alpha and non-D-PACE cells forbid it"
            )
        dpace_alpha = None
        if has_dpace_alpha:
            raw_alpha = raw_cell["dpace_alpha"]
            if (not isinstance(raw_alpha, (int, float)) or isinstance(raw_alpha, bool)
                    or not math.isfinite(float(raw_alpha))
                    or float(raw_alpha) != SOURCE_DPACE_ALPHA):
                raise RunContractError(
                    "paper-supported D-PACE control requires dpace_alpha=0.5"
                )
            dpace_alpha = float(raw_alpha)
        identifiers.add(identifier)
        cells.append(RunCellContract(identifier, policy, dpace_alpha))
    return RunContract(
        digest=hashlib.sha256(payload).hexdigest(),
        registry_sha256=registry_sha256,
        data_order_seed=data_order_seed,
        data_order_derivation=data_order_derivation,
        target_architecture=target_architecture,
        target_embedding_width=target_embedding_width,
        target_vocabulary=target_vocabulary,
        target_output_head=target_output_head,
        checkpoint_metric=checkpoint_metric,
        validation_loss=validation_loss,
        validation_aggregation=validation_aggregation,
        validation_cadence=validation_cadence,
        minimum_validation_source_clusters_per_workload=(
            minimum_validation_clusters
        ),
        root_only_interleave_algorithm=root_only_interleave_algorithm,
        root_only_interleave_clusters_per_sampler=root_only_interleave_clusters,
        root_only_origin_shuffle_buffer=root_only_shuffle_buffer,
        root_only_validation_physical_batch_size=root_only_validation_batch,
        cells=tuple(cells),
    )

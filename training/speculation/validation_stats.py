"""Pure, deterministic hierarchical aggregation for held-out validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .manifest import WORKLOADS


class ValidationStatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationObservation:
    """The three per-origin validation measurements and their split identity."""

    workload: str
    source_cluster: str
    conversation: str
    generation_seed: int
    loss: float
    deterministic_prefix: int
    common_draw_prefix: int


_SCHEMA = "local-gguf-engine/phase5-hierarchical-validation/v1"
_METRIC_NAMES = ("loss", "deterministic_prefix", "common_draw_prefix")
VALIDATION_AGGREGATION = (
    "origin_mean_within_conversation_seed/equal_seeds_within_conversation/"
    "equal_conversations_within_source_cluster/equal_clusters_within_workload/"
    "frozen_workload_weights/v1"
)


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        decoded = bytes.fromhex(value)
        return len(decoded) == 32 and decoded != bytes(32)
    except ValueError:
        return False


def _mean_metrics(values: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    rows = tuple(values)
    if not rows:
        raise ValidationStatisticsError("validation aggregation encountered an empty group")
    return tuple(
        math.fsum(sorted(row[index] for row in rows)) / len(rows)
        for index in range(len(_METRIC_NAMES))
    )  # type: ignore[return-value]


def _metrics_json(values: tuple[float, float, float]) -> dict[str, float]:
    return {name: values[index] for index, name in enumerate(_METRIC_NAMES)}


def hierarchical_validation_summary(
    observations: Iterable[ValidationObservation],
    workload_weights: dict[str, object],
    required_generation_seeds: tuple[int, ...],
    horizon: int,
) -> dict[str, object]:
    """Aggregate without allowing prolific origins or clusters to dominate.

    The frozen hierarchy is origin mean within conversation/seed, equal seeds
    within conversation, equal conversations within source cluster, equal
    source clusters within workload, then the supplied workload weights.
    """
    if (not isinstance(horizon, int) or isinstance(horizon, bool)
            or horizon <= 0 or horizon > 23
            or not isinstance(required_generation_seeds, tuple)
            or not required_generation_seeds
            or any(type(seed) is not int or seed < 0 or seed >= 2**64
                   for seed in required_generation_seeds)
            or len(set(required_generation_seeds)) != len(required_generation_seeds)):
        raise ValidationStatisticsError("hierarchical validation contract is invalid")
    if (not isinstance(workload_weights, dict)
            or set(workload_weights) != set(WORKLOADS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) or float(value) <= 0.0
                   for value in workload_weights.values())
            or not math.isclose(
                math.fsum(float(value) for value in workload_weights.values()),
                1.0, rel_tol=0.0, abs_tol=1e-12,
            )):
        raise ValidationStatisticsError("hierarchical validation workload weights are invalid")

    rows = tuple(observations)
    if not rows:
        raise ValidationStatisticsError("hierarchical validation has no observations")
    required_seeds = tuple(sorted(required_generation_seeds))
    required_seed_set = set(required_seeds)
    unit_rows: dict[
        tuple[str, str, str, int], list[tuple[float, float, float]]
    ] = {}
    conversation_identities: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, ValidationObservation):
            raise ValidationStatisticsError("hierarchical validation observation is malformed")
        if (row.workload not in WORKLOADS
                or not _valid_digest(row.source_cluster)
                or not _valid_digest(row.conversation)
                or type(row.generation_seed) is not int
                or row.generation_seed not in required_seed_set
                or isinstance(row.loss, bool) or not isinstance(row.loss, (int, float))
                or not math.isfinite(float(row.loss)) or float(row.loss) < 0.0
                or type(row.deterministic_prefix) is not int
                or not 0 <= row.deterministic_prefix <= horizon
                or type(row.common_draw_prefix) is not int
                or not 0 <= row.common_draw_prefix <= horizon):
            raise ValidationStatisticsError("hierarchical validation observation is malformed")
        identity = (row.workload, row.source_cluster)
        previous_identity = conversation_identities.setdefault(row.conversation, identity)
        if previous_identity != identity:
            raise ValidationStatisticsError(
                "hierarchical validation conversation changes workload/cluster identity"
            )
        key = (
            row.workload, row.source_cluster, row.conversation, row.generation_seed
        )
        unit_rows.setdefault(key, []).append((
            float(row.loss), float(row.deterministic_prefix),
            float(row.common_draw_prefix),
        ))

    units = {
        key: _mean_metrics(values) for key, values in sorted(unit_rows.items())
    }
    conversation_units: dict[
        tuple[str, str, str], dict[int, tuple[float, float, float]]
    ] = {}
    for (workload, cluster, conversation, seed), metrics in units.items():
        conversation_units.setdefault((workload, cluster, conversation), {})[seed] = metrics
    incomplete = [
        (workload, cluster, conversation, tuple(sorted(seed_metrics)))
        for (workload, cluster, conversation), seed_metrics
        in sorted(conversation_units.items())
        if set(seed_metrics) != required_seed_set
    ]
    if incomplete:
        workload, cluster, conversation, present = incomplete[0]
        raise ValidationStatisticsError(
            "hierarchical validation conversation lacks the complete generation-seed product: "
            f"workload={workload}, cluster={cluster}, conversation={conversation}, "
            f"present={present}"
        )

    conversations = {
        key: _mean_metrics(seed_metrics[seed] for seed in required_seeds)
        for key, seed_metrics in sorted(conversation_units.items())
    }
    cluster_conversations: dict[
        tuple[str, str], list[tuple[float, float, float]]
    ] = {}
    for (workload, cluster, _conversation), metrics in conversations.items():
        cluster_conversations.setdefault((workload, cluster), []).append(metrics)
    clusters = {
        key: _mean_metrics(values)
        for key, values in sorted(cluster_conversations.items())
    }
    workload_clusters: dict[str, list[tuple[float, float, float]]] = {}
    for (workload, _cluster), metrics in clusters.items():
        workload_clusters.setdefault(workload, []).append(metrics)
    if set(workload_clusters) != set(WORKLOADS):
        raise ValidationStatisticsError(
            "hierarchical validation does not cover every frozen workload"
        )
    workloads = {
        workload: _mean_metrics(workload_clusters[workload])
        for workload, _code in sorted(WORKLOADS.items(), key=lambda item: item[1])
    }
    weighted = tuple(
        math.fsum(
            float(workload_weights[workload]) * workloads[workload][metric]
            for workload, _code in sorted(WORKLOADS.items(), key=lambda item: item[1])
        )
        for metric in range(len(_METRIC_NAMES))
    )

    workload_reports: dict[str, object] = {}
    for workload, _code in sorted(WORKLOADS.items(), key=lambda item: item[1]):
        workload_unit_keys = [key for key in units if key[0] == workload]
        workload_conversation_keys = [key for key in conversations if key[0] == workload]
        workload_cluster_keys = [key for key in clusters if key[0] == workload]
        seed_reports: dict[str, object] = {}
        for seed in required_seeds:
            per_cluster: list[tuple[float, float, float]] = []
            for _owner, cluster in workload_cluster_keys:
                per_conversation = [
                    units[workload, cluster, conversation, seed]
                    for owner, owned_cluster, conversation in workload_conversation_keys
                    if owner == workload and owned_cluster == cluster
                ]
                per_cluster.append(_mean_metrics(per_conversation))
            seed_unit_keys = [key for key in workload_unit_keys if key[3] == seed]
            seed_reports[str(seed)] = {
                "origin_count": sum(len(unit_rows[key]) for key in seed_unit_keys),
                "conversation_count": len(seed_unit_keys),
                "source_cluster_count": len(workload_cluster_keys),
                "metrics": _metrics_json(_mean_metrics(per_cluster)),
            }

        cluster_reports: dict[str, object] = {}
        for _owner, cluster in workload_cluster_keys:
            cluster_unit_keys = [
                key for key in workload_unit_keys if key[1] == cluster
            ]
            cluster_conversation_keys = [
                key for key in workload_conversation_keys if key[1] == cluster
            ]
            cluster_reports[cluster] = {
                "origin_count": sum(len(unit_rows[key]) for key in cluster_unit_keys),
                "conversation_seed_count": len(cluster_unit_keys),
                "conversation_count": len(cluster_conversation_keys),
                "metrics": _metrics_json(clusters[workload, cluster]),
            }
        workload_reports[workload] = {
            "weight": float(workload_weights[workload]),
            "origin_count": sum(len(unit_rows[key]) for key in workload_unit_keys),
            "conversation_seed_count": len(workload_unit_keys),
            "conversation_count": len(workload_conversation_keys),
            "source_cluster_count": len(workload_cluster_keys),
            "metrics": _metrics_json(workloads[workload]),
            "seeds": seed_reports,
            "clusters": cluster_reports,
        }

    return {
        "schema": _SCHEMA,
        "aggregation": VALIDATION_AGGREGATION,
        "horizon": horizon,
        "required_generation_seeds": list(required_seeds),
        "workload_weights": {
            workload: float(workload_weights[workload])
            for workload, _code in sorted(WORKLOADS.items(), key=lambda item: item[1])
        },
        "origin_count": len(rows),
        "conversation_seed_count": len(units),
        "conversation_count": len(conversations),
        "source_cluster_count": len({row.source_cluster for row in rows}),
        "origin_mean_metrics": _metrics_json(_mean_metrics(
            (float(row.loss), float(row.deterministic_prefix),
             float(row.common_draw_prefix))
            for row in rows
        )),
        "weighted_metrics": _metrics_json(weighted),
        "workloads": workload_reports,
    }

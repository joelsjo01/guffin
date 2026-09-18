"""Causal, source-aware streaming windows over immutable target traces."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy
import torch

from .corpus import IndexedTrace, TraceCorpusIndex
from .data_order import (
    ROOT_ONLY_INTERLEAVE_ALGORITHM,
    ROOT_ONLY_INTERLEAVE_CLUSTERS_PER_SAMPLER,
)
from .manifest import WORKLOADS
from .objectives import TargetSupport
from .trace import TraceFile, TraceRecord, discover_traces


class DatasetError(ValueError):
    pass


def _order_rank(seed: int, domain: bytes, *components: bytes) -> bytes:
    """Return a call-order-independent rank for the versioned epoch order."""
    digest = hashlib.sha256()
    digest.update(b"local-gguf-engine/phase5-root-interleave/v1\0")
    digest.update(seed.to_bytes(8, "little"))
    digest.update(len(domain).to_bytes(4, "little"))
    digest.update(domain)
    for component in components:
        digest.update(len(component).to_bytes(8, "little"))
        digest.update(component)
    return digest.digest()


def _sampler_bytes(sampler: tuple[object, ...]) -> bytes:
    if not isinstance(sampler, tuple) or len(sampler) != 7:
        raise DatasetError("root-only sampler identity is malformed")
    try:
        temperature, top_k, top_p, min_p, penalty, seed, window = sampler
        if (not isinstance(top_k, int) or isinstance(top_k, bool)
                or not isinstance(seed, int) or isinstance(seed, bool)
                or not isinstance(window, int) or isinstance(window, bool)
                or not 1 <= top_k <= 256 or not 0 <= seed < 2**64
                or not 0 <= window <= 64):
            raise TypeError("sampler integer field is malformed")
        scalars = tuple(float(value) for value in (
            temperature, top_p, min_p, penalty
        ))
        if (not all(math.isfinite(value) for value in scalars)
                or scalars[0] <= 0.0 or not 0.0 < scalars[1] <= 1.0
                or not 0.0 <= scalars[2] <= 1.0 or scalars[3] <= 0.0):
            raise ValueError("sampler floating field is malformed")
        return struct.pack(
            "<fQfffQQ", scalars[0], top_k, scalars[1],
            scalars[2], scalars[3], seed, window,
        )
    except (OverflowError, struct.error, TypeError, ValueError) as error:
        raise DatasetError("root-only sampler identity is malformed") from error


@dataclass
class TrainingExample:
    context_taps: torch.Tensor
    context_positions: torch.Tensor
    anchor_indices: torch.Tensor
    rollout_tokens: torch.Tensor
    labels: torch.Tensor
    target_support: TargetSupport
    target_selected_probabilities: torch.Tensor
    target_unit_draws: torch.Tensor
    source_clusters: tuple[str, ...]
    conversations: tuple[str, ...]
    sequence_ids: tuple[int, ...]
    workloads: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    target_positions: torch.Tensor
    sampler: tuple[object, ...]

    @property
    def origin_count(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class _Row:
    features: numpy.ndarray
    token: int
    label: int | None
    selection: object | None
    target_position: int
    source_cluster: bytes
    conversation: bytes
    sequence_id: int
    context_generation: int
    workload: int
    generation_seed: int


@dataclass(frozen=True)
class OriginReference:
    entry_index: int
    record_index: int


@dataclass(frozen=True)
class OriginMembership:
    """One immutable, content-addressed population for a training scale."""

    trace_corpus_sha256: str
    horizon: int
    partition: str
    references: tuple[OriginReference, ...]
    sha256: str
    workload_counts: tuple[tuple[int, int], ...]
    generation_seed_counts: tuple[tuple[int, int], ...]
    workload_seed_counts: tuple[tuple[int, int, int], ...]
    source_group_count: int
    source_cluster_count: int
    conversation_count: int
    workload_cluster_counts: tuple[tuple[int, int], ...]
    tap_set: tuple[str, ...]


@dataclass(frozen=True)
class _MembershipStatistics:
    workload_counts: tuple[tuple[int, int], ...]
    generation_seed_counts: tuple[tuple[int, int], ...]
    workload_seed_counts: tuple[tuple[int, int, int], ...]
    source_group_count: int
    source_cluster_count: int
    conversation_count: int
    workload_cluster_counts: tuple[tuple[int, int], ...]


def _complete_conversation_products(
    catalog: TraceCorpusIndex,
    target_generation_seeds: tuple[int, ...],
) -> frozenset[bytes]:
    """Return conversations whose full authenticated manifest product is present."""
    required_seeds = set(target_generation_seeds)
    expected: dict[bytes, dict[int, int]] = {}
    raw_expected = catalog.expected_conversation_seed_turn_counts
    if not isinstance(raw_expected, tuple) or not raw_expected:
        raise DatasetError("trace index expected conversation products are malformed")
    for product in raw_expected:
        if (not isinstance(product, tuple) or len(product) != 3):
            raise DatasetError("trace index expected conversation products are malformed")
        conversation, seed, turn_count = product
        if (not isinstance(conversation, bytes) or len(conversation) != 32
                or conversation == bytes(32)
                or type(seed) is not int or seed < 0 or seed >= 2**64
                or type(turn_count) is not int or turn_count <= 0):
            raise DatasetError("trace index expected conversation products are malformed")
        by_seed = expected.setdefault(conversation, {})
        if seed in by_seed:
            raise DatasetError("trace index expected conversation products are malformed")
        by_seed[seed] = turn_count
    if tuple(sorted(raw_expected)) != raw_expected:
        raise DatasetError("trace index expected conversation products are malformed")
    if any(
            set(by_seed) != required_seeds
            or len(set(by_seed.values())) != 1
            for by_seed in expected.values()):
        raise DatasetError("trace index expected conversation products are incompatible")

    products: dict[bytes, dict[int, list[int]]] = {}
    identities: dict[bytes, tuple[int, bytes, bytes]] = {}
    for entry in catalog.entries:
        if entry.generation_seed not in required_seeds:
            raise DatasetError("trace index contains an unregistered generation seed")
        if entry.conversation not in expected:
            raise DatasetError("trace index conversation is absent from the source manifest")
        identity = (entry.workload, entry.source_group, entry.source_cluster)
        if entry.conversation in identities and identities[entry.conversation] != identity:
            raise DatasetError("trace conversation changes source/workload identity")
        identities[entry.conversation] = identity
        by_seed = products.setdefault(entry.conversation, {})
        by_seed.setdefault(entry.generation_seed, []).append(entry.turn_index)

    complete: set[bytes] = set()
    for conversation, by_seed in products.items():
        complete_product = set(by_seed) == required_seeds
        for seed in sorted(by_seed):
            turns = sorted(by_seed[seed])
            if len(set(turns)) != len(turns) or turns != list(range(len(turns))):
                raise DatasetError("trace conversation has malformed turn identities")
            if len(turns) > expected[conversation][seed]:
                raise DatasetError("trace conversation exceeds its source-manifest product")
            if turns != list(range(expected[conversation][seed])):
                complete_product = False
        if complete_product:
            complete.add(conversation)
    return frozenset(complete)


def _membership_conversations(
    catalog: TraceCorpusIndex,
    horizon: int,
    partition: str,
    target_generation_seeds: tuple[int, ...],
) -> frozenset[bytes]:
    complete = _complete_conversation_products(catalog, target_generation_seeds)
    if partition != "validation":
        return complete
    coverage: dict[bytes, set[int]] = {}
    for entry in catalog.entries:
        if (entry.validation and entry.conversation in complete
                and entry.origins(horizon)):
            coverage.setdefault(entry.conversation, set()).add(
                entry.generation_seed
            )
    required = set(target_generation_seeds)
    return frozenset(
        conversation for conversation, seeds in coverage.items()
        if seeds == required
    )


def _membership_statistics(
    catalog: TraceCorpusIndex,
    references: list[OriginReference],
) -> _MembershipStatistics:
    workloads = tuple(sorted(WORKLOADS.values()))
    seeds = tuple(sorted({entry.generation_seed for entry in catalog.entries}))
    workload_counts = {workload: 0 for workload in workloads}
    seed_counts = {seed: 0 for seed in seeds}
    workload_seed_counts = {
        (workload, seed): 0 for workload in workloads for seed in seeds
    }
    source_groups: set[bytes] = set()
    source_clusters: set[bytes] = set()
    conversations: set[bytes] = set()
    workload_clusters = {workload: set() for workload in workloads}
    for reference in references:
        if (not isinstance(reference, OriginReference)
                or not 0 <= reference.entry_index < len(catalog.entries)):
            raise DatasetError("origin membership contains an invalid reference")
        entry = catalog.entries[reference.entry_index]
        if entry.workload not in workload_counts or entry.generation_seed not in seed_counts:
            raise DatasetError("origin membership contains unknown workload/seed identity")
        workload_counts[entry.workload] += 1
        seed_counts[entry.generation_seed] += 1
        workload_seed_counts[entry.workload, entry.generation_seed] += 1
        source_groups.add(entry.source_group)
        source_clusters.add(entry.source_cluster)
        conversations.add(entry.conversation)
        workload_clusters[entry.workload].add(entry.source_cluster)
    return _MembershipStatistics(
        workload_counts=tuple(sorted(workload_counts.items())),
        generation_seed_counts=tuple(sorted(seed_counts.items())),
        workload_seed_counts=tuple(
            (workload, seed, count)
            for (workload, seed), count in sorted(workload_seed_counts.items())
        ),
        source_group_count=len(source_groups),
        source_cluster_count=len(source_clusters),
        conversation_count=len(conversations),
        workload_cluster_counts=tuple(
            (workload, len(workload_clusters[workload])) for workload in workloads
        ),
    )


def _membership_digest(
    catalog: TraceCorpusIndex,
    horizon: int,
    partition: str,
    tap_set: tuple[str, ...],
    references: list[OriginReference],
    statistics: _MembershipStatistics,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"local-gguf-engine/phase5-origin-membership/v3\0")
    digest.update(bytes.fromhex(catalog.trace_corpus_sha256))
    digest.update(horizon.to_bytes(4, "little"))
    encoded_partition = partition.encode()
    digest.update(len(encoded_partition).to_bytes(4, "little"))
    digest.update(encoded_partition)
    digest.update(len(tap_set).to_bytes(4, "little"))
    # Tap availability is a population property, not merely a loader detail:
    # an origin that cannot supply every requested state is not a member of the
    # cell's training population.
    for tap in tap_set:
        encoded_tap = tap.encode("ascii")
        digest.update(len(encoded_tap).to_bytes(4, "little"))
        digest.update(encoded_tap)
    digest.update(len(references).to_bytes(8, "little"))
    for reference in references:
        entry = catalog.entries[reference.entry_index]
        digest.update(entry.sequence_id.to_bytes(8, "little"))
        digest.update(reference.record_index.to_bytes(8, "little"))
    for workload, count in statistics.workload_counts:
        digest.update(workload.to_bytes(4, "little"))
        digest.update(count.to_bytes(8, "little"))
    for seed, count in statistics.generation_seed_counts:
        digest.update(seed.to_bytes(8, "little"))
        digest.update(count.to_bytes(8, "little"))
    for workload, seed, count in statistics.workload_seed_counts:
        digest.update(workload.to_bytes(4, "little"))
        digest.update(seed.to_bytes(8, "little"))
        digest.update(count.to_bytes(8, "little"))
    for count in (
        statistics.source_group_count,
        statistics.source_cluster_count,
        statistics.conversation_count,
    ):
        digest.update(count.to_bytes(8, "little"))
    for workload, count in statistics.workload_cluster_counts:
        digest.update(workload.to_bytes(4, "little"))
        digest.update(count.to_bytes(8, "little"))
    return digest.hexdigest()


def _workload_quotas(maximum: int, weights: dict[str, object]) -> dict[int, int]:
    if (not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0
            or not _valid_workload_weights(weights)):
        raise DatasetError("training membership workload contract is invalid")
    exact = {
        WORKLOADS[name]: maximum * float(value) for name, value in weights.items()
    }
    quotas = {workload: math.floor(value) for workload, value in exact.items()}
    remainder = maximum - sum(quotas.values())
    order = sorted(exact, key=lambda workload: (-(exact[workload] - quotas[workload]), workload))
    for workload in order[:remainder]:
        quotas[workload] += 1
    if sum(quotas.values()) != maximum or any(value <= 0 for value in quotas.values()):
        raise DatasetError("training membership workload quota is invalid")
    return quotas


def _valid_workload_weights(weights: dict[str, object]) -> bool:
    return (isinstance(weights, dict) and set(weights) == set(WORKLOADS)
            and not any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) <= 0.0
                for value in weights.values()
            )
            and abs(sum(float(value) for value in weights.values()) - 1.0) <= 1e-12)


def build_origin_membership(
    catalog: TraceCorpusIndex,
    horizon: int,
    partition: str,
    maximum_origins: int | None,
    workload_weights: dict[str, object],
    target_generation_seeds: tuple[int, ...],
    tap_set: tuple[str, ...],
) -> OriginMembership:
    """Freeze membership independently of model seed and epoch ordering."""
    if (not isinstance(horizon, int) or isinstance(horizon, bool)
            or horizon <= 0 or horizon > 23
            or partition not in {"train", "validation"}
            or maximum_origins is not None and (
                not isinstance(maximum_origins, int) or isinstance(maximum_origins, bool)
                or maximum_origins <= 0
            )
            or not target_generation_seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool)
                   or seed < 0 or seed >= 2**64 for seed in target_generation_seeds)
            or len(set(target_generation_seeds)) != len(target_generation_seeds)):
        raise DatasetError("origin membership contract is invalid")
    if (not isinstance(tap_set, tuple) or not tap_set
            or any(not isinstance(tap, str) or not tap for tap in tap_set)
            or len(set(tap_set)) != len(tap_set)):
        raise DatasetError("origin membership tap set is invalid")
    for tap in tap_set:
        _feature_identity(tap)
    if not _valid_workload_weights(workload_weights):
        raise DatasetError("training membership workload contract is invalid")
    try:
        if (len(bytes.fromhex(catalog.registry_sha256)) != 32
                or len(bytes.fromhex(catalog.trace_corpus_sha256)) != 32):
            raise DatasetError("trace catalog identity is invalid")
    except ValueError as error:
        raise DatasetError("trace catalog identity is invalid") from error
    candidates: dict[int, list[OriginReference]] = {
        workload: [] for workload in WORKLOADS.values()
    }
    if len({entry.sequence_id for entry in catalog.entries}) != len(catalog.entries):
        raise DatasetError("trace index contains duplicate sequence identities")
    complete_conversations = _membership_conversations(
        catalog, horizon, partition, target_generation_seeds
    )
    for entry_index, entry in enumerate(catalog.entries):
        if entry.validation != (partition == "validation"):
            continue
        if entry.conversation not in complete_conversations:
            continue
        if (not isinstance(entry.available_taps, tuple)
                or len(set(entry.available_taps)) != len(entry.available_taps)
                or any(not isinstance(tap, str) or not tap for tap in entry.available_taps)):
            raise DatasetError("trace index feature availability is malformed")
        if not set(tap_set).issubset(entry.available_taps):
            continue
        if entry.workload not in candidates:
            raise DatasetError("trace index contains an unknown workload")
        origins = entry.origins(horizon)
        if len(set(origins)) != len(origins):
            raise DatasetError("trace index contains duplicate origins")
        for origin in origins:
            if not isinstance(origin, int) or isinstance(origin, bool) or origin < 0:
                raise DatasetError("trace index contains an invalid origin")
            candidates[entry.workload].append(OriginReference(entry_index, origin))

    if maximum_origins is None:
        selected = [reference for workload in sorted(candidates) for reference in candidates[workload]]
    else:
        quotas = _workload_quotas(maximum_origins, workload_weights)
        selected = []
        for workload in sorted(candidates):
            by_seed = {seed: [] for seed in target_generation_seeds}
            for reference in candidates[workload]:
                seed = catalog.entries[reference.entry_index].generation_seed
                if seed not in by_seed:
                    raise DatasetError("trace index contains an unregistered generation seed")
                by_seed[seed].append(reference)
            base, remainder = divmod(quotas[workload], len(target_generation_seeds))
            for seed_offset, seed in enumerate(sorted(target_generation_seeds)):
                seed_quota = base + (seed_offset < remainder)
                values = by_seed[seed]
                values.sort(key=lambda reference: (
                    hashlib.sha256(
                        b"local-gguf-engine/phase5-origin-rank/v1\0"
                        + horizon.to_bytes(4, "little")
                        + catalog.entries[reference.entry_index].sequence_id.to_bytes(8, "little")
                        + reference.record_index.to_bytes(8, "little")
                    ).digest(),
                    catalog.entries[reference.entry_index].sequence_id,
                    reference.record_index,
                ))
                if len(values) < seed_quota:
                    name = next(name for name, value in WORKLOADS.items() if value == workload)
                    raise DatasetError(
                        f"workload {name}/seed {seed} needs {seed_quota} H={horizon} "
                        f"{partition} origins but only {len(values)} are available"
                    )
                selected.extend(values[:seed_quota])

    if len(set(selected)) != len(selected):
        raise DatasetError("origin membership selection contains duplicate references")
    workload_counts: dict[int, int] = {workload: 0 for workload in WORKLOADS.values()}
    seed_counts: dict[int, int] = {seed: 0 for seed in target_generation_seeds}
    workload_seed_coverage: dict[int, set[int]] = {
        workload: set() for workload in WORKLOADS.values()
    }
    for reference in selected:
        entry = catalog.entries[reference.entry_index]
        workload_counts[entry.workload] += 1
        if entry.generation_seed not in seed_counts:
            raise DatasetError("trace membership contains an unregistered target-generation seed")
        seed_counts[entry.generation_seed] += 1
        workload_seed_coverage[entry.workload].add(entry.generation_seed)
    missing_workloads = [
        name for name, workload in WORKLOADS.items() if workload_counts[workload] == 0
    ]
    incomplete_seed_coverage = [
        name for name, workload in WORKLOADS.items()
        if workload_seed_coverage[workload] != set(target_generation_seeds)
    ]
    if missing_workloads or incomplete_seed_coverage:
        raise DatasetError(
            "origin membership lacks frozen workload/seed coverage: "
            f"workloads={missing_workloads}, seeds={incomplete_seed_coverage}"
        )
    selected.sort(key=lambda reference: (
        catalog.entries[reference.entry_index].sequence_id, reference.record_index
    ))
    statistics = _membership_statistics(catalog, selected)
    if (statistics.workload_counts != tuple(sorted(workload_counts.items()))
            or statistics.generation_seed_counts != tuple(sorted(seed_counts.items()))):
        raise DatasetError("origin membership statistics are internally inconsistent")
    return OriginMembership(
        trace_corpus_sha256=catalog.trace_corpus_sha256,
        horizon=horizon,
        partition=partition,
        references=tuple(selected),
        sha256=_membership_digest(
            catalog, horizon, partition, tap_set, selected, statistics
        ),
        workload_counts=statistics.workload_counts,
        generation_seed_counts=statistics.generation_seed_counts,
        workload_seed_counts=statistics.workload_seed_counts,
        source_group_count=statistics.source_group_count,
        source_cluster_count=statistics.source_cluster_count,
        conversation_count=statistics.conversation_count,
        workload_cluster_counts=statistics.workload_cluster_counts,
        tap_set=tap_set,
    )


def validate_minimum_workload_clusters(
    membership: OriginMembership,
    minimum: int,
) -> dict[int, int]:
    """Fail closed when held-out independent-cluster coverage is too small."""
    if (membership.partition != "validation"
            or not isinstance(minimum, int) or isinstance(minimum, bool)
            or minimum <= 0):
        raise DatasetError("minimum validation-cluster contract is invalid")
    counts = dict(membership.workload_cluster_counts)
    if set(counts) != set(WORKLOADS.values()) or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts.values()):
        raise DatasetError("validation membership workload-cluster receipt is malformed")
    insufficient = {
        name: counts[workload]
        for name, workload in WORKLOADS.items()
        if counts[workload] < minimum
    }
    if insufficient:
        raise DatasetError(
            f"validation membership has fewer than {minimum} independent clusters: "
            f"{dict(sorted(insufficient.items()))}"
        )
    return counts


def _tap_key(stage: int, layer: int) -> str:
    if stage == 0:
        return "embedding"
    if stage == 1:
        return f"layer:{layer}"
    if stage == 2:
        return "final"
    raise DatasetError(f"unknown trace tap stage {stage}")


def _feature_identity(key: str) -> tuple[int, int]:
    if key == "embedding":
        return 0, 0
    if key == "final":
        return 2, 0
    if key.startswith("layer:"):
        try:
            layer = int(key.split(":", 1)[1])
        except ValueError as error:
            raise DatasetError(f"invalid target tap {key}") from error
        if layer < 0 or layer >= 2**16:
            raise DatasetError(f"invalid target tap {key}")
        return 1, layer
    raise DatasetError(f"invalid target tap {key}")


def _cluster_in_partition(source_cluster: bytes, partition: str) -> bool:
    validation = int.from_bytes(source_cluster[:8], "little") % 10 == 0
    return (partition == "all" or (partition == "validation" and validation)
            or (partition == "train" and not validation))


def _eligible_window(records: list[TraceRecord], origin: int, horizon: int) -> bool:
    """Require one causal anchor plus H future target selections."""
    future = records[origin:origin + horizon + 1]
    if len(future) != horizon + 1 or any(
        record.selection is None or record.target_label is None for record in future
    ):
        return False
    first = future[0]
    for offset, record in enumerate(future):
        if (record.context_generation != first.context_generation
                or record.sequence_id != first.sequence_id
                or record.turn_index != first.turn_index
                or record.source_cluster != first.source_cluster
                or record.conversation != first.conversation
                or record.split != first.split
                or record.workload != first.workload
                or record.target_position != first.target_position + offset):
            return False
        if offset and future[offset - 1].target_label != record.token_id:
            return False
    return True


def _eligible_root_rows(rows: list[_Row], origin: int, horizon: int) -> bool:
    """Revalidate an authenticated root-only window after payload compaction."""
    future = rows[origin:origin + horizon + 1]
    if len(future) != horizon + 1 or any(
        row.selection is None or row.label is None for row in future
    ):
        return False
    first = future[0]
    for offset, row in enumerate(future):
        if (row.context_generation != first.context_generation
                or row.sequence_id != first.sequence_id
                or row.source_cluster != first.source_cluster
                or row.conversation != first.conversation
                or row.workload != first.workload
                or row.generation_seed != first.generation_seed
                or row.target_position != first.target_position + offset):
            return False
        if offset and future[offset - 1].label != row.token:
            return False
    return True


def count_eligible_origins(
    directory: Path, horizon: int, registry_digest: str, partition: str,
    stop_after: int | None = None,
) -> int:
    if horizon <= 0 or horizon > 23 or partition not in {"train", "validation", "all"}:
        raise DatasetError("eligible-origin count contract is invalid")
    expected = bytes.fromhex(registry_digest)
    count = 0
    for path in discover_traces(directory):
        with TraceFile(path) as trace:
            if trace.identity.registry_sha256 != expected:
                raise DatasetError(f"trace registry identity mismatch: {path}")
            records = list(trace.records())
            if records[0].split != 1:
                raise DatasetError(f"training opened non-development trace: {path}")
            if not _cluster_in_partition(records[0].source_cluster, partition):
                continue
            for origin in range(max(0, len(records) - horizon)):
                if _eligible_window(records, origin, horizon):
                    count += 1
                    if stop_after is not None and count >= stop_after:
                        return count
    return count


def _select_row(record: TraceRecord, tap_set: tuple[str, ...]) -> _Row:
    by_key = {_tap_key(feature.stage, feature.layer): feature for feature in record.features}
    if len(by_key) != len(record.features):
        raise DatasetError("trace row contains duplicate tap identities")
    selected = []
    width = None
    for key in tap_set:
        if key not in by_key:
            raise DatasetError(f"trace row lacks registry tap {key}")
        feature = by_key[key]
        if width is None:
            width = feature.width
        elif width != feature.width:
            raise DatasetError("selected target taps have different widths")
        selected.append(numpy.frombuffer(feature.payload, dtype="<f2").copy())
    return _Row(
        features=numpy.stack(selected, axis=0), token=record.token_id,
        label=record.target_label, selection=record.selection,
        target_position=record.target_position, source_cluster=record.source_cluster,
        conversation=record.conversation,
        sequence_id=record.sequence_id, context_generation=record.context_generation,
        workload=record.workload, generation_seed=record.generation_seed,
    )


def _example(
    context: list[_Row], rows: list[_Row], origins: list[int], horizon: int,
    sampler: tuple[object, ...], context_mode: str,
) -> TrainingExample:
    if not origins:
        raise DatasetError("training batch has no origins")
    combined = context + rows
    global_origins = [len(context) + origin for origin in origins]
    generations = {combined[index].context_generation for index in global_origins}
    if len(generations) != 1:
        raise DatasetError("training batch crosses a context generation")
    generation = next(iter(generations))
    if context_mode == "root_only":
        visible_context = [combined[index] for index in global_origins]
        anchor_indices = numpy.arange(len(origins), dtype=numpy.int64)
    elif context_mode == "full_context":
        last_origin = max(global_origins)
        generation_start = last_origin
        while (generation_start > 0
               and combined[generation_start - 1].context_generation == generation):
            generation_start -= 1
        visible_context = combined[generation_start:last_origin + 1]
        if len(visible_context) > 8192:
            visible_context = visible_context[-8192:]
            generation_start = last_origin + 1 - len(visible_context)
        anchor_indices = numpy.asarray(
            [index - generation_start for index in global_origins], dtype=numpy.int64
        )
        if numpy.any(anchor_indices < 0):
            raise DatasetError("anchor lost required full context at the 8K bound")
    else:
        raise DatasetError(f"unknown feature context mode {context_mode}")
    context_array = numpy.stack([row.features for row in visible_context], axis=0)
    context_positions = numpy.asarray(
        [row.target_position for row in visible_context], dtype=numpy.int64
    )
    windows = [rows[origin:origin + horizon + 1] for origin in origins]
    if any(len(window) != horizon + 1 or any(
            row.label is None or row.selection is None
            or row.context_generation != generation for row in window
    ) or any(
            window[index].target_position != window[0].target_position + index
            for index in range(horizon + 1)
    ) or any(
            window[index - 1].label != window[index].token
            for index in range(1, horizon + 1)
    ) for window in windows):
        raise DatasetError("attempted to form a training window across an unlabeled/terminal row")
    # A trace row binds the feature state F_t to the authoritative selection
    # x_t in row.label.  The drafter consumes F_t and x_t, then predicts the H
    # selections carried by the following rows.  row.token is x_(t-1), so
    # using it here would shift the entire learned task one edge backward.
    rollout = numpy.asarray(
        [[row.label for row in window[:-1]] for window in windows], dtype=numpy.int64
    )
    labels = numpy.asarray(
        [[row.label for row in window[1:]] for window in windows], dtype=numpy.int64
    )
    targets = [window[1:] for window in windows]
    support_count = max(len(row.selection.prepared) for target in targets for row in target)
    shape = (len(origins), horizon, support_count)
    support_tokens = numpy.zeros(shape, dtype=numpy.int64)
    support_probabilities = numpy.zeros(shape, dtype=numpy.float64)
    support_mask = numpy.zeros(shape, dtype=numpy.bool_)
    selected_probabilities = numpy.zeros((len(origins), horizon), dtype=numpy.float64)
    unit_draws = numpy.zeros((len(origins), horizon), dtype=numpy.float64)
    for batch, target in enumerate(targets):
        for depth, row in enumerate(target):
            for candidate, prepared in enumerate(row.selection.prepared):
                token, _, probability, _ = prepared
                support_tokens[batch, depth, candidate] = token
                support_probabilities[batch, depth, candidate] = probability
                support_mask[batch, depth, candidate] = True
                if token == row.label:
                    selected_probabilities[batch, depth] = probability
            if selected_probabilities[batch, depth] <= 0.0:
                raise DatasetError("selected target token has zero/absent prepared probability")
            unit_draws[batch, depth] = row.selection.unit_draw
    return TrainingExample(
        context_taps=torch.from_numpy(context_array),
        context_positions=torch.from_numpy(context_positions),
        anchor_indices=torch.from_numpy(anchor_indices),
        rollout_tokens=torch.from_numpy(rollout),
        labels=torch.from_numpy(labels),
        target_support=TargetSupport(
            token_ids=torch.from_numpy(support_tokens),
            probabilities=torch.from_numpy(support_probabilities),
            mask=torch.from_numpy(support_mask),
        ),
        target_selected_probabilities=torch.from_numpy(selected_probabilities),
        target_unit_draws=torch.from_numpy(unit_draws),
        source_clusters=tuple(window[0].source_cluster.hex() for window in windows),
        conversations=tuple(window[0].conversation.hex() for window in windows),
        sequence_ids=tuple(window[0].sequence_id for window in windows),
        workloads=tuple(window[0].workload for window in windows),
        generation_seeds=tuple(window[0].generation_seed for window in windows),
        target_positions=torch.tensor(
            [window[0].target_position for window in windows], dtype=torch.int64
        ),
        sampler=sampler,
    )


def _collate_root_only(examples: list[TrainingExample]) -> TrainingExample:
    """Pack independent one-origin examples without erasing trace identity."""
    if not examples:
        raise DatasetError("root-only collator received no origins")
    first = examples[0]
    _sampler_bytes(first.sampler)
    horizon = first.labels.shape[1] if first.labels.ndim == 2 else -1
    support_width = 0

    def same_storage(value: torch.Tensor, reference: torch.Tensor) -> bool:
        return value.dtype == reference.dtype and value.device == reference.device

    for example in examples:
        tensors = (
            example.rollout_tokens, example.labels,
            example.target_selected_probabilities, example.target_unit_draws,
        )
        if (example.origin_count != 1
                or example.context_taps.ndim != 3
                or example.context_taps.shape[0] != 1
                or example.context_taps.shape[1:] != first.context_taps.shape[1:]
                or example.context_positions.shape != (1,)
                or example.anchor_indices.shape != (1,)
                or int(example.anchor_indices[0]) != 0
                or any(tensor.shape != (1, horizon) for tensor in tensors)
                or example.target_positions.shape != (1,)
                or example.target_support.token_ids.ndim != 3
                or example.target_support.token_ids.shape[:2] != (1, horizon)
                or example.target_support.probabilities.shape
                != example.target_support.token_ids.shape
                or example.target_support.mask.shape
                != example.target_support.token_ids.shape
                or len(example.source_clusters) != 1
                or len(example.conversations) != 1
                or len(example.sequence_ids) != 1
                or len(example.workloads) != 1
                or len(example.generation_seeds) != 1):
            raise DatasetError("root-only collator input shape is malformed")
        if (not same_storage(example.context_taps, first.context_taps)
                or not same_storage(
                    example.context_positions, first.context_positions
                )
                or not same_storage(example.anchor_indices, first.anchor_indices)
                or not same_storage(example.rollout_tokens, first.rollout_tokens)
                or not same_storage(example.labels, first.labels)
                or not same_storage(
                    example.target_support.token_ids,
                    first.target_support.token_ids,
                )
                or not same_storage(
                    example.target_support.probabilities,
                    first.target_support.probabilities,
                )
                or not same_storage(
                    example.target_support.mask, first.target_support.mask
                )
                or not same_storage(
                    example.target_selected_probabilities,
                    first.target_selected_probabilities,
                )
                or not same_storage(
                    example.target_unit_draws, first.target_unit_draws
                )
                or not same_storage(
                    example.target_positions, first.target_positions
                )):
            raise DatasetError("root-only collator tensor storage changed")
        if (example.sampler != first.sampler
                or example.generation_seeds[0] != example.sampler[5]):
            raise DatasetError("root-only collator crossed a sampler identity")
        support_width = max(support_width, example.target_support.token_ids.shape[2])
    if horizon <= 0 or support_width <= 0:
        raise DatasetError("root-only collator input shape is malformed")

    shape = (len(examples), horizon, support_width)
    support_tokens = first.target_support.token_ids.new_zeros(shape)
    support_probabilities = first.target_support.probabilities.new_zeros(shape)
    support_mask = first.target_support.mask.new_zeros(shape)
    for index, example in enumerate(examples):
        width = example.target_support.token_ids.shape[2]
        support_tokens[index:index + 1, :, :width] = example.target_support.token_ids
        support_probabilities[index:index + 1, :, :width] = (
            example.target_support.probabilities
        )
        support_mask[index:index + 1, :, :width] = example.target_support.mask

    return TrainingExample(
        context_taps=torch.cat([example.context_taps for example in examples]),
        context_positions=torch.cat([
            example.context_positions for example in examples
        ]),
        anchor_indices=torch.arange(
            len(examples), dtype=first.anchor_indices.dtype,
            device=first.anchor_indices.device,
        ),
        rollout_tokens=torch.cat([example.rollout_tokens for example in examples]),
        labels=torch.cat([example.labels for example in examples]),
        target_support=TargetSupport(
            token_ids=support_tokens,
            probabilities=support_probabilities,
            mask=support_mask,
        ),
        target_selected_probabilities=torch.cat([
            example.target_selected_probabilities for example in examples
        ]),
        target_unit_draws=torch.cat([
            example.target_unit_draws for example in examples
        ]),
        source_clusters=tuple(
            cluster for example in examples for cluster in example.source_clusters
        ),
        conversations=tuple(
            conversation
            for example in examples
            for conversation in example.conversations
        ),
        sequence_ids=tuple(
            sequence for example in examples for sequence in example.sequence_ids
        ),
        workloads=tuple(
            workload for example in examples for workload in example.workloads
        ),
        generation_seeds=tuple(
            seed for example in examples for seed in example.generation_seeds
        ),
        target_positions=torch.cat([
            example.target_positions for example in examples
        ]),
        sampler=first.sampler,
    )


class TraceWindowStream:
    def __init__(
        self,
        directory: Path,
        tap_set: tuple[str, ...],
        horizon: int,
        registry_digest: str,
        seed: int,
        maximum_origins: int | None,
        shuffle_buffer: int = 4096,
        partition: str = "train",
        anchor_batch_size: int = 8,
        context_mode: str = "full_context",
        catalog: TraceCorpusIndex | None = None,
        membership: OriginMembership | None = None,
        root_only_interleave_clusters_per_sampler: int = (
            ROOT_ONLY_INTERLEAVE_CLUSTERS_PER_SAMPLER
        ),
    ) -> None:
        if (not tap_set or horizon <= 0 or horizon > 23 or shuffle_buffer <= 0
                or anchor_batch_size <= 0
                or not isinstance(seed, int) or isinstance(seed, bool)
                or seed < 0 or seed >= 2**64
                or not isinstance(root_only_interleave_clusters_per_sampler, int)
                or isinstance(root_only_interleave_clusters_per_sampler, bool)
                or root_only_interleave_clusters_per_sampler <= 0
                or context_mode not in {"root_only", "full_context"}):
            raise DatasetError("trace window shape is invalid")
        self.paths = discover_traces(directory) if catalog is None else []
        self.tap_set = tap_set
        self.horizon = horizon
        self.registry_digest = bytes.fromhex(registry_digest)
        self.seed = seed
        self.maximum_origins = maximum_origins
        # Retained in the receipt/API as the deterministic local-origin shuffle
        # bound; examples themselves are never buffered because each owns its
        # full target context.
        self.shuffle_buffer = shuffle_buffer
        self.anchor_batch_size = anchor_batch_size
        self.context_mode = context_mode
        self.root_only_interleave_clusters_per_sampler = (
            root_only_interleave_clusters_per_sampler
        )
        if partition not in {"train", "validation", "all"}:
            raise DatasetError("trace development partition is invalid")
        self.partition = partition
        self.catalog = catalog
        self.membership = membership
        if catalog is not None and catalog.registry_sha256 != registry_digest:
            raise DatasetError("trace catalog registry identity is stale")
        if membership is not None:
            if catalog is None:
                raise DatasetError("origin membership requires an authenticated trace catalog")
            if (membership.trace_corpus_sha256 != catalog.trace_corpus_sha256
                    or membership.horizon != horizon or membership.partition != partition):
                raise DatasetError("origin membership is stale or incompatible")
            if membership.tap_set != tap_set:
                raise DatasetError("origin membership tap set is stale or incompatible")
            if (maximum_origins is not None
                    and maximum_origins != len(membership.references)):
                raise DatasetError("origin membership and stream maximum differ")
            references = list(membership.references)
            if len(set(references)) != len(references):
                raise DatasetError("origin membership contains duplicate references")
            canonical_references = sorted(references, key=lambda reference: (
                catalog.entries[reference.entry_index].sequence_id,
                reference.record_index,
            )) if all(
                isinstance(reference, OriginReference)
                and isinstance(reference.entry_index, int)
                and not isinstance(reference.entry_index, bool)
                and 0 <= reference.entry_index < len(catalog.entries)
                and isinstance(reference.record_index, int)
                and not isinstance(reference.record_index, bool)
                and reference.record_index >= 0
                for reference in references
            ) else []
            if references != canonical_references:
                raise DatasetError("origin membership references are not canonical")
            catalog_seeds = tuple(sorted({entry.generation_seed for entry in catalog.entries}))
            if not catalog_seeds:
                raise DatasetError("origin membership trace catalog has no generation seeds")
            complete_conversations = _membership_conversations(
                catalog, horizon, partition, catalog_seeds
            )
            for reference in references:
                if (not isinstance(reference, OriginReference)
                        or reference.entry_index < 0
                        or reference.entry_index >= len(catalog.entries)
                        or reference.record_index < 0):
                    raise DatasetError("origin membership contains an invalid reference")
                entry = catalog.entries[reference.entry_index]
                if (entry.validation != (partition == "validation")
                        or reference.record_index not in entry.origins(horizon)
                        or entry.conversation not in complete_conversations
                        or not set(tap_set).issubset(entry.available_taps)):
                    raise DatasetError("origin membership contains an ineligible reference")
            statistics = _membership_statistics(catalog, references)
            if (membership.sha256 != _membership_digest(
                    catalog, horizon, partition, tap_set, references, statistics)
                    or membership.workload_counts != statistics.workload_counts
                    or membership.generation_seed_counts != statistics.generation_seed_counts
                    or membership.workload_seed_counts != statistics.workload_seed_counts
                    or type(membership.source_group_count) is not int
                    or membership.source_group_count != statistics.source_group_count
                    or type(membership.source_cluster_count) is not int
                    or membership.source_cluster_count != statistics.source_cluster_count
                    or type(membership.conversation_count) is not int
                    or membership.conversation_count != statistics.conversation_count
                    or membership.workload_cluster_counts
                    != statistics.workload_cluster_counts
                    or any(type(workload) is not int or type(count) is not int
                           for workload, count in membership.workload_counts)
                    or any(type(seed) is not int or type(count) is not int
                           for seed, count in membership.generation_seed_counts)
                    or any(type(workload) is not int or type(seed) is not int
                           or type(count) is not int
                           for workload, seed, count in membership.workload_seed_counts)
                    or any(type(workload) is not int or type(count) is not int
                           for workload, count in membership.workload_cluster_counts)
                    or any(count <= 0 for _, count in statistics.workload_counts)
                    or any(count <= 0 for _, _, count in statistics.workload_seed_counts)):
                raise DatasetError("origin membership receipt is malformed")

    @staticmethod
    def _feature_identity(key: str) -> tuple[int, int]:
        return _feature_identity(key)

    def _open_indexed(self, entry: IndexedTrace) -> TraceFile:
        return TraceFile(
            entry.path,
            verify_envelope=False,
            expected_envelope_sha256=entry.envelope_sha256,
            expected_size=entry.file_size,
            expected_mtime_ns=entry.mtime_ns,
            expected_ctime_ns=entry.ctime_ns,
            expected_device=entry.device,
            expected_inode=entry.inode,
        )

    def _indexed_groups(
        self,
    ) -> tuple[
        list[tuple[tuple[bytes, int], list[tuple[int, int]]]],
        dict[int, list[int]],
    ]:
        if self.catalog is None:
            raise DatasetError("authenticated trace catalog is absent")
        allowed = None if self.membership is None else {
            (reference.entry_index, reference.record_index)
            for reference in self.membership.references
        }
        origins_by_entry: dict[int, list[int]] = {}
        for entry_index, entry in enumerate(self.catalog.entries):
            if (self.partition != "all"
                    and entry.validation != (self.partition == "validation")):
                continue
            origins = [
                origin for origin in entry.origins(self.horizon)
                if allowed is None or (entry_index, origin) in allowed
            ]
            if origins:
                origins_by_entry[entry_index] = origins
        groups: dict[tuple[bytes, int], list[tuple[int, int]]] = {}
        needed_groups = {
            (self.catalog.entries[index].conversation,
             self.catalog.entries[index].generation_seed)
            for index in origins_by_entry
        }
        for entry_index, entry in enumerate(self.catalog.entries):
            key = (entry.conversation, entry.generation_seed)
            if key in needed_groups:
                groups.setdefault(key, []).append((entry.turn_index, entry_index))
        return sorted(groups.items(), key=lambda item: item[0]), origins_by_entry

    def _indexed_root_entry(
        self,
        entry_index: int,
        origins: list[int],
        selected_features: frozenset[tuple[int, int]],
    ) -> tuple[IndexedTrace, list[_Row], tuple[object, ...], dict[int, int]]:
        if self.catalog is None or not 0 <= entry_index < len(self.catalog.entries):
            raise DatasetError("indexed root-only trace identity is malformed")
        entry = self.catalog.entries[entry_index]
        with self._open_indexed(entry) as trace:
            if trace.identity.registry_sha256 != self.registry_digest:
                raise DatasetError(f"trace registry identity mismatch: {entry.path}")
            if trace.identity.sampler != entry.sampler:
                raise DatasetError(f"trace sampler identity changed: {entry.path}")
            rows: list[_Row] = []
            for record in trace.records(selected_features):
                if not rows and (
                        record.sequence_id != entry.sequence_id
                        or record.turn_index != entry.turn_index
                        or record.workload != entry.workload
                        or record.generation_seed != entry.generation_seed
                        or record.source_group != entry.source_group
                        or record.source_cluster != entry.source_cluster
                        or record.conversation != entry.conversation
                        or record.prompt_derivation != entry.prompt_derivation):
                    raise DatasetError(
                        f"trace catalog provenance changed: {entry.path}"
                    )
                rows.append(_select_row(record, self.tap_set))
            if not rows:
                raise DatasetError(f"authenticated trace has no records: {entry.path}")
            if any(not _eligible_root_rows(rows, origin, self.horizon)
                   for origin in origins):
                raise DatasetError(f"authenticated origin changed: {entry.path}")
            context_generations = {
                origin: rows[origin].context_generation for origin in origins
            }
            # Root-only examples own copied anchor/features and target support.
            # Decode records directly into compact rows so the retained lane
            # never also owns a full duplicate feature-payload record list.
            return entry, rows, trace.identity.sampler, context_generations

    def _root_only_cluster_lane(
        self,
        sampler: tuple[object, ...],
        source_cluster: bytes,
        groups: list[tuple[tuple[bytes, int], list[tuple[int, int]]]],
        origins_by_entry: dict[int, list[int]],
        selected_features: frozenset[tuple[int, int]],
    ) -> Iterator[TrainingExample]:
        ranked_groups = sorted(groups, key=lambda item: (
            _order_rank(
                self.seed, b"conversation",
                item[0][0], item[0][1].to_bytes(8, "little"),
            ),
            item[0],
        ))
        for (conversation, generation_seed), turns in ranked_groups:
            ordered_turns = sorted(turns)
            for expected_turn, (turn_index, entry_index) in enumerate(ordered_turns):
                if turn_index != expected_turn:
                    raise DatasetError("conversation trace turns are not contiguous")
                if entry_index not in origins_by_entry:
                    continue
                if self.catalog is None:
                    raise DatasetError("authenticated trace catalog is absent")
                expected_entry = self.catalog.entries[entry_index]
                if (expected_entry.conversation != conversation
                        or expected_entry.generation_seed != generation_seed
                        or expected_entry.source_cluster != source_cluster
                        or expected_entry.sampler != sampler
                        or expected_entry.sampler[5] != generation_seed):
                    raise DatasetError(
                        "root-only interleave lane crossed an authenticated identity"
                    )
                origins = list(origins_by_entry[entry_index])
                entry, rows, trace_sampler, context_generations = (
                    self._indexed_root_entry(
                        entry_index, origins, selected_features
                    )
                )
                if (entry != expected_entry or trace_sampler != sampler):
                    raise DatasetError("root-only interleave trace identity changed")
                by_generation: dict[int, list[int]] = {}
                for origin in origins:
                    by_generation.setdefault(
                        context_generations[origin], []
                    ).append(origin)
                generations = sorted(by_generation.items(), key=lambda item: (
                    _order_rank(
                        self.seed, b"context-generation",
                        entry.sequence_id.to_bytes(8, "little"),
                        item[0].to_bytes(8, "little"),
                    ),
                    item[0],
                ))
                for context_generation, generation_origins in generations:
                    for start in range(0, len(generation_origins), self.shuffle_buffer):
                        chunk = generation_origins[start:start + self.shuffle_buffer]
                        chunk.sort(key=lambda origin: (
                            _order_rank(
                                self.seed, b"origin",
                                entry.sequence_id.to_bytes(8, "little"),
                                context_generation.to_bytes(8, "little"),
                                origin.to_bytes(8, "little"),
                            ),
                            origin,
                        ))
                        for origin in chunk:
                            example = _example(
                                [], rows, [origin], self.horizon,
                                trace_sampler, "root_only",
                            )
                            if (example.source_clusters != (source_cluster.hex(),)
                                    or example.conversations != (conversation.hex(),)
                                    or example.sequence_ids != (entry.sequence_id,)
                                    or example.workloads != (entry.workload,)
                                    or example.generation_seeds
                                    != (generation_seed,)):
                                raise DatasetError(
                                    "root-only example lost authenticated provenance"
                                )
                            yield example

    def _yield_root_only_interleaved(
        self,
        ordered_groups: list[
            tuple[tuple[bytes, int], list[tuple[int, int]]]
        ],
        origins_by_entry: dict[int, list[int]],
    ) -> Iterator[TrainingExample]:
        """Mix one origin per source-cluster lane before physical collation.

        Lanes retain at most one decoded trace at a time.  The fixed number of
        active lanes is per exact sampler identity, so a packed example never
        erases which target-generation seed authenticated its unit draws.
        Ranking is hash-based rather than call-order random: changing only the
        physical batch width therefore cannot perturb the flattened order.
        """
        if self.catalog is None:
            raise DatasetError("root-only interleave requires an authenticated catalog")
        selected_features = frozenset(
            self._feature_identity(key) for key in self.tap_set
        )
        strata: dict[
            tuple[object, ...],
            dict[bytes, list[tuple[tuple[bytes, int], list[tuple[int, int]]]]],
        ] = {}
        for group_key, turns in ordered_groups:
            if (not isinstance(group_key, tuple) or len(group_key) != 2
                    or not isinstance(group_key[0], bytes)
                    or len(group_key[0]) != 32
                    or not isinstance(group_key[1], int)
                    or isinstance(group_key[1], bool)
                    or not 0 <= group_key[1] < 2**64
                    or not isinstance(turns, list) or not turns
                    or any(
                        not isinstance(turn, tuple) or len(turn) != 2
                        or type(turn[0]) is not int or turn[0] < 0
                        or type(turn[1]) is not int
                        or not 0 <= turn[1] < len(self.catalog.entries)
                        for turn in turns
                    )):
                raise DatasetError("root-only conversation group is malformed")
            try:
                all_entries = [
                    self.catalog.entries[entry_index]
                    for _, entry_index in turns
                ]
            except (IndexError, TypeError) as error:
                raise DatasetError(
                    "root-only conversation group is malformed"
                ) from error
            first = all_entries[0]
            sampler = first.sampler
            source_cluster = first.source_cluster
            _sampler_bytes(sampler)
            if (first.conversation != group_key[0]
                    or first.generation_seed != group_key[1]
                    or sampler[5] != group_key[1]
                    or any(
                        entry.conversation != group_key[0]
                        or entry.generation_seed != group_key[1]
                        or entry.source_group != first.source_group
                        or entry.source_cluster != source_cluster
                        or entry.workload != first.workload
                        or entry.sampler != sampler
                        for entry in all_entries
                    )):
                raise DatasetError(
                    "root-only conversation group crossed an authenticated identity"
                )
            relevant = [
                entry_index for _, entry_index in turns
                if entry_index in origins_by_entry
            ]
            if not relevant:
                continue
            strata.setdefault(sampler, {}).setdefault(source_cluster, []).append(
                (group_key, turns)
            )
        if not strata:
            raise DatasetError("root-only interleave has no authenticated origins")

        sampler_order = sorted(strata, key=lambda sampler: (
            _order_rank(self.seed, b"sampler", _sampler_bytes(sampler)),
            _sampler_bytes(sampler),
        ))

        def sampler_blocks(
            sampler: tuple[object, ...],
        ) -> Iterator[list[TrainingExample]]:
            by_cluster = strata[sampler]
            cluster_order = sorted(by_cluster, key=lambda cluster: (
                _order_rank(
                    self.seed, b"source-cluster",
                    _sampler_bytes(sampler), cluster,
                ),
                cluster,
            ))
            pending: list[tuple[bytes, Iterator[TrainingExample]]] = [
                (
                    cluster,
                    self._root_only_cluster_lane(
                        sampler, cluster, by_cluster[cluster], origins_by_entry,
                        selected_features,
                    ),
                )
                for cluster in cluster_order
            ]
            next_pending = 0
            active = pending[:self.root_only_interleave_clusters_per_sampler]
            next_pending = len(active)
            while active:
                block: list[TrainingExample] = []
                next_active: list[tuple[bytes, Iterator[TrainingExample]]] = []
                for cluster, lane in active:
                    while True:
                        try:
                            example = next(lane)
                        except StopIteration:
                            if next_pending >= len(pending):
                                break
                            cluster, lane = pending[next_pending]
                            next_pending += 1
                            continue
                        if (example.origin_count != 1
                                or example.source_clusters != (cluster.hex(),)
                                or len(example.conversations) != 1
                                or example.sampler != sampler):
                            raise DatasetError(
                                "root-only interleave lane emitted another identity"
                            )
                        block.append(example)
                        next_active.append((cluster, lane))
                        break
                active = next_active
                if block:
                    yield block

        active_strata = [
            (sampler, sampler_blocks(sampler)) for sampler in sampler_order
        ]
        emitted = 0
        while active_strata:
            next_strata = []
            for sampler, blocks in active_strata:
                try:
                    block = next(blocks)
                except StopIteration:
                    continue
                if self.maximum_origins is not None:
                    block = block[:self.maximum_origins - emitted]
                if not block:
                    return
                # A sampler block is fixed by the interleave algorithm, not by
                # the requested physical width. This keeps the flattened order
                # and logical accumulation membership invariant across B=1..N.
                for start in range(0, len(block), self.anchor_batch_size):
                    physical = block[start:start + self.anchor_batch_size]
                    yield _collate_root_only(physical)
                    emitted += len(physical)
                if self.maximum_origins is not None and emitted >= self.maximum_origins:
                    return
                next_strata.append((sampler, blocks))
            active_strata = next_strata
        expected_origins = (
            len(self.membership.references) if self.membership is not None
            else self.maximum_origins
        )
        if expected_origins is not None and emitted != expected_origins:
            raise DatasetError(
                f"training membership emitted {emitted} origins; expected {expected_origins}"
            )

    def _yield_batches(
        self,
        randomizer: random.Random,
        ordered_groups: list[tuple[tuple[bytes, int], list[tuple[int, object]]]],
        indexed_origins: dict[int, list[int]] | None,
    ) -> Iterator[TrainingExample]:
        emitted = 0
        selected_features = frozenset(
            self._feature_identity(key) for key in self.tap_set
        )
        for _, turns in ordered_groups:
            context: list[_Row] = []
            for expected_turn, (turn_index, source) in enumerate(sorted(turns)):
                if turn_index != expected_turn:
                    raise DatasetError("conversation trace turns are not contiguous")
                if indexed_origins is None:
                    path = source
                    trace_context = TraceFile(path)
                    entry_index = None
                else:
                    if self.catalog is None or not isinstance(source, int):
                        raise DatasetError("indexed trace group is malformed")
                    entry_index = source
                    path = self.catalog.entries[entry_index].path
                    trace_context = self._open_indexed(self.catalog.entries[entry_index])
                with trace_context as trace:
                    if trace.identity.registry_sha256 != self.registry_digest:
                        raise DatasetError(f"trace registry identity mismatch: {path}")
                    if (entry_index is not None
                            and trace.identity.sampler
                            != self.catalog.entries[entry_index].sampler):
                        raise DatasetError(f"trace sampler identity changed: {path}")
                    records = list(trace.records(
                        selected_features if indexed_origins is not None else None
                    ))
                    if entry_index is not None:
                        entry = self.catalog.entries[entry_index]
                        first = records[0]
                        if (first.sequence_id != entry.sequence_id
                                or first.turn_index != entry.turn_index
                                or first.workload != entry.workload
                                or first.generation_seed != entry.generation_seed
                                or first.source_group != entry.source_group
                                or first.source_cluster != entry.source_cluster
                                or first.conversation != entry.conversation
                                or first.prompt_derivation != entry.prompt_derivation):
                            raise DatasetError(f"trace catalog provenance changed: {path}")
                    rows = [_select_row(record, self.tap_set) for record in records]
                    if indexed_origins is None:
                        origins = [
                            origin for origin in range(max(0, len(records) - self.horizon))
                            if _eligible_window(records, origin, self.horizon)
                        ]
                    else:
                        origins = list(indexed_origins.get(entry_index, ()))
                        if any(not _eligible_window(records, origin, self.horizon)
                               for origin in origins):
                            raise DatasetError(f"authenticated origin changed: {path}")
                    origins_by_generation: dict[int, list[int]] = {}
                    for origin in origins:
                        origins_by_generation.setdefault(
                            records[origin].context_generation, []
                        ).append(origin)
                    generation_groups = list(sorted(origins_by_generation.items()))
                    randomizer.shuffle(generation_groups)
                    for _, generation_origins in generation_groups:
                        for start in range(0, len(generation_origins), self.shuffle_buffer):
                            shuffled = generation_origins[start:start + self.shuffle_buffer]
                            randomizer.shuffle(shuffled)
                            for batch_start in range(0, len(shuffled), self.anchor_batch_size):
                                batch = shuffled[
                                    batch_start:batch_start + self.anchor_batch_size
                                ]
                                if self.maximum_origins is not None:
                                    batch = batch[:self.maximum_origins - emitted]
                                if not batch:
                                    return
                                yield _example(
                                    context, rows, batch, self.horizon,
                                    trace.identity.sampler, self.context_mode,
                                )
                                emitted += len(batch)
                                if (self.maximum_origins is not None
                                        and emitted >= self.maximum_origins):
                                    return
                    context.extend(rows)
                    if len(context) > 8192:
                        context = context[-8192:]
        expected_origins = (
            len(self.membership.references) if self.membership is not None
            else self.maximum_origins
        )
        if expected_origins is not None and emitted != expected_origins:
            raise DatasetError(
                f"training membership emitted {emitted} origins; expected {expected_origins}"
            )

    def __iter__(self) -> Iterator[TrainingExample]:
        randomizer = random.Random(self.seed)
        if self.catalog is not None:
            ordered_groups, origins_by_entry = self._indexed_groups()
            if (self.context_mode == "root_only"
                    and self.partition in {"train", "validation"}):
                yield from self._yield_root_only_interleaved(
                    ordered_groups, origins_by_entry
                )
                return
            randomizer.shuffle(ordered_groups)
            yield from self._yield_batches(randomizer, ordered_groups, origins_by_entry)
            return
        groups: dict[tuple[bytes, int], list[tuple[int, Path]]] = {}
        for path in self.paths:
            with TraceFile(path) as trace:
                if trace.identity.registry_sha256 != self.registry_digest:
                    raise DatasetError(f"trace registry identity mismatch: {path}")
                records = trace.records()
                first = next(records)
                if first.split != 1:
                    raise DatasetError(f"training opened non-development trace: {path}")
                if not _cluster_in_partition(first.source_cluster, self.partition):
                    continue
                groups.setdefault((first.conversation, first.generation_seed), []).append(
                    (first.turn_index, path)
                )
        ordered_groups: list[tuple[tuple[bytes, int], list[tuple[int, object]]]] = [
            (key, list(turns)) for key, turns in sorted(groups.items(), key=lambda item: item[0])
        ]
        randomizer.shuffle(ordered_groups)
        yield from self._yield_batches(randomizer, ordered_groups, None)

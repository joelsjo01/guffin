"""Framework-free authenticated trace-corpus validation."""

from __future__ import annotations

import hashlib
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .manifest import load_split_receipt
from .registry import Registry
from .trace import TraceFile, discover_traces


class CorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ManifestTurn:
    sequence_id: int
    turn_index: int
    source_group: bytes
    source_cluster: bytes
    conversation: bytes
    prompt_derivation: bytes
    split: int
    workload: int
    generation_seed: int


@dataclass(frozen=True)
class IndexedTrace:
    """Authenticated metadata needed to reuse one trace during training."""

    path: Path
    envelope_sha256: bytes
    file_size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    sequence_id: int
    turn_index: int
    workload: int
    generation_seed: int
    source_group: bytes
    source_cluster: bytes
    conversation: bytes
    prompt_derivation: bytes
    validation: bool
    sampler: tuple[object, ...]
    eligible_origins: tuple[tuple[int, tuple[int, ...]], ...]
    available_taps: tuple[str, ...] = ()

    def origins(self, horizon: int) -> tuple[int, ...]:
        for indexed_horizon, values in self.eligible_origins:
            if indexed_horizon == horizon:
                return values
        raise CorpusError(f"trace index does not contain H={horizon}")


@dataclass(frozen=True)
class TraceCorpusIndex:
    registry_sha256: str
    trace_corpus_sha256: str
    entries: tuple[IndexedTrace, ...]
    expected_conversation_seed_turn_counts: tuple[tuple[bytes, int, int], ...]


BALANCED_RECEIPT_NAME = "balanced_collection.lgsreceipt"
BALANCED_ALGORITHM = (
    "balanced-workload-seed-horizon-origin-and-validation-cluster/v1"
)


def _feature_tap_name(stage: int, layer: int) -> str:
    if stage == 0 and layer == 0:
        return "embedding"
    if stage == 1 and 0 <= layer < 2**16:
        return f"layer:{layer}"
    if stage == 2 and layer == 0:
        return "final"
    raise CorpusError("trace feature tap identity is invalid")


def _receipt_field(line: str, name: str) -> str:
    fields = line.split("\t")
    if len(fields) != 2 or fields[0] != name or not fields[1]:
        raise CorpusError(f"balanced collection receipt field {name} is malformed")
    return fields[1]


def _receipt_digest(value: str, name: str) -> str:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise CorpusError(
            f"balanced collection receipt field {name} is malformed"
        ) from error
    if len(value) != 64 or len(decoded) != 32 or decoded == bytes(32):
        raise CorpusError(f"balanced collection receipt field {name} is malformed")
    return value


def _receipt_integer(value: str, name: str, maximum: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise CorpusError(f"balanced collection receipt field {name} is malformed")
    result = int(value)
    if result <= 0 or result > maximum:
        raise CorpusError(f"balanced collection receipt field {name} is malformed")
    return result


def _load_balanced_collection_receipt(directory: Path) -> dict[str, object]:
    path = directory / BALANCED_RECEIPT_NAME
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CorpusError(
            "non-prefix trace corpus has no balanced collection receipt"
        ) from error
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise CorpusError("balanced collection receipt is not ASCII") from error
    if not text.endswith("\n") or "\r" in text:
        raise CorpusError("balanced collection receipt line endings are noncanonical")
    lines = text.splitlines(keepends=True)
    if len(lines) < 11 or lines[0] != "LGBALANCED1\n":
        raise CorpusError("balanced collection receipt magic/version is invalid")
    claimed = _receipt_digest(
        _receipt_field(lines[-1].removesuffix("\n"), "sha256"), "sha256"
    )
    prefix = "".join(lines[:-1]).encode("ascii")
    if hashlib.sha256(prefix).hexdigest() != claimed:
        raise CorpusError("balanced collection receipt digest is invalid")
    source_manifest = _receipt_digest(
        _receipt_field(lines[1].removesuffix("\n"), "source_manifest_sha256"),
        "source_manifest_sha256",
    )
    registry = _receipt_digest(
        _receipt_field(lines[2].removesuffix("\n"), "registry_sha256"),
        "registry_sha256",
    )
    feature_contract = _receipt_digest(
        _receipt_field(lines[3].removesuffix("\n"), "feature_contract_sha256"),
        "feature_contract_sha256",
    )
    if _receipt_field(lines[4].removesuffix("\n"), "algorithm") != BALANCED_ALGORITHM:
        raise CorpusError("balanced collection receipt algorithm is unsupported")
    horizon = _receipt_integer(
        _receipt_field(lines[5].removesuffix("\n"), "horizon"), "horizon", 23
    )
    quota = _receipt_integer(
        _receipt_field(
            lines[6].removesuffix("\n"),
            "training_origins_per_workload_seed",
        ),
        "training_origins_per_workload_seed", 2**64 - 1,
    )
    cluster_minimum = _receipt_integer(
        _receipt_field(
            lines[7].removesuffix("\n"),
            "minimum_validation_clusters_per_workload",
        ),
        "minimum_validation_clusters_per_workload", 2**32 - 1,
    )
    sequence_count = _receipt_integer(
        _receipt_field(lines[8].removesuffix("\n"), "sequence_count"),
        "sequence_count", 2**64 - 1,
    )
    if len(lines) != 10 + sequence_count:
        raise CorpusError("balanced collection receipt sequence count is inconsistent")
    sequences = tuple(
        _receipt_integer(
            _receipt_field(line.removesuffix("\n"), "sequence"),
            "sequence", 2**64 - 1,
        )
        for line in lines[9:-1]
    )
    if tuple(sorted(set(sequences))) != sequences:
        raise CorpusError("balanced collection receipt sequences are not canonical")
    return {
        "source_manifest_sha256": source_manifest,
        "registry_sha256": registry,
        "feature_contract_sha256": feature_contract,
        "algorithm": BALANCED_ALGORITHM,
        "horizon": horizon,
        "training_origins_per_workload_seed": quota,
        "minimum_validation_clusters_per_workload": cluster_minimum,
        "sequence_ids": sequences,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_balanced_collection_schedule(
    receipt: dict[str, object],
    manifest_turns: dict[int, _ManifestTurn],
    manifest_order: tuple[int, ...],
    indexed: list[IndexedTrace],
    target_generation_seeds: set[int],
) -> None:
    selected_sequences = tuple(entry.sequence_id for entry in indexed)
    if receipt["sequence_ids"] != selected_sequences:
        raise CorpusError("balanced collection receipt does not match trace files")
    horizon = int(receipt["horizon"])
    quota = int(receipt["training_origins_per_workload_seed"])
    cluster_minimum = int(receipt["minimum_validation_clusters_per_workload"])
    entries = {entry.sequence_id: entry for entry in indexed}
    conversation_rows: list[list[int]] = []
    for sequence in manifest_order:
        if (not conversation_rows
                or manifest_turns[conversation_rows[-1][0]].conversation
                != manifest_turns[sequence].conversation):
            conversation_rows.append([])
        conversation_rows[-1].append(sequence)

    training = {
        workload: {seed: 0 for seed in target_generation_seeds}
        for workload in range(1, 9)
    }
    validation = {workload: set() for workload in range(1, 9)}

    def complete() -> bool:
        return all(
            all(count >= quota for count in training[workload].values())
            and len(validation[workload]) >= cluster_minimum
            for workload in range(1, 9)
        )

    for rows in conversation_rows:
        turn = manifest_turns[rows[0]]
        is_validation = int.from_bytes(turn.source_cluster[:8], "little") % 10 == 0
        needed = (
            len(validation[turn.workload]) < cluster_minimum
            and turn.source_cluster not in validation[turn.workload]
        ) if is_validation else any(
            training[turn.workload][seed] < quota
            for seed in target_generation_seeds
        )
        actual = any(sequence in entries for sequence in rows)
        if actual != needed:
            raise CorpusError(
                "trace subset differs from the canonical balanced collection schedule"
            )
        if not actual:
            continue
        if any(sequence not in entries for sequence in rows):
            raise CorpusError(
                "balanced collection does not contain a whole conversation/seed/turn product"
            )
        origins = {seed: 0 for seed in target_generation_seeds}
        for sequence in rows:
            entry = entries[sequence]
            origins[entry.generation_seed] += len(entry.origins(horizon))
        if is_validation:
            if all(origins[seed] > 0 for seed in target_generation_seeds):
                validation[turn.workload].add(turn.source_cluster)
        else:
            for seed in target_generation_seeds:
                training[turn.workload][seed] += origins[seed]
    if not complete():
        raise CorpusError("balanced collection receipt did not reach its frozen coverage goal")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(value: str, field: str, line: int) -> bytes:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise CorpusError(f"development manifest line {line} has invalid {field}") from error
    if len(value) != 64 or len(decoded) != 32 or decoded == bytes(32):
        raise CorpusError(f"development manifest line {line} has invalid {field}")
    return decoded


def _manifest_integer(value: str, field: str, line: int, maximum: int) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise CorpusError(f"development manifest line {line} has invalid {field}")
    result = int(value)
    if result > maximum:
        raise CorpusError(f"development manifest line {line} has invalid {field}")
    return result


def _load_development_manifest(
    split_receipt: dict[str, object], split_receipt_path: Path, registry: Registry,
) -> tuple[
    dict[int, _ManifestTurn],
    tuple[int, ...],
    tuple[tuple[bytes, int, int], ...],
]:
    if split_receipt.get("target_generation_seeds") != list(
            registry.raw["target_generation_seeds"]):
        raise CorpusError("split receipt target-generation seeds are incompatible")
    raw_entry = split_receipt["manifests"]["training_development"]
    if not isinstance(raw_entry, dict):
        raise CorpusError("development manifest receipt is malformed")
    raw_name = raw_entry["path"]
    if (not isinstance(raw_name, str) or Path(raw_name).name != raw_name
            or Path(raw_name).is_absolute()):
        raise CorpusError("development manifest path is not a local filename")
    path = split_receipt_path.parent / raw_name
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CorpusError(f"development manifest is unreadable: {path}") from error
    if hashlib.sha256(payload).hexdigest() != raw_entry["sha256"]:
        raise CorpusError("development manifest digest differs from its split receipt")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusError("development manifest is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise CorpusError("development manifest line endings are noncanonical")
    lines = text.splitlines()
    if not lines or lines[0] != f"LGSOURCE4\t{registry.digest}":
        raise CorpusError("development manifest header is incompatible")
    expected_rows = raw_entry["turn_seed_rows"]
    if len(lines) - 1 != expected_rows:
        raise CorpusError("development manifest row count differs from its receipt")
    turns: dict[int, _ManifestTurn] = {}
    order: list[int] = []
    previous_sequence = 0
    previous_by_group: dict[tuple[bytes, int], int] = {}
    conversation_identity: dict[bytes, tuple[bytes, bytes, int, int]] = {}
    source_group_clusters: dict[bytes, bytes] = {}
    conversation_seed_turn_counts: dict[bytes, dict[int, int]] = {}
    registered_seeds = set(int(seed) for seed in registry.raw["target_generation_seeds"])
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 11:
            raise CorpusError(
                f"development manifest line {line_number} does not have eleven fields"
            )
        sequence_id = _manifest_integer(fields[0], "sequence ID", line_number, 2**64 - 1)
        turn_index = _manifest_integer(fields[1], "turn index", line_number, 2**32 - 1)
        split = _manifest_integer(fields[6], "split", line_number, 255)
        workload = _manifest_integer(fields[7], "workload", line_number, 255)
        generation_seed = _manifest_integer(
            fields[8], "generation seed", line_number, 2**64 - 1
        )
        _manifest_integer(fields[9], "generation cap", line_number, 2**64 - 1)
        if (sequence_id != previous_sequence + 1 or split != 1
                or workload not in range(1, 9)
                or generation_seed not in registered_seeds):
            raise CorpusError(f"development manifest line {line_number} has invalid provenance")
        source_group = _manifest_digest(fields[2], "source group", line_number)
        source_cluster = _manifest_digest(fields[3], "source cluster", line_number)
        conversation = _manifest_digest(fields[4], "conversation", line_number)
        prompt_derivation = _manifest_digest(fields[5], "prompt derivation", line_number)
        group = (conversation, generation_seed)
        wanted_turn = previous_by_group.get(group, -1) + 1
        if turn_index != wanted_turn:
            raise CorpusError("development manifest conversation turns are not contiguous")
        previous_by_group[group] = turn_index
        conversation_seed_turn_counts.setdefault(conversation, {})[
            generation_seed
        ] = turn_index + 1
        identity = (source_group, source_cluster, split, workload)
        if conversation in conversation_identity and conversation_identity[conversation] != identity:
            raise CorpusError("development manifest conversation provenance changes across rows")
        conversation_identity[conversation] = identity
        if (source_group in source_group_clusters
                and source_group_clusters[source_group] != source_cluster):
            raise CorpusError("development manifest source group crosses source clusters")
        source_group_clusters[source_group] = source_cluster
        turn = _ManifestTurn(
            sequence_id, turn_index, source_group, source_cluster, conversation,
            prompt_derivation, split, workload, generation_seed,
        )
        turns[sequence_id] = turn
        order.append(sequence_id)
        previous_sequence = sequence_id
    if not turns:
        raise CorpusError("development manifest contains no turns")
    for seed_turn_counts in conversation_seed_turn_counts.values():
        if (set(seed_turn_counts) != registered_seeds
                or len(set(seed_turn_counts.values())) != 1):
            raise CorpusError(
                "development manifest does not contain the full seed/turn product"
            )
    expected_products = tuple(sorted(
        (conversation, seed, turn_count)
        for conversation, seed_turn_counts in conversation_seed_turn_counts.items()
        for seed, turn_count in seed_turn_counts.items()
    ))
    return turns, tuple(order), expected_products


def _validate_trace_corpus(
    directory: Path,
    registry: Registry,
    model_path: Path | None,
    auditor: Path | None,
    split_receipt_path: Path,
) -> tuple[dict[str, object], TraceCorpusIndex]:
    unpublished = sorted(directory.glob("*.partial"))
    if unpublished:
        raise CorpusError(
            "trace corpus contains unpublished partial state: "
            + unpublished[0].name
        )
    paths = discover_traces(directory)
    split_receipt = load_split_receipt(split_receipt_path, registry.digest)
    expected_manifest_digest = split_receipt["manifests"]["training_development"]["sha256"]
    manifest_turns, manifest_order, expected_conversation_seed_turn_counts = (
        _load_development_manifest(
            split_receipt, split_receipt_path, registry
        )
    )
    model_digest = file_sha256(model_path) if model_path is not None else None
    corpus_digest = hashlib.sha256()
    corpus_digest.update(b"local-gguf-engine/phase5-trace-corpus/v1\0")
    corpus_digest.update(len(paths).to_bytes(8, "little"))
    common: tuple[object, ...] | None = None
    common_feature_contract: bytes | None = None
    common_feature_layout: tuple[tuple[int, int, int, int], ...] | None = None
    records = selections = feature_bytes = 0
    horizons = (3, 4, 7, 8, 15, 16, 23)
    eligible_by_horizon = Counter({horizon: 0 for horizon in horizons})
    train_by_horizon = Counter({horizon: 0 for horizon in horizons})
    validation_by_horizon = Counter({horizon: 0 for horizon in horizons})
    clusters: set[str] = set()
    source_groups: set[str] = set()
    conversations: set[str] = set()
    sequence_ids: set[int] = set()
    trace_counts_by_workload = Counter({workload: 0 for workload in range(1, 9)})
    selection_counts_by_workload = Counter({workload: 0 for workload in range(1, 9)})
    eligible_by_workload_and_horizon = {
        horizon: Counter({workload: 0 for workload in range(1, 9)})
        for horizon in horizons
    }
    train_by_workload_and_horizon = {
        horizon: Counter({workload: 0 for workload in range(1, 9)})
        for horizon in horizons
    }
    validation_by_workload_and_horizon = {
        horizon: Counter({workload: 0 for workload in range(1, 9)})
        for horizon in horizons
    }
    indexed: list[IndexedTrace] = []
    sampler_contract = registry.raw["target_generation_sampler"]
    target_generation_seeds = set(registry.raw["target_generation_seeds"])
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    for path in paths:
        if auditor is not None:
            process = subprocess.run(
                [str(auditor), "audit", str(path)], check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if process.returncode:
                raise CorpusError(f"native trace audit failed for {path}: {process.stderr.strip()}")
        with TraceFile(path) as trace:
            encoded_name = path.name.encode()
            corpus_digest.update(len(encoded_name).to_bytes(8, "little"))
            corpus_digest.update(encoded_name)
            corpus_digest.update(trace.envelope_sha256)
            identity = trace.identity
            if identity.registry_sha256.hex() != registry.digest:
                raise CorpusError(f"trace registry digest mismatch: {path}")
            if model_digest is not None and identity.model_sha256.hex() != model_digest:
                raise CorpusError(f"trace target model digest mismatch: {path}")
            if identity.source_manifest_sha256.hex() != expected_manifest_digest:
                raise CorpusError(f"trace source manifest differs from its split receipt: {path}")
            expected_sampler_without_seed = (
                f32(sampler_contract["temperature"]), sampler_contract["top_k"],
                f32(sampler_contract["top_p"]), f32(sampler_contract["min_p"]),
                f32(sampler_contract["repetition_penalty"]),
                sampler_contract["repetition_window"],
            )
            if identity.sampler[:5] + identity.sampler[6:] != expected_sampler_without_seed:
                raise CorpusError(f"trace sampler contract differs from registry: {path}")
            if identity.sampler[5] not in target_generation_seeds:
                raise CorpusError(f"trace uses an unregistered target-generation seed: {path}")
            signature = (
                identity.model_bytes, identity.source_tensor_count,
                identity.model_sha256, identity.canonical_metadata_sha256,
                identity.source_tensors_sha256, identity.resident_allocation_count,
                identity.resident_bytes, identity.resident_sha256,
                identity.feature_contract_sha256, identity.tokenizer_sha256,
                identity.chat_template_sha256, identity.source_manifest_sha256,
                identity.registry_sha256, identity.sampler[:5] + identity.sampler[6:],
            )
            if common is None:
                common = signature
                common_feature_contract = identity.feature_contract_sha256
            elif signature != common:
                raise CorpusError(f"trace corpus mixes incompatible identities: {path}")
            causal_run = 0
            previous = None
            first_record = None
            origins_by_horizon: dict[int, list[int]] = {horizon: [] for horizon in horizons}
            trace_selections = 0
            trace_feature_layout: tuple[tuple[int, int, int, int], ...] | None = None
            for record_index, record in enumerate(trace.records()):
                if record.split != 1:
                    raise CorpusError(f"training corpus contains non-development record: {path}")
                if first_record is None:
                    first_record = record
                    if path.name != f"{record.sequence_id:020d}.lgtrace":
                        raise CorpusError(f"trace filename does not match sequence ID: {path}")
                    if record.sequence_id in sequence_ids:
                        raise CorpusError(f"duplicate trace sequence ID: {record.sequence_id}")
                    sequence_ids.add(record.sequence_id)
                    expected_turn = manifest_turns.get(record.sequence_id)
                    if expected_turn is None or (
                            record.turn_index != expected_turn.turn_index
                            or record.source_group != expected_turn.source_group
                            or record.source_cluster != expected_turn.source_cluster
                            or record.conversation != expected_turn.conversation
                            or record.prompt_derivation != expected_turn.prompt_derivation
                            or record.split != expected_turn.split
                            or record.workload != expected_turn.workload
                            or record.generation_seed != expected_turn.generation_seed):
                        raise CorpusError(f"trace source provenance differs from manifest: {path}")
                elif (record.sequence_id != first_record.sequence_id
                      or record.turn_index != first_record.turn_index
                      or record.source_group != first_record.source_group
                      or record.source_cluster != first_record.source_cluster
                      or record.conversation != first_record.conversation
                      or record.prompt_derivation != first_record.prompt_derivation
                      or record.split != first_record.split
                      or record.workload != first_record.workload
                      or record.generation_seed != first_record.generation_seed):
                    raise CorpusError(f"trace mixes source/turn provenance: {path}")
                records += 1
                if record.generation_seed != identity.sampler[5]:
                    raise CorpusError(f"trace record seed differs from sampler identity: {path}")
                selections += record.selection is not None
                trace_selections += record.selection is not None
                feature_bytes += sum(len(feature.payload) for feature in record.features)
                feature_layout = tuple(
                    (feature.stage, feature.layer, feature.width, feature.dtype)
                    for feature in record.features
                )
                if len({item[:2] for item in feature_layout}) != len(feature_layout):
                    raise CorpusError(f"trace row contains duplicate feature taps: {path}")
                for stage, layer, _, _ in feature_layout:
                    _feature_tap_name(stage, layer)
                if trace_feature_layout is None:
                    trace_feature_layout = feature_layout
                elif feature_layout != trace_feature_layout:
                    raise CorpusError(f"trace feature layout changes between records: {path}")
                source_groups.add(record.source_group.hex())
                clusters.add(record.source_cluster.hex())
                conversations.add(record.conversation.hex())
                eligible = record.selection is not None and record.target_label is not None
                if (eligible and previous is not None and previous.selection is not None
                        and previous.target_label is not None
                        and previous.sequence_id == record.sequence_id
                        and previous.turn_index == record.turn_index
                        and previous.source_cluster == record.source_cluster
                        and previous.conversation == record.conversation
                        and previous.split == record.split
                        and previous.workload == record.workload
                        and previous.context_generation == record.context_generation
                        and previous.target_position + 1 == record.target_position
                        and previous.target_label == record.token_id):
                    causal_run += 1
                elif eligible:
                    causal_run = 1
                else:
                    causal_run = 0
                for horizon in horizons:
                    # H proposal edges require one selected anchor x_t plus H
                    # subsequent selected rows x_(t+1)..x_(t+H).
                    if causal_run >= horizon + 1:
                        origin = record_index - horizon
                        origins_by_horizon[horizon].append(origin)
                        eligible_by_horizon[horizon] += 1
                        eligible_by_workload_and_horizon[horizon][record.workload] += 1
                        target = (validation_by_horizon
                                  if int.from_bytes(record.source_cluster[:8], "little") % 10 == 0
                                  else train_by_horizon)
                        target[horizon] += 1
                        workload_target = (
                            validation_by_workload_and_horizon
                            if int.from_bytes(record.source_cluster[:8], "little") % 10 == 0
                            else train_by_workload_and_horizon
                        )
                        workload_target[horizon][record.workload] += 1
                previous = record
            if first_record is None:
                raise CorpusError(f"trace contains no records: {path}")
            if trace_feature_layout is None:
                raise CorpusError(f"trace feature layout is absent: {path}")
            if common_feature_layout is None:
                common_feature_layout = trace_feature_layout
            elif trace_feature_layout != common_feature_layout:
                raise CorpusError(f"trace corpus mixes incompatible feature layouts: {path}")
            trace_counts_by_workload[first_record.workload] += 1
            selection_counts_by_workload[first_record.workload] += trace_selections
            indexed.append(IndexedTrace(
                path=path,
                envelope_sha256=trace.envelope_sha256,
                file_size=trace.file_size,
                mtime_ns=trace.mtime_ns,
                ctime_ns=trace.ctime_ns,
                device=trace.device,
                inode=trace.inode,
                sequence_id=first_record.sequence_id,
                turn_index=first_record.turn_index,
                workload=first_record.workload,
                generation_seed=first_record.generation_seed,
                source_group=first_record.source_group,
                source_cluster=first_record.source_cluster,
                conversation=first_record.conversation,
                prompt_derivation=first_record.prompt_derivation,
                validation=int.from_bytes(first_record.source_cluster[:8], "little") % 10 == 0,
                sampler=identity.sampler,
                eligible_origins=tuple(
                    (horizon, tuple(origins_by_horizon[horizon])) for horizon in horizons
                ),
                available_taps=tuple(
                    _feature_tap_name(stage, layer)
                    for stage, layer, _, _ in trace_feature_layout
                ),
            ))
    expected_prefix = manifest_order[:len(paths)]
    actual_sequences = tuple(entry.sequence_id for entry in indexed)
    balanced_receipt: dict[str, object] | None = None
    if actual_sequences != expected_prefix:
        balanced_receipt = _load_balanced_collection_receipt(directory)
        if (balanced_receipt["source_manifest_sha256"] != expected_manifest_digest
                or balanced_receipt["registry_sha256"] != registry.digest
                or common_feature_contract is None
                or balanced_receipt["feature_contract_sha256"]
                    != common_feature_contract.hex()):
            raise CorpusError("balanced collection receipt identity is incompatible")
        _validate_balanced_collection_schedule(
            balanced_receipt, manifest_turns, manifest_order, indexed,
            target_generation_seeds,
        )
        balanced_receipt = {
            key: value for key, value in balanced_receipt.items()
            if key != "sequence_ids"
        } | {"sequence_count": len(actual_sequences)}
    receipt = {
        "trace_count": len(paths), "record_count": records,
        "selection_count": selections,
        "eligible_origins_by_horizon": {
            str(key): eligible_by_horizon[key] for key in horizons
        },
        "training_origins_by_horizon": {
            str(key): train_by_horizon[key] for key in horizons
        },
        "validation_origins_by_horizon": {
            str(key): validation_by_horizon[key] for key in horizons
        },
        "trace_counts_by_workload": {
            str(key): trace_counts_by_workload[key] for key in range(1, 9)
        },
        "selection_counts_by_workload": {
            str(key): selection_counts_by_workload[key] for key in range(1, 9)
        },
        "eligible_origins_by_workload_and_horizon": {
            str(horizon): {
                str(workload): eligible_by_workload_and_horizon[horizon][workload]
                for workload in range(1, 9)
            } for horizon in horizons
        },
        "training_origins_by_workload_and_horizon": {
            str(horizon): {
                str(workload): train_by_workload_and_horizon[horizon][workload]
                for workload in range(1, 9)
            } for horizon in horizons
        },
        "validation_origins_by_workload_and_horizon": {
            str(horizon): {
                str(workload): validation_by_workload_and_horizon[horizon][workload]
                for workload in range(1, 9)
            } for horizon in horizons
        },
        "feature_bytes": feature_bytes,
        "feature_contract_sha256": (
            common_feature_contract.hex() if common_feature_contract is not None else None
        ),
        "feature_layout": [
            {"tap": _feature_tap_name(stage, layer), "width": width, "dtype": dtype}
            for stage, layer, width, dtype in (common_feature_layout or ())
        ],
        "source_group_count": len(source_groups),
        "source_cluster_count": len(clusters), "conversation_count": len(conversations),
        "model_sha256": model_digest,
        "source_manifest_sha256": expected_manifest_digest,
        "trace_corpus_sha256": corpus_digest.hexdigest(),
        "split_receipt_sha256": split_receipt["receipt_sha256"],
        "balanced_collection": balanced_receipt,
    }
    return receipt, TraceCorpusIndex(
        registry_sha256=registry.digest,
        trace_corpus_sha256=str(receipt["trace_corpus_sha256"]),
        entries=tuple(indexed),
        expected_conversation_seed_turn_counts=(
            expected_conversation_seed_turn_counts
        ),
    )


def validate_trace_corpus(
    directory: Path,
    registry: Registry,
    model_path: Path | None,
    auditor: Path | None,
    split_receipt_path: Path,
) -> dict[str, object]:
    return _validate_trace_corpus(
        directory, registry, model_path, auditor, split_receipt_path
    )[0]


def validate_trace_corpus_with_index(
    directory: Path,
    registry: Registry,
    model_path: Path | None,
    auditor: Path | None,
    split_receipt_path: Path,
) -> tuple[dict[str, object], TraceCorpusIndex]:
    """Authenticate once and retain bounded metadata for all later epochs."""
    return _validate_trace_corpus(
        directory, registry, model_path, auditor, split_receipt_path
    )

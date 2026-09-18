"""Independent bounded reader for authenticated LGTRC5/schema-2 traces."""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAGIC = b"LGTRC5\0\0"
SCHEMA_VERSION = 2
MAX_FILE_BYTES = 1 << 36
MAX_RECORDS = 100_000_000
MAX_FEATURES = 64
MAX_FEATURE_BYTES = 64 * 1024 * 1024
HISTORY_CAPACITY = 64
RAW_DRAW_CAPACITY = 4
NO_PARENT = 2**64 - 1


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class TraceIdentity:
    model_bytes: int
    source_tensor_count: int
    model_sha256: bytes
    canonical_metadata_sha256: bytes
    source_tensors_sha256: bytes
    resident_allocation_count: int
    resident_bytes: int
    resident_sha256: bytes
    feature_contract_sha256: bytes
    tokenizer_sha256: bytes
    chat_template_sha256: bytes
    source_manifest_sha256: bytes
    registry_sha256: bytes
    sampler: tuple[object, ...]
    record_count: int


@dataclass(frozen=True)
class TraceFeature:
    stage: int
    layer: int
    target_position: int
    width: int
    dtype: int
    payload: bytes


@dataclass(frozen=True)
class TraceSelection:
    pre_draw_count: int
    post_draw_count: int
    raw_draws: tuple[int, ...]
    consumes_random_draw: bool
    draw_index: int
    unit_draw: float
    mapped_draw: float
    compact: tuple[tuple[int, float], ...]
    prepared: tuple[tuple[int, float, float, float], ...]
    total_weight: float
    request_top_k: int
    request_history: tuple[int, ...]


@dataclass(frozen=True)
class TraceRecord:
    sequence_id: int
    turn_index: int
    kind: int
    terminal: bool
    parent: int | None
    target_position: int
    token_id: int
    target_label: int | None
    source_group: bytes
    source_cluster: bytes
    conversation: bytes
    prompt_derivation: bytes
    split: int
    workload: int
    context_generation: int
    context_flags: int
    generation_seed: int
    remaining_budget: int
    retrieval_eligible: bool
    retrieval_proposal: int | None
    selection: TraceSelection | None
    features: tuple[TraceFeature, ...]


class _Reader:
    def __init__(self, data: mmap.mmap) -> None:
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset > len(self.data) or count > len(self.data) - self.offset:
            raise TraceError("truncated trace field")
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def skip(self, count: int) -> None:
        if count < 0 or self.offset > len(self.data) or count > len(self.data) - self.offset:
            raise TraceError("truncated trace field")
        self.offset += count

    def unpack(self, form: str) -> tuple[object, ...]:
        size = struct.calcsize(form)
        return struct.unpack(form, self.take(size))

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def u64(self) -> int:
        return self.unpack("<Q")[0]

    def f32(self) -> float:
        return self.unpack("<f")[0]

    def f64(self) -> float:
        return self.unpack("<d")[0]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _close(left: float, right: float) -> bool:
    return (math.isfinite(left) and math.isfinite(right)
            and abs(left - right) <= 2e-13 * max(1.0, abs(left), abs(right)))


class TraceFile:
    def __init__(
        self,
        path: Path,
        *,
        verify_envelope: bool = True,
        expected_envelope_sha256: bytes | None = None,
        expected_size: int | None = None,
        expected_mtime_ns: int | None = None,
        expected_ctime_ns: int | None = None,
        expected_device: int | None = None,
        expected_inode: int | None = None,
    ) -> None:
        self.path = path
        self._file = path.open("rb")
        try:
            status = os.fstat(self._file.fileno())
            size = status.st_size
            self.file_size = size
            self.mtime_ns = status.st_mtime_ns
            self.ctime_ns = status.st_ctime_ns
            self.device = status.st_dev
            self.inode = status.st_ino
            self._fingerprint = (
                self.device, self.inode, self.file_size, self.mtime_ns, self.ctime_ns,
            )
            if size <= 0 or size > MAX_FILE_BYTES:
                raise TraceError(f"trace size is invalid: {path}")
            expected_fingerprint = (
                expected_device, expected_inode, expected_size,
                expected_mtime_ns, expected_ctime_ns,
            )
            labels = ("device", "inode", "size", "modification time", "change time")
            for label, expected, actual in zip(
                    labels, expected_fingerprint, self._fingerprint):
                if expected is not None and actual != expected:
                    raise TraceError(
                        f"trace {label} changed after corpus authentication: {path}"
                    )
            if not verify_envelope and expected_envelope_sha256 is None:
                raise TraceError("skipping envelope verification requires an authenticated digest")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            if verify_envelope:
                self._verify_envelope()
            self.envelope_sha256 = bytes(self._map[12:44])
            if (expected_envelope_sha256 is not None
                    and self.envelope_sha256 != expected_envelope_sha256):
                raise TraceError(f"trace envelope identity changed: {path}")
            self._reader = _Reader(self._map)
            self.identity = self._read_header()
        except BaseException:
            self.close()
            raise

    def assert_unchanged(self) -> None:
        """Fail if an opened trace changed while it was being consumed."""
        if not hasattr(self, "_file") or not hasattr(self, "_fingerprint"):
            return
        status = os.fstat(self._file.fileno())
        current = (
            status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns,
        )
        if current != self._fingerprint:
            raise TraceError(f"trace changed while open: {self.path}")

    def close(self) -> None:
        if hasattr(self, "_map"):
            self._map.close()
            del self._map
        if hasattr(self, "_file"):
            self._file.close()
            del self._file

    def __enter__(self) -> "TraceFile":
        return self

    def __exit__(self, exception_type: object, *_: object) -> None:
        try:
            if exception_type is None:
                self.assert_unchanged()
        finally:
            self.close()

    def _verify_envelope(self) -> None:
        if len(self._map) < 44:
            raise TraceError("trace is shorter than its envelope")
        expected = self._map[12:44]
        digest = hashlib.sha256()
        digest.update(memoryview(self._map)[:12])
        digest.update(bytes(32))
        for offset in range(44, len(self._map), 8 * 1024 * 1024):
            digest.update(memoryview(self._map)[offset:min(len(self._map), offset + 8 * 1024 * 1024)])
        if digest.digest() != expected:
            raise TraceError("trace envelope SHA-256 mismatch")

    def _read_header(self) -> TraceIdentity:
        reader = self._reader
        if reader.take(8) != MAGIC or reader.u32() != SCHEMA_VERSION:
            raise TraceError("trace magic/schema is unsupported")
        reader.take(32)
        if reader.u32() != 1:
            raise TraceError("model receipt version is unsupported")
        model_bytes = reader.u64()
        source_tensor_count = reader.u32()
        model_digest = reader.take(32)
        canonical_metadata_digest = reader.take(32)
        source_tensors_digest = reader.take(32)
        if (model_bytes == 0 or source_tensor_count == 0
                or any(value == bytes(32) for value in (
                    model_digest, canonical_metadata_digest, source_tensors_digest
                ))):
            raise TraceError("model receipt is incomplete")
        if reader.u32() != 1:
            raise TraceError("resident receipt version is unsupported")
        allocation_count = reader.u32()
        resident_bytes = reader.u64()
        resident_digest = reader.take(32)
        if allocation_count == 0 or resident_bytes == 0 or resident_digest == bytes(32):
            raise TraceError("resident receipt is incomplete")
        identities = tuple(reader.take(32) for _ in range(5))
        if any(value == bytes(32) for value in identities):
            raise TraceError("trace identity digest is zero")
        sampler = (
            reader.f32(), reader.u64(), reader.f32(), reader.f32(), reader.f32(),
            reader.u64(), reader.u64(),
        )
        if (not math.isfinite(sampler[0]) or sampler[0] < 0.0
                or sampler[1] == 0 or sampler[1] > 256
                or not math.isfinite(sampler[2]) or not 0.0 < sampler[2] <= 1.0
                or not math.isfinite(sampler[3]) or not 0.0 <= sampler[3] <= 1.0
                or not math.isfinite(sampler[4]) or sampler[4] <= 0.0
                or sampler[6] > HISTORY_CAPACITY):
            raise TraceError("trace sampler identity is invalid")
        count = reader.u64()
        if count == 0 or count > MAX_RECORDS:
            raise TraceError("trace record count is invalid")
        return TraceIdentity(
            model_bytes=model_bytes, source_tensor_count=source_tensor_count,
            model_sha256=model_digest,
            canonical_metadata_sha256=canonical_metadata_digest,
            source_tensors_sha256=source_tensors_digest,
            resident_allocation_count=allocation_count,
            resident_bytes=resident_bytes, resident_sha256=resident_digest,
            feature_contract_sha256=identities[0],
            tokenizer_sha256=identities[1],
            chat_template_sha256=identities[2],
            source_manifest_sha256=identities[3],
            registry_sha256=identities[4],
            sampler=sampler,
            record_count=count,
        )

    def records(
        self,
        selected_features: frozenset[tuple[int, int]] | None = None,
    ) -> Iterator[TraceRecord]:
        """Read records, optionally retaining only authenticated feature taps.

        A selective read is allowed only after the caller opened this file with
        an envelope digest established by a full corpus-authentication pass.
        The parser still consumes and bounds every field; it merely avoids
        copying tens or hundreds of megabytes of unused feature payloads.
        """
        reader = self._reader
        previous: TraceRecord | None = None
        for index in range(self.identity.record_count):
            sequence_id = reader.u64()
            turn_index = reader.u32()
            kind = reader.u8()
            terminal_raw = reader.u8()
            parent_raw = reader.u64()
            target_position = reader.u32()
            draft_position = reader.u32()
            token_id = reader.u32()
            has_label = reader.u8()
            label_raw = reader.u32()
            sampler_draw_count = reader.u64()
            source_group, source_cluster, conversation, prompt_derivation = (
                reader.take(32) for _ in range(4)
            )
            split, workload = reader.u8(), reader.u8()
            context_generation, context_flags = reader.u32(), reader.u32()
            generation_seed = reader.u64()
            remaining_budget = reader.u32()
            retrieval_eligible, has_retrieval = reader.u8(), reader.u8()
            retrieval_raw = reader.u32()
            has_selection = reader.u8()
            if (terminal_raw > 1 or has_label > 1 or retrieval_eligible > 1
                    or has_retrieval > 1 or has_selection > 1 or draft_position != target_position + 1
                    or kind not in (0, 1, 2) or split not in (1, 2, 3, 4)
                    or sequence_id == 0 or target_position == 2**32 - 1
                    or workload < 1 or workload > 8 or context_flags & ~0x7F
                    or (has_retrieval and not retrieval_eligible)
                    or any(value == bytes(32) for value in (
                        source_group, source_cluster, conversation, prompt_derivation
                    ))):
                raise TraceError(f"record {index} metadata is invalid")
            selection = self._read_selection(
                reader, label_raw if has_label else None
            ) if has_selection else None
            feature_count = reader.u16()
            if feature_count == 0 or feature_count > MAX_FEATURES:
                raise TraceError(f"record {index} feature count is invalid")
            decoded_features = (
                self._read_feature(reader, target_position, kind, selected_features)
                for _ in range(feature_count)
            )
            features = tuple(feature for feature in decoded_features if feature is not None)
            if selected_features is not None:
                actual = {(feature.stage, feature.layer) for feature in features}
                if actual != selected_features:
                    raise TraceError(f"record {index} lacks a selected feature tap")
            record = TraceRecord(
                sequence_id=sequence_id, turn_index=turn_index, kind=kind,
                terminal=bool(terminal_raw),
                parent=None if parent_raw == NO_PARENT else parent_raw,
                target_position=target_position, token_id=token_id,
                target_label=label_raw if has_label else None,
                source_group=source_group, source_cluster=source_cluster,
                conversation=conversation, prompt_derivation=prompt_derivation,
                split=split, workload=workload, generation_seed=generation_seed,
                context_generation=context_generation, context_flags=context_flags,
                remaining_budget=remaining_budget,
                retrieval_eligible=bool(retrieval_eligible),
                retrieval_proposal=retrieval_raw if has_retrieval else None,
                selection=selection, features=features,
            )
            if index == 0:
                if record.parent is not None:
                    raise TraceError("root record has a parent")
            elif (record.parent != index - 1 or previous is None or previous.terminal
                  or previous.sequence_id != record.sequence_id
                  or previous.turn_index != record.turn_index
                  or previous.source_group != record.source_group
                  or previous.source_cluster != record.source_cluster
                  or previous.conversation != record.conversation
                  or previous.prompt_derivation != record.prompt_derivation
                  or previous.split != record.split
                  or previous.workload != record.workload
                  or previous.context_generation != record.context_generation
                  or previous.generation_seed != record.generation_seed
                  or previous.target_position + 1 != record.target_position
                  or previous.target_label != record.token_id):
                raise TraceError(f"record {index} causal edge is invalid")
            if selection is not None and (record.target_label is None
                    or sampler_draw_count != selection.pre_draw_count):
                raise TraceError(f"record {index} sampler evidence is inconsistent")
            if selection is not None:
                if selection.consumes_random_draw:
                    cumulative = 0.0
                    reproduced = selection.prepared[-1][0]
                    for candidate in selection.prepared:
                        cumulative += candidate[3]
                        if selection.mapped_draw < cumulative:
                            reproduced = candidate[0]
                            break
                else:
                    reproduced = selection.prepared[0][0]
                if reproduced != record.target_label:
                    raise TraceError(f"record {index} raw draw does not reproduce selected token")
            if selection is None and sampler_draw_count != 0:
                raise TraceError(f"record {index} unlabeled row consumed a draw")
            if selection is None and not (
                    record.kind == 0 or (
                        record.kind == 1 and record.terminal and record.target_label is None
                    )):
                raise TraceError(f"record {index} unlabeled row kind is invalid")
            previous = record
            yield record
        if previous is None or not previous.terminal or reader.offset != len(reader.data):
            raise TraceError("trace did not end exactly at one terminal record")

    def _read_selection(self, reader: _Reader, selected_token: int | None) -> TraceSelection:
        top_k, base_count, history_count = reader.u32(), reader.u32(), reader.u32()
        request_history_storage = tuple(reader.u32() for _ in range(HISTORY_CAPACITY))
        pre = self._read_history(reader)
        pre_draw = reader.u64()
        post = self._read_history(reader)
        post_draw = reader.u64()
        raw_count = reader.u32()
        raw_storage = tuple(reader.u64() for _ in range(RAW_DRAW_CAPACITY))
        draw_index, unit, mapped = reader.u64(), reader.f64(), reader.f64()
        consumes = reader.u8()
        temperature, configured_top_k, _, _, repetition_penalty, _, window = self.identity.sampler
        expected_top_k = 1 if temperature == 0.0 else configured_top_k
        expected_history = self._selection_history(pre, repetition_penalty)
        if (selected_token is None or top_k == 0 or top_k > 256
                or top_k != expected_top_k or history_count > HISTORY_CAPACITY
                or base_count != top_k + history_count or raw_count > RAW_DRAW_CAPACITY
                or tuple(request_history_storage[:history_count]) != expected_history
                or any(request_history_storage[history_count:])
                or pre[2] != window or post[2] != window
                or consumes > 1 or bool(consumes) != (temperature != 0.0)
                or (bool(consumes) and raw_count == 0)
                or (not bool(consumes) and raw_count != 0)
                or any(raw_storage[raw_count:])
                or draw_index != pre_draw or post_draw != pre_draw + consumes
                or not math.isfinite(unit) or unit < 0.0 or unit >= 1.0
                or not math.isfinite(mapped)):
            raise TraceError(
                "selection request/draw bounds are invalid: "
                f"top_k={top_k}/{expected_top_k}, history={history_count}/"
                f"{len(expected_history)}, windows={pre[2]}/{post[2]}/{window}, "
                f"draws={raw_count}/{consumes}, indices={pre_draw}/{post_draw}/{draw_index}, "
                f"unit={unit}, mapped={mapped}, request_tail={any(request_history_storage[history_count:])}, "
                f"raw_tail={any(raw_storage[raw_count:])}"
            )
        if self._append_history(pre, selected_token) != post:
            raise TraceError("selection sampler history transition is invalid")
        compact_count = reader.u32()
        if compact_count == 0 or compact_count > 384:
            raise TraceError("compact candidate count is invalid")
        compact = tuple((reader.u32(), reader.f32()) for _ in range(compact_count))
        prepared_count = reader.u32()
        if prepared_count == 0 or prepared_count > 256:
            raise TraceError("prepared candidate count is invalid")
        prepared = tuple(
            (reader.u32(), reader.f32(), reader.f64(), reader.f64())
            for _ in range(prepared_count)
        )
        total = reader.f64()
        if pre[2] > HISTORY_CAPACITY or post[2] > HISTORY_CAPACITY:
            raise TraceError("sampler history window is invalid")
        selection = TraceSelection(
            pre_draw_count=pre_draw, post_draw_count=post_draw,
            raw_draws=raw_storage[:raw_count], consumes_random_draw=bool(consumes),
            draw_index=draw_index,
            unit_draw=unit, mapped_draw=mapped, compact=compact,
            prepared=prepared, total_weight=total, request_top_k=top_k,
            request_history=request_history_storage[:history_count],
        )
        self._validate_prepared(selection)
        if (selection.consumes_random_draw
                and not 0.0 <= selection.mapped_draw < selection.total_weight):
            raise TraceError("selection mapped draw is outside its distribution")
        if (not selection.consumes_random_draw
                and (selection.unit_draw != 0.0 or selection.mapped_draw != 0.0)):
            raise TraceError("greedy selection contains a random draw")
        return selection

    def _validate_prepared(self, selection: TraceSelection) -> None:
        temperature, _, top_p, min_p, repetition_penalty, _, _ = self.identity.sampler
        candidates: dict[int, float] = {}
        for token, logit in selection.compact:
            if not math.isfinite(logit):
                raise TraceError("compact candidate logit is nonfinite")
            if token in candidates and struct.pack("<f", candidates[token]) != struct.pack("<f", logit):
                raise TraceError("duplicate compact candidate logits differ")
            candidates[token] = logit
        if len(candidates) < selection.request_top_k:
            raise TraceError("compact candidate set is smaller than requested top-k")
        history = set(selection.request_history)
        adjusted = []
        for token, logit in candidates.items():
            if token in history:
                logit = _f32(logit * repetition_penalty if logit < 0.0 else logit / repetition_penalty)
            adjusted.append((token, logit))
        adjusted.sort(key=lambda item: (-item[1], item[0]))
        adjusted = adjusted[:selection.request_top_k]
        if temperature == 0.0:
            expected = [(adjusted[0][0], 1.0, 1.0)]
            normalization = 1.0
        else:
            maximum = adjusted[0][1]
            pre_weights = [math.exp(float(_f32(logit - maximum))) for _, logit in adjusted]
            full_total = sum(pre_weights)
            retained = 0.0
            count = 0
            while True:
                retained += pre_weights[count]
                count += 1
                if count >= len(adjusted) or retained / full_total >= top_p:
                    break
            if min_p > 0.0:
                min_count = 1
                while min_count < count and pre_weights[min_count] >= min_p:
                    min_count += 1
                count = min_count
            adjusted = adjusted[:count]
            weights = [math.exp(float(_f32(logit - maximum)) / temperature) for _, logit in adjusted]
            normalization = sum(weights)
            expected = [
                (item[0], weight / normalization, weight)
                for item, weight in zip(adjusted, weights)
            ]
        if len(expected) != len(selection.prepared) or not _close(normalization, selection.total_weight):
            raise TraceError("prepared distribution denominator changed")
        compact_map = dict(selection.compact)
        for wanted, stored in zip(expected, selection.prepared):
            token, compact_logit, probability, weight = stored
            if (token != wanted[0] or token not in compact_map
                    or struct.pack("<f", compact_logit) != struct.pack("<f", compact_map[token])
                    or not _close(probability, wanted[1]) or not _close(weight, wanted[2])):
                raise TraceError("prepared distribution entry changed")
    @staticmethod
    def _read_history(reader: _Reader) -> tuple[int, int, int, tuple[int, ...]]:
        head, length, window = reader.u32(), reader.u32(), reader.u32()
        storage = tuple(reader.u32() for _ in range(HISTORY_CAPACITY))
        if (length > window or window > HISTORY_CAPACITY
                or (window == 0 and head != 0) or (window != 0 and head >= window)):
            raise TraceError("sampler history bounds are invalid")
        return head, length, window, storage

    @staticmethod
    def _selection_history(
        history: tuple[int, int, int, tuple[int, ...]], repetition_penalty: float,
    ) -> tuple[int, ...]:
        head, length, window, storage = history
        if repetition_penalty == 1.0:
            return ()
        result: list[int] = []
        for offset in range(length):
            token = storage[(head + offset) % window]
            if token not in result:
                result.append(token)
        return tuple(result)

    @staticmethod
    def _append_history(
        history: tuple[int, int, int, tuple[int, ...]], token: int,
    ) -> tuple[int, int, int, tuple[int, ...]]:
        head, length, window, immutable_storage = history
        storage = list(immutable_storage)
        if window != 0:
            if length < window:
                storage[(head + length) % window] = token
                length += 1
            else:
                storage[head] = token
                head = (head + 1) % window
        return head, length, window, tuple(storage)

    @staticmethod
    def _read_feature(
        reader: _Reader,
        record_position: int,
        record_kind: int,
        selected_features: frozenset[tuple[int, int]] | None,
    ) -> TraceFeature | None:
        execution_epoch, arena_epoch = reader.u64(), reader.u64()
        bank, child = reader.u32(), reader.u32()
        stage, layer = reader.u8(), reader.u16()
        target_position, draft_position = reader.u32(), reader.u32()
        rows, width, dtype, byte_count = reader.u32(), reader.u32(), reader.u8(), reader.u32()
        expected_digest = reader.take(32)
        if (execution_epoch == 0 or bank == 0 or child == 0
                or stage not in (0, 1, 2) or (stage != 1 and layer != 0)
                or (record_kind == 2) != (arena_epoch != 0)
                or rows != 1 or width == 0 or dtype != 1
                or target_position != record_position or draft_position != target_position + 1
                or byte_count != width * 2 or byte_count > MAX_FEATURE_BYTES):
            raise TraceError("feature metadata is invalid")
        retained = selected_features is None or (stage, layer) in selected_features
        if retained:
            payload = reader.take(byte_count)
            if hashlib.sha256(payload).digest() != expected_digest:
                raise TraceError("feature payload SHA-256 mismatch")
        else:
            reader.skip(byte_count)
            return None
        return TraceFeature(
            stage=stage, layer=layer, target_position=target_position,
            width=width, dtype=dtype, payload=payload,
        )


def discover_traces(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.lgtrace"))
    if not paths:
        raise TraceError(f"no .lgtrace files found in {directory}")
    return paths

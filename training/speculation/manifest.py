"""Immutable source-cluster splitting and collection-manifest publication."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "local-gguf-engine/source-split/v1"
MANIFEST_MAGIC = "LGSOURCE4"
SPLITS = (
    ("training_development", 1, 0.80),
    ("architecture_selection", 2, 0.10),
    ("calibration", 3, 0.05),
    ("sealed_admission", 4, 0.05),
)
WORKLOADS = {
    "instruction_chat": 1,
    "factual_grounded_qa": 2,
    "code": 3,
    "mathematics_reasoning": 4,
    "summarization_translation": 5,
    "structured_tool": 6,
    "rare_non_english": 7,
    "repeated_retrieval": 8,
}
DEFAULT_GENERATION_SEEDS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
)


class ManifestError(ValueError):
    pass


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def load_split_receipt(path: Path, registry_digest: str) -> dict[str, object]:
    def is_sha256(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            return len(bytes.fromhex(value)) == 32
        except ValueError:
            return False

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"split receipt is unreadable or malformed: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ManifestError("split receipt schema is unsupported")
    claimed = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if (not is_sha256(claimed)
            or hashlib.sha256(canonical_json(unsigned)).hexdigest() != claimed):
        raise ManifestError("split receipt digest is invalid")
    if value.get("registry_sha256") != registry_digest:
        raise ManifestError("split receipt registry identity is stale")
    manifests = value.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {name for name, _, _ in SPLITS}:
        raise ManifestError("split receipt manifest set is incomplete")
    for name, entry in manifests.items():
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256", "turn_seed_rows"}
                or not isinstance(entry["path"], str) or not entry["path"]
                or not is_sha256(entry["sha256"])
                or not isinstance(entry["turn_seed_rows"], int) or entry["turn_seed_rows"] < 0):
            raise ManifestError(f"split receipt entry {name} is invalid")
    return value


def digest_text(domain: str, *values: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode())
    digest.update(b"\0")
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def shingles(text: str, width: int = 5) -> frozenset[str]:
    words = re.findall(r"[\w']+", normalize_text(text), flags=re.UNICODE)
    if len(words) < width:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[index:index + width]) for index in range(len(words) - width + 1))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


@dataclass(frozen=True)
class Turn:
    prompt: str
    maximum_new_tokens: int = 0


@dataclass(frozen=True)
class Conversation:
    source_group: str
    root_source: str
    prompt_family: str
    topic: str
    conversation_id: str
    workload: str
    turns: tuple[Turn, ...]

    @property
    def joined_text(self) -> str:
        return "\n".join(turn.prompt for turn in self.turns)


def _required_text(raw: dict[str, object], key: str, line: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ManifestError(f"line {line}: {key} must be nonempty NUL-free text")
    return value


def load_source_jsonl(path: Path) -> list[Conversation]:
    conversations: list[Conversation] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ManifestError(f"line {line_number}: blank rows are forbidden")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ManifestError(f"line {line_number}: row must be an object")
            identifier = _required_text(raw, "conversation_id", line_number)
            if identifier in seen:
                raise ManifestError(f"line {line_number}: duplicate conversation_id {identifier!r}")
            seen.add(identifier)
            workload = _required_text(raw, "workload", line_number)
            if workload not in WORKLOADS:
                raise ManifestError(f"line {line_number}: unknown workload {workload!r}")
            raw_turns = raw.get("turns")
            if not isinstance(raw_turns, list) or not raw_turns:
                raise ManifestError(f"line {line_number}: turns must be a nonempty list")
            turns: list[Turn] = []
            for raw_turn in raw_turns:
                if isinstance(raw_turn, str):
                    prompt, maximum = raw_turn, 0
                elif isinstance(raw_turn, dict) and set(raw_turn) == {"prompt", "maximum_new_tokens"}:
                    prompt, maximum = raw_turn["prompt"], raw_turn["maximum_new_tokens"]
                else:
                    raise ManifestError(
                        f"line {line_number}: each turn must be text or exact prompt/cap object"
                    )
                if (not isinstance(prompt, str) or not prompt or "\0" in prompt
                        or not isinstance(maximum, int) or isinstance(maximum, bool)
                        or maximum < 0 or maximum >= 2**64):
                    raise ManifestError(f"line {line_number}: turn prompt/cap is invalid")
                turns.append(Turn(prompt, maximum))
            conversations.append(Conversation(
                source_group=_required_text(raw, "source_group", line_number),
                root_source=_required_text(raw, "root_source", line_number),
                prompt_family=_required_text(raw, "prompt_family", line_number),
                topic=_required_text(raw, "topic", line_number),
                conversation_id=identifier,
                workload=workload,
                turns=tuple(turns),
            ))
    if not conversations:
        raise ManifestError("source JSONL is empty")
    return conversations


def source_components(conversations: list[Conversation], near_duplicate_threshold: float) -> list[list[int]]:
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ManifestError("near-duplicate threshold must be in (0,1]")
    union = UnionFind(len(conversations))
    exact_owners: dict[tuple[str, str], int] = {}
    text_sets = [shingles(item.joined_text) for item in conversations]
    shingle_owners: dict[str, list[int]] = {}
    empty_texts: list[int] = []
    for index, item in enumerate(conversations):
        identities = (
            ("root_source", normalize_text(item.root_source)),
            ("source_group", normalize_text(item.source_group)),
            ("prompt_family_topic", normalize_text(item.prompt_family) + "\0" + normalize_text(item.topic)),
            ("exact_text", normalize_text(item.joined_text)),
        )
        for identity in identities:
            if identity in exact_owners:
                union.union(index, exact_owners[identity])
            else:
                exact_owners[identity] = index
        # Jaccard-positive pairs must share at least one shingle. An inverted
        # index therefore gives exact candidate discovery without the false
        # negatives of probabilistic LSH banding. Empty shingle sets retain the
        # explicit Jaccard(empty, empty)=1 convention.
        candidates = set(empty_texts if not text_sets[index] else ())
        for shingle in text_sets[index]:
            candidates.update(shingle_owners.get(shingle, ()))
        for other in sorted(candidates):
            intersection = len(text_sets[index] & text_sets[other])
            combined = len(text_sets[index] | text_sets[other])
            similarity = intersection / combined if combined else 1.0
            if similarity >= near_duplicate_threshold:
                union.union(index, other)
        if text_sets[index]:
            for shingle in text_sets[index]:
                shingle_owners.setdefault(shingle, []).append(index)
        else:
            empty_texts.append(index)
    components: dict[int, list[int]] = {}
    for index in range(len(conversations)):
        components.setdefault(union.find(index), []).append(index)
    return sorted(components.values(), key=lambda group: tuple(
        conversations[index].conversation_id for index in group
    ))


def assign_splits(
    conversations: list[Conversation],
    components: list[list[int]],
    split_salt: str,
) -> dict[int, tuple[str, int, str]]:
    assignments: dict[int, tuple[str, int, str]] = {}
    thresholds = []
    cumulative = 0.0
    for name, number, weight in SPLITS:
        cumulative += weight
        thresholds.append((name, number, cumulative))
    for component in components:
        members = sorted(conversations[index].conversation_id for index in component)
        cluster = digest_text("source-cluster/v1", *members)
        draw = int(digest_text("source-split-draw/v1", split_salt, cluster)[:16], 16) / 2**64
        name, number = thresholds[-1][:2]
        for candidate_name, candidate_number, threshold in thresholds:
            if draw < threshold:
                name, number = candidate_name, candidate_number
                break
        for index in component:
            assignments[index] = (name, number, cluster)
    return assignments


def atomic_publish(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def publish_manifests(
    source_path: Path,
    output_directory: Path,
    registry_path: Path,
    split_salt: str,
    near_duplicate_threshold: float = 0.82,
) -> dict[str, object]:
    conversations = load_source_jsonl(source_path)
    registry_bytes = registry_path.read_bytes()
    registry_digest = hashlib.sha256(registry_bytes).hexdigest()
    registry = json.loads(registry_bytes)
    seeds = tuple(registry.get("target_generation_seeds", ()))
    if seeds != DEFAULT_GENERATION_SEEDS:
        raise ManifestError("registry target_generation_seeds do not match the frozen five-seed protocol")
    components = source_components(conversations, near_duplicate_threshold)
    assignments = assign_splits(conversations, components, split_salt)
    rows_by_split: dict[str, list[tuple[tuple[object, ...], tuple[object, ...]]]] = {
        name: [] for name, _, _ in SPLITS
    }
    for index, item in enumerate(conversations):
        split_name, split_number, cluster = assignments[index]
        source_group = digest_text("source-group/v1", item.source_group)
        conversation = digest_text("conversation/v1", item.conversation_id)
        for seed in seeds:
            for turn_index, turn in enumerate(item.turns):
                derivation = digest_text(
                    "prompt-derivation/v1", item.conversation_id, str(turn_index),
                    str(turn.maximum_new_tokens), turn.prompt
                )
                fields = (
                    turn_index, source_group, cluster, conversation, derivation,
                    split_number, WORKLOADS[item.workload], seed, turn.maximum_new_tokens,
                    base64.b64encode(turn.prompt.encode()).decode("ascii"),
                )
                rows_by_split[split_name].append(((item.conversation_id, seed, turn_index), fields))
    manifest_receipts: dict[str, dict[str, object]] = {}
    sequence_id = 1
    for split_name, _, _ in SPLITS:
        lines = [f"{MANIFEST_MAGIC}\t{registry_digest}"]
        for _, fields in sorted(rows_by_split[split_name]):
            lines.append("\t".join(str(value) for value in (sequence_id,) + fields))
            sequence_id += 1
        payload = ("\n".join(lines) + "\n").encode()
        path = output_directory / f"{split_name}.lgsmanifest"
        atomic_publish(path, payload)
        manifest_receipts[split_name] = {
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "turn_seed_rows": len(lines) - 1,
        }
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "source_jsonl_sha256": source_digest,
        "registry_sha256": registry_digest,
        "split_salt_sha256": hashlib.sha256(split_salt.encode()).hexdigest(),
        "near_duplicate_threshold": near_duplicate_threshold,
        "conversation_count": len(conversations),
        "source_cluster_count": len(components),
        "target_generation_seeds": list(seeds),
        "manifests": manifest_receipts,
    }
    receipt_payload = canonical_json(receipt)
    receipt["receipt_sha256"] = hashlib.sha256(receipt_payload).hexdigest()
    atomic_publish(output_directory / "split_receipt.json", canonical_json(receipt))
    return receipt

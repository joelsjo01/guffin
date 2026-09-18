"""Pinned public-source download and deterministic Phase-5 prompt-corpus construction."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

from .manifest import WORKLOADS, canonical_json


SOURCE_SCHEMA = "local-gguf-engine/phase5-corpus-sources/v1"
RECEIPT_SCHEMA = "local-gguf-engine/phase5-prompt-corpus/v1"
SELECTION_DOMAIN = "local-gguf-engine/phase5-public-corpus-selection/v1"
MAXIMUM_SOURCE_BYTES = 1 << 30
MAXIMUM_PROMPT_BYTES = 32 * 1024
REQUIRED_SOURCE_IDS = frozenset({
    "stanford-alpaca",
    "code-alpaca-20k",
    "databricks-dolly-15k",
    "gsm8k-train",
    "hermes-function-calling-singleturn",
    "hermes-json-mode-singleturn",
    "openassistant-oasst2-ready-messages",
    "squad-v2-train",
})


class CorpusBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceDefinition:
    identifier: str
    revision: str
    url: str
    filename: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class PromptCandidate:
    source_id: str
    source_key: str
    root_key: str
    prompt_family: str
    topic: str
    workload: str
    prompts: tuple[str, ...]
    stratum: str = ""


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_definitions(path: Path) -> tuple[str, tuple[SourceDefinition, ...]]:
    payload = path.read_bytes()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CorpusBuildError(f"source registry is malformed: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"schema", "sources"} \
            or raw.get("schema") != SOURCE_SCHEMA or not isinstance(raw.get("sources"), list):
        raise CorpusBuildError("source registry schema is unsupported")
    definitions = []
    identifiers: set[str] = set()
    filenames: set[str] = set()
    required_fields = {"id", "revision", "url", "filename", "byte_count", "sha256"}
    for item in raw["sources"]:
        if not isinstance(item, dict) or set(item) != required_fields:
            raise CorpusBuildError("source registry entry has an invalid field set")
        identifier = item["id"]
        revision = item["revision"]
        url = item["url"]
        filename = item["filename"]
        byte_count = item["byte_count"]
        sha256 = item["sha256"]
        if (not isinstance(identifier, str) or not identifier
                or identifier in identifiers or not isinstance(revision, str) or not revision
                or not isinstance(url, str) or not url.startswith("https://")
                or not isinstance(filename, str) or not filename
                or Path(filename).name != filename or filename in filenames
                or not isinstance(byte_count, int) or isinstance(byte_count, bool)
                or byte_count <= 0 or byte_count > MAXIMUM_SOURCE_BYTES
                or not _is_sha256(sha256)):
            raise CorpusBuildError("source registry entry is malformed or unsafe")
        identifiers.add(identifier)
        filenames.add(filename)
        definitions.append(SourceDefinition(
            identifier, revision, url, filename, byte_count, sha256
        ))
    if identifiers != REQUIRED_SOURCE_IDS:
        raise CorpusBuildError(
            f"source registry coverage differs: {sorted(identifiers ^ REQUIRED_SOURCE_IDS)}"
        )
    return hashlib.sha256(payload).hexdigest(), tuple(definitions)


def _validate_cached_source(source: SourceDefinition, path: Path) -> None:
    if not path.is_file():
        raise CorpusBuildError(f"source cache entry is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.byte_count:
        raise CorpusBuildError(
            f"source byte count mismatch for {source.identifier}: "
            f"expected {source.byte_count}, found {actual_size}"
        )
    actual_digest = file_sha256(path)
    if actual_digest != source.sha256:
        raise CorpusBuildError(
            f"source SHA-256 mismatch for {source.identifier}: "
            f"expected {source.sha256}, found {actual_digest}"
        )


def _download_source(
    source: SourceDefinition,
    cache_directory: Path,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    destination = cache_directory / source.filename
    if destination.exists():
        _validate_cached_source(source, destination)
        return destination
    cache_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=source.filename + ".", suffix=".partial", dir=cache_directory
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "local-gguf-engine-phase5-corpus/1"},
        )
        with opener(request, timeout=120) as response, os.fdopen(descriptor, "wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                total += len(chunk)
                if total > source.byte_count:
                    raise CorpusBuildError(f"download exceeded pinned size for {source.identifier}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != source.byte_count or digest.hexdigest() != source.sha256:
            raise CorpusBuildError(
                f"download identity mismatch for {source.identifier}; upstream bytes drifted"
            )
        os.replace(temporary, destination)
        return destination
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def acquire_sources(
    definitions: Iterable[SourceDefinition], cache_directory: Path,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Path]:
    result = {}
    for source in definitions:
        result[source.identifier] = _download_source(source, cache_directory, opener)
    return result


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value or "\0" in value or len(value.encode("utf-8")) > MAXIMUM_PROMPT_BYTES:
        return None
    return value


def _instruction_prompt(instruction: object, context: object) -> str | None:
    instruction_text = _clean_text(instruction)
    context_text = _clean_text(context) if context else None
    if instruction_text is None:
        return None
    prompt = instruction_text
    if context_text is not None:
        prompt += "\n\nAdditional input:\n" + context_text
    return _clean_text(prompt)


def _json_list(path: Path) -> list[dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusBuildError(f"source JSON is unreadable: {path}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CorpusBuildError(f"source JSON must be an object list: {path}")
    return value


def _json_lines(path: Path) -> list[dict[str, object]]:
    result = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise CorpusBuildError(f"blank JSONL row at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CorpusBuildError(f"non-object JSONL row at {path}:{line_number}")
                result.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusBuildError(f"source JSONL is unreadable: {path}") from error
    return result


def _candidate(
    source_id: str,
    source_key: str,
    root_key: str,
    prompt_family: str,
    topic: str,
    workload: str,
    prompts: Iterable[object],
    stratum: str = "",
) -> PromptCandidate | None:
    cleaned = tuple(_clean_text(prompt) for prompt in prompts)
    if (not cleaned or any(prompt is None for prompt in cleaned)
            or workload not in WORKLOADS):
        return None
    return PromptCandidate(
        source_id, source_key, root_key, prompt_family, topic, workload,
        tuple(prompt for prompt in cleaned if prompt is not None), stratum,
    )


def _oasst_candidates(path: Path) -> tuple[list[PromptCandidate], list[PromptCandidate]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusBuildError(f"OpenAssistant source is unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise CorpusBuildError("OpenAssistant source contains a non-object row")
    nodes = {
        row.get("message_id"): row for row in rows
        if isinstance(row.get("message_id"), str)
        and row.get("role") in {"prompter", "assistant"}
        and row.get("deleted") is False
        and row.get("review_result") is True
        and _clean_text(row.get("text")) is not None
    }
    children: dict[str, list[dict[str, object]]] = defaultdict(list)
    roots = []
    for row in nodes.values():
        parent = row.get("parent_id")
        if parent is None and row.get("role") == "prompter":
            roots.append(row)
        elif isinstance(parent, str) and parent in nodes:
            children[parent].append(row)

    def child_key(row: dict[str, object]) -> tuple[int, str]:
        rank = row.get("rank")
        return (rank if isinstance(rank, int) else 1 << 30, str(row["message_id"]))

    english = []
    multilingual = []
    for root in roots:
        root_id = str(root["message_id"])
        language = str(root.get("lang", "unknown"))
        prompts = [str(root["text"])]
        if len(prompts[0].encode("utf-8")) < 16:
            continue
        current = root
        while len(prompts) < 3:
            assistant_children = sorted(
                (item for item in children.get(str(current["message_id"]), ())
                 if item.get("role") == "assistant"),
                key=child_key,
            )
            if not assistant_children:
                break
            assistant = assistant_children[0]
            prompter_children = sorted(
                (item for item in children.get(str(assistant["message_id"]), ())
                 if item.get("role") == "prompter" and item.get("lang") == language),
                key=child_key,
            )
            if not prompter_children:
                break
            current = prompter_children[0]
            prompts.append(str(current["text"]))
        item = _candidate(
            "openassistant-oasst2-ready-messages", root_id, root_id,
            "oasst2-conversation", f"oasst2-tree:{root_id}",
            "instruction_chat" if language == "en" else "rare_non_english",
            prompts if language == "en" else prompts[:1],
            ("multi_turn" if len(prompts) > 1 else "single_turn")
            if language == "en" else language,
        )
        if item is not None:
            (english if language == "en" else multilingual).append(item)
    return english, multilingual


def _alpaca_translation_candidates(path: Path) -> list[PromptCandidate]:
    result = []
    pattern = re.compile(r"\b(?:translate|translation)\b", re.IGNORECASE)
    for index, row in enumerate(_json_list(path)):
        instruction = _clean_text(row.get("instruction"))
        if instruction is None or pattern.search(instruction) is None:
            continue
        prompt = _instruction_prompt(instruction, row.get("input"))
        item = _candidate(
            "stanford-alpaca", str(index), str(index), "alpaca-translation",
            f"alpaca:{index}", "summarization_translation", (prompt,), "translation",
        )
        if item is not None:
            result.append(item)
    return result


def _code_candidates(path: Path) -> list[PromptCandidate]:
    result = []
    for index, row in enumerate(_json_list(path)):
        prompt = _instruction_prompt(row.get("instruction"), row.get("input"))
        item = _candidate(
            "code-alpaca-20k", str(index), str(index), "code-alpaca-instruction",
            f"code-alpaca:{index}", "code", (prompt,),
        )
        if item is not None:
            result.append(item)
    return result


def _dolly_summary_candidates(path: Path) -> list[PromptCandidate]:
    result = []
    for index, row in enumerate(_json_lines(path)):
        if row.get("category") != "summarization":
            continue
        prompt = _instruction_prompt(row.get("instruction"), row.get("context"))
        item = _candidate(
            "databricks-dolly-15k", str(index), str(index), "dolly-summarization",
            f"dolly:{index}", "summarization_translation", (prompt,), "summarization",
        )
        if item is not None:
            result.append(item)
    return result


def _math_candidates(path: Path) -> list[PromptCandidate]:
    result = []
    for index, row in enumerate(_json_lines(path)):
        question = _clean_text(row.get("question"))
        prompt = None if question is None else "Solve this problem carefully.\n\n" + question
        item = _candidate(
            "gsm8k-train", str(index), str(index), "gsm8k-problem",
            f"gsm8k:{index}", "mathematics_reasoning", (prompt,),
        )
        if item is not None:
            result.append(item)
    return result


def _hermes_candidates(path: Path, source_id: str, family: str, stratum: str) \
        -> list[PromptCandidate]:
    result = []
    for index, row in enumerate(_json_list(path)):
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            continue
        system = next((item.get("value") for item in conversations
                       if isinstance(item, dict) and item.get("from") == "system"), None)
        human = next((item.get("value") for item in conversations
                      if isinstance(item, dict) and item.get("from") == "human"), None)
        system_text = _clean_text(system)
        human_text = _clean_text(human)
        prompt = None if system_text is None or human_text is None else (
            system_text + "\n\nUser request:\n" + human_text
        )
        if prompt is not None and len(prompt.encode("utf-8")) >= 8 * 1024:
            continue
        identifier = row.get("id")
        key = str(identifier) if isinstance(identifier, str) and identifier else str(index)
        topic = ":".join(str(row.get(name, "unknown")) for name in ("category", "subcategory"))
        item = _candidate(
            source_id, key, key, family, f"hermes:{source_id}:{key}:{topic}",
            "structured_tool", (prompt,), stratum,
        )
        if item is not None:
            result.append(item)
    return result


def _squad_prompt(context: str, question: str, first: bool) -> str:
    if first:
        return (
            "Use only the following reference text when answering.\n\n"
            f"Reference text:\n{context}\n\nQuestion:\n{question}"
        )
    return "Using the same reference text from the first turn, answer:\n" + question


def _squad_candidates(path: Path) -> tuple[list[PromptCandidate], list[PromptCandidate]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusBuildError(f"SQuAD source is unreadable: {path}") from error
    articles = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(articles, list):
        raise CorpusBuildError("SQuAD source has no data array")
    factual = []
    repeated = []
    for article_index, article in enumerate(articles):
        if not isinstance(article, dict) or not isinstance(article.get("paragraphs"), list):
            continue
        title = _clean_text(article.get("title")) or f"article-{article_index}"
        article_root = f"squad-article:{article_index}:{title}"
        paragraph_material: list[tuple[str, list[tuple[str, str]]]] = []
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            if not isinstance(paragraph, dict):
                continue
            context = _clean_text(paragraph.get("context"))
            qas = paragraph.get("qas")
            if context is None or not isinstance(qas, list):
                continue
            questions = []
            for qa in qas:
                if not isinstance(qa, dict) or qa.get("is_impossible") is True:
                    continue
                question = _clean_text(qa.get("question"))
                identifier = qa.get("id")
                if question is None or not isinstance(identifier, str) or not identifier:
                    continue
                questions.append((identifier, question))
                prompt = _squad_prompt(context, question, True)
                item = _candidate(
                    "squad-v2-train", identifier,
                    article_root, "squad-grounded-qa",
                    f"{article_root}:paragraph:{paragraph_index}", "factual_grounded_qa",
                    (prompt,),
                )
                if item is not None:
                    factual.append(item)
            if questions:
                paragraph_material.append((context, questions))
            if len(questions) >= 2:
                chosen = sorted(questions, key=lambda value: value[0])[:3]
                prompts = [_squad_prompt(context, question, offset == 0)
                           for offset, (_, question) in enumerate(chosen)]
                item = _candidate(
                    "squad-v2-train", f"article-{article_index}-paragraph-{paragraph_index}",
                    article_root, "squad-repeated-paragraph", article_root,
                    "repeated_retrieval", prompts, "short",
                )
                if item is not None:
                    repeated.append(item)
        for stratum, limit in (("medium", 6_000), ("long", 26_000)):
            contexts = []
            questions = []
            total = 0
            for context, paragraph_questions in paragraph_material:
                added = len(context.encode("utf-8")) + (2 if contexts else 0)
                if total + added > limit:
                    break
                contexts.append(context)
                questions.extend(paragraph_questions)
                total += added
            if len(contexts) < 2 or len(questions) < 2:
                continue
            document = "\n\n".join(contexts)
            chosen = sorted(
                questions,
                key=lambda value: hashlib.sha256(
                    (SELECTION_DOMAIN + "\0squad-question\0" + value[0]).encode()
                ).digest(),
            )[:4]
            prompts = [_squad_prompt(document, question, offset == 0)
                       for offset, (_, question) in enumerate(chosen)]
            item = _candidate(
                "squad-v2-train", f"article-{article_index}-{stratum}", article_root,
                f"squad-repeated-{stratum}", article_root,
                "repeated_retrieval", prompts, stratum,
            )
            if item is not None:
                repeated.append(item)
    return factual, repeated


def _selection_key(candidate: PromptCandidate, purpose: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(SELECTION_DOMAIN.encode())
    for value in (purpose, candidate.source_id, candidate.source_key, *candidate.prompts):
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return digest.digest()


def _take(candidates: Iterable[PromptCandidate], count: int, purpose: str) \
        -> list[PromptCandidate]:
    unique: dict[str, PromptCandidate] = {}
    for candidate in candidates:
        normalized = "\0".join(" ".join(prompt.casefold().split()) for prompt in candidate.prompts)
        unique.setdefault(normalized, candidate)
    ordered = sorted(unique.values(), key=lambda item: _selection_key(item, purpose))
    if len(ordered) < count:
        raise CorpusBuildError(
            f"{purpose} needs {count} unique conversations but only {len(ordered)} are available"
        )
    return ordered[:count]


def _take_split(
    candidates: Iterable[PromptCandidate], count: int, purpose: str,
    first_stratum: str, first_count: int,
) -> list[PromptCandidate]:
    values = list(candidates)
    first = _take(
        (item for item in values if item.stratum == first_stratum), first_count,
        purpose + ":" + first_stratum,
    )
    remainder = _take(
        (item for item in values if item.stratum != first_stratum),
        count - len(first), purpose + ":remainder",
    )
    return first + remainder


def _take_balanced_strata(
    candidates: Iterable[PromptCandidate], count: int, purpose: str,
) -> list[PromptCandidate]:
    groups: dict[str, list[PromptCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.stratum].append(candidate)
    for stratum in groups:
        groups[stratum] = sorted(
            groups[stratum], key=lambda item: _selection_key(item, purpose + ":" + stratum)
        )
    strata = sorted(groups, key=lambda value: hashlib.sha256(
        (SELECTION_DOMAIN + "\0" + purpose + "\0" + value).encode()
    ).digest())
    selected = []
    index = 0
    while len(selected) < count and strata:
        stratum = strata[index % len(strata)]
        if groups[stratum]:
            selected.append(groups[stratum].pop(0))
        else:
            strata.remove(stratum)
            index = 0
            continue
        index += 1
    if len(selected) != count:
        raise CorpusBuildError(
            f"{purpose} needs {count} conversations but only {len(selected)} are available"
        )
    return selected


def _take_retrieval_mix(
    candidates: Iterable[PromptCandidate], count: int,
) -> list[PromptCandidate]:
    values = list(candidates)
    long_count = count // 16
    medium_count = (count * 3) // 16
    short_count = count - medium_count - long_count
    selected = []
    for stratum, requested in (
        ("short", short_count), ("medium", medium_count), ("long", long_count)
    ):
        selected.extend(_take(
            (item for item in values if item.stratum == stratum), requested,
            "repeated_retrieval:" + stratum,
        ))
    return selected


def _turn_cap(candidate: PromptCandidate, turn_index: int) -> int:
    digest = hashlib.sha256(
        (SELECTION_DOMAIN + "\0cap\0" + candidate.source_id + "\0"
         + candidate.source_key + "\0" + str(turn_index)).encode()
    ).digest()
    return (32, 64, 128, 256, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)[digest[0] & 15]


def _row(candidate: PromptCandidate) -> dict[str, object]:
    identity = hashlib.sha256(
        (SELECTION_DOMAIN + "\0conversation\0" + candidate.source_id + "\0"
         + candidate.source_key).encode()
    ).hexdigest()
    root = f"{candidate.source_id}:{candidate.root_key}"
    return {
        "source_group": root,
        "root_source": root,
        "prompt_family": candidate.prompt_family,
        "topic": candidate.topic,
        "conversation_id": identity,
        "workload": candidate.workload,
        "turns": [
            {"prompt": prompt, "maximum_new_tokens": _turn_cap(candidate, index)}
            for index, prompt in enumerate(candidate.prompts)
        ],
    }


def _build_rows(paths: dict[str, Path], count: int) -> tuple[
    list[dict[str, object]], dict[str, int], dict[str, dict[str, int]], dict[str, int]
]:
    instruction, multilingual = _oasst_candidates(
        paths["openassistant-oasst2-ready-messages"]
    )
    factual, repeated = _squad_candidates(paths["squad-v2-train"])
    translations = _alpaca_translation_candidates(paths["stanford-alpaca"])
    summaries = _dolly_summary_candidates(paths["databricks-dolly-15k"])
    functions = _hermes_candidates(
        paths["hermes-function-calling-singleturn"],
        "hermes-function-calling-singleturn", "hermes-function-calling", "function_call",
    )
    json_mode = _hermes_candidates(
        paths["hermes-json-mode-singleturn"],
        "hermes-json-mode-singleturn", "hermes-json-mode", "json_mode",
    )
    pools = {
        "instruction_chat": instruction,
        "factual_grounded_qa": factual,
        "code": _code_candidates(paths["code-alpaca-20k"]),
        "mathematics_reasoning": _math_candidates(paths["gsm8k-train"]),
        "summarization_translation": summaries + translations,
        "structured_tool": functions + json_mode,
        "rare_non_english": multilingual,
        "repeated_retrieval": repeated,
    }
    selected: dict[str, list[PromptCandidate]] = {}
    selected["instruction_chat"] = _take_split(
        instruction, count, "instruction_chat", "multi_turn", count // 2
    )
    selected["summarization_translation"] = _take_split(
        pools["summarization_translation"], count, "summarization_translation",
        "translation", count // 2,
    )
    selected["structured_tool"] = _take_split(
        pools["structured_tool"], count, "structured_tool", "function_call", count // 2
    )
    selected["rare_non_english"] = _take_balanced_strata(
        multilingual, count, "rare_non_english"
    )
    selected["repeated_retrieval"] = _take_retrieval_mix(repeated, count)
    for workload in ("factual_grounded_qa", "code", "mathematics_reasoning"):
        selected[workload] = _take(pools[workload], count, workload)
    rows = [_row(candidate) for workload in sorted(selected) for candidate in selected[workload]]
    if len(rows) != count * len(WORKLOADS):
        raise CorpusBuildError("selected corpus count is incomplete")
    identifiers = [str(row["conversation_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise CorpusBuildError("selected corpus conversation identities collide")
    rows.sort(key=lambda row: str(row["conversation_id"]))
    selected_strata = {
        workload: dict(sorted(Counter(item.stratum for item in values).items()))
        for workload, values in sorted(selected.items())
    }
    selected_sources = Counter(
        item.source_id for values in selected.values() for item in values
    )
    return (
        rows,
        {workload: len(pool) for workload, pool in sorted(pools.items())},
        selected_strata,
        dict(sorted(selected_sources.items())),
    )


def _receipt_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".receipt.json")


def _load_completed(
    output_path: Path, receipt_path: Path, registry_digest: str,
    builder_digest: str, count: int,
) -> dict[str, object] | None:
    if not output_path.exists() and not receipt_path.exists():
        return None
    if output_path.is_file() and not receipt_path.exists():
        return None
    if not output_path.is_file() or not receipt_path.is_file():
        raise CorpusBuildError("corpus output/receipt publication state is invalid")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusBuildError("existing corpus receipt is unreadable") from error
    claimed = receipt.get("receipt_sha256") if isinstance(receipt, dict) else None
    unsigned = dict(receipt) if isinstance(receipt, dict) else {}
    unsigned.pop("receipt_sha256", None)
    if (receipt.get("schema") != RECEIPT_SCHEMA or not _is_sha256(claimed)
            or hashlib.sha256(canonical_json(unsigned)).hexdigest() != claimed
            or receipt.get("source_registry_sha256") != registry_digest
            or receipt.get("builder_code_sha256") != builder_digest
            or receipt.get("conversations_per_workload") != count
            or receipt.get("source_jsonl_sha256") != file_sha256(output_path)):
        raise CorpusBuildError("existing corpus output/receipt is stale or malformed")
    return receipt


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise CorpusBuildError(f"refusing to overwrite immutable output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_public_corpus(
    source_registry_path: Path,
    cache_directory: Path,
    output_path: Path,
    conversations_per_workload: int,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, object]:
    if conversations_per_workload <= 0 or conversations_per_workload > 4096:
        raise CorpusBuildError("conversations per workload must be in [1,4096]")
    registry_digest, definitions = load_source_definitions(source_registry_path)
    builder_digest = file_sha256(Path(__file__))
    receipt_path = _receipt_path(output_path)
    completed = _load_completed(
        output_path, receipt_path, registry_digest, builder_digest,
        conversations_per_workload,
    )
    if completed is not None:
        return completed
    paths = acquire_sources(definitions, cache_directory, opener)
    rows, candidate_counts, selected_strata, selected_source_counts = _build_rows(
        paths, conversations_per_workload
    )
    payload = b"".join(canonical_json(row) for row in rows)
    payload_digest = hashlib.sha256(payload).hexdigest()
    workload_counts = Counter(str(row["workload"]) for row in rows)
    turn_caps = Counter(
        int(turn["maximum_new_tokens"])
        for row in rows for turn in row["turns"]  # type: ignore[union-attr]
    )
    strata = Counter()
    for row in rows:
        strata["multi_turn" if len(row["turns"]) > 1 else "single_turn"] += 1
    source_receipts = [{
        "id": source.identifier,
        "revision": source.revision,
        "url": source.url,
        "filename": source.filename,
        "byte_count": source.byte_count,
        "sha256": source.sha256,
    } for source in definitions]
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "selection_domain": SELECTION_DOMAIN,
        "source_registry_sha256": registry_digest,
        "builder_code_sha256": builder_digest,
        "conversations_per_workload": conversations_per_workload,
        "conversation_count": len(rows),
        "turn_count": sum(len(row["turns"]) for row in rows),
        "workload_counts": dict(sorted(workload_counts.items())),
        "source_candidate_counts": candidate_counts,
        "selected_source_counts": selected_source_counts,
        "selected_stratum_counts": selected_strata,
        "conversation_shape_counts": dict(sorted(strata.items())),
        "maximum_new_tokens_counts": {
            str(key): value for key, value in sorted(turn_caps.items())
        },
        "source_jsonl_sha256": payload_digest,
        "sources": source_receipts,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    if output_path.exists():
        if not output_path.is_file() or file_sha256(output_path) != payload_digest:
            raise CorpusBuildError("unreceipted corpus output differs from deterministic rebuild")
    else:
        _atomic_write(output_path, payload)
    try:
        _atomic_write(receipt_path, canonical_json(receipt))
    except BaseException:
        # The data file is immutable and its digest is known. A retry cannot
        # safely guess whether publication completed, so leave the mismatch
        # visible for explicit operator recovery.
        raise
    return receipt

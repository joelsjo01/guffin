# Frozen chat workload

`corpus.json` is Guffin's release **performance workload**, not an official
MT-Bench evaluation or a model-quality leaderboard. It contains 32 public
MT-Bench conversations and six separately labeled project-owned cases.
The core has 64 user turns; the complete workload has 38 conversations and
80 user turns. Each conversation starts with fresh chat state, and its later
turns continue from that configuration's own generated answers.

## Core selection

The source is the original MT-Bench question release in
[FastChat revision b494d0c6b4e7935f1764f8439e75da3e66beccc7](https://github.com/lm-sys/FastChat/blob/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl).
Four conversations per original category were selected on October 6, 2026,
for task and input variety, before any timed execution of this workload. Both
published user turns are preserved verbatim, including spelling and wording.

| Category | Published question IDs | Covered tasks |
| --- | --- | --- |
| Writing | 81, 82, 87, 90 | Travel writing, concise email, fiction, editing |
| Roleplay | 95, 96, 97, 100 | Translation, technical explanation, teaching, creative perspective |
| Reasoning | 101, 105, 106, 109 | Trick question, constraint puzzle, logic, spatial reasoning |
| Math | 111, 114, 116, 119 | Geometry, probability, algebra, arithmetic |
| Coding | 121, 123, 124, 126 | File processing, HTML/CSS, debugging, algorithm design |
| Extraction | 131, 133, 134, 140 | Classification, entity extraction, numerical comparison, CSV/JSON |
| STEM | 141, 143, 145, 149 | Physics, biology, chemistry, machine learning |
| Humanities | 151, 154, 155, 157 | Economics, lesson planning, art, argument analysis |

This balanced subset supplies breadth, not a statistical sample of all chat
traffic. No case was selected for speed, draft acceptance, or favorable answers.
The upstream source contains 80 conversations. Only the selected JSONL rows
are preserved in [sources/mt-bench-selected.jsonl](sources/mt-bench-selected.jsonl).

## Supplementary cases

| ID | Purpose | Turns |
| --- | --- | --- |
| `guffin-short-chat` | Greeting and brief answer latency | 2 |
| `guffin-swedish-chat` | Swedish explanation and follow-up | 2 |
| `guffin-chinese-chat` | Chinese explanation and a Python follow-up | 2 |
| `guffin-document-review` | Review frozen README and engineering principles | 2 |
| `guffin-code-review` | Review frozen CLI helpers/help text and the options declaration | 2 |
| `guffin-sustained-design` | Continue one software-design conversation through six stages | 6 |

The long cases contain meaningful project documents and code, without random
or repeated padding. Their first user inputs are 19,124 and 7,008 UTF-8 bytes,
respectively. **Bytes are not model tokens.** Native tokenizer-only preflight
on the local Gemma 2 9B Q4 and Gemma 3 12B Q4_K_M files found 4,715 / 4,611
rendered tokens for the document input and 3,049 / 2,060 for the code input.
Those starting prompts fit the engine's default capacities; generated replies
and follow-ups can still exhaust context. Check other model variants against
their effective capacity before timing; do not silently truncate a case. The six-turn case tests continuation through a longer
session. The runner's normal pauses apply to every turn, including this case;
it is not an uninterrupted thermal stress test.

Frozen inputs and campaign records retain their original `Coach6/guffin` URLs;
the current repository is [joelsjo01/guffin](https://github.com/joelsjo01/guffin).

The documents and code are snapshots from Guffin revision
`644312e18197379cb5e6d3ef25f1e90dd3ed5553`, rather than live files. The code case
uses a contiguous excerpt of `options.cpp` lines 1–130, ending after the complete
`write_help` function, plus the complete `options.hpp` declaration. The full
original source snapshot remains available for provenance; excerpt bounds and
hash are explicit in the corpus. Future README
or source edits therefore do not change an existing workload. The original
supplementary instructions are versioned as `2026-10-06-v1`.

## Provenance and format

The schema version is 1. `sources` records upstream URLs, revisions, licenses
and SHA-256 identities. `conversations` records stable IDs, category, source,
suite (`core` or `supplementary`), and ordered user `turns`. Supplementary
`groups` label their purpose; `context_source_ids` identifies context sources,
and optional `context_excerpts` declares an inclusive source line range and hash.
The declared default suite is `all`; the runner may also select the core alone.

The downloaded upstream question file's SHA-256 is
`119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7`.
The selected-row snapshot's SHA-256 is
`866d4a0088280d53adc6448fc5a0d0b65d5f5df2dbb6edef933e23ef07779df1`.
Each conversation's `turns_sha256` hashes its UTF-8 JSON list with
`ensure_ascii=False` and compact separators `(',', ':')`.
The original-prompt source digest hashes the ordered list of supplementary
`{id, turns}` objects in that same encoding. Snapshot and license hashes cover
raw file bytes. The runner records the complete corpus-file hash with results.

Saved answers support inspection of stopping and obvious failures; they do not
establish lossless sampling or model quality. Sampling settings, output caps,
timing, cooldowns and configuration order belong to the runner's protocol.
Do not implicitly inherit FastChat's category-dependent sampling defaults.
Organic follow-ups contain each backend's own preceding answer, so their
histories and token counts can differ legitimately. Report those counts rather
than describing them as identical work.

## Attribution

MT-Bench was introduced by Zheng et al. in
[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).
The source question file is distributed in FastChat's Apache-2.0 repository;
the exact pinned [FastChat license](sources/FastChat-LICENSE) is included.
Retain the source attribution and license when redistributing these fixtures.

The six supplementary cases and embedded Guffin snapshots are project-owned
and covered by Guffin's [MIT license](../../LICENSE). The upstream MT-Bench
fixtures retain their Apache-2.0 license and attribution.

Offline integrity checks require only Python's standard library:

```sh
python3 -m unittest discover -s tests/scripts -p 'chat_corpus_tests.py'
```

They verify balance, verbatim source turns, IDs, hashes and frozen context
snapshots. They do not run inference or fetch data.

# Generation Benchmark Milestone Map

Purpose: add a durable, compact, real-path benchmark foundation to
`local-gguf-generate` before optimization work or llama.cpp comparison.

The interactive frontend is complete. Use this file as the benchmark roadmap
and `docs/NEXT_TASK.md` as the baton for only the first incomplete gate.

Status legend: `Done`, `Current`, `Next`, `Planned`.

## Non-Negotiable Measurement Contract

Benchmarking must exercise the same engine-owned path as normal generation:
the production GGUF loader, tokenizer, Gemma chat template, resident Metal
plan, prefill schedules, KV cache, decode schedules, stop handling, and greedy
or sampled token selection. Do not create a second inference loop for
benchmarking.

Responsibilities stay separate:

- Runtime/session records monotonic timing boundaries, token/cache counts, stop
  reason, context events, selection mode, and resident memory facts.
- Benchmark code owns case orchestration, repetitions, and statistics.
- CLI code owns option parsing and human/JSONL presentation.

Normal chat prints one permanent concise line after every completed response:

```text
[ Prompt: 19.0 t/s | Generation: 15.6 t/s ]
```

There is no `--timings` switch. Prompt throughput counts only newly evaluated
tokens, never the resident KV prefix. Generation throughput uses the generated
tokens completed by the steady-state decode loop and its runtime-recorded decode
duration. Terminal write/flush time is not used for either throughput number.

Two TTFT values remain distinct:

- Runtime TTFT: prefill start through first token selection.
- End-to-end TTFT: accepted turn through the first visible text callback. If no
  visible text is emitted, report it as unavailable rather than inventing a
  timestamp.

All durations use a monotonic clock. Tests use an injected/fake clock and assert
derived values and event ordering, not real machine speed.

## Final Command Shape

```text
local-gguf-generate models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." 128 \
  --backend metal --greedy --bench \
  --warmup 1 --repetitions 5 --output human

local-gguf-generate models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." 128 \
  --backend metal --bench --warmup 1 --repetitions 5 --output jsonl \
  > benchmarks/results/gemma2-baseline.jsonl
```

`--bench` belongs to `local-gguf-generate`; do not add a
`local-gguf-bench` executable. Results go to stdout so files remain a shell
redirection concern. Diagnostics and errors remain on stderr so JSONL stdout
stays parseable.

## Milestones

| Status | Gate | Exit Proof |
| --- | --- | --- |
| Done | Persistent production frontend | Chat, streaming, commands, context management, greedy generation, and seeded sampling pass CI with 297 tests. |
| Done | 1. Runtime measurement contract | Real prefill/selection/decode boundaries and complete per-turn facts are returned by the runtime/session with deterministic clock tests. |
| Done | 2. Permanent CLI timing line | Every successful normal chat response prints prompt and generation throughput derived from the per-turn metrics. |
| Done | 3. Benchmark records and statistics | Stable result/case schemas and mean, median, min, max, and sample standard deviation are deterministically tested. |
| Done | 4. Real-path benchmark runner | Cold, warm-empty, cached-second-turn, and seeded-sampled cases reuse production generation with correct plan/KV lifecycle. |
| Done | 5. Human and JSONL CLI surface | `--bench`, warmups, repetitions, formats, stdout discipline, validation, and summaries work end to end. |
| Done | 6. Release baseline and handoff | CI passes, a real release Metal smoke benchmark is documented, and optimization work can consume the schema without redesign. |

## Gate 1: Runtime Measurement Contract

Status: Done on 2026-07-12.

Implemented:

- `Gemma2GenerationMetrics` records monotonic prefill, first-selection, decode,
  completion, token-count, and stop-reason facts in the shared with-plan loop.
- Decode duration accumulates production token-selection/decode intervals and
  excludes streaming callback/terminal time. Sampled selection overhead after
  the first token is included; first selection belongs to prefill/runtime TTFT.
- `Gemma2ChatTurnMetrics` adds accepted/visible/completed boundaries, end-to-end
  TTFT, total latency, cache reuse, selection mode, context rebuild facts, and
  resident plan byte counts.
- The production clock defaults to `steady_clock`; injected clocks make timing
  and event ordering deterministic in tests.
- Stop reasons distinguish token limit, model stop token, callback stop,
  reverse stop, and interruption without changing generation behavior.
- CI passes with 298 tests.

Implement the raw facts before formatting or statistics:

- Add engine-owned timing/result types rather than CLI-owned stopwatch logic.
- Add an injectable monotonic clock seam with a steady-clock production
  default. Keep clock ownership small; do not thread benchmark policy through
  tensor operators.
- In the shared with-plan generation loop, record prefill start/end, first token
  selection, decode start/end, and generation end around the actual production
  calls. Greedy and sampled generation must populate the same schema.
- Preserve the current Metal execution and synchronization behavior. Timing
  instrumentation must not add command-buffer waits or per-kernel fences.
- Extend the runtime result with newly evaluated prefill tokens, generated
  tokens attributable to decode, durations, runtime TTFT, and an explicit stop
  reason: model stop token, reverse stop, callback cancellation, token limit,
  or interruption as applicable.
- In `Gemma2ChatSession`, record turn acceptance and first visible callback,
  then attach end-to-end TTFT, total latency, initial cache position, reused KV
  token count, context rebuild status, trimmed-turn count, and selection mode.
- Expose the resident weight, KV-cache, and scratch byte counts already owned by
  `Gemma2MetalModelPlan`; do not recount buffers in the CLI.
- Define zero-duration/zero-token rates as unavailable, never infinity or NaN.

Likely touch points:

- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `src/engine/gemma2_runtime.cpp`
- `include/local_gguf_engine/engine/gemma2_chat_session.hpp`
- `src/engine/gemma2_chat_session.cpp`
- focused engine/session tests under `tests/engine`

End this gate only when:

- Fake-clock tests prove exact boundary ordering and throughput inputs for
  greedy and sampled paths.
- A cached second turn reports a nonzero initial cache position, reused tokens,
  and only the appended suffix as newly evaluated prompt tokens.
- Context rebuild and each stop path produce explicit facts without changing
  token ids, KV positions, streaming, or sampler state.
- Existing generation behavior remains unchanged and `bash scripts/ci.sh`
  passes.

## Gate 2: Permanent CLI Timing Line

Status: Done on 2026-07-12.

Implemented:

- Every successful normal chat turn and `/regen` prints the permanent compact
  prompt/generation timing line with one decimal place.
- Prompt rate uses only newly evaluated prompt tokens and runtime prefill time;
  generation rate uses decode token count and accumulated decode time.
- Zero-token or zero-duration rates print `n/a`; NaN and infinity are never
  emitted.
- The line follows assistant output and precedes context-rebuild notices.
  Failed/context-rejected turns do not print a timing line.
- Formatter and CLI transcript tests cover rounding, unavailable values, first
  and cached turns, regeneration, sampled output, reverse stops, and context
  rebuild ordering. CI passes with 299 tests.

Implement:

- Format the compact line from `Gemma2ChatTurnResult` metrics after each
  successful normal chat response and `/regen` response.
- Print one decimal place and a stable unavailable representation when a rate
  cannot be computed.
- Keep timing below assistant text and before any context-rebuild notice.
- Do not estimate timing from streamed bytes, decoded text length, terminal
  flushes, or CLI wall-clock wrappers.
- Preserve clean cancellation and error behavior; failed turns print no fake
  performance line.

End this gate only when deterministic CLI tests cover first turn, cached turn,
regeneration, zero/absent measurements, cancellation, and error paths, and
`bash scripts/ci.sh` passes.

## Gate 3: Benchmark Records And Statistics

Status: Done on 2026-07-12.

Implemented:

- `benchmarks/generation_benchmark.hpp` defines schema version 1, the four
  stable case identities, and setup, repetition, statistics, and summary
  records.
- Repetition records retain complete turn metrics, setup timings/memory,
  model/backend identity, sampler settings/seed, warmup state, and index.
- Stored units remain integer nanoseconds/tokens/bytes. Prompt and generation
  throughput are optional derived values.
- Case summaries exclude warmups and other cases, retain measured repetitions
  in input order, and compute mean, median, min, max, and sample standard
  deviation from finite available values only.
- One-sample standard deviation is `0.0`; empty/unavailable summaries remain
  absent rather than becoming zero, NaN, or infinity.
- Focused tests are wired into the single binary. CI passes with 304 tests.

Implement a small benchmark layer, separate from engine and CLI formatting:

- Define stable case identities: `cold_first_turn`, `warm_first_turn`,
  `cached_second_turn`, and `seeded_sampled_turn`.
- Define one repetition record containing model/plan setup durations, per-turn
  metrics, memory facts, sampler mode/seed, warmup flag, and repetition index.
- Compute mean, median, min, max, and sample standard deviation over measured
  repetitions only. Preserve individual records.
- Aggregate durations and throughput fields explicitly. Do not average absent
  values into zero, and do not recompute mean throughput from an ambiguous mix
  of cached/prefill cases.
- Choose and document stable units: integer token/byte counts and durations in
  nanoseconds internally; seconds and tokens/s only as derived presentation
  values.

Likely add focused files under `include/local_gguf_engine/benchmarks`,
`src/benchmarks`, and `tests/benchmarks`, then wire them into the existing
single test binary.

End this gate only when table-driven tests cover odd/even medians, one and many
repetitions, sample standard deviation, unavailable metrics, warmup exclusion,
and stable case/field names.

## Gate 4: Real-Path Benchmark Runner

Status: Done on 2026-07-12.

Implemented:

- `run_generation_benchmark` loads/parses/binds the model once, times that
  separately from one Metal plan creation/upload, then moves the prepared state
  into one persistent `Gemma2ChatSession` without duplicate upload.
- The runner records one true cold first turn. Warm-first, cached-second, and
  seeded-sampled cases each run configured warmups plus measured repetitions.
- Repetition reset uses `clear()` and sampler reconfiguration only. The plan
  revision remains unchanged; `/reset` and production full replay are not used.
- Cached-second cases establish a production first turn before measuring the
  incremental second turn at nonzero cache position for every repetition.
- Seeded sampled repetitions reset identical sampler state and retain generated
  token IDs to prove reproducibility.
- The runner returns raw setup/repetition records and all four summaries without
  printing. Invalid options fail before model IO.
- Deterministic fake-clock/tiny-model tests cover setup boundaries, record and
  warmup counts, summary exclusion, cache facts, resident memory, sampled mode,
  and token reproducibility. CI passes with 306 tests.

Implement lifecycle orchestration without duplicating inference:

1. Time file-backed GGUF read/model construction separately from Metal plan
   creation/upload.
2. Load the model and build the resident plan once for the suite.
3. Run configured warmups through the same session path, mark them as warmups,
   and exclude them from summaries.
4. For `cold_first_turn`, report the one process-level load and plan timings
   separately from the first measured turn; never fold setup into prompt t/s.
5. For `warm_first_turn`, clear logical history/cache position between
   repetitions while retaining the resident plan and uploaded weights.
6. For `cached_second_turn`, establish the first turn through production chat,
   then measure the second incremental turn with resident KV and a nonzero
   initial cache position. Re-establish the same state for every repetition.
7. For `seeded_sampled_turn`, use explicit sampler settings and seed, keep its
   records/statistics separate from greedy, and exercise the full-vocabulary
   logit readback plus CPU filtering/RNG path.

Session reset semantics are important: benchmark repetition reset means
`clear()` or an equivalent logical reset that keeps `Gemma2MetalModelPlan`
resident. It must not call the existing `/reset` behavior, which rebuilds the
plan, and it must not introduce production full replay for ordinary chat.

End this gate only when injected runner/session tests prove one load/plan,
warmup exclusion, resident plan identity, empty-prefix warm cases, nonzero
cached-prefix cases, deterministic repetition state, and separate sampled
results.

## Gate 5: Human And JSONL CLI Surface

Status: Done on 2026-07-12.

Implemented:

- The one-shot command accepts `--bench`, `--warmup`, `--repetitions`,
  `--output human|jsonl`, and sampled-case controls while preserving existing
  one-shot/chat behavior.
- Human output reports setup, every warmup/measured record, all four cases, and
  complete per-metric statistical summaries.
- JSONL emits schema-versioned setup, repetition, and summary objects with
  explicit units and `null` unavailable values. Repetitions include sampler,
  stop/cache/context, memory, generated-token, TTFT, latency, and throughput
  facts; summaries remain attributable to model/backend.
- JSONL stdout stays machine-clean, errors stay on stderr, and no files are
  created by the CLI.
- CLI tests cover both formats and failure paths. CI passes with 308 tests.

Implement:

- Extend the one-shot command parser with `--bench`, `--warmup N`,
  `--repetitions N`, and `--output human|jsonl` while preserving existing
  one-shot and chat commands.
- Reject benchmark-only options without `--bench`, invalid counts/formats,
  incompatible chat usage, and unsupported backends with polite errors.
- Human output prints setup facts, each case/repetition, then compact summary
  rows for all required metrics.
- JSONL emits one self-contained object per setup/repetition/summary record with
  a schema version, case, record kind, units, and explicit `null` for
  unavailable values. Escape strings correctly; never hand-build invalid JSON.
- Keep stdout machine-clean in JSONL mode. Progress, warnings, and errors go to
  stderr.
- Include model identity/path label, backend, greedy/sampled mode, sampler
  controls/seed, stop reason, cache/context facts, and resident byte counts.

End this gate only when CLI tests cover usage, every option, failure paths,
warmup/repetition counts, all four cases, human summaries, parseable JSONL,
stdout/stderr separation, and backward compatibility. Run
`bash scripts/ci.sh`.

## Gate 6: Release Baseline And Optimization Handoff

Status: Done on 2026-07-12.

Validation findings on 2026-07-12:

- Release build and real host-Metal chat timing succeeded; the bounded smoke
  turn reported prompt 1.6 t/s and generation 0.3 t/s.
- The first suite exposed a cached-second failure after token-limited output:
  decoded assistant text could re-tokenize differently from exact KV token IDs.
  The session now appends only the next Gemma template suffix to exact cached
  IDs, without replay; a deterministic `▁Hello` fixture covers the boundary.
- The corrected release suite completed one warmup and two measured repetitions
  for repeatable cases. Warm greedy prompt throughput averaged 1.750 t/s and
  generation averaged 0.287 t/s; cached generation averaged 0.292 t/s; sampled
  generation averaged 0.281 t/s.
- Setup reported 0.073 s model load/bind, 12.230 s Metal plan creation/upload,
  6,837,368,832 resident weight bytes, 5,637,144,576 KV bytes, and 2,394,112
  scratch bytes.
- A bounded JSONL validation produced 9 valid objects: one setup, four
  repetitions, and four summaries. Both artifacts remain ignored.
- The first optimization target is shared Metal decode execution. Sampled and
  greedy results are close, while source inspection shows repeated per-op
  command-buffer commit/wait boundaries on the decode path.

Complete the foundation:

- Rebuild release and run the benchmark from a normal Terminal with Metal
  available against `models/gemma-2-9b-Q4.gguf`.
- Use a fixed prompt, max-new-token count, sampler settings, seed, warmup count,
  and repetition count. Record the exact command and environment metadata
  needed to interpret results.
- Confirm JSONL can be redirected under ignored `benchmarks/results` and that
  no benchmark output is added to source control.
- Check that normal chat always prints the permanent timing line and that
  benchmark mode uses the same token outputs/stops as the corresponding normal
  path.
- Update `docs/NEXT_TASK.md` to the first benchmark-driven optimization only
  after the measured bottleneck is known.

End this gate only when CI passes, the release smoke run produces all four case
families and complete summaries, and the result schema is sufficient for later
optimization sweeps and llama.cpp comparison without redesign.

## Explicitly Deferred

- llama.cpp comparison or parity claims
- per-kernel or Metal command-buffer profiling (`--profile` may be designed
  later using the same case runner)
- synthetic kernel suites
- context shifting changes
- new sampling algorithms
- dashboards, databases, result-file management, or automatic persistence
- a separate benchmark executable

## Baton Rule

Copy only the first incomplete gate into `docs/NEXT_TASK.md`. Complete and
verify that gate before advancing the baton. This file is the durable roadmap,
not an implementation log.

# Gemma 2 Engine Bottleneck Findings

## Purpose

This document defines the current performance diagnosis and optimization
direction for the selected Gemma 2 9B Q4 Metal path on the base Apple M4. The
goal is to make the inference engine itself faster than the pinned `llama.cpp`
reference, not to improve presentation-layer measurements.

Performance decisions must use engine-owned intervals:

1. prefill command-buffer GPU duration;
2. synchronous prefill wall duration;
3. runtime time from inference entry through first token selection;
4. decode duration and generated tokens per second; and
5. prompt tokens per second at fixed, identical token counts and cache
   positions.

The current evidence says decode is approximately at parity, some prefill
shapes are competitive, and the local prefill routing has a clear 9-15-row
geometry hole. It does not yet identify which decode operation has enough
remaining cost to make the complete engine decisively faster. Exact operation
timing is therefore the next required evidence.

## Decisions From the Current Review

### Why the two chat sessions feel equally fast

Both engines currently generate at roughly 14 tokens per second. A 100-token
answer therefore spends about 7.1 seconds in decode:

```text
100 tokens / 14 tokens per second = 7.1 seconds
```

A modest difference before the first token is difficult to notice beside
seven seconds of generation. Raising sustained decode throughput from 14 to
17 tokens per second would reduce the same response to about 5.9 seconds and
would be clearly perceptible. Decode throughput is therefore the first product
performance target for ordinary chat.

### Metric priority

| Priority | Metric | Purpose |
|---:|---|---|
| 1 | Generation tokens per second | Measures the dominant cost of ordinary responses |
| 2 | Runtime first-token-selected duration | Measures engine responsiveness through token selection |
| 3 | Prefill wall duration | Measures actual synchronous prompt-processing latency |
| 4 | Prefill GPU duration | Attributes prompt cost to Metal execution |
| 5 | Prompt tokens per second at fixed rows | Compares kernel efficiency only when shapes match |
| 6 | Total engine request duration | Proves the complete inference path improved |

Prompt throughput is not a single chat-session score. It must be reported as
a curve over fixed evaluated-token counts and cache positions.

### KV-cache decision

Keep the current exact Metal-resident KV cache. It prevents processed
conversation tokens from being evaluated again and is the correct policy for
the single-user local product. It does not explain or solve short-context
decode throughput because both engines already use KV during generation.

Do not imitate a general server's slot/checkpoint policy. Reconsider the local
cache only if measured long-context KV traffic becomes a leading decode cost
or its approximately 2.114-GB allocation prevents the target workload.

### Kernel-organization decision

Use three conceptual projection families:

1. one-row packed decode matvec;
2. small-batch packed projection for approximately 2-8 rows; and
3. tiled SIMD-group prefill for nine or more rows.

One source family may produce several compiled geometries through function
constants. The objective is a small maintainable algorithm set with no bad
shape ranges, not one universal pipeline and not a separate handwritten
kernel for every row count.

### Optimization decision

Do not attempt to make every isolated kernel faster without knowing its share
of inference time. A kernel that owns 0.2% of decode cannot materially improve
the engine even if its own duration is halved. A locally faster fused kernel
can also reduce occupancy and make the complete path slower.

Maintain a scorecard for every important production family:

- exact shapes and dtypes;
- GPU duration and percentage of complete prefill/decode;
- bytes read and written;
- effective bandwidth;
- dispatch and barrier counts;
- threadgroup/SIMD geometry; and
- measured complete-engine effect after replacement.

Optimize the family with the largest verified complete-path ceiling and remove
any candidate that wins only in isolation.

### Work order

1. Establish the fixed engine-only local/reference benchmark and operation
   timing.
2. Optimize the largest measured decode family first.
3. Sweep prefill rows 2-32 and fix the verified 9-15-row hole.
4. Measure attention across cache positions before considering fusion.
5. Keep exact KV reuse unless its measured context-dependent cost changes the
   decision.

## Measurement Contract

### What counts

- **Prefill GPU duration:** time reported for the production Metal command
  buffers that evaluate a fixed prompt suffix.
- **Prefill wall duration:** time from resident prefill entry through its
  synchronous completion and compact candidate readback.
- **Interactive TTFT and prompt duration:** time after chat rendering and
  tokenization, before KV reconciliation, through selection of the first token.
- **Interactive generation throughput:** all sampled tokens divided by the time
  from first sampled token through final sampled token, exactly matching pinned
  llama.cpp `result_timings`.
- **Resident decode throughput:** decoded tokens divided by raw resident decode
  duration; this remains a secondary local diagnostic.
- **Resident prompt throughput:** newly evaluated prompt tokens divided by raw
  prefill wall duration; this remains a secondary local diagnostic.
- **Total engine request duration:** prefill, selection, and all requested
  decode steps, excluding tokenization, chat rendering, text decoding, output
  callbacks, transport, and terminal work.

### Required controls

Kernel and fixed-token diagnostics must match:

- exact model bytes;
- exact input token IDs;
- cache position and reusable prefix length;
- prompt row count and microbatch/chunk shape;
- context length and attention position;
- greedy or seeded sampling configuration;
- requested decode-token count and stop handling;
- Metal offload and Flash Attention configuration; and
- explicitly labelled execution state.

The real-product headline deliberately does not require identical sampled
tokens, prompt suffix lengths, or cache positions. Each executable owns its
normal chat rendering, sampler, stops, and KV policy; those differences are
part of the product being compared. It instead fixes the model, user messages,
seed, fresh-process state, and thermal protocol, and reports each engine's
actual counts beside its rates.

Prompt throughput is meaningful only beside its evaluated-token count. A large
replayed prompt can report more tokens per second than a small cached suffix
while taking substantially longer in absolute prefill time.

### Targeted diagnostic sweeps

The canonical quick benchmark intentionally stays small. When prompt routing
or long-context attention becomes the active task, extend diagnostics—not the
default benchmark—over representative row counts such as:

```text
1, 2, 4, 8, 9, 15, 16, 17, 25, 32, 64, 128, 256, 512
```

For an active attention task, use cache positions such as:

```text
0, 256, 1024, 4096, and the largest valid target context
```

Use the five-pair retention rule only after the single-pair real-chat screen
has identified a candidate. Interleave alternatives and allow thermal
cooldowns when drift could bias the result.

### Verified local quick benchmark

The canonical runner launches the actual local chat executable and feeds it
this exact conversation:

```text
Hello, is AI getting smarter?
Okay, so will it always progress this fast?
```

Both turns use the normal sampled chat configuration with seed 42, persistent
follow-up state, streaming, text decoding, and ordinary stop handling. Natural
responses are not capped. The local path has no hidden warm-up. One fresh
process plus 30-second cooldowns before chat and diagnostics is the quick
screen. The five projection probes run only afterward in a separate process.

When invoked from a terminal, the driver renders the child streams in the
terminal's alternate screen. This preserves the compositor/system contention
present during manual streaming while restoring a concise screen for the final
tables. A non-terminal invocation remains useful for automation but must be
identified as captured-stream rather than terminal-rendered evidence.
The follow-up is written only after the first response and a fixed five-second
input pause. That wait is excluded from TTFT, but it preserves the ordinary
between-turn power-state transition instead of preloading both input lines.

The benchmark does not recreate either timing interval. Local rates and TTFT
come from the same metrics printed by interactive chat, with an additional
detail line for durations in seconds and raw counts. The parser rejects a record
when raw counters do not reproduce the rounded interactive rate. The pinned
reference uses the same boundaries but is run separately; its identity,
protocol, result, invalidation rule, and command live only in
`docs/LLAMA_CPP_REFERENCE_BASELINE.md`.

Chat Prompt t/s cannot be read as latency without evaluated and cached counts.
Manual local follow-up prefill varied from roughly 511 to 746 ms with shorter,
uncontrolled input gaps, while immediate preloaded input was roughly 275 ms.
Short cached prefill is therefore power-state-sensitive; retain the fixed pause
and compare raw latency as well as throughput.

## Finding 1: Decode Speed Is the Main Perceived Chat Metric

The current local and pinned-reference runs both produce roughly 14 generated
tokens per second for the selected model. A response of 100 tokens therefore
spends approximately seven seconds in decode. Differences of a few hundred
milliseconds before generation are much less noticeable than a sustained
decode improvement.

The next meaningful target is a stable production-chat generation win over the
pinned reference, not a faster isolated helper. A candidate should improve the
headline end-to-end generation duration by at least 2% across retained paired
runs without regressing prompt shapes by more than 2%. Raw decode duration is
diagnostic evidence for the mechanism, not the retention target. A 5% or larger
reference win would be clearly useful.

## Finding 2: Decode Is Close to the Weight-Bandwidth Ceiling

The resident model payload is 6,837,368,832 bytes. A simplified full-weight
traversal estimate at 14 generated tokens per second is approximately 95.7
GB/s. The base M4 is specified at 120 GB/s memory bandwidth, so the theoretical
headroom from executing the same byte stream more efficiently is limited.

This estimate is not literal DRAM traffic. Cache behavior, metadata loads,
activation traffic, KV reads, and the tied output matrix complicate the actual
bytes. It is still useful system evidence: a large decode improvement requires
one or more of these outcomes:

- higher useful bandwidth through better packed Q4_K/Q6_K kernels;
- a faster exact-shape F16 output-head traversal;
- less avoidable model or activation traffic;
- better occupancy without duplicated weight reads; or
- a justified representation change that reduces bytes per token while
  preserving the required numerical behavior.

Dispatch reduction alone is unlikely to produce a large win while the same
multi-gigabyte weights are still traversed.

## Finding 3: Decode Must Be Profiled by Operation Family

Whole-token throughput does not reveal which operation now limits decode. The
production path needs attributable GPU intervals for:

- Q4_K projections, separated by exact Gemma shapes;
- the Q6_K down projection;
- the tied F16 output projection;
- attention at several cache positions;
- normalization, residual, RoPE, and cache-store work;
- compact candidate selection; and
- idle gaps, barriers, and command scheduling.

For each family, record:

- GPU duration and percentage of the complete decode step;
- weight and activation bytes read/written;
- effective bandwidth;
- dispatch and barrier counts;
- threadgroup and SIMD-group geometry; and
- a plausible whole-token improvement ceiling.

Optimize the family with the largest verified whole-token ceiling. Do not
choose work from source-code complexity, dispatch count, or an isolated
microbenchmark alone.

## Finding 4: The Tied F16 Output Head Is a High-Value Suspect

The tied F16 output matrix is approximately 1.835 GB and is traversed for every
selected token. An earlier cooperative-kernel rewrite improved real decode by
approximately 3.2-3.6%, proving that this operation is material.

The retained kernel already uses multiple SIMD groups, vector loads, and SIMD
reductions, so another rewrite is not automatically justified. Exact GPU
timing must establish its current share of decode and achievable bandwidth.
If it remains the largest single interval, compare exact-shape geometry,
multiple output rows per SIMD group, read coalescing, accumulator layout, and
selector integration without changing logits or reading the weight matrix
twice.

## Finding 5: Q4_K Is the Dominant Quantized Format in the Exact Model

The current GGUF inspection reports:

```text
Q4_K tensors: 252
Q6_K tensors: 42
```

Runtime prefill counters confirm six Q4_K and one Q6_K projection per layer.
For this file, Q, K, V, attention output, gate, and up are Q4_K;
`ffn_down.weight` is Q6_K.

The superseded diagnosis incorrectly classified `attn_v.weight` as Q6_K.
Future planning must use the current GGUF directory and runtime counters. If
the decode profile attributes comparable time to individual projections, Q4_K
has the larger repeated system ceiling and should be optimized first. Q6_K
must still be measured independently because its unpacking and arithmetic
balance differ.

## Finding 6: The Current Kernel Specialization Strategy Is Reasonable but Has a Hole

One universal projection kernel can be correct for every row count, but it is
unlikely to be fastest. One-row decode is a bandwidth-bound matrix-vector
problem, a few prompt rows need bounded register accumulation, and larger
prefill batches benefit from SIMD-group matrix multiplication and on-chip tile
reuse.

A maintainable target is three conceptual source families:

1. one-row packed decode matvec;
2. small-batch packed projection for approximately 2-8 rows; and
3. tiled prefill matrix multiplication for nine or more rows.

Function constants may compile a source family into a small number of pipeline
variants. This preserves optimized geometries without duplicating the complete
algorithm for every token count.

The current quantized routing is:

```text
1 row       packed decode matvec
2-8 rows    packed small-batch matmul, four-row tiles
exactly 16  two-SIMD-group 16-row tiled matmul
all others  four-SIMD-group 32-row tiled matmul
```

The measured warm production path exposed this discontinuity:

| Prompt rows | Selected family | Prefill GPU duration |
|---:|---|---:|
| 15 | generic 32-row tile | 240.7 ms |
| 16 | exact 16-row tile | 176.3 ms |
| 21 | generic 32-row tile | 248.8-252.1 ms |
| 25 | generic 32-row tile | 266.4 ms |

Exactly 16 rows was about 27% faster than 15 despite doing more useful work.
Rows 9-15 are therefore a verified routing/geometry problem. Screen the
16-row geometry across that range.

Rows 17-31 are not yet a proven kernel defect. A 16-row dispatch plus a tail
may reread weights and lose to one partially occupied 32-row tile. Sweep every
row count and retain another variant only if the complete prefill improves.

## Finding 7: Prompt Throughput Needs Counts and Latency

The local session retains an exact token/KV prefix and evaluates only the newly
appended suffix. The separately stored pinned reference baseline shows that
`llama-cli` reprocesses much more of its independently sampled history. This is
an observed product-policy difference, not a controlled fixed-shape comparison;
the exact reference counts live in `docs/LLAMA_CPP_REFERENCE_BASELINE.md`.

More prompt rows reuse each loaded and dequantized weight tile. A large batch
can therefore report higher prompt tokens per second while consuming more
absolute time. Prompt throughput can be a headline product observation only
when evaluated count, cache reuse, raw duration, and TTFT are printed beside
it. For kernel-efficiency decisions it must still be reported as a curve over
fixed row counts.

Existing controlled evidence suggests:

- the exact 16-row specialization is strong;
- 32-row and roughly 128-row tiled paths are competitive with the pinned
  reference; and
- the immediate local gap is tail utilization, especially rows 9-15.

Re-run these claims through the benchmark matrix before retaining further
prefill changes.

## Finding 8: Keep the Exact Resident KV Cache

The current chat session retains exact processed token IDs and their
Metal-resident KV entries. It validates that the newly rendered conversation
has the cached token sequence as a prefix, then evaluates only the appended
suffix at the existing cache position.

This is the correct policy for the current single-user local product because
it:

- avoids replaying processed conversation tokens;
- minimizes subsequent prompt work;
- preserves exact cache-position ownership; and
- keeps generation state model-resident.

The tradeoffs are approximately 2.114 GB of KV allocation, session-specific
ownership, invalidation after edits to old messages, and added complexity for
context shifting or multiple simultaneous sessions. Those tradeoffs are
acceptable for the selected product path.

Do not redesign KV ownership merely to imitate a general multi-slot server.
Change it only if long-context profiling proves that KV traffic or capacity is
a leading decode bottleneck. Any compressed or reduced KV representation must
preserve attention correctness and demonstrate a complete-path improvement.

## Finding 9: Attention Is a Context-Dependent Bottleneck

The local prefill path separately dispatches attention scores, softmax, and
attention values and materializes their intermediates. The retained prefill
plan allocates 181,146,892 scratch bytes. At short contexts, projection-weight
traversal dominates and earlier Flash Attention controls changed performance
only slightly.

Attention cost grows with cache position. Measure it at the required context
matrix before choosing a fused implementation. A retained fused path must
cover Gemma's sliding and global layers, F16 ring-cache behavior, partition
boundaries, and unavailable-device behavior. Do not infer a short-context win
from long-context asymptotics or a reference engine's feature list.

## Finding 10: Broad Decode Fusion Was a Measured Dead End

The rejected seven-wave decode candidate reduced the short-token graph from
634 dispatches and 465 explicit barriers to 297 dispatches and zero barriers.
It removed 147,460 scratch bytes and preserved full logits, selected tokens,
F16 cache contents, and real-model output.

After correcting its slow projection subregion, its best cooled conversation
screen was only about 0.27% faster than the 21.094-second baseline median,
below the repository's 2% retention gate. A more aggressive gate/up variant
regressed from register pressure and occupancy loss.

Do not reimplement this design unchanged. It reorganized the same dominant
weight stream without reducing its bytes. Revisit fusion only when per-family
timing gives a specific fused boundary a greater-than-2% whole-decode ceiling.

## Priority Order

1. Build the fixed-shape engine-only benchmark and per-operation Metal timing.
2. Select and optimize the decode family with the largest measured complete-
   token ceiling.
3. Sweep prefill rows 2-32 and fix the verified 9-15-row geometry gap.
4. Measure attention across cache positions and implement fusion only beyond a
   verified crossover.
5. Keep exact resident KV reuse unless the context-dependent profile proves it
   is a leading cost.
6. Leave broad decode fusion rejected unless new timing invalidates the prior
   result.

## Acceptance Rules

- Use each product's own displayed timing intervals for the real-chat headline.
- For fixed-token diagnostics, compare identical model bytes, token IDs, cache
  positions, prompt shapes, selection mode, output budgets, and labelled state.
- Report evaluated and reused prompt counts beside every prompt rate.
- Require at least five paired retained repetitions, four paired wins, and a
  median improvement of at least 2%.
- Preserve full-logit or independent golden evidence, selected tokens, F16 KV
  contents, unavailable-Metal behavior, model residency, and allocation
  identities.
- Reject an isolated-kernel win that does not improve raw production prefill,
  runtime first-token selection, or decode duration.
- Run `bash scripts/ci.sh` after implementation changes.

## Source Anchors

Local implementation:

- `src/engine/gemma2_chat_session.cpp`: exact cached-prefix validation and
  suffix extraction.
- `src/engine/gemma2_runtime.cpp`: resident prefill/decode, timing boundaries,
  selection, and execution counters.
- `src/tensor/backends/metal/metal_context.mm`: quantized row-family routing.
- `src/tensor/backends/metal_backend.mm`: packed decode, small-batch, and tiled
  quantized kernels plus the F16 output projection.
- `src/tensor/backends/metal/metal_command_context.mm`: production dispatches,
  barriers, compact selection, and command statistics.
- `src/benchmarks/quick_benchmark.cpp`: secondary Metal projection probes only.
- `src/frontends/cli/generate.cpp`: local interactive timing and auditable raw
  count/duration output.
- `scripts/quick_benchmark.py`: direct product-process driver, pinned-version
  validation, timing parser, and compact two-table comparison.

Pinned reference source:

- Metal matmul routing and graph operations:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- Metal kernels:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal.metal>
- Metal encoder and synchronization wrapper:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-device.m>
- Interactive timing output:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/tools/cli/cli.cpp>
- Raw CLI timing boundaries and cache accounting:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/tools/server/server-context.cpp>

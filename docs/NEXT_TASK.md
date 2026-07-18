# Completed Task: Collapse Decode Into One Dependency-Aware Metal Compute Pass

## Verified Result

Accepted on 2026-07-18. Production decode now uses one concurrent Metal
compute encoder, explicit fixed-schedule dependency barriers, direct K/V cache
row writes, one command buffer/commit/wait, and zero transient allocations.
The staged decode executor and its activation hierarchy, schedule-owned buffer
copies, decode K/V scratch buffers, and their legacy tests are absent.

The 42-layer top-logit path now records `803` compute dispatches in `1` compute
encoder, `0` blit encoders, and `634` memory barriers. The nonzero two-layer
fixture records `43/1/0/34`; disabling top-logit records `42/1/0/33` for
dispatches/compute encoders/blit encoders/barriers.

An isolated pre-change `HEAD` build and the candidate were benchmarked one at a
time with Gemma 2 9B Q4, the same prompt, 8 generated tokens, 1 warmup, and 5
measured repetitions:

| Case | Baseline mean / median | Candidate mean / median | Mean change |
| --- | ---: | ---: | ---: |
| Warm first-turn decode | 4.577 / 4.548 t/s | 5.100 / 5.113 t/s | +11.4% |
| Cached second-turn decode | 4.427 / 4.481 t/s | 5.014 / 5.061 t/s | +13.3% |
| Seeded sampled decode | 4.149 / 4.188 t/s | 4.332 / 4.324 t/s | +4.4% |

Warm first-turn prefill was unchanged within noise: `2.150` t/s baseline versus
`2.152` t/s candidate. Later sampled-prefill measurements showed thermal drift,
but decode improved in every measured scenario. The greedy decode gains exceed
the observed repetition variance, so the implementation stays.

Correctness is independently covered by a deterministic nonzero CPU-reference
test with F16/Q4_K/Q6_K weights, two layers, sliding and global attention, a
nonzero cache position, full-logit comparison, top-logit comparison, and every
K/V cache element. `bash scripts/ci.sh` passes all `315` tests and
`git diff --check` is clean.

## Finding

The next unresolved item in `LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md` is decode
encoder fragmentation. The production path already submits one command buffer
and waits once per token, but it opens and closes an encoder for every dispatch.

For Gemma 2 9B, one token currently encodes:

- `1` embedding dispatch;
- `19` compute dispatches and `1` KV blit in each of `42` layers;
- `4` final dispatches when resident top-logit selection is enabled.

That is `803` compute encoders plus `42` blit encoders per token. The two-layer
fixture proves the same formula with `43` compute encoders and `2` blit
encoders. The math kernels and command-buffer count are not the cause of these
counts: every `MetalCommandContext::encode_*` method creates an encoder and
immediately calls `endEncoding`.

This likely costs CPU-side Objective-C encoding time, GPU pass transitions, and
idle gaps between very small kernels. It is a high-confidence structural flaw,
but its end-to-end magnitude still requires an A/B release benchmark.

There is also a complete older staged decode executor alongside production:
`run_gemma2_metal_decode_step` builds and retains an all-intermediate activation
object graph through operation-at-a-time tensor wrappers. Production does not
call it; one resident-decode test keeps it alive as an oracle. The public
activation structs, helper chain, validation functions, schedule-owned copies
of resident buffers, and tests coupled to those objects are legacy code. This
task must delete them rather than maintain two decode pipelines.

## Upstream Design

The comparison target is llama.cpp commit
`b15ca938ad00aa6b3ee6c2edda7363fd02826b18`.

Its Metal graph slice creates one compute encoder in `ggml_metal_op`, encodes
many graph nodes through it, and ends it only when the slice is destroyed. By
default it requests `MTLDispatchTypeConcurrent`. Before encoding a node,
llama.cpp compares that node's source and destination memory ranges with the
currently active set. Independent nodes remain in the same concurrent wave;
an overlapping read/write dependency inserts
`memoryBarrierWithScope:MTLBarrierScopeBuffers` and starts a new wave.

The local engine does not need llama.cpp's generic graph or range tracker. Its
decode schedule is fixed, so dependency waves can be expressed directly and
more simply. Apple documents that one compute encoder may contain multiple
kernel dispatches in one compute pass, and that compute memory barriers order
writes before dependent later reads.

Primary upstream locations:

- `ggml/src/ggml-metal/ggml-metal-context.m`: command-buffer slicing and
  concurrency default;
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`: one encoder per graph slice,
  dependency-range tracking, and barriers;
- `ggml/src/ggml-metal/ggml-metal-device.m`: concurrent encoder and buffer
  barrier wrappers.

## Target Design

Use one concurrent compute encoder for the complete resident decode token, zero
blit encoders, one command buffer, one commit, and one final wait.

There must be exactly one Metal decode implementation when the task finishes:
`run_gemma2_metal_resident_decode_step`. Do not keep the staged executor behind
a diagnostic name, compile flag, test-only call, or alternate public API.

Do not translate each KV blit into a compute-copy dispatch. Remove the copies:

- bind the current layer's V-cache row as the output of the V projection;
- bind the current layer's K-cache row as the output of K RoPE;
- keep raw K in scratch because RoPE cannot safely overwrite the current raw-K
  input under its paired-lane implementation;
- continue reading the full cache buffer from the attention kernels.

This removes `42` encoder transitions, `42` copy operations, and the resident
decode `key` and `value` scratch buffers. It also avoids adding a copy kernel
that would become dead work after a better direct-write implementation.

`MetalCommandContext` should lazily create and retain one
`MTLComputeCommandEncoder`, using
`computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent`. Every
compute operation rebinds its pipeline and arguments on that encoder. `commit`
ends the encoder exactly once before submitting; destruction and error paths
must not double-end it or submit an unfinished encoder.

Add an explicit barrier operation and count it in structural statistics. Use
resource-specific barriers when the dependent buffers are known; use the
all-buffer scope only if profiling shows that constructing resource lists costs
more without changing GPU scheduling. Do not insert a barrier automatically
after every `encode_*` call, because that would serialize independent work.

The intended decode dependency waves are:

1. embedding;
2. attention RMSNorm;
3. Q, K, and V projections together;
4. Q RoPE and K RoPE together;
5. attention scores after the direct K-cache write;
6. softmax;
7. attention values after the direct V-cache write;
8. attention output projection;
9. post-attention RMSNorm;
10. attention residual add;
11. feed-forward RMSNorm;
12. gate and up projections together;
13. GELU-gate multiply;
14. down projection;
15. post-feed-forward RMSNorm;
16. feed-forward residual add, which feeds the next layer;
17. final RMSNorm, output projection, softcap, and optional top-logit as four
    dependent waves.

Place one barrier between dependent waves. Q/K/V and gate/up must not have
barriers between their dispatches: they share read-only input and have disjoint
outputs, so the concurrent encoder may overlap them. A fixed wave schedule is
preferred over adding a generic graph scheduler in this task.

## Implementation Path

1. Extend `MetalCommandStats` with `compute_dispatch_count` and
   `memory_barrier_count`. Keep encoder count separate from dispatch count so a
   test cannot pass by accidentally deleting work.
2. Refactor `MetalCommandContext::Impl` to own one active compute encoder.
   Replace per-operation creation and `endEncoding` with a shared accessor.
   Add a barrier method and a single finalization path used by `commit`.
3. Add checked destination byte offsets to the resident matmul and RoPE
   encoders. Include the offset in capacity validation and bind it through
   `setBuffer:offset:atIndex:`. Default to zero so prefill behavior remains
   unchanged.
4. In `run_gemma2_metal_resident_decode_step`, calculate the current cache-row
   offsets with checked 64-bit arithmetic. Send V projection directly to
   `cache.value` and K RoPE directly to `cache.key`; delete the decode
   `encode_kv_cache_write` call.
5. Insert barriers at the fixed wave boundaries above. Preserve the independent
   Q/K/V and gate/up groups.
6. Remove `key` and `value` from `Gemma2MetalDecodeScratch`, its allocation,
   descriptor/byte accounting, and tests. Keep the corresponding prefill
   scratch and blit path untouched.
7. Delete the old decode pipeline from `gemma2_runtime.cpp` and
   `gemma2_runtime.hpp`, including `run_gemma2_metal_decode_step`,
   `run_gemma2_metal_scheduled_decode_block`, the layer-handoff/input/KV-write/
   output-norm/logits helpers, their validation chain, and every staged decode
   activation struct. Remove now-unused includes and utility functions.
8. Simplify `Gemma2MetalSchedule` to metadata production actually reads. Delete
   schedule-owned copies/accessors for weights, KV buffers, resident decode
   scratch, and resident byte counts. Delete decode-only scratch descriptors;
   if `Gemma2MetalScheduleScratch` and `phase_scratch_byte_count` have no
   production consumer after the staged executor is removed, delete them too
   rather than preserving a test-only abstraction.
9. Delete the old tests whose purpose is staged-pipeline parity, retained
   intermediate shapes/lifetimes, schedule buffer-copy ownership, or obsolete
   `43/2` and `803/42` encoder counts. Do not rename those tests and point them
   at the new executor.
10. Add a new deterministic, nonzero tiny-model decode test. Build expected
    logits and appended K/V rows with CPU/reference tensor operations in test
    code, not with another Metal execution pipeline. Exercise at least two
    layers so both sliding and global attention run, use a nonzero cache
    position, and use nonzero F16/Q4_K/Q6_K weights so stale data or a missing
    barrier cannot pass as all-zero output.
11. Add focused resident structural/error tests for top-logit enabled and
    disabled, cache-row destination offsets, context overflow, invalid token,
    and Metal-unavailable behavior. The structural test must require, for both
    the tiny fixture and the real 42-layer formula:
    `1` command buffer, `1` commit, `1` wait, `1` compute encoder, `0` blit
    encoders, the expected compute dispatch count, the expected barrier count,
    and `0` transient allocations.
12. Audit with `rg` and delete any remaining symbol, comment, test helper, or
    assertion that describes the staged decode path or treats production as
    “resident” in contrast to another decode implementation. Rename the sole
    executor only if that makes the API clearer without broadening the change.
13. Delete any temporary timing logs, alternate executor, compute-copy kernel,
    or benchmark-only switch after the new path is proven.

The new tests replace, rather than extend, the old decode tests. Retain
independent tensor-kernel parity tests, prefill tests, and generation/chat tests
that exercise production behavior without depending on staged decode objects.

The main files are:

- `src/tensor/backends/metal/metal_command_context.hpp`
- `src/tensor/backends/metal/metal_command_context.mm`
- `include/local_gguf_engine/engine/gemma2_metal_plan.hpp`
- `src/engine/gemma2_metal_plan.cpp`
- `src/engine/gemma2_runtime.cpp`
- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`

Before deleting the old oracle, use it once during development to diagnose the
new implementation if useful. It must be removed in the final tree; correctness
must ultimately be proven by the new independent reference/golden tests.

## Correctness Gates

- Production logits, selected token, selected logit, and every layer's appended
  K/V row match the independent nonzero CPU/reference result at a justified
  tolerance.
- Tests cover both sliding-window and global-attention layers.
- Offset validation rejects cache overflow and mis-sized destinations.
- `select_top_logit=false` remains correct and omits only its final dispatch.
- Metal-unavailable and invalid-input behavior remains unchanged.
- Prefill structural counts and parity remain unchanged.
- `bash scripts/ci.sh` and `git diff --check` pass.

Run Metal validation during development if available. A missing barrier can
produce plausible but stale values, so output-token parity alone is not enough;
full logits and every written KV row are required.

The cleanup gate is an `rg` audit showing no staged decode executor, activation
hierarchy, diagnostic parity test, or schedule-owned resident buffer copies.

## Performance Gate

Before editing, preserve the release generator as a baseline binary outside the
repo. Benchmark baseline and candidate with the exact same model, prompt,
greedy settings, generated-token count, warmup, and repetitions. Interleave
runs when practical to limit M4 thermal drift.

Record:

- end-to-end generation tokens/s and decode duration from
  `local-gguf-generate --bench`;
- compute encoder, blit encoder, dispatch, and barrier counts;
- per-token GPU duration from the existing command completion statistics;
- CPU wall time minus GPU duration as a rough encoding/wait overhead signal.

The structural acceptance target is exactly one compute encoder and zero blit
encoders per decode token. Keep the change only if repeated release results
improve end-to-end decode beyond noise without a prefill regression. If the
counts improve but throughput does not, revert the complexity and record that
encoder fragmentation was not the dominant measured bottleneck; do not proceed
directly into fused attention in the same change.

## Boundaries and Stop Condition

Do not change attention arithmetic, RMSNorm/softmax kernels, Q4_K/Q6_K math, KV
dtype or global layout, prefill scheduling, sampling, tokenizer behavior, or
chat orchestration. Concurrent dispatch is allowed only within the explicit
independent waves above.

Stop when production decode uses one concurrent compute encoder, zero blit
encoders, direct K/V cache-row writes, one command buffer/commit/wait, and zero
transient allocations; the old staged executor, its public types/helpers,
unused schedule state, old tests, old decode KV scratch, and obsolete assertions
are absent; replacement nonzero reference and structural/error tests pass; CI
passes; and comparable release benchmarks decide whether the change stays.

# Next Task: Finish Gemma Chat Parity

## Correctness Gate

Pause performance tuning until sampled multi-turn chat has deterministic
token, logit, and stability gates. Automated tests must remain self-contained;
reference-runtime comparisons are manual diagnostics, not test dependencies.

The collapse had three proven correctness causes, all now fixed:

- Gemma 2 requires split-half NeoX RoPE; the engine used adjacent-pair RoPE.
- The SentencePiece path used longest-prefix matching without escaping spaces,
  turning ordinary words such as `how are you` into `<unk>` tokens.
- Metal's fast `tanh` produced non-finite GELU results for normal gate values;
  Gemma GELU and softcaps now use stable precise `tanh`.

For `Hello, how are you?`, engine and `llama.cpp` now produce the same 15
prompt token IDs and the same top-10 next-token ranking. The measured full-logit
comparison is cosine `0.999978`, RMSE `0.109`, with `39/40` top-token overlap.
The formerly failing sampled decode now remains finite for the full 128-token
probe.

SentencePiece byte fallback now preserves text that has no normal vocabulary
piece, including raw UTF-8 bytes, instead of collapsing the whole piece to
`<unk>`. Sampling now applies the supported transforms in the fixed order
repetition penalty, top-k, top-p, min-p, temperature, then distribution. Local
fixtures lock both behaviors without invoking `llama.cpp`.

Before returning to performance work:

1. make the exact known prompt token IDs a self-contained regression fixture;
2. capture logit parity at an incremental second-turn boundary and
   after full replay of the same prefix;
3. run the reported three-turn chat without unknown prompt tokens, non-finite
   logits, repetition collapse, or context loss.

Do not tune sampling thresholds to conceal tokenization or forward-logit
divergence. Resume the performance task below only after these gates pass.

# Queued Task: Complete Resident Decode Bottleneck Attribution

## Objective

Measure where one complete steady-state Gemma 2 decode token spends wall time
after the Q4_K/Q6_K SIMD work. First reconcile host preparation, command
encoding, GPU execution, and readback. Only then divide GPU time among kernel
families and select the next optimization from measured aggregate cost.

Do not optimize another kernel or runtime subsystem until this attribution is
complete.

## Why This Is Next

The tied-F16 projection bake-off is complete, and its result changes the
priority. The scalar `[1,3584] x [3584,256000]` projection is slower than a
SIMD implementation in isolation, but it is not large enough to explain the
remaining end-to-end decode time.

The temporary real-weight probe loaded the model once and timed one projection
per command buffer with `GPUStartTime`/`GPUEndTime`, using 10 warmups and 50
measured repetitions. Every pipeline reported execution width `32`, maximum
`1024` threads, and zero static threadgroup memory.

At 256 threads per threadgroup, the interleaved result was:

| Mapping | Median | Mean | Minimum | Stddev | Effective weight bandwidth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing scalar | `21.242 ms` | `21.302 ms` | `21.125 ms` | `0.171 ms` | `86.384 GB/s` |
| 1 output/SIMD | `17.690 ms` | `17.702 ms` | `17.539 ms` | `0.122 ms` | `103.734 GB/s` |
| 2 outputs/SIMD | `17.761 ms` | `17.851 ms` | `17.669 ms` | `0.196 ms` | `103.314 GB/s` |
| 4 outputs/SIMD | `17.846 ms` | `17.922 ms` | `17.677 ms` | `0.270 ms` | `102.823 GB/s` |

The one-output mapping won under the predeclared tie rule. Its threadgroup
medians were `17.939 ms` at 64 threads, `17.799 ms` at 128 threads, and
`17.687 ms` at 256 threads. Deterministic parity passed for mixed signs,
cancellation, output tails `1` through `5`, K values `127`, `128`, `129`, and
the real `3584` width.

However, the exact required production benchmark did not confirm an
end-to-end improvement over the established `3.130 t/s` mean and `3.000 t/s`
median warm baseline:

- first post-bake-off run: `2.912 t/s` mean, `2.900 t/s` median;
- clean post-CI repeat: `2.737 t/s` mean, `2.748 t/s` median, with high
  run-to-run variance.

The isolated winner saves about `3.55 ms` from a token whose measured decode
duration is roughly hundreds of milliseconds. It therefore failed the task's
production acceptance rule. The F16 candidate pipelines, diagnostic probe,
environment switches, tests specific to the removed path, and integration were
deleted. Production remains on the existing scalar F16 fallback; Q4_K/Q6_K
and one-submit behavior are unchanged.

The current throughput timer does not cover the complete token path. It starts
after `create_gemma2_metal_decode_schedule`, so schedule construction is absent
from reported generation throughput. That construction creates a layer vector
and copies schedule-owned vectors of `BackendBuffer` handles for resident
weights and KV buffers, causing allocations and shared-ownership reference
updates even though tensor payloads remain resident.

Inside the timed resident step, the command context creates about `803` compute
encoders and `42` blit encoders per token. `GPUStartTime`/`GPUEndTime` excludes
the CPU cost of creating and labeling those encoders. The resident path has no
transient tensor-buffer allocations or payload copies, but schedule ownership,
descriptor validation, command encoding, queue delay, and result readback must
still be ruled out as material costs before blaming GPU kernels.

## Measurement Task

### 1. Reconcile Complete Inter-Token Wall Time

Measure a real warm greedy token from immediately before decode-schedule
creation until the selected result is available. Attribute it to:

1. decode-schedule construction, including allocations and handle copies;
2. CPU command-buffer and encoder construction before commit;
3. commit-to-GPU-start queue delay;
4. complete GPU execution from `GPUStartTime` to `GPUEndTime`;
5. completion wait, top-logit readback, and other post-GPU work.

Report both the current `decode_duration_ns` interval and the wider true
inter-token interval. Reconcile the latter closely enough that tensor, plan,
and Metal command-encoding overhead cannot be hidden outside GPU timing.

### 2. Attribute GPU Execution

Attribute one resident decode token across these aggregate families:

1. all Q4_K projection dispatches across 42 blocks;
2. all Q6_K projection dispatches across 42 blocks;
3. attention score, softmax, and attention-value dispatches;
4. RMSNorm, residual add, SiLU multiply, RoPE, and KV writes;
5. final F16 vocabulary projection, softcap, and top-1 selection;
6. encoder transitions, command-buffer scheduling gaps, or other unattributed
   GPU time.

Measure totals per family and counts per token, not only one representative
kernel. The result must reconcile closely with the complete command buffer's
GPU duration, which must in turn reconcile with the wider inter-token wall
time.

## Method Constraints

- Use the real resident decode schedule, real model buffers, real cache state,
  and exact production tensor shapes.
- Load/upload the model once. Warm the path before recording measurements.
- Measure the schedule and host path without changing their ownership or
  allocation behavior first.
- Keep the production one-command-buffer path unchanged outside a temporary
  diagnostic build.
- Prefer Metal counter samples or another same-command-buffer timestamp method
  if the M4 and current API support reliable per-encoder boundaries.
- If reliable in-buffer timestamps are unavailable, use a temporary diagnostic
  executor that isolates already-resident operations. Clearly account for the
  synchronization distortion and reconcile isolated totals against whole-token
  GPU time.
- Use at least 10 warmups and 50 measured tokens. Report median, mean, minimum,
  standard deviation, dispatch count, and aggregate milliseconds per family.
- Interleave comparisons and keep thermal conditions stable.
- Remove all diagnostic timing code, switches, and output after the report is
  captured. Do not add permanent kernel-level timing.

## Decision Rule

Choose the next implementation only from the largest actionable aggregate
cost. Prefer a change capable of removing at least 10% of true inter-token wall
time; do not select another isolated micro-optimization whose theoretical
whole-token gain is below measurement noise.

If schedule construction or CPU command encoding is material, fix that host
architecture before tuning kernels. Candidate areas include borrowing plan
state instead of copying schedule-owned buffer graphs, reusing immutable decode
schedule structure, or reducing encoder creation. Select among them only from
the measured breakdown.

If Q4_K or Q6_K remains dominant in aggregate, optimize the measured format and
real shapes further. If attention or small elementwise dispatches dominate,
research fusion or shape-specific kernels. Do not return to the F16 projection
unless attribution shows a materially larger production share than the
`~21 ms` isolated measurement.

## Verification

After the diagnostic work:

- run `bash scripts/ci.sh` after removing all temporary code;
- run `git diff --check`;
- repeat the exact release benchmark command from the completed bake-off;
- record the host/GPU reconciliation, GPU-family attribution, and selected next
  optimization in this file;
- stop and report the recommendation before implementing it.

## Do Not

- Do not retain the rejected F16 SIMD kernel.
- Do not alter schedule ownership, tensor abstractions, command encoding,
  Q4_K/Q6_K, prefill, attention, KV, sampling, context handling, model loading,
  or CLI behavior during attribution.
- Do not split production decode into extra submissions or waits.
- Do not add a benchmark executable, runtime autotuning, dashboards, databases,
  or a llama.cpp comparison.

## Stop Condition

Stop after true inter-token wall time reconciles with host and GPU costs, GPU
time reconciles with its aggregate families, the largest actionable cost is
identified, all diagnostic code is removed, CI passes, and this baton contains
a concrete research/implementation target. Wait for approval before
implementing that optimization.

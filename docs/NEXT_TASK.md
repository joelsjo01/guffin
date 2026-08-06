# Next Task: Attribute Resident Decode Cost

## Verified Current State

The engine now owns a default-enabled, state-neutral Gemma 2 Metal startup
warm-up. `Gemma2ChatSession` runs it once after construction and once for a
new plan created by `/reset`; `/clear` does not rerun it. Production chat has
no warm-up policy switch: every plan is warmed. The quick benchmark records
the mandatory policy and its 16-prefill/one-decode shape in both human and
JSON output.

The warm-up is deliberately outside all turn measurements. Construction
finishes before the CLI banner, while interactive TTFT starts later inside
`generate_user_turn()` after chat rendering and tokenization. The warm-up calls
resident execution primitives directly and never calls a generation wrapper
or the injected turn clock. Automated coverage proves the constructor leaves
that clock untouched.

The exact lifecycle is:

```text
capture resident allocation identities
CPU-zero every byte of every shared Metal KV buffer
run production resident prefill for 16 repeated valid tokens at cache position 0
run production compact candidate selection and readback
sample with a copy of the configured sampler and observe only in that copy
run production one-token resident decode at cache position 16
run production compact candidate selection and readback
CPU-zero every byte of every shared Metal KV buffer
verify allocation identities and byte counts
discard candidates, token, logits, and sampler copy
```

For the target model this executes the exact-16 Q4_K/Q6_K projection routes,
prefill attention and KV stores, F16 output projection, one-row Q4_K/Q6_K
decode projections, concurrent decode attention/barriers, and both selector
readbacks. The host zero fill is synchronous, occurs only when no command is
in flight, and does not replace any allocation.

After construction, messages, cached token IDs, turn history, stop state, and
the production sampler are unchanged. KV is physically byte-zero, logical
cache position is zero, resident allocation identities are unchanged, and the
retained warm-up statistics are startup-only. A failed replacement-plan
warm-up leaves the old plan installed.

## Correctness Evidence

`bash scripts/ci.sh` passes:

- 298 C++ tests;
- 8 Python benchmark-script tests; and
- repository formatting/check rules.

The Metal warm-up test independently verifies:

- exact 16-row prefill plus one-row decode route counters;
- Q4_K, Q6_K, F16, attention, KV-store, selector, and readback execution;
- full-byte zero KV after completion;
- unchanged resident allocation identities;
- unchanged caller sampler RNG/history behavior; and
- identical first production selected tokens and final F16 KV bytes between
  warmed and cold plans.

Production chat exposes no lifecycle-policy branch. Existing
unavailable-Metal coverage still fails through the Metal backend without a CPU
fallback.

## Real-Path Screen

The following attribution used a temporary development opt-out before that
switch was removed from production chat. Release model:
`models/gemma-2-9b-Q4.gguf`; prompt:
`Hello, is AI getting smarter?`; seed 42; eight selected tokens; fresh
processes with a 30-second cooldown before each alternative.

| Policy | TTFT / prompt | Prompt rate | Generation interval | Generation rate |
| --- | ---: | ---: | ---: | ---: |
| warm | 0.945210 s | 16.9 t/s | 0.522198 s | 15.3 t/s |
| cold | 1.037510 s | 15.4 t/s | 0.486619 s | 16.4 t/s |

The selected text/IDs were identical. This single attribution pair shows a
92.300 ms, 8.9% first-turn TTFT reduction. The short generation difference is
not retention evidence; eight-token generation is too noisy and warm-up cannot
be credited with a steady-state bandwidth improvement.

The canonical natural-stop quick benchmark subsequently completed in a fresh
release process with captured pipes and the required 30-second cooldown. It
reported 16 evaluated tokens, 173.6 ms TTFT, and 92.2 prompt tokens/s on the
first turn; the follow-up reused 238 cached tokens and reported 19 evaluated
tokens, 472.8 ms TTFT, and 40.2 prompt tokens/s. Generation was 13.9 and 13.7
tokens/s. This single run verifies the warm production path and timing
plumbing, but the repository's five-pair warmed-local versus warmed-reference
retention rule remains unsatisfied.

### Memory result

`/usr/bin/time -l` construction-only samples reported:

| Policy | Maximum RSS | Peak footprint |
| --- | ---: | ---: |
| cold | 8,963,424,256 bytes | 6,898,739,328 bytes |
| warm | 9,256,255,488 bytes | 9,240,882,816 bytes |

The model, KV, and scratch virtual allocation sizes are unchanged. Warm-up is
a latency optimization, not a memory-size optimization: full KV prefaulting
makes approximately the 2.114-GB KV allocation physically committed and raises
peak footprint by about 2.34 GB in this screen. Do not describe it as reducing
memory. Reconsider full-KV prefault policy if memory pressure or compression is
more important than roughly 0.1-second cold TTFT on the target machine.

Construction wall samples ranged from roughly 6 to 12 seconds and were highly
order/file-cache dependent. They are not suitable for a retention claim. The
warm-up's own duration is retained by the session as
`Gemma2MetalWarmupStats::wall_duration_ns` and remains outside turn metrics.

## Pinned llama.cpp Comparison

For pinned llama.cpp commit
`d48a56effbba99944a677938d4beb63a0065ecdf` (build 9430), normal startup runs a
real decoder evaluation on `[BOS, EOS]`, clears logical/physical KV, waits,
resets performance counters, disables warm-up state, and resets samplers.
Backend sampling is not exercised. Large model, KV, and scheduler buffers are
allocated before that pass.

On the inspected Gemma 2 9B Q4 run it lazily inserted 14 Metal pipeline states
covering unary scaling/softcap, RMSNorm fusions, Q4_K/Q6_K matvec, NeoX RoPE,
F16 KV rows, vector Flash Attention/reduction, GELU gating, row lookup, and F16
output projection. The complete pass was about 143 ms. It removes relevant
pipeline compilation, first submission/encoder/scheduler execution, initial
weight traversal, and KV-page first touch from measured work. Its two-token
shape is not equivalent to local: local eagerly creates pipelines and routes
two rows through its 2-8 small-batch path, while the canonical local prompt is
16 rows and local decode has distinct one-row projections and attention.

That is why local warms 16-row prefill plus one decode rather than copying the
upstream token count. A portfolio of extra full-model warm-up shapes remains
rejected because it adds multi-gigabyte weight streams, startup delay, and
thermal load without raising steady-state memory bandwidth.

## Active Objective

Attribute the complete resident one-token decode on the target 9B Q4 model by
operation family and identify the largest optimization ceiling. Warm-up has
made lifecycle policy comparable; the dominant product metric remains
sustained generation throughput.

Capture, without changing numerical execution:

1. GPU duration and share for embedding, normalization/residual, Q/K/V/O
   Q4_K projection, gate/up Q4_K projection, down Q6_K projection, RoPE/KV
   store, decode attention, GELU gating, final F16 vocabulary projection, and
   compact selection;
2. CPU encoding/submission/wait gaps, buffer-binding counts, dispatches,
   barriers, and per-token host allocation;
3. effective bytes and bandwidth for each projection family; and
4. complete warmed real-chat generation throughput with exact workload facts.

Choose the family with the largest measured whole-path ceiling. Implement only
one candidate, require token/logit/KV parity, run `bash scripts/ci.sh`, and use
the five-pair interleaved retention rule after the candidate passes a single
real-path screen.

Do not begin residency sets, packed arenas, argument buffers, private weights,
indirect command encoding, GPU sampling, or additional warm-up shapes in this
baton. Full-KV prefault removal is a separate lifecycle/memory decision and
must not be mixed into decode kernel attribution.

## Stop Condition

Stop when one complete warmed resident decode has trustworthy per-family GPU
and host attribution, the largest end-to-end optimization ceiling is named,
and the next implementation slice is narrow enough to prove with independent
parity and a real-product benchmark.

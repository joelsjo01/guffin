# Next Task

## Status

Stage 0 is complete as of 2026-08-06. No implementation task is active. Stop
here until the user selects the next stage; Stage 0 has no remaining work or
further subdivisions.

## Verified measurement foundation

The canonical chat now requests a versioned opt-in attribution record emitted
by the unchanged production path. Every turn reports raw decode wall/GPU and
selection time, command/encoder/dispatch/barrier/allocation counts, kernel
families, cache positions, cumulative attention-token work, compact selection,
and checked scheduled-weight byte categories. The Python driver correlates each
record with the existing headline timing lines and rejects schema, count,
duration, cache, byte-sum, and overflow inconsistencies. Ordinary interactive
chat output remains unchanged without `--decode-diagnostics`.

The target plan schedules 6,837,376,000 logical weight bytes per decode token:

| Category | Bytes/token |
| --- | ---: |
| Selected F16 embedding row | 7,168 |
| Attention projections | 1,080,188,928 |
| FFN gate/up | 2,427,715,584 |
| FFN down | 1,492,033,536 |
| F16 output head | 1,835,008,000 |
| Norms | 2,422,784 |
| **Total** | **6,837,376,000** |

The old 6,837,368,832 figure counted projection and norm weights but omitted the
single 7,168-byte F16 embedding row actually read. Scheduled bytes are logical
operation accounting, not a claim about physical hardware traffic.

The thermal-safe canonical run completed with the unchanged prompts, seed,
natural stopping, startup warm-up, five-second inter-turn pause, and separate
cooled kernel process:

| Turn | Eval | Cache | Decode | TTFT | Prompt t/s | Generation t/s | GPU/decode token | Wall/decode token | Scheduled GB/s | Attention tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 16 | 0 | 222 | 175.2 ms | 91.3 | 14.0 | 70.98 ms | 71.94 ms | 96.3 | 1,188,810 |
| Follow-up | 19 | 238 | 245 | 609.6 ms | 31.2 | 13.7 | 71.91 ms | 73.16 ms | 95.1 | 3,910,200 |

The first turn averaged 644.0 dispatches per decoded token and the follow-up
686.0 as longer-context fused-attention partitioning increased. Production
scheduled bandwidth is compatible with the separately measured sustained
projection regime rather than the misleading isolated-dispatch values.

## Verified Metal profiling path

`scripts/capture_decode_gpu_trace.py` warms the ordinary runtime and captures
exactly the next command buffer labeled `Gemma2 resident decode`. Capture starts
before command-buffer creation, stops after completion, refuses overwrites, and
disarms after one command. Stable debug groups identify embedding, per-layer
attention projections, attention/KV, attention output, FFN gate/up/GELU, FFN
down, final norm/output, and selection without adding dispatches or barriers.

A real target-model capture completed successfully with
`MTL_CAPTURE_ENABLED=1`. The `.gputrace` was an 8.5 GB directory because Metal
snapshots resident resources referenced by the one captured command buffer; it
was verified nonempty and then removed from `/tmp`. No generated trace is
tracked. Full Xcode is not installed, so occupancy, limiter, register, cache,
and ALU inspection remains an optional manual follow-up rather than invented
counter data.

The local Apple M4 exposes only the runtime `timestamp` / `GPUTimestamp` counter
set. Stage-boundary sampling is supported; draw, dispatch, tile, and blit
boundary sampling are not. Runtime counter buffers therefore cannot supply the
detailed limiter evidence required by later kernel decisions; use the captured
GPU trace and Xcode profiler.

## Frozen correctness foundation

The independent CPU decoder was moved from the monolithic Metal-plan test into
reusable test support. Its sequential trace records every position's canonical
post-softcap logits, fully sorted candidate set, logical K/V state, and cache
position. A three-token Metal parity test checks every logits row and live K/V
transition while crossing the 8-token sliding-window boundary and allowing the
global layer to continue growing. Existing nonzero-cache, stale-row, context,
prefill/incremental, selector, and unavailable-Metal coverage remains intact.

Sampling now has one pure preparation function shared by `TokenSampler::sample`.
It freezes retained token IDs and normalized probabilities after the existing
repetition-penalty, top-k, top-p, min-p, and temperature order. Tests cover
greedy and sampled distributions, ties, duplicate candidates, repetition
history, filter boundaries, normalization, and seeded behavior. Speculative
acceptance was not added.

## Reproducibility and planning state

The quick-benchmark JSON includes a validated environment manifest with engine
and diagnostic schema identity, real-path executable/build mode, absolute model
identity and size, GGUF version/tensor/metadata counts, architecture, machine,
Metal device, macOS product/build, and the complete workload protocol. Optional
unavailable descriptive fields remain explicit rather than being fabricated.

`docs/DECODE_ACCELERATION_MASTER_PLAN.md` now uses the corrected sustained
width ratios and the exact scheduled-byte definition. Its fixed `<1.35x`
width-two verifier assumption was removed; a future verifier must establish its
economic gate from full causal latency, proposal cost, and expected committed
tokens. The pinned llama.cpp reference remains valid because the model, commit,
hardware, macOS, prompts, sampling, and timing definitions did not change, so it
was not rerun.

## Verification completed

- `bash scripts/ci.sh`: passed; 14 Python tests and 306 C++ tests.
- `make MODE=release benchmark`: passed.
- `scripts/quick_benchmark.py --model models/gemma-2-9b-Q4.gguf`: passed with
  both attribution records, environment validation, and all sustained kernel
  diagnostics.
- One real bounded target-model `.gputrace`: created, validated nonempty, and
  removed after recording its 8.5 GB resource-snapshot cost.
- `git diff --check`: rerun at final handoff.

This work adds measurement and correctness foundations, not an optimization.
No decode speedup is claimed and no production inference dependency, alternate
scheduler, replayed attribution pass, generated benchmark output, trace, model,
or environment file was added.

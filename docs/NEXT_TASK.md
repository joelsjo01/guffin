# Next Task

The six-gate benchmark foundation is complete. Use this as the first
benchmark-driven optimization baton.

## Objective

Reduce per-token Metal decode latency by removing unnecessary per-operator
command-buffer commit/wait boundaries from one production decode step. Keep the
slice narrow and prove improvement with the integrated benchmark.

## Baseline Evidence

- Release: arm64 macOS 26.5.2, `-O3 -DNDEBUG -mcpu=apple-m4`.
- Model: `models/gemma-2-9b-Q4.gguf`.
- Warm greedy prompt: mean 1.750 t/s for 21 prompt tokens.
- Warm greedy generation: mean 0.287 t/s for 8 generated tokens.
- Cached-second generation: mean 0.292 t/s.
- Seeded sampled generation: mean 0.281 t/s.
- Greedy and sampled decode are close, so CPU sampling/full-logit readback is
  not the primary bottleneck.
- `metal_tensor_ops.mm` contains repeated command-buffer `commit` plus
  `waitUntilCompleted` boundaries across individual operations. At roughly
  3.5 seconds per generated token, the shared decode execution path is the
  first target.

Exact baseline command:

```bash
./build/release/local-gguf-generate models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." 8 --backend metal --greedy --bench \
  --warmup 1 --repetitions 2 --temperature 0.8 --top-k 40 --top-p 0.95 \
  --repeat-penalty 1.0 --seed 42 --output human
```

## Implement

- Trace one `run_gemma2_metal_decode_step` through Metal tensor operations and
  identify which synchronization boundaries are required for CPU readback or
  resource lifetime versus merely separating dependent GPU operations.
- Add the smallest reusable command-buffer/encoder ownership seam needed to
  submit a coherent decode unit with fewer waits.
- Keep weights, KV, and resident scratch on Metal. Preserve the final greedy
  selected-token readback and sampled final-logit readback boundaries.
- Start with one coherent decode block/step; do not broadly rewrite every Metal
  operator at once.
- Add deterministic parity tests for token IDs, logits where applicable, KV
  writes/cache positions, and unavailable-Metal behavior.
- Re-run the exact release benchmark and report before/after decode throughput
  and correctness.

## Do Not

- Do not change kernels, quantization math, sampling, context policy, benchmark
  schema, or chat behavior in the same slice.
- Do not remove synchronization required before CPU readback.
- Do not compare llama.cpp yet or add a broad graph/command scheduler.

## Verify

- Run `bash scripts/ci.sh` and `git diff --check`.
- Run the exact release human baseline command from a host Metal Terminal.
- Require identical greedy token IDs and seeded sampled reproducibility.
- Keep benchmark artifacts ignored.

## Stop When

One production decode unit uses fewer command-buffer waits, correctness/CI
pass, and the real benchmark shows whether the change improves decode. Stop
before further kernel or scheduler work.

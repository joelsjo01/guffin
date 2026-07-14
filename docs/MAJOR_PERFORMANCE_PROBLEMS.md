# Major Performance Problems

Gemma 2 9B Q4 now runs through a model-resident Metal path with one command
buffer per decode token and specialized Q4_K/Q6_K SIMD matvec kernels. Those
structural milestones are complete; overall inference is still far from the
performance target.

## Current Priorities

### 1. Attribute The Complete Decode Token

The established warm greedy baseline is `3.130 t/s` mean and `3.000 t/s`
median. The rejected F16 vocabulary-projection rewrite improved its isolated
kernel by about `3.55 ms` but did not improve end-to-end inference. Do not
return to it without evidence that it has become a material share of token
time.

Measure one real resident token across:

- all `252` Q4_K and `42` Q6_K projections;
- attention score, softmax, and value work;
- norm, residual, activation, RoPE, and KV operations;
- final projection and token selection;
- gaps between the roughly `845` compute/blit encoder sections.

Select the next decode change only from the largest actionable aggregate
family. Prefer work capable of removing at least 10% of total decode GPU time.
If projections dominate, improve the aggregate Q4_K/Q6_K workload. If gaps or
small operations dominate, investigate encoder consolidation and fusion. If no
such opportunity exists, stop decode tuning and move to prefill.

### 2. Replace The Scalar Quantized Prefill Kernels

Prompt processing remains around `1.8-1.9 t/s`. Multi-row Q4_K/Q6_K projection
still assigns one output element to one thread, serially walks the full inner
dimension, and repeatedly unpacks quantized values. It lacks the cooperative
execution and data reuse added to decode.

Build shape-specific small-batch prefill kernels with SIMD cooperation and
tiled activation reuse. Measure prompt throughput and time to first token on
the real generation path.

### 3. Decompose And Reduce Plan Creation

Plan creation has measured about `12.2 s`, versus about `0.07 s` for model
parsing and binding. Measure Metal compilation, pipeline creation, weight
upload, KV allocation, and other plan work separately before changing the
architecture.

Likely solutions depend on the measured cause: precompiled Metal libraries,
less weight copying, deferred work, or cheaper cache allocation.

### 4. Reduce The KV Cache Footprint

The plan reserves about `5.64 GB` for an F32 cache covering all `8192`
positions and `42` layers. This raises startup cost and memory pressure and
will become more important at long contexts.

Investigate F16 KV, incremental capacity, and a `4096`-position ring for
sliding-window layers while preserving full history for global-attention
layers. Treat this as a separate correctness and long-context performance
project, not a speculative short-context decode optimization.

## Decision Order

1. Complete resident decode attribution.
2. Implement one measured decode change only if it has material whole-token
   potential.
3. Specialize quantized prefill.
4. Profile and reduce plan creation.
5. Reduce KV memory and optimize long-context attention from dedicated
   measurements.

Use end-to-end benchmarks as the acceptance test. Isolated kernel wins are
diagnostic evidence, not completed performance improvements.

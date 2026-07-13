# Major Performance Problems

The integrated benchmark exposes several large problems. These are not small
kernel-tuning opportunities:

- **Decode kernels remain extremely slow:** after removing per-operation
  submissions, warm greedy generation is about `0.930 t/s`, or roughly `1.08 s`
  per token. Q4_K/Q6_K matvec is the next major target.
- **Prefill is also very slow:** warm prompt processing is about `1.92 t/s`, so
  first-token latency remains high even after the model is resident.
- **Startup plan creation is excessive:** model parsing and binding takes about
  `0.073 s`, while Metal plan creation/upload takes about `12.23 s`.
- **First visible output takes far too long:** plan creation plus slow prefill
  still puts useful output well beyond interactive startup expectations.
- **The KV allocation is suspiciously large:** the plan reserves about
  `5.64 GB` for KV cache up front. This likely increases startup and memory
  pressure and should be investigated separately after the current decode
  kernel work.

## Baseline Context

These measurements use the release Gemma 2 9B Q4 Metal path with an 8-token
generation limit. The first major fix reduced decode from `27.87 s` to `8.60 s`
for eight tokens by replacing hundreds of commits/waits and transient buffers
with one resident submission. The structural synchronization issue is solved;
the remaining decode problem is now primarily kernel execution.

The earlier `llama.cpp` CLI example reported `19.0 t/s` prompt processing and
`15.6 t/s` generation. That is an informal reference rather than a controlled
comparison, but the gap is large enough to confirm that the current problems
are architectural, not benchmark noise.

Kernel-level profiling has not yet divided these totals among individual GPU
kernels. The benchmark proves the end-to-end severity; it does not yet prove
the exact internal share of every startup or decode cost.

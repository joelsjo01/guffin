# Next Task: Awaiting the Next Authorized Pipeline Finding

## Verified Current State

Finding 6 is implemented. Gemma 2 Metal generation no longer copies or sorts a
full vocabulary on the CPU and no longer has separate sampled and resident
top-logit representations.

The production path now has one exact compact-candidate contract:

- `TokenSampler` validates `1 <= top_k <= 256` and
  `repetition_window <= 64`, owns RNG/history, emits a unique-history request,
  and completes sampling in fixed-capacity CPU arrays.
- Metal applies final softcap while selecting. A 512-element stable bitonic
  tile kernel retains `K + U`, deterministic truncated merge stages reduce the
  tile runs, and one final kernel appends the `U` history logits.
- The exact sparse-change bound is used:
  `top_K(adjusted) <= top_(K+U)(base) union history`.
- Greedy uses a specialized 512-thread resident top-1 reduction through the
  same result contract.
- The candidate workspace is persistent in `Gemma2MetalModelPlan`, shared by
  prefill/decode, included in scratch accounting, and performs no per-token
  Metal allocation.
- Final prefill/decode still use one existing command buffer, one commit, and
  one synchronous wait. The standalone final-softcap dispatch is absent.
- Benchmark records expose CPU completion time, compact readback bytes,
  selector dispatches, maximum candidate count, and generated token IDs.

The old callback selector, tensor-backed sampler, vocabulary-sized candidate
and weight allocations, full sort, chat logits copy, top-logit staging,
resident top-logit encoder/reader, and generic backend top-logit API and tests
have been deleted. The independent pure-CPU `top_logit_f32` oracle and general
backend tensor-copy facility remain.

## Correctness Evidence

`bash scripts/ci.sh` passes all 303 tests on the M4 Metal host.

Coverage includes:

- compact sampler ordering, transform order, positive/negative repetition,
  promotion below one, history eviction/deduplication, seeded replay, copied
  sampler replay, reset, and product-bound failures;
- a randomized sparse-change-superset property test at `K=1/40/256`,
  `U=0/1/64`, and penalties below/equal/above one;
- Metal parity against an independent CPU full sort at vocabulary sizes from 1
  through 256000, including 511/512/513 tails and stable ties across tiles;
- exact compact count/byte accounting, one commit/wait, zero transient buffer
  allocations, raw resident-logit oracle parity, prefill/decode integration,
  session replay, seeded `/regen`, stops, streaming, and unavailable Metal;
- identical seeded token IDs between the preserved baseline and candidate:
  `[651, 8203, 8149, 3868, 1861, 576, 476, 27171]` in every measured run.

## Performance Evidence

Release build, `models/gemma-2-9b-Q4.gguf`, prompt
`Explain why the sky is blue.`, eight tokens, one warmup and five measured
repetitions, seed 42, temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, and
repetition penalty 1.1:

| Metric | Preserved baseline | Compact candidate | Change |
|---|---:|---:|---:|
| sampled generation | 7.6471 tok/s | 8.7084 tok/s | +13.88% |
| sampled decode, 8 tokens | 1,046.15 ms | 918.65 ms | -12.19% |
| sampled total turn | 2,893.94 ms | 2,742.99 ms | -5.22% |
| warm greedy generation | 8.7052 tok/s | 8.7169 tok/s | +0.13% |
| full-logit CPU readback | 1,024,000 B/token | 0 B/token | removed |

The measured sampled compact union read 4,512 bytes across eight selections:
564 bytes/token on average, with a maximum observed candidate count of 76
(608 bytes). This is below the 1,344-byte default bound. CPU compact completion
averaged about 9.8 microseconds/token. The selector used 11 embedded dispatches
per sampled selection and added no commit, wait, or transient allocation.

The command-buffer GPU timestamp covers the complete resident model step, not
individual selector dispatches. The directly comparable decode measurement
therefore includes the candidate GPU work; it is the authoritative evidence
that the combined selector/model path improved beyond variance. The old
baseline did not expose an isolated sampler timer, so its removed selection
cost is bounded by the identical-token decode delta rather than reported as a
fictional per-kernel timestamp.

### Note for later deep research: why the whole turn improved by only 5.22%

Finding 6 removed nearly all of the old sampling overhead, but sampling was
only one small part of producing a token. On the measured eight-token turn:

- prompt prefill still took about 1.82 seconds;
- old sampled decode took about 1.046 seconds for eight tokens;
- compact-candidate decode took about 0.919 seconds for eight tokens;
- the change therefore saved about 127 milliseconds, or 15.9 milliseconds per
  generated token.

This is a 13.88% improvement in generation throughput, but only a 5.22%
improvement in total turn time because the unchanged prompt prefill dominates
such a short generation. With longer generations, prefill is paid once and the
generation improvement becomes a larger share of the total.

In simple terms, the old runtime moved a one-megabyte list containing the score
of every possible token from the GPU to the CPU, then sorted all 256000 scores.
The replacement asks the GPU for only the small set that can still win, usually
around 70 candidates, and the CPU finishes the exact same seeded choice from
that set. The output is unchanged; the unnecessary transfer and full sort are
gone.

The remaining roughly 115 milliseconds per generated token is primarily model
computation rather than sampling. Future deep research seeking a much larger
gain should measure and attack the dominant transformer and output-projection
work, especially the quantized LM-head/output projection, rather than further
micro-optimizing the now roughly 9.8-microsecond CPU sampling completion. Apply
the same whole-path rule before choosing that work: profile projection totals,
memory traffic, dispatches, and achievable end-to-end gain first. Do not assume
LM-head fusion is automatically the correct next implementation.

Benchmark output is kept outside the repository at:

- `/private/tmp/finding6-baseline.jsonl`
- `/private/tmp/finding6-candidate.jsonl`

## Constraints for the Next Baton

- Preserve the exact compact-candidate theorem and transform order: final
  softcap, repetition, top-k, top-p, min-p, temperature, draw.
- Preserve one resident submission/wait and zero full-vocabulary CPU readback.
- Preserve Finding 5's F16 exact-ring KV-cache contract.
- Do not restore deleted sampling or backend top-logit compatibility surfaces.
- Do not commit benchmark output, model weights, or binaries.

No further implementation finding is authorized in this document. Select and
research the next pipeline finding only when requested; do not silently broaden
Finding 6 into LM-head fusion, GPU RNG, speculative decoding, or another
pipeline rewrite.

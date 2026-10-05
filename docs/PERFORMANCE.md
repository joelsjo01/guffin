# Performance evidence

Guffin's retained changes have measured benefits on an Apple M4: smaller Gemma 3 KV storage, faster large-prompt processing, and a modest Gemma 2 retrieval improvement with unchanged sampled output. Some speculative candidates lost after passing correctness. This page separates those recorded results from hypotheses and from the older llama.cpp reference.

These are development snapshots from September 29 and October 4, 2026, rather than a measurement of today's edited executable. Each configuration has one cooled two-turn conversation, with natural stopping and no output cap. There are no error bars or repeat-run variance estimates. Successful conversations were not rerun for this documentation update.

## What the numbers measure

The primary rate is `sum(generated_tokens) × 1e9 / sum(generation_completion_ns)`. Completion includes final KV publication, callbacks and result assembly. It excludes prompt processing, startup/model loading and the five-second input pause. The aggregate is weighted by elapsed time; it is not the average of the two displayed rates.

All selected engine receipts used an arm64 Mac16,12 with Apple M4 Metal, macOS **26.6.2 (25G83)**, seed 42, captured pipes, startup warmup of 16 prefill tokens + one decode token + two verifier tokens, and **240 seconds idle before each fresh process**. Kernel diagnostics were disabled during chat. The target used the normal CLI sampler: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, repetition penalty 1.0 and a 64-token history. Manifests have no target sampler overrides; the quick-benchmark schema did not independently serialize all target sampler scalars.

The standard short conversation is:

1. `Hello, is AI getting smarter?`
2. `Okay, so will it always progress this fast?`

Changing a mode can change both the natural response and the context seen by the follow-up. A fixed seed does not make those runs identical workloads.

## Gemma 3: ordinary decoding and native H1 speculation

The September 29 snapshot uses `gemma-3-12b-it-Q4_K_M.gguf`, native Q4_K/Q6_K weights, a Q6_K output head and a **16,384-token allocation**. Retrieval is off. H1 uses the native Gemma 3 270M Q8_0 drafter, its F32 cache, full target verification and lossless speculative sampling.

| Configuration | First: tokens / generation seconds / t/s | Follow-up: tokens / generation seconds / t/s | Combined t/s |
| --- | --- | --- | ---: |
| Ordinary, F32 KV | 1,506 / 124.674288834 / 12.08 | 1,026 / 116.288566375 / 8.82 | **10.51** |
| Ordinary, F16 KV | 1,493 / 119.748809000 / 12.47 | 967 / 103.972833708 / 9.30 | **11.00** |
| H1 lossless, F32 KV | 1,344 / 78.131872042 / 17.20 | 1,024 / 95.735071542 / 10.70 | **13.62** |
| H1 lossless, F16 KV | 1,253 / 68.374112625 / 18.33 | 1,041 / 76.986912500 / 13.52 | **15.78** |

F16's observed aggregate increase is **4.64% in ordinary mode** and **15.87% in H1 mode**. Within the F16 snapshot, H1 observes **43.52% higher aggregate throughput** than ordinary decoding. Those figures describe these natural-stop conversations: the modes produced different text, lengths and follow-up histories. They do not establish a constant speedup for identical token sequences or arbitrary workloads.

The memory effect has a clearer structural explanation: the target KV allocation falls from **2,818,572,288 to 1,409,286,144 bytes**. Scratch also shrinks. Ordinary total owned storage falls from 10,553,663,712 to 9,079,628,000 bytes; H1 total owned storage, including its drafter, falls from 11,077,685,408 to 9,602,076,832 bytes. These are allocation ledgers, not physical-memory measurements.

| Configuration | Native physical-footprint high-water, bytes | System swap-outs during run, 16-KiB pages |
| --- | ---: | ---: |
| Ordinary, F32 KV | 10,647,022,608 | 116,312 |
| Ordinary, F16 KV | 9,194,777,640 | 50,196 |
| H1, F32 KV | 11,166,920,368 | 151,952 |
| H1, F16 KV | 9,720,213,120 | 54,124 |

F16 changes the selected target's arithmetic. Lossless speculation preserves that target distribution; it does not promise the output of a different F32 target. Physical-footprint values are `proc_pid_rusage` lifetime maxima observed while the native child was alive, including Metal allocations. A late peak after the final live sample could be missed. Swap counters are system-wide and cannot attribute pages to Guffin. The `pmset` records reported no warnings, but did not measure GPU clocks, temperature or throttling.

The owner also observed approximately 30–40% faster Gemma 3 chat than llama.cpp in manual use. That remains an informal observation: no frozen, comparable Gemma 3 llama.cpp performance pair is recorded. The four-mode measurements above support a concrete same-engine story without converting that observation into a controlled competitor result.

## Gemma 3: useful ordinary-path improvements

Before the cache change, the September 26 target integration preserved both ordinary F32 responses: 1,506 and 1,026 tokens, with hashes `260ae35592bb6060` / `dd4aadff1d4f599f`. Aggregate completion throughput rose from **9.71 to 10.34 t/s (+6.48%)**. First/follow-up prompt times fell from 0.789661/1.181695 to 0.311648/0.652287 seconds. This is a separate frozen before/after result, not a baseline to mix with the later four-mode chart.

The retained M32 prompt kernel addresses larger prompts. Its October 4 candidate preserves the frozen long conversation's evaluated work and sampled outputs, with an F16 target, no retrieval and no learned speculation. The first prompt evaluates 2,018 new tokens; the follow-up evaluates 1,349 and reuses 2,940. The prompts are frozen documentation excerpts, rather than the two short prompts above.

| Turn | Frozen prompt time | M32 prompt time | Prompt throughput increase | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| First | 36.176861 s | 31.659791 s | 14.27% | 922 |
| Follow-up | 28.886381 s | 23.924771 s | 20.74% | 852 |

M32 saves **9.478680 seconds of prompt processing** across this conversation, with unchanged hashes `17bd8f1b4dd6d6d8` / `1e96156f5ec5d879`. Completion throughput also rises from 11.1446 to 11.4275 t/s (+2.54%), but scalar decode kernels did not change: that observation is not evidence of a new scalar decode mechanism. Physical-footprint high-water stays effectively flat, 9,265,605,672 → 9,266,539,560 bytes; neither run adds system swap-outs.

The measured trial also included M16 for short prompts. M16 increased short-chat internal TTFT by about 3.9% and was removed. The measured long route uses M32 throughout its 128-row chunks and 98/69-row tails, so removing M16 leaves that route intact. The retained post-removal executable passed correctness checks; the successful conversation was not repeated. This is why the result is attributed to the measured candidate, not assigned a fresh measurement date to the current build.

## Gemma 2: a small retained gain and a rejected drafter

The October 4 comparison uses `gemma-2-9b-Q4.gguf`, **8,192-token capacity**, F16 KV and the existing default retrieval gate. “Default” here includes retrieval; these are not ordinary `--no-speculative` results. Packed Q4 projections and the shared two-row verification path are retained. The frozen and updated retrieval runs have identical prompt work, token counts, selected-token hashes and retrieval decisions.

| Configuration | First: tokens / generation seconds / t/s | Follow-up: tokens / generation seconds / t/s | Combined t/s |
| --- | --- | --- | ---: |
| Frozen default retrieval | 276 / 16.560730250 / 16.67 | 322 / 19.657151916 / 16.38 | **16.51** |
| Updated target + retrieval | 276 / 16.216601500 / 17.02 | 322 / 19.274319958 / 16.71 | **16.85** |
| Rejected Arm R H1 candidate | 258 / 17.149000541 / 15.04 | 274 / 17.977660417 / 15.24 | **15.15** |

The retained port observes **+2.05% aggregate generation**, winning both turns. First/follow-up prompt times improve from 165.231/270.842 to 152.693/270.503 ms; first visible text improves from 225.783/339.899 to 214.615/339.395 ms. Packed weights add 127,647,744 bytes, and scratch grows by 17,636,652 bytes. Native physical-footprint high-water rises 1.61%, from 10,872,319,400 to 11,047,546,400 bytes. This small gain comes with a memory cost and only one cooled comparison.

Arm R passed correctness but observed **10.11% lower aggregate throughput** than the updated retrieval path. It accepted 104 of 411 proposed H1 edges (25.30%); proposal cost was about 11.7 ms per cycle on top of roughly 70 ms target verification. Its different, shorter responses also change follow-up history, so the rejected run is not identical work. It ran first and system swap-outs were 45,212 pages, versus 0 for frozen retrieval and 3,616 for updated retrieval. Those counters cannot isolate its memory-pressure effect. The implementation and receipts are retained locally; the recurrent runtime was removed from production.

## What the llama.cpp reference can establish

The [pinned external reference](LLAMA_CPP_REFERENCE_BASELINE.md) records Gemma 2 on August 6, 2026, with Apple M4, **macOS 26.5.2 (25F84)** and llama.cpp `d48a56effbba99944a677938d4beb63a0065ecdf` / llama-cli 9430. Its natural responses contain 239 and 257 sampled tokens, at the reported **14.1 and 13.9 t/s**; the recorded generation durations are 16.9686 and 18.5227 seconds. This is a historical reference, not a same-condition competitor bar beside October's 16.85 t/s.

There are substantive differences beyond the date. Guffin's Gemma 2 executes Q6 down weights requantized to Q4 and an F16 output head requantized to Q8, retaining a separate F16 embedding. The file name therefore does not mean both engines executed identical weight precision. The responses and prompt-cache work differ, and llama.cpp's sample-to-final-sample timing is not Guffin's complete-generation clock. There is no controlled current percentage advantage or quality-equivalence claim here, and no recorded Gemma 3 llama.cpp timing to fill that gap. The pinned document remains unchanged.

## Receipts and calculation trail

The raw conversations, frozen executables and per-process traces are local development artifacts under ignored `build/` and `benchmarks/results/`. The filenames below identify provenance; they are deliberately plain text rather than links to files absent from a public checkout. These tables retain counts, timing scopes and identities without publishing the full raw datasets. [Development history](archive/DEVELOPMENT_HISTORY.md) retains the experiment decisions; [the benchmark driver](../scripts/quick_benchmark.py) describes the current collection and validation protocol.

<details>
<summary>Selected receipt identities and exact calculation inputs</summary>

The cache/M32/port receipts record engine build `dff16f08a7e3439e82b6f7eaf660ee43f547716c-dirty`; the early integration baseline appends `-kernel-baseline`. Those labels do not identify clean source revisions; use the executable SHA below to distinguish configurations. The current executable's name and source may have changed since collection.

| Snapshot | Local receipt directory and filenames |
| --- | --- |
| Gemma 3 cache modes | `benchmarks/results/gemma3-f16-kv-2026-09-29/`: `awake16-baseline-ordinary.json`, `native16-candidate-ordinary.json`, `native16-baseline-h1-full.json`, `native16-candidate-h1-full.json` |
| Gemma 3 early integration | `benchmarks/results/gemma3-target-integration-2026-09-26/`: `baseline.json`, `final.json` |
| Gemma 3 M32 | `build/gemma3-ordinary-implementation-2026-09-29/results/`: `baseline-long.json`, `matrix-long.json` |
| Gemma 2 target port | `build/gemma2-recurrent-h1-2026-10-04/results/`: `baseline-stage4.json`, `candidate-stage4.json`, `candidate-rh1.json` |

The selected cache/M32/port receipts have adjacent `.manifest.json`, `.resources.json` and `.native-rusage.json` files. Binary hashes agree across the receipts, frozen manifests, resource reports and launch records. The original `native16-baseline-ordinary.json` was invalidated by a 174.078-second system suspension; the `awake16` replacement is the ordinary F32 row above. It is not cherry-picked as an additional successful sample.

| Frozen executable | SHA-256 |
| --- | --- |
| Gemma 3 F32 | `b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab` |
| Gemma 3 F16 | `1432247a6eb8edf94a93390e591f2efd43a5ec3bb0401055a89cd3ad08c42216` |
| Gemma 3 early baseline | `66b3b14521b2b7ce97a7c0fe9e170d99bdc49e0d412e70215bbd3ab6b72de187` |
| Gemma 3 early integration | `326fcd150716e39838f835c4ab2ce85ca4e8bb779d8d646fa41adabf905d5d75` |
| M32 frozen baseline | `29914d243941dd7513917e97d87aa005d618c1f5506b861d2ddb3edb526c6fc7` |
| Measured matrix trial | `0d828c5578c8ed4f851ab1ead4b7029081dc24c31a233bd6e24f861c62c710d3` |
| Gemma 2 frozen retrieval | `b4c6041322e5361bf469368d1785bbe8cf1285734bb9f1a8faad332926906300` |
| Measured Gemma 2 port / Arm R | `14fee23ab7c0df08fad137bdde428ae09001b0a2771f80cc044af20285cfd1f3` |

Model identities: Gemma 3 target SHA-256 `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5` is authenticated separately in `build/gemma3-target-reference/identity.json`; its GGUF has 7,300,574,976 bytes, 626 tensors and 40 metadata entries. Gemma 2 has 6,843,425,696 bytes, 464 tensors and 33 metadata entries; an earlier identity receipt, `benchmarks/results/gemma3-tli-W2vqJl/stage4.json`, records SHA-256 `0452e54ba9b6c5b2e466fbd7f28c089fefb7c3dceddcc2b75becab6b16cae523`. The selected quick-benchmark receipts record target path, size and metadata, rather than freshly attesting a target SHA for every run. H1 directly records drafter SHA-256 `0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3`.

| Configuration | First / follow-up selected-token hashes | Total generated tokens | Total completion nanoseconds |
| --- | --- | ---: | ---: |
| Gemma 3 ordinary F32 | `260ae35592bb6060` / `dd4aadff1d4f599f` | 2,532 | 240,962,855,209 |
| Gemma 3 ordinary F16 | `8b3b0b12c0cc1348` / `86b87b255be9444c` | 2,460 | 223,721,642,708 |
| Gemma 3 H1 F32 | `3a9063eeaaadedd9` / `ecc10d1344ecd5e7` | 2,368 | 173,866,943,584 |
| Gemma 3 H1 F16 | `c111f86600fc6b29` / `f04dd01aadbb08aa` | 2,294 | 145,361,025,125 |
| Gemma 3 early baseline | `260ae35592bb6060` / `dd4aadff1d4f599f` | 2,532 | 260,760,855,542 |
| Gemma 3 early integration | same as early baseline | 2,532 | 244,894,256,083 |
| M32 frozen baseline | `17bd8f1b4dd6d6d8` / `1e96156f5ec5d879` | 1,774 | 159,179,846,292 |
| M32 candidate | same as M32 baseline | 1,774 | 155,239,559,209 |
| Gemma 2 frozen retrieval | `49080569719b1cea` / `369b09d9b01ff1e1` | 598 | 36,217,882,166 |
| Gemma 2 updated retrieval | same as frozen retrieval | 598 | 35,490,921,458 |
| Gemma 2 Arm R | `1939edd3e0e92a51` / `066af2b16076914d` | 532 | 35,126,660,958 |

The cache-mode snapshots evaluate 16/19 new prompt tokens. Follow-up cached tokens are 1,522, 1,509, 1,360 and 1,269 respectively, reflecting the different first answers. Gemma 2 retrieval evaluates 16/19 with 292 cached tokens on the follow-up; Arm R reuses 274. The H1 drafter records the same sampler scalars as the target, represented as F32 values, and a 64-token repetition window.

Some historical cache-comparison tables use the narrower printed `generation_ns` despite declaring completion as their primary clock. This page recomputes every engine rate from `decode_attribution.generation_completion_ns`; the historical archive is preserved rather than silently rewritten. For example, the F16 H1 cache gain here is 15.87%, not the older printed-clock 15.67%. A repeat of the current benchmark protocol can check present behavior, but cannot recreate these historical executable/configuration snapshots merely by building today's source.

</details>

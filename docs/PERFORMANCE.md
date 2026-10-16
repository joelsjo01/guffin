# Performance evidence

The record contains three original default-chat campaigns and a later Gemma 3
H1-versus-H1 comparison on a **16 GiB MacBook Air M4** (`Mac16,12`), macOS
**26.6.2 (25G83)**. Each completed 38 conversations / 80 turns per engine with
no failures, interruptions,
unattempted cases or reported memory-collection errors. The retained evidence
includes manifests, atomic receipts, raw logs, reports and standalone charts.

These are single-pass observations of instrumented local chat. They establish
the measured workload results below; they do not establish answer quality,
identical executed numerics, repeat-run stability or a universal advantage.

## Final release results

The core has 32 balanced two-turn MT-Bench conversations. Six supplementary
conversations cover short replies, multilingual input, document/code review
and sustained discussion; their 16 turns remain a separate suite.

| Model and mode | Core Guffin / llama.cpp | Core rate ratio | Supplementary Guffin / llama.cpp | Supplementary ratio |
| --- | ---: | ---: | ---: | ---: |
| Gemma 2 9B default | 16.76 / 13.63 t/s | 1.23× | 15.40 / 12.64 t/s | 1.22× |
| Gemma 3 12B ordinary | 12.57 / 12.40 t/s | 1.01× | 12.62 / 12.51 t/s | 1.01× |
| Gemma 3 12B speculative defaults (H1 / H3) | 18.43 / 8.43 t/s | 2.19× | 16.63 / 7.81 t/s | 2.13× |
| Gemma 3 12B speculative, both H1 | 18.53 / 13.26 t/s | 1.40× | 16.99 / 12.94 t/s | 1.31× |

Rates are summed generation steps divided by summed generation durations over
jointly complete cases. All planned cases qualify in these campaigns. The
ordinary Gemma 3 difference is small (**1.40%** on the core); one pass does not
demonstrate a stable lead. The speculative comparisons use the same 270M
drafter checkpoint. Guffin's H1 lossless path is compared with llama.cpp's
`draft-simple` path, first at default horizons, then with both using H1.

The same Guffin binary appears in all three Gemma 3 campaigns. In the original
default campaigns, its core speculative rate is 46.61% higher than its ordinary rate; llama.cpp's speculative rate is
32.03% lower than its ordinary rate. Those are cross-campaign observations on
different natural responses and histories, not identical-work speedups or a
general finding about the engines' other speculative configurations.

The H1 follow-up sets `LLAMA_ARG_SPEC_DRAFT_N_MAX=1` for llama.cpp; Guffin
already defaults to H1. All 38 llama.cpp logs confirm `n_max=1`, and all 80
Guffin turn diagnostics confirm horizon 1, full verification and lossless
sampling. Core generation is **39.75% higher in Guffin**, supplementary
**31.24% higher**, with conversation-level throughput wins in all 38 cases.
llama.cpp's core rate is **57.34% higher** than in its H3 default campaign;
Guffin's differs by **+0.53%** from its earlier H1 campaign. The binaries,
models and corpus match the earlier campaigns. Horizon is matched, while
algorithms and other defaults remain engine-specific: context allocations
are 16,384 tokens for Guffin and 43,008 for llama.cpp.

### Latency and memory

| Model and mode | Core median internal TTFT, Guffin / llama.cpp | Core peak physical footprint, Guffin / llama.cpp | Supplementary peak, Guffin / llama.cpp |
| --- | ---: | ---: | ---: |
| Gemma 2 default | 399.82 / 1,179.89 ms | 9.15 / 2.58 GiB | 9.17 / 5.49 GiB |
| Gemma 3 ordinary | 597.63 / 966.35 ms | 8.62 / 5.06 GiB | 8.62 / 7.26 GiB |
| Gemma 3 speculative defaults (H1 / H3) | 597.94 / 1,020.54 ms | 9.15 / 4.47 GiB | 9.16 / 7.10 GiB |
| Gemma 3 speculative, both H1 | 592.95 / 931.52 ms | 9.15 / 4.47 GiB | 9.15 / 7.05 GiB |

Guffin has lower internal TTFT on all 42 follow-ups in each campaign. The
engines reuse different amounts of prefix work and generate different first
answers, so those wins do not isolate a prompt-kernel advantage. All 38 first
prompt token counts match within each campaign. Guffin wins first-turn TTFT on
36/38 Gemma 2 cases, 24/38 in the original Gemma 3 campaigns and 25/38 in
the H1-versus-H1 campaign. In that last campaign, core first-reply medians
are 0.613 / 0.811 seconds and follow-up medians are 0.454 / 1.304 seconds
(Guffin / llama.cpp; 32 observations per median). Core peak sampled RSS is
12.29 / 11.30 GiB.

Large first prompts remain slower in Guffin:

| Model and mode | Code review first TTFT, Guffin / llama.cpp | Document review first TTFT, Guffin / llama.cpp |
| --- | ---: | ---: |
| Gemma 2 default | 13.683 / 13.295 s | 34.350 / 31.953 s |
| Gemma 3 ordinary | 20.726 / 17.092 s | 47.687 / 38.603 s |
| Gemma 3 speculative defaults (H1 / H3) | 20.665 / 17.805 s | 47.696 / 39.124 s |
| Gemma 3 speculative, both H1 | 20.676 / 17.134 s | 47.319 / 38.310 s |

Internal TTFT excludes model loading, terminal transport and tokenization.
Memory is the native process physical-footprint lifetime maximum observed
while the CLI is alive, including Metal and draft allocations. It differs
from the sampled resident-size maximum and from an engine allocation ledger.
A peak after the last live observation can be missed. Guffin reports a higher physical footprint, but macOS accounting differs for
copied and mapped weights; the gap does not isolate an equal amount of extra
RAM consumption or memory pressure. RSS remains a separate measurement, and
system swap counters cannot attribute pages to an individual engine.

### Retained evidence and identities

| Campaign | Manifest | Full report |
| --- | --- | --- |
| Gemma 2 default | [gemma2-release-v2](../benchmarks/results/gemma2-release-v2/manifest.json) | [Results and charts](../benchmarks/results/gemma2-release-v2/report/summary.md) |
| Gemma 3 ordinary | [gemma3-normal-release-v2](../benchmarks/results/gemma3-normal-release-v2/manifest.json) | [Results and charts](../benchmarks/results/gemma3-normal-release-v2/report/summary.md) |
| Gemma 3 speculative defaults | [gemma3-speculative-release](../benchmarks/results/gemma3-speculative-release/manifest.json) | [Results and charts](../benchmarks/results/gemma3-speculative-release/report/summary.md) |
| Gemma 3 H1 versus H1 | [gemma3-h1-vs-h1](../benchmarks/results/gemma3-h1-vs-h1/manifest.json) | [Results and charts](../benchmarks/results/gemma3-h1-vs-h1/report/summary.md) |

An October 14 offline audit of the original three campaigns checked all
76 receipts / 160 turns per campaign, answer hashes and log references. Regenerating each report produced identical
JSON, CSV, Markdown and SVG bytes; no inference was rerun. An October 16 audit
checked the H1 campaign's 76 receipts / 160 turns, answer hashes, log references,
raw metrics and draft horizons; its aggregates match the retained report.

Each report links its JSON, turn-level CSV and SVG charts. The adjacent
`receipts/` and `logs/` directories retain answers, clocks, counts, stop
information, memory samples and failure details. The manifest pins the corpus,
measurement scripts, environment, command lines, binary/model hashes and
linked-library identities; it is the authority for recreating the configuration.

| Artifact | SHA-256 |
| --- | --- |
| Gemma 2 GGUF | `0452e54ba9b6c5b2e466fbd7f28c089fefb7c3dceddcc2b75becab6b16cae523` |
| Gemma 3 target GGUF | `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5` |
| Gemma 3 drafter GGUF | `0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3` |
| Guffin, Gemma 2 campaign | `aaf1adbcd672a59679f24cd69327fec023525d8f96876e1ce7b47a1875adb595` |
| Guffin, all Gemma 3 campaigns | `dd18c4f5213bd9b6d3de877b165e4462972aafdfdad975b02955bd59a1f3440b` |
| llama-cli, all campaigns | `9765459b6789303e8f2233796df1778adef17a34d5c2396fd3a92e4ae6264757` |

The campaigns were created on October 7, 12, 13 and 15 UTC respectively. Guffin's
reported source identity is `644312e18197379cb5e6d3ef25f1e90dd3ed5553-dirty`;
the different binary hashes identify the measured builds. llama.cpp is
0.5.0, build 11146, commit `7fe450e19`. Later source fixes or documentation
commits do not turn these into measurements of a newly built executable.

## Methodology and comparison limits

[The benchmark guide](BENCHMARKING.md) owns commands and recovery. The runner
drives the actual installed CLIs through a pseudoterminal, with one fresh
process per conversation and rotating configuration order. It waits 300 seconds
before each launch and 120 seconds between messages in the live process.
Cooling reduces heat accumulation on this fanless machine but does not prove
equal GPU clocks, temperature, VM state or background activity.

The CLI inference defaults remain intact: no benchmark seed, context,
sampler, output-cap, GPU-placement or warmup override. The original three
campaigns also keep default horizons; the H1 follow-up overrides only
llama.cpp's maximum draft length through the recorded environment. Measurement
logging adds Guffin `--decode-diagnostics` and llama-cli `--log-verbosity 4`;
that instrumentation has overhead, especially during prefill. Follow-ups use
each engine's own preceding answers. Equal user messages do not imply equal
generated work or identical cached histories.

- Common generation rate is `sum(max(selected_tokens − 1, 0)) / sum(generation_seconds)`.
  Draft proposals and rejected tokens do not count as emitted generation.
  Native printed rates remain available separately.
- Internal TTFT runs from post-tokenization prompt handling to the first
  target-token selection. Generation runs from first to last selection;
  startup, cooling pauses and input waits are excluded.
- Native counts include their own stopping-token conventions. Guffin reports
  `stop_token` for all release turns. llama-cli's exact per-turn stop reason
  is unexposed and remains `unknown`; frontend completion is not proof of EOS.
  Known limited, interrupted or incomplete replies do not enter headline rates.
- The core and supplementary suites remain separate. Aggregates sum work and
  time before division; they do not average rounded per-turn rates.
- There are no repetitions, confidence intervals or measured variance. Answers
  are retained for inspection, but the campaign has no answer-quality scoring
  or independent numerical oracle.

Gemma 2's same GGUF hash does not establish equal executed weights: Guffin
requantizes Q6 down projections to Q4 and an F16 output head to Q8, while keeping
its separate F16 embedding. Default suffix retrieval is also part of that
product path. Gemma 3 retains native target quantization, but precision,
sampling and speculative implementation still require their own contracts.

The August [pinned llama.cpp reference](LLAMA_CPP_REFERENCE_BASELINE.md) uses
another executable, OS version, prompt pair and timing convention. It remains
a historical planning reference, not a bar in the final release comparison.

## Numerical limits

Performance and numerical fidelity are different evidence. Independent
checkpoint, primitive, sampler and lifecycle tests qualify particular contracts;
successful benchmark conversations do not replace them.

### F16 KV feedback

The retained Gemma 3 F16 path exceeds its original independent full-model
bounds on a reproduced natural 128-token prompt. The September baseline,
later native execution and original CPU oracle reproduce exactly, so later
packing/loading/prompt work did not introduce this disagreement.

| Original CPU comparison | Maximum logit difference | Logit violations | KV component violations | Head-RMS violations |
| --- | ---: | ---: | ---: | ---: |
| Native F16 KV | 0.00402546 | 657 | 23,569 | 168 |
| Native F32 KV | 0.00009996 | 0 | 0 | 0 |

The logit gate is `0.002 + 0.0003 × abs(reference)`. F16 cache components use
`0.0005 + (0.0003 + 2^-10) × max(abs(reference), reference_head_RMS)`;
head RMS uses the same coefficient on reference head RMS. These are regression
budgets, not analytical transformer-error or answer-quality guarantees.

Same-input FP64 operator checks found small local arithmetic errors. Captured
half stores matched independent nearest-even rounding; native and CPU batch
prefixes each matched their own smaller-batch runs. Changing only the CPU RoPE
frequency expression from `base**(-exponent)` to `1/(base**exponent)` also made
two mathematically equivalent CPU F16 executions exceed the original bounds.
The controls support amplified rounding feedback without establishing a wrong
model equation, cache index or half conversion. They do not exclude every
possible defect or resolve the original gate.

The default prepared next-token law happened to match in the inspected final
conditional; that single conditional cannot establish answer-quality or
arbitrary-context fidelity. Lossless H1 preserves the configured native target
law, including its precision. Neither reference equations nor tolerances were
changed to hide the failure.

### Half prompt operands did not qualify

The complete half-operand prompt implementation used native quantized weights,
nearest-even operands, F32 accumulation and an independent matching-policy CPU
oracle. Both full-body and MLP-only policies failed the original four-token
full-vocabulary logit/cache gates:

| Candidate | Maximum logit difference | Logit violations | KV component violations |
| --- | ---: | ---: | ---: |
| All seven body projections half | 0.00743341 | 21,881 | 4,411 |
| Gate/up/down only half | 0.00661278 | 12,585 | 2,288 |

Captured same-input projections matched independent CPU products bit for bit;
the observed full-model disagreement developed through rounding feedback.
Small prepared-law differences in one context did not qualify fidelity or
quality. Both candidates and their exclusive runtime machinery were removed.
The retained F32 prompt/F16 KV four-token control passes unchanged bounds.

## Historical development evidence

The [retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md) explains the design
decisions and the [development archive](archive/DEVELOPMENT_HISTORY.md) preserves
the longer research record. Old local benchmark outputs were removed when the
three final campaigns became the retained release evidence; the findings below
survive as documented historical results, not reproducible receipt bundles.

- **Gemma 2 retrieval:** the cooled exact-match comparison generated the same
  598 tokens in 37.233 rather than 38.293 seconds, a 2.77% reduction. The later
  target port improved complete-generation rate 16.51 → 16.85 t/s (+2.05%),
  with unchanged hashes and decisions and a 1.61% physical-footprint increase.
- **Gemma 3 F16/H1:** the September four-mode snapshot observed ordinary
  F32/F16 at 10.51/11.00 t/s and H1 F32/F16 at 13.62/15.78 t/s, using total
  selected tokens divided by complete-generation time. Natural responses
  differed. F16 halved target KV allocation 2,818,572,288 → 1,409,286,144 bytes;
  that structural saving is distinct from process footprint and model fidelity.
- **Gemma 3 prompt work:** the original exact M32 adoption saved 9.48 seconds
  over a long two-turn prompt, with unchanged output hashes. Later larger tiles,
  shared bulk attention, terminal-work removal and exact direct activation
  loads reduced targeted code/document first TTFT from 31.951/75.241 to
  20.638/46.210 seconds. Those are successive retention screens, not release
  figures to add together. Final release TTFT is reported above.
- **A prompt gain has a decode caveat:** the final direct-load screen improved
  first code/document TTFT 3.66%/5.99%, but observed code generation −2.29%,
  document −0.05% and pooled generation −1.17% on unequal natural workloads.
  It was retained for exact prompt-latency improvement, not a universal
  no-regression or scalar-decode claim.
- **Gemma 2 prompt/loading:** fixing tokenizer boundaries removed excess
  input work; fused bulk attention then cut document first TTFT 63.129 →
  33.786 seconds on equal input counts and removed a 64 MiB score workspace.
  Direct final-allocation preparation reduced duplicate storage. Gemma 3
  source-contiguous packing and read advice cut an unprimed-file startup
  13.558 → 8.344 seconds in one APFS-clone diagnostic. Warm-file launch times
  are a different condition; no universal cold-start guarantee follows.
- **Negative results constrain the next design:** trained compact heads and
  native Gemma 2 recurrent H1 did not repay verifier cost. Broad Stage 5R heads
  were stopped before training; Arm P stayed incomplete. Complete H3/four-row,
  native32 and long-context attention candidates failed their actual gates.
  Kernel wins, untrained designs and partial probes do not establish chat wins.

Historical development used seed-42 two-turn chats, normally with 240-second
cooldowns, and often a complete-generation clock including publication,
callbacks and result assembly. Some archive tables use narrower printed clocks;
their original timing boundaries remain visible. They must not be mixed with
the release N−1 generation convention or treated as current measurements.

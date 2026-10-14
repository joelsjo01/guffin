# Development history

This archive preserves the original research, implementation decisions and
failed alternatives through October 5, 2026. It is a historical record, not the
current operating contract or active baton. Use [current state](../NEXT_TASK.md),
[the retrospective](../SPECULATIVE_DECODING_RETROSPECTIVE.md) and
[performance evidence](../PERFORMANCE.md) for today's handoff and release claims.

Earlier local benchmark outputs were deleted when the three final release
campaigns became the retained evidence. Original artifact paths and availability
claims below describe what existed at the time; they are preserved as provenance,
not live evidence links. Historical pending work and stop instructions do not
set current authorization. Findings, counts, numerical limits and incomplete
design distinctions remain in their original context; new task diaries are not
relocated here.

## Guffin CLI and installation complete (2026-10-05)

The installed production command is `guffin`. Chat is the default:
`guffin -m MODEL`; `-m/--model` selects the GGUF, and
`--draft-model FILE` retains the optimized native pairing.
`-p/--prompt` gives one answer and exits through the same session, target
precision and sampler as chat. The former numeric positional generation
syntax and `--chat` flag are retired. Sampling defaults, Gemma 2 retrieval
behavior, Gemma 3 F16/H1/full/lossless pairing, and context capacities remain
the engine's existing policies.

Basic help is concise, `--help-all` documents advanced controls, and
`--version` reports the compiled commit identity. The chat UI uses a short
banner and `/help`. Interactive chat shows prompt tokens/s, generation tokens/s
and internal TTFT by default. `--stats` also enables timing for one-shot and piped output;
raw counters and attribution remain under `--decode-diagnostics`. Redirected
stdout contains answers, with requested statistics and errors on stderr. `--interactive`
allows diagnostic conversation drivers to retain their prompt/completion
protocol. Updated benchmark and trace scripts use that option and the new
binary names. `guffin inspect MODEL` reuses GGUF inspection without inference.

`make` builds the M4 release executable. `make install` copies it into the
existing Apple Silicon Homebrew prefix when present, otherwise `/usr/local`;
`PREFIX` and `DESTDIR` support custom/staged installation. Benchmark sources
are linked into tests and `guffin-bench` rather than the production CLI.
CI explicitly builds debug mode, independent of the new release default.

Native `bash scripts/ci.sh` passes **507 C++ tests** and the **64-test Python
suite** (15 skips), without compiler warnings. Regressions cover help/version
without a model, validation before loading, inspection, default chat, common
aliases, shared one-shot sampling, redirected output and pipeline failures.
Staged install/uninstall passed with a prefix containing spaces and preserved
an unrelated file. The installed file matches the release binary and links
only system libraries/frameworks.

The release executable is installed at `/opt/homebrew/bin/guffin` and resolves
on this user's PATH. Short installed smoke tests from `/tmp` pass Gemma 3
ordinary and speculative two-turn chat and Gemma 2 one-shot generation.
Speculative diagnostics confirm F16, horizon 1 and lossless sampling.
No performance benchmark or new speed claim was made.

This phase is complete. Repository-wide dead-code/dependency auditing and
the release research/results documentation remain separate follow-up work.
Do not broaden this CLI/install task into inference optimization.

## Hosted CI residency compatibility fix (2026-10-05)

The hosted `f69364e` run finishes with 158 C++ failures: 147 report
`Metal residency set creation failed: unknown error`, and the remaining CLI
failures follow from that initialization error. Unlike the September hang,
this is an optional Metal feature being requested without checking GPU support.
[Apple's Metal feature table](https://developer.apple.com/metal/capabilities/)
lists residency sets as requiring Apple6 or later;
macOS 15 API availability alone does not establish hardware support.

`MetalBufferResidency` now checks the device family before creating a set.
Unsupported GPUs keep ordinary Metal buffers without starting the residency
worker or tracking allocations. Supported GPUs, including the M4, retain the
500 ms refresh and three-minute idle window, and actual creation failures still
propagate. Two isolated fake-device tests cover unsupported-device lifecycle
and supported-device error propagation without requiring GPU hardware.

Native `bash scripts/ci.sh` passes **503 C++ tests** and the **64-test Python
suite** (15 skips), including the new regressions and existing numerical,
cache, speculative and CLI coverage. No compiler warnings were reported.
This is a compatibility fix; no performance benchmark was run. Changes are
local, and a hosted rerun of this fix is still pending. The fix is complete;
stop without further inference optimization.

## September CI hang resolved locally (2026-10-05)

The `dff16f0` snapshot reproduces a native Metal hang. A process stack shows
`MetalCommandContext::restart()` blocked requesting a command buffer because
discarded buffers retain command-queue slots. Buffered test output explains
why the GitHub log stops earlier, after the CLI diagnostics test.

The existing current-source ownership fix releases the old buffer before
replacement and drains temporary Objective-C ownership. Backporting only that
command-context fix to the isolated September snapshot makes all **423 C++
tests pass**. Current native `bash scripts/ci.sh` also passes **501 C++ and 64
Python tests** (15 Python skips), including the abandoned-command regression.
No production change was needed. CI now has a 20-minute job limit, and existing
test output flushes immediately without extra lines. The subsequent hosted
`f69364e` run reaches completion and exposes the separate residency compatibility
failure described above.

## Gemma 2 recurrent H1 trial and retained target improvements (2026-10-04)

**Completed evaluation: retain the Gemma 2 target improvements; reject the
recurrent H1 candidate as a production speedup.** The complete native Arm R
implementation passed correctness, but lost the cooled chat comparison. Its
source, executable and evidence remain in the ignored task directory. Production
contains the target port only. Gemma 3 is treated as complete; no training,
model downloads or new Gemma 3 optimization were performed in this task.

The frozen worktree, isolated trials and reports are under
`build/gemma2-recurrent-h1-2026-10-04/`. The baseline includes all pre-existing
user changes: never restore these files from Git HEAD. The original executable
SHA is `b4c6041322e5361bf469368d1785bbe8cf1285734bb9f1a8faad332926906300`.
The fully tested/measured candidate SHA is
`14fee23ab7c0df08fad137bdde428ae09001b0a2771f80cc044af20285cfd1f3`.
`candidate-source/` and `candidate-source-sha256.json` preserve the complete
recurrent implementation; `before/` and `before-source-sha256.json` preserve the
starting state. No benchmark output or weights should be committed.

### Retained target mechanism

Gemma 2 now packs its Q4 body/down matrices once and uses compact R1/R2
projections with producer-owned input sums in shared verification. Existing
retrieval uses that R2 path. The original prompt schedule, half-operand prompt
matrix policy, direct ordinary cache writes, Q6-down-to-Q4 policy and Q8 output
head remain. Q6 V matrices stay canonical to preserve their original reduction
arithmetic; no duplicate body/head copy is retained. Gemma 3's precise kernels
are unchanged. The generic target workspace getter and retrieval entry point
now name their shared Gemma 2/3 role.

Retained code is the isolated 11-file `target-port-candidate.patch` plus three
files containing getter renames. Source review confirms it matches the tested
isolated target implementation and the measured candidate's numerical kernels.
The recurrent plan, feature-handoff API, artifact loader, CLI options, benchmark
options and recurrent tests were removed from production. Successful chat runs
were not repeated after removing that inactive integration.

Correctness rejected two earlier target variants. Switching the entire prompt
graph changed full-checkpoint logits by up to 0.01063; packing Q6 V matrices
also changed continuation arithmetic. Both were removed. The final retained
implementation matches the frozen target byte for byte for a 16-token prefill,
two continuation logits and all 42 layers' F16 K/V. Joint R2 verification matches
ordinary logits and K/V exactly. The 29 focused native cases include actual
matrix widths and 1/2/4/8/9/16/17/33 rows, original Q6 arithmetic, independent
CPU/F16 bounds, short contexts, and retrieval/graph cache parity.

Diagnostic R2 body medians fell from 66.721 to 54.070 ms (18.96% less GPU time);
the two-row Q8 head still costs about 17.859 ms. These diagnostics explain a
mechanism, not the entire chat gain. Ordinary decoding keeps 633 dispatches.
Resident packed weights add 127,647,744 bytes, and measured chat scratch grows
from 108,716,572 to 126,353,224 bytes. See `target-comparison.json`,
`target-port-summary.json` and `quant-focused-native.log`.

### Cooled actual-chat evidence

One fresh process per configuration, each after 240 seconds idle, with no
competing build or GPU diagnostic. The canonical two-turn sampled conversation
uses seed 42, natural stopping, an 8,192 context allocation, F16 KV and the
existing retrieval gate. All rates below use `generation_completion_ns`, which
includes final cache publication, callbacks and result assembly. Standard here
means the existing default Gemma 2 retrieval mode, not `--no-speculative`.

| Configuration | First turn tokens / seconds / tokens/s | Follow-up tokens / seconds / tokens/s | Aggregate tokens/s |
| --- | --- | --- | --- |
| Frozen standard | 276 / 16.560730250 / 16.6659 | 322 / 19.657151916 / 16.3808 | 16.5112 |
| Updated standard | 276 / 16.216601500 / 17.0196 | 322 / 19.274319958 / 16.7062 | 16.8494 |
| Updated target + Arm R H1 | 258 / 17.149000541 / 15.0446 | 274 / 17.977660417 / 15.2411 | 15.1452 |

The target port improves aggregate generation by **2.0483%**, winning both
turns and narrowly passing the 2% retention gate. The standard token hashes
remain `49080569719b1cea` / `369b09d9b01ff1e1`, with identical token counts,
retrieval decisions and prompt work. Prompt times improve from
165.231/270.842 to 152.693/270.503 ms; first visible text improves from
225.783/339.899 to 214.615/339.395 ms. No headline latency regresses.
This is one cooled comparison, not proof of the same gain on every chat.

Arm R H1 is **8.2731% slower than the frozen standard** and **10.1142% slower
than the updated standard**. Its hashes are `1939edd3e0e92a51` /
`066af2b16076914d`; these are different natural-stop responses and must not be
called an exact-work comparison. Prompt times are 152.413/265.253 ms and first
visible text 212.962/336.933 ms; its shorter first response also changes the
follow-up history. Latency was not the failure here.

Native physical-footprint high-water was 10,872,319,400 bytes for frozen
standard, 11,047,546,400 for updated standard (+1.61%), and 11,040,615,992 for
Arm R H1. System-wide swap-out deltas were 0, 3,616 and 45,212 16-KiB pages,
respectively. The recurrent run occurred first and substantially changed system
swap usage; these counters cannot attribute swaps to the engine, nor establish
identical memory pressure across runs. No thermal/performance warning was
recorded by `pmset`; this does not measure GPU temperature or clocks. Preserve
these limits rather than claiming a universal architecture slowdown.

Raw reports, binary manifests, per-process memory samples and the computed
comparison are in `results/{baseline-stage4,candidate-stage4,candidate-rh1}.*`
and `results/comparison.json`.

### What the recurrent trial established

This was a complete implementation, not a timing substitute. It authenticates
the existing `phase6-h3-v2/seed{1,2,3}.lgweights` exports and executes the first
edge of the trained width-512 root-only gated recurrence. It preserves F16
feature capture, BF16 boundaries and learned normalization. The runtime borrows
the target embedding/norm/head, runs full-vocabulary proposals, and supplies the
actual emitted integer probability masses to lossless correction. The Q8 draft
head is an explicit adaptation from the original F16 head; an F16 comparison
option exists in the archived candidate. Q8 changes q, not the corrected target
sampling law. Only seed1/Q8 was measured in live chats; this result does not
establish that every checkpoint or proposal policy fails.

Each cycle starts from the authentic target predecessor. Checked feature
ownership/epoch/position allows a rejected correction to enter the next joint
H1 cycle without an extra ordinary target pass. No recurrent state or draft KV
is carried between cycles, no unused later-edge/confidence work runs, and there
are no per-cycle transient allocations. The first token uses ordinary target
execution to preserve first-text latency. Existing retrieval, stopping,
callbacks, cache rollback, regeneration and context lifecycle remain intact.

The candidate passed native CI with **513 C++ cases and 66 Python cases**
(15 existing Python skips), both independent Gemma 3 F32/F16 golden sets, and
the public real-checkpoint ordinary/recurrent greedy chat lifecycle comparison.
The separate unavailable-Metal run passed all 513 C++ cases. Independent
first-edge CPU/Metal states were exact; all 256,000 F16/Q8 head logits passed
independent F64 equations (maximum absolute error 2.28882e-05). Actual q support,
CDF, exact integer mass normalization and draw inversion passed. Lifecycle
coverage includes rejection, owner/epoch checks, callback failure/recovery,
regeneration, clearing, sampler changes, resetting and context admission.
See `final-native-ci.log` and `final-unavailable-tests.log`.

Live H1 accepted 50/203 and 54/208 proposals: **104/411 = 25.3041%**. There
were no numerical refusals or censored cycles. Mean drafting took 11.669 ms,
target verification 70.172 ms GPU, and the complete learned cycle 83.561 ms.
Against the updated standard's average token time, a simple H1 cost estimate
needs about **41% acceptance to break even**, or about 44% for a 2% gain. At
observed acceptance, cycle time would need to fall to roughly 74.37 ms merely
to break even. This estimate holds timing fixed and is not a new benchmark.
It rules out expecting small host/dispatch tweaks to rescue this checkpoint.

The old seed1 offline first-edge common-draw result was 36.8347% over 83,340
validation origins with an F16 head; seed2/3 were 37.0854%/36.7183%. Those differ
from live on-policy H1 origins and Q8 proposals. The archived live H3 figure
74/178 = 0.41573 counted all accepted edges per cycle, not first-edge acceptance.
Old rejection carries also took ordinary decode and never entered its learned
cycle population. No feature chronology, embedding scaling, normalization or
sampler-order mismatch was found. On identical p/q, lossless expected acceptance
is at least common-draw matching probability; independent draws alone do not
explain the lower live result.

There is no justified live confidence threshold in the inspected artifacts.
Precomposing the large vocabulary head with the 512-wide expansion could remove
substantial draft traffic, but intermediate BF16 rounding prevents an exact
algebraic substitution. That would be a new proposal law requiring its own
numerical, acceptance and chat evidence, not a missing faithful implementation
of this trial. No such follow-up or training is started here.

The archived candidate remains runnable for manual inspection:

```sh
./build/gemma2-recurrent-h1-2026-10-04/candidate-generate \
  models/gemma-2-9b-Q4.gguf --chat --context-size 8192 \
  --recurrent-draft-weights artifacts/learned_speculation/phase6-h3-v2/seed1.lgweights
```

This optional archived executable is not the retained production command.
The retained improvements apply automatically to the existing Gemma 2 command.
Final native CI for the retained source passed **501 C++ cases and 64 Python
cases** (15 existing Python skips), including both independent Gemma 3 F32/F16
checkpoint goldens. The unavailable-Metal run also passed all 501 C++ cases.
Logs are `retained-native-ci.log` and `retained-unavailable-tests.log`.
The rebuilt retained CLI SHA is
`a6507c3ac675add1098b5876c6b137abc005f9bcc42d2d84c46e94acf971322a`;
its source manifest is `retained-changed-source-sha256.json`.
The only post-measurement implementation change removes the inactive recurrent
integration; numerical kernels match the measured candidate. This baton is
complete. No successful chat benchmark needs to be rerun, and no additional
drafter, head rewrite or training task is started here.

## Ordinary Gemma 3 implementation trials (2026-09-29)

**Completed (2026-10-04). Retain M32 for larger prompts and the
original M8 path for short ones; unsuccessful candidates are excluded from production.**
The user authorized all four audited mechanisms, correctness checks and removal
of failures. All four were implemented in isolated source copies. Production
currently contains only the larger prompt matrix tiles and their tests; no
attention, FFN fusion or lookahead candidate code was merged. No speculative
optimization, training, model download or unrelated cleanup is included.

The complete initial worktree, candidate source/diffs, raw diagnostics and
binary identities are preserved under
`build/gemma3-ordinary-implementation-2026-09-29/`. The frozen baseline CLI SHA is
`29914d243941dd7513917e97d87aa005d618c1f5506b861d2ddb3edb526c6fc7`.
The measured M8/M16/M32 trial CLI SHA is
`0d828c5578c8ed4f851ab1ead4b7029081dc24c31a233bd6e24f861c62c710d3`.
The final release CLI after M16 removal is
`b4c6041322e5361bf469368d1785bbe8cf1285734bb9f1a8faad332926906300`.
These snapshots include pre-existing user changes; reverting to Git HEAD would
incorrectly remove unrelated work.

### Candidate decisions and fidelity

| Candidate | Complete mechanism and correctness evidence | Decision |
| --- | --- | --- |
| Larger prompt projection tiles | Precise F32 variants, same N32/K64 and per-row K8 reduction order; original packed weights; 22 native tensor cases and all seven actual model matrix geometries bit-identical to original M8 | Retain M32 for >=32 useful rows; remove M16 after short-chat regression screen; keep original M8 for 4–31 rows |
| Multiquery prompt attention | BQ32/BK16 split-D F32 matrix QK and PV, F16 KV widened immediately, causal/local/ring/private masks, checked indexing, 20 KiB shared scratch; compensated block accumulation passes unchanged F64 bounds at 8K and existing short full-model goldens | Rejected: new independent natural 16-token check fails original bounds where baseline passes; full-body performance offers no clear retained gain |
| Cooperative ordinary R1 gate/up/GELU | Two four-SIMD banks in a 256-thread group, unchanged Q4 dense arithmetic, precise parallel GELU and ordered receipts, separate persistent output receipts with correct admission accounting | Rejected: exact outputs and expected dispatch savings, but no meaningful full-target time saving |
| Bounded prompt lookahead | Two persistent command/metadata slots; CPU encodes successor while predecessor executes, then waits/checks F16 status before successor submission; shared activation/KV owners remain ordered | Rejected: correct complete implementation, but measured CPU preparation budget is far below the retention threshold |

The matrix implementation keeps the existing R1/R2/verifier dispatches. M32
reuses each decoded weight tile over more real prompt rows; no activation
rounding, alternate weight format or extra resident matrix copy is introduced.
The all-336-body-matrix diagnostic at 128 useful rows averaged M8 2385.759 ms,
M16 2052.292 ms and M32 1995.031 ms: 13.98% and 16.38% less GPU time. These are
interleaved mechanism measurements, **not actual chat speedups**. Native tests
cover row/column tails, nonzero buffer views, guards and values that discriminate
F32 from F16 operands. A new real 32-row causal-prefix test projects hidden row 3
and checks the first four all-layer KV rows against the existing authenticated
full-vocabulary/F32/F16 goldens, exercising the actual M32 transformer route.
This independent checkpoint covers that causal prefix, not all 32 model rows.
Final independent source review found no blocking indexing, synchronization,
dispatch or ownership issue in the six retained implementation/test files.

FFN fusion passed 35 primitive/runtime cases including both full-model golden
sets. Real-weight segment outputs/receipts and complete target logits/private KV
were byte-exact. It removed 96 dispatches and 48 barriers per full target token.
Median gate/up/GELU stream time was 34.400 -> 34.569 ms; body plus Q6 head was
80.245 -> 80.467 ms. Paired samples also showed no saving near the required
1.6–2.1 ms. These short diagnostics do not prove a universal slowdown, but do
not justify a cooled chat or geometry sweep. Its additional kernel, API and
7,680-byte receipt allocation are absent from production.

Lookahead passed F32/F16 multi-chunk, replay, continuation, exact cache/logit,
selector-owner, allocation, invalid-input, F16-overflow and in-flight exception
drain checks. Independent review verified disjoint host staging and GPU ordering.
The fresh original-path 2,048-token prompt took 42,039.9 ms wall/42,008.5 ms GPU;
even removing **every** non-GPU interval saves only 0.075%. Warm 128-row chunks
at prefixes 0/2,048 had 0.053%/0.065% non-GPU shares. This rules out a worthwhile
lookahead gain at the current scale; no cooled candidate chat was run. Its
extra metadata/command owners and admission changes are absent from production.

### Attention rejection and newly exposed baseline uncertainty

The attention implementation was complete. An 8K synthetic error was traced to
long PV numerator accumulation, not masks or QK scores. F32 compensated sums of
16-key matrix block results plus the denominator fixed it without relaxing the
independent 2e-7 oracle bound. All 23 tensor and 11 target cases, including the
original F32/F16 goldens and real 32-row causal-prefix check, then passed.
The corrected full-body diagnostic was flat for fresh 4/16/128-row prompts;
128 rows after a 1,024-token real prefix showed an apparent 1.93% gain within
measured drift. No cooled attention chat was run or retained.

A fresh independent **128-token natural-prompt CPU oracle** was generated with
the existing streamed GGUF/PyTorch CPU equations and exact native token IDs.
It stores all 48 layers' logical F16 KV and logits at positions 3/15/127.
Both GPU paths were checked separately against original independent bounds;
we did **not** double a mutual-comparison tolerance to make a failure pass.

| Natural prompt prefix | Original baseline | Attention candidate |
| --- | --- | --- |
| 4 tokens | Pass | Pass |
| 16 tokens | Pass; worst logit/KV component budget ratios 0.885/0.949 | Fail; 9 logits and 1 KV component, ratios 1.207/1.072 |
| 128 tokens | Fail; logit/component/head-RMS ratios 1.654/8.956/2.892 | Fail; ratios 2.060/4.282/1.135 |

Top-1 agreed with the CPU reference in every compared case, but that does not
waive the failures. The 16-token result is a candidate-specific failure under
the agreed bounds. The 128-token result exposes an **existing baseline
numerical-validation gap**, not an effect introduced by the bit-identical matrix
tiles. Its cause remains unresolved: do not assert an indexing defect, harmless
roundoff or answer-quality degradation without diagnosis. A focused follow-up
should distinguish reference/batch arithmetic and F16 rounding sensitivity from
an engine defect before changing tolerances or pursuing more attention rewrites.
This task does not silently expand into that separate investigation.

Evidence: `attention-independent-comparison.json`, `attention-parity/`,
`natural-prompt-oracle/identity.json`, `natural-prompt-reference.py` and the
original reference script SHA recorded in that identity. Attention and combined
candidate trees are archived as rejected, not production alternatives.

### Verification and actual-chat results

The measured matrix trial passed native and unavailable-device CI,
including the M32 causal-prefix check: 494 C++ cases and 64 Python cases in each
run, both authenticated F32/F16 target golden sets on native Metal, and retained
Gemma 2 and speculation lifecycle coverage. No independent tolerance was relaxed.
Those trial logs are `final-ci-native.log` and `final-ci-unavailable.log`.
After removing M16, final native and unavailable-device CI also passed all
494 C++ and 64 Python cases, including both authenticated native golden sets;
see `retained-ci-native.log` and `retained-ci-unavailable.log`.

The frozen long chat uses 2,018 newly evaluated first-turn tokens and 1,349
follow-up tokens, a 16,384 allocation, natural stopping and the canonical sampled
chat driver. Baseline prompt times were 36.176861/28.886381 seconds, generation
counts 922/852, generation durations 75.423946/83.598588 seconds, selected hashes
`17bd8f1b4dd6d6d8`/`1e96156f5ec5d879`, and native physical-footprint high-water
9,265,605,672 bytes. Candidate comparisons must preserve these workloads and
hashes, which the measured candidate preserved. Each fresh process received at
least 240 seconds idle, with no competing
builds or GPU work. Long-prompt TTFT/throughput establishes the intended benefit;
ordinary generation and canonical short-chat headlines enforced regression guards.
The long-chat candidate preserved both workloads and selected-token hashes.
Prompt times fell to 31.659791/23.924771 seconds: throughput improved
14.27%/20.74%, or 17.05% overall, and TTFT fell 12.49%/17.18%. Generation times
were 73.947376/81.112804 seconds, with no regression. Native physical-footprint
high-water was 9,266,539,560 bytes, effectively unchanged. Neither run added
system swap-outs; these counters do not attribute swap activity to the engine.
The canonical generation-completion clock gives 2.00%/3.03% gains, 2.54%
overall. Scalar decode kernels did not change, so this is an observed chat
result, not evidence of a new scalar decode mechanism.

Work resumed on 2026-10-04 with production source and executable hashes verified
unchanged. The earlier canonical short baseline completed. The user explicitly
asked not to repeat successful runs, so that result is reused and only the
missing matrix candidate short chat was completed. The gap between
measurement dates limits interpretation of small timing differences.
The short candidate preserved both workloads and hashes: 16/19 new prompt tokens,
1,493/967 generated tokens, hashes `8b3b0b12c0cc1348`/`86b87b255be9444c`.
M16 TTFT was 301.495/479.075 ms versus baseline 290.263/460.857 ms:
3.87%/3.95% worse, outside the 2% guard. Generation completion throughput was
0.63%/11.18% higher, but the cross-date comparison and unchanged scalar kernels
do not justify attributing that to the matrix change. Native footprint high-water
was 9,256,217,664 versus 9,254,612,056 bytes; neither run added system swap-outs.

M16 was therefore removed entirely from production, including its four pipeline
specializations. Final dispatch is M8 for 4–31 prompt rows and M32 for >=32;
ordinary/verifier dispatch is unchanged. The successful long comparison uses
128-row chunks and 98/69-row tails, all on the retained M32 computation. The
short route returns to the original M8 computation. Per the user's instruction,
successful benchmark runs will not be repeated: the final binary after M16
removal is checked by CI, not described as a newly benchmarked executable.
Final source/test changes are limited to the Metal matrix shader, persistent
pipeline table/construction, prompt dispatch, tensor parity coverage and the
target causal-prefix golden check. `docs/GRAPH.md` was updated only as a locator.
`retained-matrix-source.diff`, `remove-m16.diff`,
`retained-source-identity.json`, `completed-chat-summary.json`, and the raw
`results/` records preserve source, binary, workload and measurement identities.
Pre-existing user changes were preserved; no rejected implementation or temporary
instrumentation remains in production.

**Stop condition reached.** All four authorized mechanisms were implemented and
screened, unsuccessful variants excluded, retained source checked, and evidence
recorded. No further tuning, benchmark repetition or speculative work is active.
The separate 128-token baseline numerical investigation above remains a known
gap and a possible next task, not a silently started extension of this one.

## Ordinary Gemma 3 pipeline audit and Gemma 2 comparison (2026-09-29)

**Completed prior audit, superseded by the implementation above.** The user
asked to set speculation aside, scrutinize the ordinary
Gemma 3 pipeline and concrete upstream implementations, account for rejected
experiments, then compare Gemma 2. This audit changes documentation only.
No new inference benchmark, production change, model download or training ran.
The release executable remains the retained long-chat baseline,
`29914d243941dd7513917e97d87aa005d618c1f5506b861d2ddb3edb526c6fc7`.

**Decision:** the strongest distinct opportunities concern prompt throughput:
more reuse in FP32 projection tiles and true multiquery prompt attention.
For steady ordinary generation, a carefully structured gate/up/GELU fusion is
an unmeasured, bounded candidate. There is no newly demonstrated large decode
speedup or justification for another generic Q4/attention tuning sweep.

### Scope, current mechanism and evidence quality

The actual GGUF metadata/tensor directories were reread. Gemma 3 has 48 layers,
hidden/FFN widths 3,840/15,360, 16 query and eight KV heads of dimension 256,
40 local layers with a 1,024-token window and eight global layers. The local
file contains 288 Q4_K, 49 Q6_K and 289 F32 tensors; V and down projections are
mixed Q4/Q6. The tied Q6 embedding/output allocation is 825,753,600 bytes.
The file's advertised 131,072 context is distinct from a session's allocation.

| Pipeline part | Current ordinary implementation | Audit conclusion |
| --- | --- | --- |
| Loading/residency | Once-only columns8 body packing; one tied canonical Q6 head; persistent Metal owners | No full-model upload per token or duplicate resident output head |
| Prompt projections | 128-token chunks; genuine F32 M8/N32/K64 SIMD-group matrix multiplication for >=4 rows | Matrix prefill exists; reuse across only eight rows is the gap |
| Prompt attention | Per-query online attention; two GQA heads share each K/V load | Missing reuse across many known prompt rows |
| Scalar projections | Dedicated packed R1 Q4/Q6 kernels; producer-fused K/8 activation sums | Fused sums already benefit ordinary decoding |
| Scalar attention | F16 KV widened to F32; online softmax; local rings and global scans; GQA reuse | No missing GQA optimization; tested blocked replacement did not help R1 |
| Norms/RoPE/activation | Native Q/K norms and RoPE, fused norm-add, precise GELU; gate/up still separate | Most small passes are not a large measured cost; one fusion candidate remains |
| Head/selection | Final prompt row only; full native Q6 head; GPU compact candidates and tiny CPU readback | No vocabulary projection for every prompt token or full-logit CPU transfer |
| Submission/streaming | One concurrent encoder/commit/wait per scalar token; incremental detokenization | CPU-only rewrites have a small measured ceiling |
| KV/publication | Persistent F16 cache; private prompt rows then ordered publication; checked half stores | Context semantics and poisoning must survive any scheduling change |

Relevant call path: `gemma2_runtime.cpp:251` dispatches ordinary Gemma 3 to
`gemma_target_runtime.cpp:377`; its shared body begins at `:14`. Q4 receipts are
enabled at `:42`; projection dispatch is `metal_command_context.mm:406` and the
matrix shader is `metal_backend.mm:1148`. Prompt execution is
`gemma_target_runtime.cpp:407`; it runs the head only on the last prompt row.
Terminal decode already elides its unused head (`gemma2_runtime.cpp:1956`).

The newest retained ordinary F16 receipt is
`benchmarks/results/gemma3-f16-kv-2026-09-29/native16-candidate-ordinary.json`.
It predates idle-residency maintenance and is **not a fresh current-binary
profile**. Its first/follow-up GPU time was 79.246/106.370 ms per decoded token;
full generation-completion time was 80.207/107.521 ms. Eliminating every
non-GPU interval would improve throughput only about 1.21%/1.08%. CPU selection
was about 0.0707/0.0700 ms/token. It recorded one submission/wait, zero transient
**device** allocations, about 918/926 dispatches and 629/637 barriers per token.
GPU time includes scheduling, stalls and dependencies; it does not establish
memory-bandwidth saturation or make every dispatch removable.

The resident projection footprint is 7,433,994,240 bytes per full weight stream:
5,292,195,840 Q4 body, 1,316,044,800 Q6 body and 825,753,600 Q6 head. At Apple's
advertised [120 GB/s M4 Air bandwidth](https://support.apple.com/en-au/122209),
one ideal read of that footprint takes 61.95 ms, before attention and other
costs. This is a deliberately optimistic streaming model, not a measured DRAM
counter, promised 16.1 tok/s result or proof that the kernels are optimal. It
argues against expecting an effortless doubling with the same dense weights.

### Strongest prompt opportunities

**A. Reuse each decoded weight tile across more real prompt rows.**
The current M8/N32/K64 kernel stages 8 KiB of F32 weights plus 2 KiB of F32
activations; each of four SIMD groups accumulates one 8x8 output. A full
128-row chunk executes sixteen body-equivalent dequantization passes. The
packed body footprint is 6,608,240,640 bytes, so footprint times passes is
105.73 GB/chunk. This is logical reuse accounting, not literal shader-load or
DRAM traffic; caches and repeated metadata loads matter.

M16 or M32 can retain N32/K64 and the existing per-row eight-wide MMA order,
while accumulating two/four independent 8x8 outputs per SIMD group. Scratch
becomes 12/16 KiB and accumulator state rises. This reduces repeated decoding
without rounding F32 operands to half or keeping another dequantized weight
copy. Actual register allocation, occupancy and numerical equality need checks.
A cache-local launch walk is a secondary option: current grid.x is the N tile
and grid.y the M tile. Changing independent tile order need not change arithmetic,
but physical scheduler/cache behavior cannot be inferred from grid dimensions.

The old padded matrix decode failures do not reject this mechanism: R1/R2 paid
for mostly empty M8 tiles; prompt chunks contain real rows and already use MMA.
Larger prompt tiles were left unmeasured in the earlier pipeline trial. Do not
claim a steady-generation gain from a prompt-only dispatch.

**B. Give prompt attention its own multiquery tiled operation.**
The current attention grid is KV heads x partitions x individual query rows
(`metal_command_context.mm:2021`, `metal_backend.mm:2948`). K/V is shared between
two GQA heads, but read again independently for each of up to 128 prompt rows.
A tiled QK, block-softmax and PV computation can reuse KV across many queries,
without allocating the full attention-score matrix or changing the transformer.

The concrete reference is MLX's current [F32 D256 instantiation](https://github.com/ml-explore/mlx/blob/7916d8b0c8877b8a10f231363aa49f2f83acb34a/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal)
and [attention kernel](https://github.com/ml-explore/mlx/blob/7916d8b0c8877b8a10f231363aa49f2f83acb34a/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h).
Its BQ32/BK16/D256, WM4/WN2 configuration keeps F32 queries in registers,
splits the head dimension across two SIMD groups, combines partial scores and
reuses K/V scratch. This addresses D256 resource pressure: a naive padded F32
query tile alone exceeds 32 KiB. It is a concrete dataflow reference, not proof
that its timings or precision defaults transfer.

Adapt it to mixed F32 queries/F16 stored KV, immediate widening, precise
exponentials and the engine's resident-ring/private-chunk addressing. Preserve
causal and per-row sliding masks, empty tiles, odd query tails, extreme scores,
F16 nonfinite handling, regeneration and exact replay of the same prompt path.
The original independent logit/KV bounds remain the acceptance criteria; no
blanket fast-math change or whole-cache conversion is proposed.

[llama.cpp's current Metal dispatcher](https://github.com/ggml-org/llama.cpp/blob/6a2743f028f78bfb88a7189607b49bde30df3769/ggml/src/ggml-metal/ggml-metal-ops.cpp#L2949)
also separates vector attention below 20 aligned queries from matrix attention.
Its [F16 attention implementation](https://github.com/ggml-org/llama.cpp/blob/6a2743f028f78bfb88a7189607b49bde30df3769/ggml/src/ggml-metal/kernels/fa_f16.metal)
includes half-query staging, so it is not a direct precision-preserving port.

The rejected one/two-row attention trials do not settle a 128-query prompt
kernel: the available reuse, matrix arithmetic and intended metric differ.
Nevertheless, there is no current F16 prompt-stage attribution establishing
whether A or B saves more whole-prompt time. Measure that once before choosing
which implementation carries the main effort; do not promise a percentage.

### Remaining ordinary-generation candidate: cooperative gate/up fusion

The current per-layer gate, up and GELU/product occupy three dispatches
(`gemma_target_runtime.cpp:147`, `metal_backend.mm:673`, `:1875`). Both matrices
are Q4_K with matching shapes. A faithful fused segment can retain the existing
columns8 buffers, native quantization and exact dense reduction order:

- One 256-thread group contains two four-SIMD banks. One bank computes eight
  gate outputs; the other computes the matching eight up outputs using the
  existing two-column workers and four K partitions.
- Each bank has separate partial storage; all eight SIMD groups reach every
  threadgroup barrier uniformly. The barrier-containing helper must not sit
  inside divergent gate/up branches.
- After unchanged reductions, exchange only sixteen reduced scalars. Apply the
  existing precise GELU/product and, where the down projection is Q4, emit the
  same sequential K/8 activation-sum receipt.

This avoids a new weight format and need not double dense per-thread live
state. Shared partial storage grows from 1 KiB to 2 KiB plus 64 bytes for the
reduced values. Eight-SIMD occupancy, extra local synchronization and epilogue
scheduling can still erase the gain; compiler register behavior is unmeasured.
The existing helper's output epilogue would need to expose its reduced values.

The structural saving is **96 dispatches and 48 global barriers per token**.
Weight bytes and total dense worker count remain approximately unchanged;
eliminating gate/up materialization removes only 11,796,480 bytes/token.
That is about 0.098 ms at ideal peak bandwidth. A >=2% throughput improvement
at the historical 80–107 ms/token budget requires roughly **1.6–2.1 ms saved**,
or 33–44 microseconds per layer. Therefore the hypothesis concerns GPU
scheduling/issue/cache efficiency, not a large weight-bandwidth reduction.

[llama.cpp CUDA's fused gate/up GLU](https://github.com/ggml-org/llama.cpp/blob/6a2743f028f78bfb88a7189607b49bde30df3769/ggml/src/ggml-cuda/mmvq.cu#L708)
is an engineering precedent for doing both reductions and the activation in one
operation; its Q8 activation packing and integer-dot hardware are not this
engine's arithmetic. The two-bank Metal adaptation above is a design proposal,
not a novel-algorithm claim or an implemented/measured improvement. Current
llama.cpp Metal does not provide an equivalent fused matmul/GEGLU path.

A worthwhile test must include the **whole gate+up+GELU segment using all actual
layer weights**, followed by a complete ordinary target pass and cooled real
ordinary chats if it earns continuation. A toy activation kernel or dispatch
count alone is insufficient. Stop if the projection penalty consumes the
possible savings; do not start an open-ended geometry search.

### What previous experiments actually tell us

| Previous mechanism | Evidence and applicable lesson |
| --- | --- |
| Native32 Q4 reduction | Actual all-Q4 R1 stream 52.084 -> 60.848 ms (+16.82%); this is direct ordinary negative evidence, not only an H1 result |
| Packed tied Q6 head | Complete R1 head 8.0104 -> 8.1778 ms with exact parity and unchanged bytes; repacking the head is already tested |
| Four output columns/SIMD | Earlier exact shape/checkpoint screen found no established scalar gain; current MLX four-column qmv is not a newly overlooked shortcut |
| Wider/single-column Q4 variants | Mostly R2 probes; informative warnings, not universal R1 disproof |
| Q/K norm+RoPE fusion | Faithfully implemented with 143 fewer dispatches/48 fewer barriers; combined ordinary candidate -6.26%; not an untried idea, but fusion alone was not isolated |
| Scratch/publication rewrite | Separate resource-only trial saved 178.49 MiB but ordinary aggregate throughput fell 2.95%; allocation reduction did not establish speed |
| Broad F32 blocked attention | Ordinary follow-up -4.30%; no reason to restore it for ordinary generation |
| Recent long F16 attention | R1 replay +0.20%/+0.68% time at 8K/near16K; integrated paired chat also failed; no new R1 attention win demonstrated |
| Gate/up/GELU fusion | Previously rejected at budgeting, not implemented/tested; the cooperative design above addresses a specific live-state/receipt-ownership concern |

The older stage profile
`benchmarks/results/gemma3-pipeline-optimization-2026-09-27/stage-profile-summary.json`
was F32, before compact packing and with perturbed encoder boundaries. It found
FFN gate/up ~32 ms and down ~20 ms, versus Q/K norm+RoPE ~0.49 ms, selection
~0.32 ms and private-KV publication ~0.017 ms. It identifies the order of costs,
not current F16 stage shares. Do not subtract sums of category medians from a
median total to manufacture an unmeasured GPU scheduling budget.

Current [MLX qmv](https://github.com/ml-explore/mlx/blob/7916d8b0c8877b8a10f231363aa49f2f83acb34a/mlx/backend/metal/kernels/quantized.h#L824)
uses affine quantization and four output columns per SIMD. Its [wide dispatcher](https://github.com/ml-explore/mlx/blob/7916d8b0c8877b8a10f231363aa49f2f83acb34a/mlx/backend/metal/quantized.cpp#L1825)
explicitly requires M>=2; those gains do not show a missing R1 path. The current
[llama.cpp Metal Q4/Q6 matvec](https://github.com/ggml-org/llama.cpp/blob/6a2743f028f78bfb88a7189607b49bde30df3769/ggml/src/ggml-metal/kernels/mul_mv.metal#L1511)
uses closely related native-group factoring. Our accepted core already has
compact body packing and producer-fused sums. No superior unadopted ordinary
algorithm was established by comparing names or headline upstream results.

### Gemma 2: useful transfer and misleading comparisons

The actual Gemma 2 file has 42 layers, widths 3,584/14,336, alternating global/
local attention, a 4,096-token local window and a tied F16 embedding/head.
Both production paths already use F16 KV (`model.cpp:581`), resident weights,
compact selection, one full-token command and terminal head elision.

Gemma 2's retained 9B policy **requantizes Q6 down weights to Q4** and creates
a **Q8 output head from its F16 table**, retaining the separate F16 embedding
(`gemma2_metal_plan.cpp:352`, `:408`, `:445`, `:756`). Gemma 3 deliberately keeps
its native Q6 down/head; tests enforce this (`gemma3_model_plan_tests.cpp:65`,
`:398`). Gemma 2 also uses the old compiler defaults, while Gemma 3 requests
safe arithmetic and precise functions (`metal_context.mm:73`). Copying these
policies changes the numerical model rather than just improving execution.
Source-derived scheduled weight work is about 5.70 GB/token for retained Gemma 2
versus 7.44 GB for Gemma 3, independently of their different attention costs.
Raw cross-model tokens/s therefore cannot establish a missing Gemma 3 shortcut.

A genuine transferable scheduling idea is **bounded prompt lookahead**. Gemma 2
queues chunks and waits only at the end without feature capture
(`gemma2_runtime.cpp:987`, `:1353`, `:1376`); Gemma 3 commits and waits after every
128-row chunk (`gemma_target_runtime.cpp:348`, `:415`). Two in-flight staging
slots could overlap host encoding with GPU work while keeping shared scratch
ordered on one queue. Simply removing waits is incorrect: Gemma 3 rewrites
tokens, positions, ancestors, selected paths, schedules and F16 status at offset
zero. Preserve per-command status, poisoning, capture, cancellation and cache
completion ownership. Measure recoverable host/GPU gaps first; this affects
multi-chunk prefill, not steady generation or single-chunk prompts.

### Secondary work and explicit exclusions

- Trace-only `sampler_pre`, request and ticket construction currently occurs
  unconditionally (`gemma2_runtime.cpp:1668`), and ordinary Gemma 3 builds a
  Gemma 2 layer schedule then separately stages its own (`:1956`, `:251`). These
  are real removable host costs; the ~1.1% total host ceiling prevents using
  them as a promised >=2% decode project. Zero device allocations does not mean
  zero host allocations.
- Whole-history rendering/tokenization occurs before cache reconciliation
  (`gemma2_chat_session.cpp:440`). Incremental prompt handling might help long
  input latency, but must preserve BPE/template boundaries and generated token
  IDs. Per-token output detokenization is already incremental.
- Direct packing into final shared storage could reduce startup staging; bounded
  global-KV growth or unused-scratch removal could reduce allocation/pressure.
  None establishes less steady attention work. Untouched allocation and physical
  footprint are distinct; do not repeat the rejected ownership rewrite on nominal
  byte counts alone.
- Metal 4 is not a new neural accelerator for M4. Apple's [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
  offers useful tile-locality/cooperative-epilogue ideas but discusses M5 neural
  acceleration. A [June 2026 M4 Max characterization](https://arxiv.org/abs/2606.12765)
  reports shader execution and emulated FP8; it is not this Air's benchmark.
  Replacing APIs, adding explicit software prefetch, or using indirect commands
  is not a throughput result.
- Head shortlists, approximate sparsity, dropped global keys, further weight/KV
  quantization, half activations or blanket fast math are separate numerical
  changes. This audit recommends no silent change to the target distribution.

### Proposed next phase and stop condition

1. If prompt throughput is the priority, capture one current ordinary-F16
   long-prompt attribution separating matrix work, attention and interchunk gaps.
   Select the larger-M projection or multiquery attention mechanism with the
   larger recoverable budget. Keep R1 kernels unchanged; do not bundle a broad
   workspace rewrite or speculative changes.
2. If steady generation is the priority, the cooperative gate/up/GELU segment is
   the most concrete remaining unimplemented candidate found here. Check current
   R1 cost and the whole real-weight fused segment against its 1.6–2.1 ms/token
   requirement before full integration. No percentage gain is established now.
3. Any authorized implementation must retain native weights/F32 arithmetic,
   independent F16/F32 full-model goldens, tail/cancellation/reference tests,
   context/rebuild/failure behavior and unavailable-device behavior. Use existing
   tests and disposable diagnostics; run full CI after code changes. Retention
   needs separately cooled real ordinary chats with >=2% aggregate gain, both
   turns winning and <=2% other headline regressions. A prompt-only change needs
   actual long-prompt TTFT/throughput evidence with unchanged decode behavior;
   a 16/19-token prompt pair cannot establish a long-prefill gain. Preserve exact
   token-work comparisons where the numerical path is unchanged and report
   differing workloads honestly when prompt arithmetic changes within bounds.

**Stop here.** Only `docs/NEXT_TASK.md` changes in this audit; no implementation
or new benchmark was performed and doc-only work needs no CI. The research
receipts, metadata-derived budgets, source hashes and pinned dense upstream
snapshots are ignored under `build/gemma3-ordinary-audit-2026-09-29/`.
Upstream identities: llama.cpp `6a2743f028f78bfb88a7189607b49bde30df3769`
(2026-09-29T18:09:10Z), MLX `7916d8b0c8877b8a10f231363aa49f2f83acb34a`
(2026-09-29T12:34:07Z). Previous trials below are historical evidence, not active
implementation instructions.

## Long-global F16 attention trial: not retained (2026-09-29)

**Current baton: complete. The attention candidate failed the actual-chat
performance gate and has been removed from production. Final CI passed on the
retained tree. Stop and report; no further kernel or context-interval tuning is
authorized by this baton.**
The earlier target-pass improvement did not establish a retained chat gain.

### Implementation and correctness

The candidate faithfully integrated the measured 32-key blocked F16 attention
algorithm. Eligible query pairs shared K/V across both rows and both GQA heads;
odd rows used matching scalar arithmetic. Routing depended on each row's visible
keys (at least 8,192), zero-based cache range and uncapped scores, independently
of allocated context and verifier batch width. Private ancestry, partition
reduction and producer-fused Q4 sums were preserved. The kernel body matched
the measured prototype byte-for-byte after renaming, and dispatch assertions
confirmed the intended path was used. No integration mistake was identified.

The candidate passed native release CI: 495 engine tests and 64 Python tests
(15 optional skips), including both existing independent F32/F16 12B golden
sets. Added numerical tests covered the threshold, 16K/32K/64K capacities,
private siblings, offset views, extreme scores, exact scalar/batch and fused/
unfused parity, and an independent F64 oracle. A subsequent five-row extension
covered multiple pairs and an odd tail and passed CI too. A reference-fixture
error was corrected to use exactly half-representable inputs; tolerances were
not relaxed. Those candidate-specific tests are archived with the rejected code.

### Actual chat result

One baseline and one candidate ran in separate fresh processes, each after
240 seconds of idle cooldown. Both used Gemma 3 12B Q4_K_M, F16 KV, the native
270M drafter, H1 full lossless verification, context 16,384, seed 42, natural
stopping and a five-second inter-turn pause. The frozen long prompt evaluated
9,577 tokens after templating; the follow-up reused 10,806 and evaluated 42.
Both builds included the retrieval-history fix described below.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| First-turn generation | 8.78135 tok/s | 8.63624 tok/s | -1.65% |
| Follow-up generation | 10.23545 tok/s | 10.19295 tok/s | -0.42% |
| Aggregate generation, raw tokens / total generation time | 9.37878 tok/s | 9.27112 tok/s | **-1.15%** |
| First-turn TTFT | 238.902 s | 242.117 s | +1.35% |
| Follow-up TTFT | 1.14576 s | 1.01504 s | -11.41% |

Work was exactly comparable: both produced 1,229 and 999 decoded tokens, with
selected-token hashes `40636cda9acf413f` and `6ba81e69a2e9ae84`. Accepted drafts
were 411/816 cycles and 350/647 cycles in both builds. Generation durations were
139.955635 / 97.602003 seconds for baseline and 142.307292 / 98.008949 seconds for
candidate. This fails the required >=2% aggregate improvement and both-turn win.
The follow-up TTFT improvement does not rescue the failed generation gate.

Target GPU time per cycle was 131.505 -> 132.678 ms on the first turn and
133.256 -> 132.959 ms on the follow-up. The unchanged drafter's GPU time also
rose about 4.0% and 5.8%. Run-state or thermal interaction is possible, but no
GPU clocks, power or temperature traces establish the cause. No macOS thermal
warnings were recorded; that does not rule out throttling. Peak native footprint
was 9,841,733,248 / 9,829,805,720 bytes; system swap did not grow. The isolated
verifier savings were not reproduced as a sustained chat gain in this pair.
This does not prove the mechanism cannot help any workload. Actual 32K/64K chats
were not tested. Further ordinary/short-chat controls were unnecessary once the
primary gate failed and the candidate was removed.

### Retained change, artifacts and stop condition

`src/engine/gemma2_chat_session.cpp` now rebuilds the retrieval proposer only
when retrieval is enabled. Previously its independent 8,192-token history limit
could reject a longer Gemma 3 chat even when retrieval was disabled. This fix
is retained, with a public-session regression in
`tests/frontends/cli/generate_cli_tests.cpp` covering >8K history, follow-up cache
reuse, regeneration and oversize rejection without state mutation. Both real
long chats above also exercised it. It changes no model context limit or
attention algorithm; production keeps the prior F16/H1 fused-sums defaults.

All five attention-related source/test files and `docs/GRAPH.md` were restored
to their start-of-turn bytes, preserving pre-existing work. The complete
candidate source, tests, hashes, diffs, frozen executables and raw benchmark
receipts remain ignored in `build/gemma3-long-global-production-2026-09-29/`.
Use `candidate-source/`, `candidate-source-sha256.json`, `chat-audit.py`,
`compare.py` and `results/candidate-long-h1.comparison.json` for review.
Baseline executable SHA:
`29914d243941dd7513917e97d87aa005d618c1f5506b861d2ddb3edb526c6fc7`.
Candidate executable SHA:
`f22b075ef75bf85974163ac89978e4d53b8c211a02ddb22b6c1ed42bb972cba5`.
Final retained-tree release CI passed **494 engine tests and 64 Python tests
(15 optional skips)**, including both independent F32/F16 12B golden sets and
the new long-chat regression; see `ci-retained.log`. The rebuilt release chat
executable is byte-identical to the measured baseline above. No training, new
models or broader optimization work is part of this completed trial.

## Historical F16 long-context attention measurement (2026-09-29)

**Historical pre-implementation measurement, superseded by the actual-chat
trial above.** These replay results motivated the implemented candidate, which
was subsequently removed after failing the chat retention gate. The source and
binary statements below describe the earlier measurement only. The work used an ignored copy of current sources and the
saved faithful blocked-attention mechanism, adapted to F16 KV. This supersedes
the preceding audit's recommendation to measure the opportunity first.

### Decision and directly measured payoff

There is recoverable time in long-global attention. This is now supported by
complete, unprofiled two-row target passes, including the model body, output
head and candidate selection, rather than just an isolated attention estimate:

| Populated prefix | Current verifier GPU | Candidate verifier GPU | GPU time saved | Current / candidate verifier wall |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 83.069 ms | 82.988 ms | 0.081 ms; unchanged-path control | 84.330 / 83.917 ms |
| 8,192 tokens | 92.676 ms | 88.144 ms | 4.532 ms (4.89% less time) | 93.366 / 88.807 ms |
| 16,352 tokens | 102.303 ms | 93.678 ms | 8.625 ms (8.43% less time) | 103.155 / 94.493 ms |

These are medians of four alternating baseline/candidate pairs at each fixed
prefix. Every long-context pair favored the candidate: GPU savings ranged from
4.344–6.443 ms at 8K and 8.407–9.470 ms near 16K. Median wall times imply 5.13%
and 9.17% greater verifier-pass throughput, respectively; those percentages
are **not measured whole-chat gains**. Drafting, acceptance/rejection, endpoint
work, callbacks, stopping and thermal behavior during sustained chat remain
outside this replay. Other cycle costs dilute a verifier-only saving.

The matching single-row kernel has a cost, which was measured rather than
assumed away. Ordinary full-logit decode GPU medians were 83.578 -> 83.744 ms
at 8K and 88.571 -> 89.171 ms near 16K: approximately 0.20% and 0.68% more time.
Wall time rose 0.35% and 0.75%. These three-pair diagnostics exclude ordinary
token selection and are not a substitute for the ordinary-chat regression
guard. They do not reproduce the old generalized F32 attention regression.

Historical recommendation (completed and rejected above): implement and validate
shared long-global attention for H1,
preserving the current short/local path. Do not restart a general verifier,
Q4, head or sampler search on the strength of this result. Do not claim a
short-chat gain, a constant-cost attention algorithm, or a production speedup
already delivered.

### Where the time went

Separate Metal stage-counter runs measured the following sums across layers:

| Prefix | Current global attention | Candidate global attention | Current / candidate local attention |
| --- | ---: | ---: | ---: |
| 2,048 | 1.653 ms | 1.655 ms | 5.177 / 5.175 ms |
| 8,192 | 10.517 ms | 6.206 ms | 5.101 / 5.384 ms |
| 16,352 | 20.519 ms | 11.930 ms | 5.241 / 5.038 ms |

Global-attention savings of 4.311 and 8.590 ms closely track the unprofiled
whole-verifier savings of 4.532 and 8.625 ms. At 2K, eliminating even the entire
measured global stage would remove only about 1.65 ms before adding draft and
other cycle costs. That does not support a worthwhile general short-chat claim.

Counter instrumentation splits compute encoders and increased complete GPU
duration by roughly 8–11%. Stage numbers identify the cost and mechanism;
unprofiled totals above establish the observed complete-pass saving. The stage
numbers must not be presented as an unperturbed production timeline.

The candidate shares F16 K/V loads across the two query rows and both GQA heads
per KV head, and uses the saved 32-key blocked softmax with F32 arithmetic.
Private ancestors and each row's causal visibility remain explicit; the current
producer-fused Q4 sums remain enabled. The isolated F16 probe also compared
paired and unpaired executions of the same blocked algorithm: at 8K/near 16K,
paired medians were approximately 744/1,517 microseconds per attention layer,
versus 1,258/2,650 microseconds unpaired. The isolated baseline was approximately
1,213/2,475 microseconds. This supports row sharing as the useful mechanism;
running blocked attention independently for each row did not deliver the gain.
These synthetic, cache-hot numbers are not physical DRAM or occupancy counters.

### Correctness and scope of the diagnostic

The real target was `models/gemma-3-12b-it-Q4_K_M.gguf`, with current mixed
Q4_K/Q6_K weights, F16 KV, fused Q4 sums and a 16,384-token plan. Prefixes came
from the prior F16 check's frozen 16,361 technical-text tokens. Production
prefill populated the cache incrementally; each comparison replayed the same
two tokens from identical physical KV state. No artificial zero KV was used
for complete-model timing. No drafter or sampled conversation was run.

At all three prefixes, both variants passed exact full-vocabulary parity between
two sequential ordinary rows and joint verification, exact private K/V parity
in all 48 layers, and exact restoration of overwritten physical cache slots
after provisional publication followed by finalize-zero rollback. Candidate
versus baseline maximum logit differences were 0, 0.000256777 and 0.000288486;
every logit passed the existing 12B mixed tolerance
`0.002 + 0.0003 * abs(reference)`. These are scoped checks, not complete
production lifecycle or independent model-golden coverage for the candidate.

The separate kernel probe passed twelve short, mixed-partition, local-ring and
long-global cases. It checked an independent F64 reference at eight coordinates
of every query head, exact fused-sum receipts and exact candidate R1/R2 parity.
F16 loads widen immediately; queries, scores, softmax and accumulation stay F32.
No attention positions were dropped and no approximation to the sampling law
was introduced by the diagnostic.

Four minutes of GPU-idle cooldown preceded the timed group at each checkpoint,
with CPU-only residency renewal. Comparisons used alternating short replays
within one process, not separately cooled fresh-process chats. System swap use
was unchanged at 1,398.56 MiB before and after; macOS reported no thermal or
performance warnings, which does not prove absence of throttling. Peak process
footprint was about 9.305 GB including diagnostics. Memory pressure and thermals
during actual long chat still require validation.

### Historical proposed implementation boundary and verification

1. Integrate only the demonstrated long-global F16 R1/R2 mechanism into the
   Metal attention dispatch. Keep existing local/short attention and the fused
   Q4 producer contract. Evidence supports the 8K and near-16K points; the exact
   crossover is not established by this measurement.
2. Make routing independent of verifier batch width, preserve ordinary/verifier
   numerical consistency, and cover a pair straddling the routing threshold.
   The ignored diagnostic deliberately accepts only a 16K global cache and
   throws on a mixed threshold; these shortcuts must not become production
   restrictions. Preserve private ancestry, supported contexts, F16 bounds,
   rollback, stopping and failure handling. Run the required existing golden,
   lifecycle, unavailable-device and full CI checks after implementation.
3. Resolve the already identified unconditional retrieval-proposer rebuild in
   `Gemma2ChatSession` before validating actual chats above 8K. Its independent
   8,192-token limit remains unchanged here. The near-16K result exercises the
   production target runtime below that frontend, not a working 16K chat turn.
4. Judge retention with separately cooled fresh-process actual H1 conversations
   at long populated contexts, plus short-context and ordinary controls. Apply
   the repository's >=2% aggregate generation gain, both-turn win and <=2%
   other-headline-regression requirements. Stop if the target-pass gain fails
   to survive that test; this result does not authorize an endless tuning loop.

Ignored reproducibility artifacts live in
`build/gemma3-f16-long-attention-2026-09-29/`: `PROTOCOL.md`, `replay.cpp`,
`replay.jsonl`, `replay.log`, `summary.json`, `attention-probe.mm`,
`attention-probe.jsonl`, preparation/build scripts, `source/`,
`production-source-sha256.json` and `artifact-sha256.json`. The frozen-token SHA
is `948e99193aabe61f59703c0f4c614688509745a0e5b7c9c153c4e3e195390d5e`.
The production source hash check found no changes; the release executable
remains `5d5e4622056895a1c8b8928f1273ad83ed179282e85f8dd45e47185e2b3f81ec`.
Only this durable document changed during the investigation; no production CI
run was needed for that documentation-only change. Diagnostic compilation and
the numerical/replay checks above completed successfully.

## Whole-pipeline audit and proposed next work (2026-09-29)

**Preceding audit, superseded by the F16 measurement above: investigation
complete; that audit request did not authorize implementation.** The user
reports that the idle-residency change worked.
That is user validation, not a new controlled timing result. Production remains
F16 target KV, producer-fused Q4 sums, and H1/full/lossless with the full native
270M drafter. This audit changed documentation only. No inference benchmark,
training, model download, or production implementation was run or changed.

The recommendation is to fix the newly identified chat-context restriction.
**The audit does not establish another worthwhile general H1 chat-speed gain.**
Attention is the leading remaining long-context hypothesis, conditional on a
current F16 cost measurement establishing enough addressable time. Prefill and
memory opportunities solve different user-visible problems. Another general
verifier/controller rewrite is not supported by the evidence. The findings below
distinguish source facts, retained measurements, and unmeasured proposals; later
historical sections must not be read as instructions to restart rejected
experiments.

### Confidence correction after scrutinizing actual H1 payoff

The user challenged whether the remaining mechanisms justify more optimization
work. The initial audit recommendation gave attention too much priority without
a current F16 whole-chat benefit. Missing reuse in source is a concrete mechanism,
not a concrete performance gain. None of this audit's proposed new optimizations
has established a retained gain over current H1.

At the recorded F16 cycle costs, holding acceptance/work fixed:

| Desired whole-chat throughput gain | Required saving per first-turn cycle | Required saving per follow-up cycle |
| --- | ---: | ---: |
| 2% | 1.76 ms | 2.31 ms |
| 5% | 4.27 ms | 5.61 ms |
| 10% | 8.15 ms | 10.71 ms |

These use `saved_time = cycle_time * (1 - 1/(1+gain))`, not the approximation
that time and throughput percentages are interchangeable. Endpoint/prefill costs
are excluded, as in the cycle model below.

The old F32 attention probes indicated only 0.294–1.021 ms of weighted savings
around 2K, and paired global attention at 2,090 keys was slower. Those probes
do not support an ordinary short/medium-chat speed claim. The combined old
attention/sampler chat's +4.34% was not isolated attribution, failed the ordinary
regression guard and preceded F16. At 8K the old synthetic 9.6–10.4 ms saving is
more interesting, but it is neither an actual long-chat result nor a current
F16 estimate. F16 has already changed the addressable memory cost.

For a 5% whole-chat gain through drafting alone, total draft wall time would
need to fall about 46–50%. For even 2%, it needs roughly 19–21%. Body row sharing
or attention head sharing addresses only part of drafting; there is no evidence
either reaches those thresholds. The failed Q4, head and sampler implementations
further weaken the case for another broad search on the common short/medium
H1 workload, although they do not prove a hardware optimum.

Decision: preserve the useful H1 baseline. Pursue long-global attention only if
long conversations are the intended workload and current F16 attribution reveals
a worthwhile budget. Pursue tiled prefill for slow long prompts, and physical KV
growth for memory pressure, without calling either a proven steady-generation
speedup. Draft alignment is a separate possible way to reduce target passes per
output token, with no demonstrated gain for this model pair and no training
authorized. This clarification changes the recommendation's confidence, not the
underlying measurements or mechanisms.

### Model and evidence checked

The actual local GGUF directories and scalar metadata were inspected again,
using `gguf-inspect` and the repository's independent GGUF reader:

| Property | Target | Drafter |
| --- | --- | --- |
| File | `models/gemma-3-12b-it-Q4_K_M.gguf` | `models/gemma-3-270m-it-Q8_0.gguf` |
| Architecture / layers | Gemma 3 / 48 | Gemma 3 / 18 |
| Hidden / FFN width | 3,840 / 15,360 | 640 / 2,048 |
| Query / KV heads, head width | 16 / 8, 256 | 4 / 1, 256 |
| Local window / global layers | 1,024 / 8 | 512 / 3 |
| Advertised context | 131,072 | 32,768 |
| Tensor types | 288 Q4_K, 49 Q6_K, 289 F32 | 127 Q8_0, 109 F32 |
| Supported tensor payload | 7,294,024,704 bytes | 285,018,624 bytes |

The target has a tied Q6_K embedding/output table, not an all-Q4 body/head.
Its global RoPE metadata specifies base 1,000,000 and linear factor 8. Existing
model binding and the shared block encoder implement the Gemma 3 Q/K norms,
hybrid windows, local/global RoPE and residual/norm ordering. No new architecture
mismatch was established in this audit. These details agree with the relevant
[Gemma 3 report](https://arxiv.org/html/2503.19786v1#S2); the file's own metadata
and validated conversion conventions remain authoritative for this checkpoint.

The newest retained native F16 receipts are
`benchmarks/results/gemma3-f16-kv-2026-09-29/native16-candidate-ordinary.json`
and `native16-candidate-h1-full.json`. They predate residency maintenance and
contain different naturally stopped outputs. They describe those executions,
not a new equal-work or current-binary benchmark:

| H1 measurement, per cycle unless stated | First turn | Follow-up |
| --- | ---: | ---: |
| Proposals / accepted | 761 / 490 | 652 / 387 |
| Acceptance | 64.39% | 59.36% |
| Cycle wall time | 89.659 ms | 117.836 ms |
| Target GPU time | 79.723 ms | 103.877 ms |
| Draft wall time, including finalization/gather | 8.561 ms | 12.317 ms |
| Remaining cycle time by subtraction | 1.375 ms | 1.642 ms |
| Target GPU / draft wall share | 88.92% / 9.55% | 88.15% / 10.45% |

Target-only GPU duration was 98.90% / 99.00% of reported decode wall duration.
This does **not** prove every kernel is bandwidth-saturated: GPU time includes
kernel dispatch, barriers, under-occupancy and memory stalls. It does establish
that removing CPU selection alone cannot explain a large throughput gain.
The H1 subtraction is not an independently instrumented host-stage measurement.

For intuition, with fixed other costs and `A` the acceptance probability,
H1 progress is approximately `1+A` tokens per cycle, apart from endpoints.
A 10% target-time reduction would improve overall throughput about 9.7%;
halving all draft time about 5.0–5.5%; a same-cost ten-percentage-point acceptance
increase about 6.1–6.3%. Perfect acceptance at the same cost would provide
21.7–25.5%. These are cost-model sensitivities, not promised gains: acceptance
also changes how often the drafter ingests two known tokens next cycle.

The older per-stage measurements were made on the F32 path. They identified
FFN projections as the largest short-context component and attention as the
growing long-context component. They cannot establish current F16 stage shares.
Likewise `scheduled_bytes_basis=one_row_equivalent_not_physical_bandwidth`
explicitly prevents using H1's reported scheduled bytes as measured DRAM traffic.

### Concrete correctness issue: disabled retrieval still imposes 8K

`Gemma2ChatSession::generate_user_turn`, in `gemma2_chat_session.cpp`, calls
`proposer_.rebuild(rendered_token_ids)` unconditionally inside `run_generation`.
It only conditionally passes that proposer into generation afterward.
`retrieval_proposer_max_context` is still 8,192, and `RetrievalProposer::rebuild`
throws above it. Therefore a rendered prompt above 8,192 tokens can fail even
with retrieval disabled, in ordinary Gemma 3 or standalone H1, despite passing
the target's 16K admission check. This is a direct source-level finding, not a
new full-model reproduction. It affects a long first prompt as well as a later
turn whose retained history crosses that boundary.

The earlier filled-16K harness validated runtime/cache arithmetic directly;
it did not establish that this chat entry point accepts those prompts. Keep that
evidence, but do not call it complete long-context product coverage.

Proposed repair: keep retrieval initialization/history maintenance behind the
actual retrieval-enabled path. Preserve or explicitly validate the optional
retrieval mode's own capacity; do not silently enlarge every old experimental
structure. Check ordinary and standalone chat just above 8K, cache reuse on a
follow-up, and enabled-retrieval admission. This repair preserves the intended
16K context; it is not a request to reduce it to 8K.

### Target attention: strongest next generation-kernel candidate

`verifier_attention_cache_kernel` in `src/tensor/backends/metal_backend.mm`
already reads half KV directly, keeps F32 arithmetic, shares K/V across the
two GQA query heads, uses stable online softmax, and emits fused Q4 sums.
It avoids a full score matrix. It is incorrect to describe it as an unfused
naive attention implementation or as missing GQA reuse altogether.

The remaining work is specific. `group.z` owns just one token row, so H1's
two queries scan their largely common history independently. Every SIMD stream
updates its maximum, exponentials, denominator and output for each key. A
fixed token-count rule selects 1/2/4/8/16/32 partitions, with the same rule for
half and float caches. This exposes three related opportunities:

1. Reuse each loaded K/V fragment across both H1 rows as well as both GQA heads,
   with short-lived fragments rather than large decoded tiles in registers.
2. Process key tiles with a stable block-softmax recurrence, reducing repeated
   rescaling of the accumulated output and increasing useful parallel work.
3. Choose partition/tile geometry for M4, head width 256 and F16 storage, with
   explicit short/local versus long/global routing.

For a tile of scores `s`, stable state `(m,l,u)` can be updated as
`m'=max(m,max(s))`, `a=exp(m-m')`,
`l'=a*l+sum(exp(s-m'))`,
`u'=a*u+sum(exp(s-m')*V)`, with final output `u/l`.
Empty states and fully masked tiles need explicit handling to avoid evaluating
`-infinity - -infinity`; a fully masked tile leaves the running state unchanged.
This is exact dense attention in real arithmetic; different floating-point
reduction orders still require numerical validation. The current kernel already
implements its tokenwise version. The proposal changes tiling/reuse, not which
keys belong to the model's attention.

Concrete references inspected, beyond paper descriptions:

- Current llama.cpp's `ggml-metal-tuning.cpp` chooses query rows per group and
  elements per thread using GPU family, KV dtype, head dimensions and context
  buckets. Its F16 width-256 Apple9 entries include different scalar/batch
  choices, and short KV keeps a baseline. The associated
  [merged tuning work](https://github.com/ggml-org/llama.cpp/pull/26570)
  supports shape-specific dispatch, not copying one universal tile.
- Current vLLM-Metal's `pagedattention.metal` has a two-row verification window,
  loads a K fragment once into per-row accumulators, and separately masks each
  row's causal/window boundary. Its
  [shared-row implementation](https://github.com/vllm-project/vllm-metal/pull/534)
  is a concrete register-lifetime reference. Paging and its numerical types
  are not requirements for our contiguous ring/private-cache layout.

There is relevant negative evidence: the earlier F32 blocked-attention candidate
was faithfully implemented. Broad replacement regressed ordinary follow-up
generation by 4.30%; the combined H1 candidate gained 4.34% aggregate but was
not retained. Isolated long-global attention was promising around 8K, while
short/local and scalar results were inconsistent. Thus the proposed change is
**F16, selectively long-context attention with complete ordinary/H1 integration**,
not blindly restoring the old globally dispatched candidate. F16 reduces memory
traffic and may move the crossover or eliminate part of the old benefit.

Every logical row must have the same arithmetic in ordinary and speculative
execution from the same cache. A threshold based on batch size alone can silently
break that property. Route by logical row properties; crossing a threshold in
an H1 pair needs correctly split/mixed execution, not a different target law.
Preserve ancestor/sibling isolation, wrapped local windows, private half-cache
rounding, all-masked partition handling, extreme-score stability, exact Q4
receipts, and publication/rollback. Retain the fast short-context path wherever
the new geometry does not earn its cost.

Forty layers already have bounded local work. Eight global layers still perform
dense attention over increasing history. Sharing and tiling reduce the cost;
they do not make arbitrary unchanged-model global attention constant-time.
Reusing an earlier query's attention output or dropping old global keys is not
an exact solution. Approximate attention/KV policies would change the target.

### Prompt processing: a distinct, underexplored opportunity

`run_gemma3_resident_prefill` processes chunks of up to 128 tokens. Its projection
path already uses F32 SIMD-group matrix multiplication, M8/N32/K64, and computes
the output head only for the final prompt row. Neither a missing matrix path nor
projecting the entire vocabulary for every prompt token is the problem.

Attention still uses the row-independent verifier kernel for the whole chunk.
This leaves two genuine prefill candidates:

- A dedicated multi-query tiled attention path that shares KV across prompt rows
  and computes QK/PV tiles, preserving causal and sliding-window masks across
  durable and private cache. Reuse the common attention semantics and selected
  precision policy; there is no reason to build another model executor.
- Larger projection row tiles for sufficiently long chunks. M16/M32 could
  amortize the same weight decode over more rows than M8. F32 accumulators,
  threadgroup memory and occupancy set the tradeoff; larger is not automatically
  faster. Keep short-prompt dispatch separate and avoid new full-weight copies.

MLX's `steel/attn/kernels/steel_attention.h` concretely performs tiled score
reduction, block softmax and PV accumulation; llama.cpp also distinguishes vector
attention from matrix attention. These apply the IO-reuse principle of
[FlashAttention](https://arxiv.org/abs/2205.14135). Adapt the dataflow, not their
half-query/fast-math defaults: target queries and accumulation are currently F32.
The existing F32 projection implementation demonstrates that native SIMD-group
matrix operations are already available in this engine on M4.

This work principally improves prompt latency, context reconstruction and long
follow-up prompts. It does not make steady R1/R2 generation faster by itself.
There is no current F16 prefill-stage profile to justify a percentage estimate.
The usual 16/19-new-token chat pair alone cannot validate a long-prefill claim.

### Dense projections, norms, RoPE and target output head

The ordinary/H1 body already uses interleaved columns8 Q4_K/Q6_K storage,
R1/R2 weight-decode sharing, and producer-fused K/8 activation sums. Gate/up and
Q/K/V project separately but their independent dispatches share barriers.
Post-branch normalization plus residual is already fused. The output head is
the sole canonical Q6 table also used for embedding.

The tested native-32 Q4 reduction, one-column 256-thread kernel, wider K-worker
prototype, four-row H3 changes and packed tied-head replacement did not establish
retained wins. They were not all incomplete placeholders. MLX's
[qmv_wide implementation](https://github.com/ml-explore/mlx/pull/3764) reinforces
small-batch weight reuse, which the target already has. Its affine quantization,
activation types and M4 Pro timings are not this GGUF's arithmetic or this Air's
performance. No basis was found for declaring another generic Q4 retile the
highest-confidence next step.

Untried or separately addressable fusion candidates include Q/K head norm with
RoPE/checked KV store, and paired gate/up projection with GELU/product and Q4-sum
production. They save dispatches and intermediate traffic, not the dominant
weight stream. Both must preserve the existing quantizer/reduction equations;
gate/up fusion increases simultaneous register state. Fusing the entire FFN
through down projection requires a global dependency or recomputation and is
not justified merely by calling it fusion.

The first ordinary receipt recorded about 918 compute dispatches and 629 barriers
per generated token. That is enough to justify **measuring GPU-side small-kernel
cost**, despite very little exposed CPU time. It does not justify treating all
non-matmul launches as removable or promising a large speedup from CPU encoding.
Use existing stage/capture mechanisms or a disposable harness, not a permanent
profiling framework.

The retained earlier scalar/joint Q6-head diagnostic was approximately
8.010/8.105 ms. Delaying H1's bonus head would reload it and serialize a dependency
for almost no saved joint work. Full-vocabulary target projection also cannot be
replaced by a heuristic shortlist while claiming the same target distribution.
Certified pruning remains a research possibility with no demonstrated useful
bound/cost here; it is not an implementation recommendation.

### Drafter execution: real reuse gaps, bounded overall payoff

`gd_q8` in `src/engine/gemma3_draft.mm` assigns a separate threadgroup to each
token row. `propose()` already batches a saved pending token with the known root,
but their weight decode is not shared. A two-row Q8 body specialization can load
one scale/quant fragment and use it for both rows while retaining each row's
F32 dequantization/FMA/reduction order. The head already projects just one row.
Acceptance determines how often there are two known body rows to exploit.

`gd_attention` independently scans KV for each of four query heads even though
all four share one KV head. This is a second concrete gap: share fragments across
two or four heads, then consider row sharing where it also helps. Do not force
eight simultaneous row/head states if registers destroy occupancy. MLX's current
`sdpa_vector_2pass_1_gqa` is a concrete head-sharing reference, but its single-query,
unmasked specialization does not handle our ring/private causal layout unchanged.

At a 16K prepared context the drafter's F32 durable KV is 111 MiB; moving it to
F16 would save 55.5 MiB. That is much smaller than the target's next memory
opportunity, and adds another precision policy. It is not the leading memory fix.
The Q8 tied head is 178,257,920 bytes, 62.5% of draft weight payload; this is not a
measured time fraction. Retain its full vocabulary and the actual integer-CDF
proposal law. Draft-only vocabulary restriction is mathematically compatible
with lossless target correction if the true restricted q is reported, but earlier
shortlist work and acceptance loss make it a separate quality/cost experiment,
not a free optimization or a reason to restart compact-drafter architectures.

Profile draft body, attention, head and sampling at representative short/long
contexts before choosing between Q8-row and GQA-head sharing. Even halving the
entire drafter gives only the 5.0–5.5% whole-cycle sensitivity above; either isolated
kernel can save only part of that. Drafter improvements do not accelerate
target-only generation.

### Speculative schedule, sampling, publication and streaming

The current H1 flow is structurally sound: select a root from retained target
candidates; ingest known pending/root draft tokens together; sample one proposal;
verify root/proposal with shared target R2 execution; apply acceptance/residual
sampling; publish only the selected KV path; retain the bonus/root candidates.
No hidden second target forward for an accepted prefix was found. The final
draft proposal stays pending rather than executing an unnecessary draft body.

Receipts show approximately four command buffers per H1 cycle: draft proposal,
target verification, target publication, and draft publication. Target publication
backs up endangered durable rows and scatters selected private rows; draft
`commit_prefix` also publishes its private rows. These are real lifecycle work,
not proof of redundant full inference. Deferred or combined publication could
reduce submissions, but callback-visible state, stop endpoints, sampler rollback,
and errors after partial delivery make it a transactional change. Preserve those
guarantees and demand measured publication cost before taking on that rewrite.

The normal compact sampler does not read all target logits to the CPU. Draft
gather exposes actual integer-CDF masses on the target's retained support, and
lossless correction depends on those masses, not an idealized softmax. The earlier
draft sampler dispatch-removal candidate did not establish a retained chat win.
Avoid replacing this with token equality merely because another project does so.
The inspected MLX-LM `speculative_generate_step` checks independently selected
target/draft token equality; that can preserve target sampling with its own
construction, but has a different acceptance coupling from our p/q residual
algorithm. Its acceptance or throughput is not an interchangeable baseline.

For the present lossless coupling, conditional acceptance is
`sum_v min(p(v),q(v)) = 1-TV(p,q)` for the actual transformed sampling laws.
Improving same-cost draft alignment can reduce expensive target passes per output
token. The recent
[acceptance-aware training paper](https://arxiv.org/html/2609.24150v1) supports
matching the loss to sampled acceptance; it does not prove an improvement for this
exported Q8 Gemma pair. A later user-run pilot should compare ordinary distillation
with overlap-aware training on the quantized F16-KV target, using held-out chat
cost and exported q. H1 does not require a multi-token training framework.

Streaming has another distinct cost: reported TTFT ends at first selection,
while the first visible text follows materializing/committing that token's KV.
In the retained F16 runs the gap was roughly 76–115 ms, including other work in
the visible-time clock. Publishing before commit would change the callback/cache
contract. The draft prompt synchronization is already deferred until after first
target publication. Label and measure visible latency; do not manufacture a TTFT
win by moving work outside the reported interval.

### Memory, startup and host work

Target durable F16 KV currently allocates all configured global capacity at plan
creation. Its byte formula is `2 * 2048 * 2 * (40*1024 + 8*C)`.
At 16K that is 1.3125 GiB. Starting the eight global layers with 4K physical
capacity would use 0.5625 GiB and save **0.75 GiB / 768 MiB**, while retaining
the same logical 16K limit and growing before it is needed. This is the strongest
remaining concrete memory-saving proposal without another numerical change.

Growth must occur at a completed transaction boundary, before any prefill or
proposal crosses capacity. Distinguish logical context from physical row masks;
copy existing half bits, update all bound views/admission accounting/residency,
and handle allocation failure without publishing a partial cache. Whole-cache
doubling briefly keeps old and new buffers; staggered growth or fixed-size slabs
may reduce the peak but add indirection/transaction complexity. Choose using peak
memory and latency evidence. MLX's incremental `KVCache` is an existence proof,
not a reason to import its framework or blindly copy repeated concatenation.

This improves early memory pressure, not attention work over an already filled
prefix. Residency maintenance may make avoiding unused capacity more valuable;
that interaction is an inference, not a new physical-memory measurement. At 128K,
target F16 KV alone would be 8.3125 GiB; model metadata support is not proof that
this 16 GiB machine can admit the full model/cache/runtime working set. The
drafter's own 32K limit also remains a separate constraint.

Resident target weights are about 7.44 GB after packing. Upload already stages
one repacked tensor at a time and keeps a single tied head. Hashing/repacking and
shader initialization are startup costs, not per-token inference. Persistent
packed-weight/shader caches could improve launch time but require identity and
format invalidation; no evidence makes them the current generation bottleneck.
No full durable-KV checkpoint mirror should be reintroduced.

Chat currently renders/tokenizes history before reconciling exact cached IDs.
Incremental template/tokenization work could improve host latency at long history,
but must preserve BPE boundaries, special-token handling, truncation, reverse stops
and regeneration. Fix disabled retrieval first. Existing receipts do not assign
a meaningful generation bottleneck to tokenization or detokenization.

Metal indirect command buffers / command reuse can reduce encoding cost, as
[Apple documents](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu).
They do not eliminate GPU weight reads or automatically fuse dispatches. Current
llama.cpp deliberately disables its tensor-API kernels by default before M5/A19,
with an implementation comment reporting no significant M4 benefit. That is an
engineering result, not a claim that M4 lacks all Metal 4 support. A wholesale
Metal 4 migration is not the leading hypothesis here.

### Proposed order, verification and stop conditions

1. **Repair the unintended 8K chat restriction**, including actual session-path
   coverage. This is a correctness prerequisite, not a claimed speedup.
2. **Establish whether current F16 attention warrants implementation**: ordinary
   R1 and H1 R2 at common populated prefixes, plus short/long prefill. Separate
   local/global attention from FFN/head time and record instrumentation overhead.
   If the budget is worthwhile, prioritize selective long-global sharing for
   generation and tiled multi-row attention for prefill; keep dispatch decisions
   tied to measured crossovers. Do not treat this as a commitment to another
   general H1 kernel optimization pass.
3. **Consider the drafter only where its own profile justifies the payoff**, using real Q8 row reuse
   and/or GQA reuse. Do not substitute an impoverished draft architecture.
4. **Add physical KV growth if memory pressure is the objective**, preserving
   logical context and exact stored values. Do not conflate this with faster
   dense attention at a filled context.
5. Consider targeted norm/FFN fusion or user-run acceptance training only with
   a credible remaining cost budget. Leave H3, generic async drafting, repeated
   head repacking and rejected Q4 layouts outside this proposed phase.

Reuse `tests/tensor/gemma3_metal_tests.cpp`, Gemma 3 runtime/draft tests,
`tests/engine/standalone_chat_tests.cpp`, lossless sampling tests and session
coverage in `tests/frontends/cli/generate_cli_tests.cpp`. Candidate attention
needs ordinary/verifier numerical agreement from the same physical prefix,
independent reference checks, nonzero/extreme queries, long global history,
local wrap, sibling isolation, mixed routing boundaries, fused receipts,
F16 overflow refusal and byte-exact rollback. Memory growth needs boundary,
allocation-failure, outstanding-lease and reset/regeneration checks. No generic
new testing subsystem is proposed.

A future implementation must pass the repository checks and establish benefit
on actual chat, with the documented fresh-process cooldown protocol. For a
long-context or prefill claim, include that actual workload in addition to the
standard two-turn chat and report visible TTFT, generation, memory and thermals.
Keep fixed-state comparisons for numerical/cost diagnosis distinct from sampled
natural-stop timing. If current F16 stage measurements do not reveal useful
attention headroom, stop that candidate rather than spending repeated rounds
tuning secondary effects. No new percentage speedup is established by this audit.

### Reproducible external source identities

Public source snapshots were read on September 29 and pinned before inspection:

- [llama.cpp `ba0ba54d`](https://github.com/ggml-org/llama.cpp/tree/ba0ba54d93b25faf1e149f4ccedd3e9d84798563):
  `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml-metal-tuning.{cpp,h}`,
  `ggml-metal-device.m`, `kernels/fa_vec_common.metal`, `fa_vec_f16.metal`,
  and `src/models/gemma3.cpp`.
- [MLX `7916d8b0`](https://github.com/ml-explore/mlx/tree/7916d8b0c8877b8a10f231363aa49f2f83acb34a):
  Metal `quantized.cpp`, `kernels/quantized.h`, `kernels/sdpa_vector.h`,
  `kernels/steel/attn/kernels/steel_attention.h`, and attention dispatch.
- [MLX-LM `94cdcae1`](https://github.com/ml-explore/mlx-lm/tree/94cdcae13b266c337bcaca09b97b9c5a9c0e2cde):
  `mlx_lm/generate.py`, `models/cache.py`, and `models/gemma3_text.py`.
- [vLLM-Metal `b2facc1f`](https://github.com/vllm-project/vllm-metal/tree/b2facc1f1d208f6ec7b43e7e4a00cf91109752e7):
  `vllm_metal/metal/kernels_v2/pagedattention.metal` and `pagedattention_tiled.metal`.

These are reference implementations, not production dependencies or newly run
competitor baselines. The installed llama-cli's ggml 0.13.1 identity belongs to
the preceding idle-residency comparison and is distinct from current upstream.

## Idle Metal residency maintenance (2026-09-29)

The user requested the idle-residency mechanism found in the installed
llama-cli 9430 / ggml 0.13.1 backend. The engine now registers owned Metal
buffers in one residency set and requests residency every 500 ms for three
minutes after allocation or the latest target/drafter command submission.
After inactivity the refresh requests stop. This is enabled automatically on
macOS 15 and later and leaves the retained F16/H1/full/lossless defaults intact.

The buffer owner tracks references to each underlying allocation, so packed
weight views and cache views neither duplicate registration nor prematurely
remove a shared parent. The last owner removes and commits the allocation;
an empty set ends residency. Shutdown stops and joins the maintenance worker.
The mechanism changes neither sampling nor numerical execution, adds no GPU
commands and does not elevate process priority. Residency creation errors fail
explicitly. Retaining residency can leave more memory physically occupied while
the user pauses between messages.

The user subsequently reported that this fixed the observed idle behavior;
the idle TTFT penalty has not been remeasured in a controlled benchmark. At the
user's request, no tests or benchmarks were added or run for this change.
Verification is limited to the release build and
source review. Task-local source snapshots and the build log are ignored under
`build/gemma3-idle-residency-2026-09-29/`. The follow-up pipeline audit and
proposed next work are above. Earlier correctness/performance evidence below
belongs to the versions described there, before residency maintenance was added.

## Native F16 target KV implemented (2026-09-29)

The approved F16 implementation is complete for ordinary Gemma 3 and H1,
including the retained producer-fused K/8 Q4 sums. F16 is now the automatic
target-cache policy for Gemma 3 and Gemma 2; explicit `--kv-cache f32` retains
the previous Gemma 3 numerical target. Supplying `--draft-model` now selects
H1, full verification and lossless sampling without additional tuning flags.
The Gemma 3 270M drafter remains F32. This section owns the current state;
the design below is the pre-implementation investigation.

### Implemented mechanism and boundaries

- One engine precision policy is resolved before admission and allocation.
  All factories, prepared plans, session compatibility, reset, regeneration,
  context reconstruction and diagnostics use that same resolved policy.
  Loaded metadata and authenticated weight identity are unchanged. Unsupported
  policies and explicit prepared-plan mismatches fail before inference.
- Durable target KV, ordinary/chunk private KV, verifier private KV and undo
  all use the selected dtype. K is projected, normalized and rotated in F32,
  then stored as half; projected V is stored as half. Attention reads those
  rounded values even for the current token and uncommitted proposal rows.
  Publication and rollback preserve the stored bits. No F32 cache mirror or
  separate cache-dequantization pass was introduced.
- The precise half-cache attention specialization produces the existing fused
  Q4 sums. Queries, attention/softmax accumulation, partition partials,
  activations, projection math and receipts remain F32. Packed body weights,
  the shared tied Q6 head, per-row partition routing and H1 scheduling remain
  intact. Gemma 2's existing compilation policy is unchanged.
- A checked half-store specialization ORs status bit 8 only when narrowed K/V
  is nonfinite. It shares the existing dispatch and completion boundary:
  no additional GPU submission or per-layer host wait. Ordinary execution
  poisons an already-published invalid cache; width-two verification restores
  its backups; full/conditional graph verification rejects private invalid
  state before host sampling/publication and releases the lease. There is no
  silent clamping or fallback to another precision.
- `--kv-cache auto|f32|f16` is exposed through chat and the canonical benchmark.
  Receipts report the resolved dtype. An explicit benchmark request must match
  both turns' actual execution receipts.

This is a different numerical target from F32 storage. Lossless speculation
preserves the selected F16 target's sampling law; it does not promise equality
with the old F32 distribution or identical seeded output.

### Correctness evidence and the reference discrepancy

`MODE=release` full native `bash scripts/ci.sh` passed **492 C++ tests**,
with both `LOCAL_GGUF_GEMMA3_TARGET_GOLDENS=build/gemma3-target-reference`
and `LOCAL_GGUF_GEMMA3_TARGET_F16_GOLDENS=build/gemma3-target-reference-f16`.
The Python suite reported 62 tests total with 15 optional skips; the separate
reference environment passed all 10 reference-specific checks without skips. The
unavailable-device run reported 492 passing test entries, with hardware-only
cases returning through their unavailable-device guards. The release build had
no compiler warnings, and `git diff --check` passed.

Native primitive coverage includes independent nearest-even normal/subnormal
ties, signed zeros, maximum finite values and overflow; K rounding after RoPE;
all-head precise half attention with nonzero queries; wrapped and offset caches;
tree/sibling isolation; extreme finite scores; partition threshold crossings;
and exact fused receipts. Runtime coverage includes independently rounded
six-layer equations, ordinary/verifier exactness, accept/rollback paths,
KV-only endpoints, full/conditional schedules, callback/stop behavior,
sampler/cache recovery, reset, regeneration, context shifting, policy admission
and CLI failure paths. The new overflow fixture was corrected to establish a
valid prefix and advance proposal epochs on retry; production validation was
not bypassed to make it pass.

The first 12B F16 cache assertion failed at a nearly cancelling layer-14 V
coordinate. A separate instrumented copy captured F32 intermediates without
changing production source or goldens. All **393,216 native V stores** matched
NumPy and PyTorch bitwise and independent C++ mathematical RNE values;
**98,304 position-zero K stores** matched independent rotation followed by
NumPy RNE. The first V
difference already occurs in layer 0: CPU/native raw values
31.2578525543/31.2577915192 correctly round to 31.265625/31.25. At the failing
layer-14 coordinate, the upstream hidden-state effect was -5.79599e-4, versus
only -2.31341e-6 from the remaining native normalization/projection arithmetic.

The F16-specific cache regression check therefore uses each 256-coordinate
head's reference RMS to scale cancelling components. Its component bound is
`5e-4 + (3e-4 + 2^-10) * max(abs(reference_coordinate), reference_head_RMS)`;
it also bounds the head RMS error by `5e-4 + (3e-4 + 2^-10)*reference_head_RMS`
and rejects all nonfinite values. The worst observed component/RMS budget
fractions across scalar, matrix, H3, mixed continuation and rollback phases
were 0.758648/0.488827. This is an explicit precision-specific regression
contract, not a universal analytic transformer error bound. Existing F32
cache/logit bounds, authenticated goldens and exact same-conditional checks
were preserved. The general CPU half converter was left unchanged; the new
conversion oracles do not rely on its ties-away behavior.

The full 12B F16 matrix-prefill versus CPU maximum logit error was 0.00167990,
within the unchanged logit bounds. The largest recorded filtered sampler TV
in those four-token reference comparisons was 0.00006044, with unchanged
support. This evidence checks implementation fidelity; it does not establish
quality for arbitrary conversations.

### Filled-context numerical and physical-memory evidence

A separate ignored harness used the production prefill, scalar and H1 graph
paths with exactly the same frozen 16,361 token IDs in both precisions. It
populated through 16,360 tokens. At each 1K, 2K, 8K and 16K checkpoint, the
initial two-row H1 pair matched ordinary full logits and all private KV bytes
exactly from the same
prompt cache; provisional publication followed by accept-zero restored physical
cache bytes exactly, and every populated cache value and all inspected
full-logit rows remained finite.

| Prefix checkpoint | F32 physical GiB | F16 physical GiB | Maximum full-vocabulary T=1 TV | Maximum filtered sampler TV |
| --- | ---: | ---: | ---: | ---: |
| 1,023 | 9.8521 | 8.4931 | 0.000774573 | 0.000781064 |
| 2,048 | 9.8976 | 8.5270 | 0.000577136 | 0.000401700 |
| 8,192 | 9.9301 | 8.5437 | 0.0000317303 | 0.0000344020 |
| 16,352 | 10.0557 | 8.6084 | 0.0000520979 | 0.0000436686 |

Each checkpoint has eight teacher-forced predictions over all 262,144 logits.
Across these 32 positions, maximum absolute logit change was 0.01750565;
mean next-token NLL was 3.10751275 F32 versus 3.10755324 F16 (delta +0.00004049
nats/token). One argmax changed, one top-40 set changed, and no filtered support
set changed. Filtering follows the engine's top-k/top-p/min-p-before-temperature
order with 40/0.95/0.05, T=0.8 and repetition penalty 1.0. This frozen
technical-text sample is a numerical sensitivity check, not a broad
model-quality evaluation or a cooled speed benchmark.

Physical samples were taken before each host cache scan. Later lifetime peaks
can include earlier diagnostic readbacks, and this harness is not the chat
process. Initial plan allocation had nearly identical physical footprint
(7.056 GiB) in both modes despite different Metal allocation totals; after
the first 1K prefill, physical use was close to the whole Metal allocation.
This supports the hypothesis that unused reserved global capacity becomes
resident during Metal execution. It does not prove the speed or footprint of
an as-yet-unimplemented growth allocator.

### Cooled real-chat retention result

Four valid runs used fresh frozen native executables, the canonical sampled
16K two-turn conversation, seed 42, natural stops, five seconds between turns
and at least 240 seconds of idle cooldown before each process. No builds or
diagnostics overlapped those cooldowns or chats. H1 used the unchanged 270M
F32 drafter, horizon 1, full schedule and lossless sampling.

| Mode | F32 first / follow-up tok/s | F16 first / follow-up tok/s | Aggregate throughput change | Sampled lifetime high-water GiB, F32 → F16 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 12.09 / 8.83 | 12.48 / 9.31 | +4.64% | 9.916 → 8.563 |
| H1 full | 17.23 / 10.71 | 18.33 / 13.52 | +15.67% | 10.400 → 9.053 |

Aggregate throughput is total decoded tokens divided by total generation
duration, not the mean of rounded per-turn rates. Ordinary improved 3.21% and
5.42% by turn; H1 improved 6.33% and 26.20%. Both exceed the 2% aggregate
retention screen, win on both turns and have no greater-than-2% regression
in another recorded headline metric. These are same-protocol observations,
not universal gains or exact-work comparisons: natural-stop text, lengths,
follow-up cache positions and H1 proposal acceptance differ.

| Mode / KV | First tokens / generation seconds | Follow-up tokens / generation seconds | First / follow-up TTFT ms | First / follow-up visible-text latency ms |
| --- | ---: | ---: | ---: | ---: |
| Ordinary / F32 | 1506 / 124.593054 | 1026 / 116.189238 | 293.237 / 600.525 | 392.965 / 719.010 |
| Ordinary / F16 | 1493 / 119.674671 | 967 / 103.879871 | 291.105 / 609.165 | 393.458 / 724.390 |
| H1 full / F32 | 1344 / 77.983649 | 1024 / 95.568223 | 292.259 / 631.040 | 374.620 / 732.629 |
| H1 full / F16 | 1253 / 68.372774 | 1041 / 76.982233 | 283.962 / 544.252 | 359.659 / 631.541 |

Latency is engine-reported per-turn latency after startup warmup. Ordinary
visible-text latency changed +0.13%/+0.75%; H1 improved 3.99%/13.80%. Ordinary
follow-up prompt throughput fell 1.42% (TTFT +1.44%), within the 2% guard.
H1 accepted/proposed edges were 537/805 and 378/644 for F32, versus 490/761
and 387/652 for F16. All runs evaluated 16/19 new prompt tokens; follow-up
cached positions were 1,522/1,509 for ordinary F32/F16 and 1,360/1,269 for H1.

The actual allocation receipts confirm savings of **1,474,035,712 bytes
(1.372803 GiB) ordinary** and **1,475,608,576 bytes (1.374268 GiB) H1**.
Durable target KV is 1,409,286,144 rather than 2,818,572,288 bytes. H1 total
owned bytes fell from 11,077,685,408 to 9,602,076,832. Separately, sampled
native lifetime-high-water physical footprint fell 1.353 GiB ordinary and
1.347 GiB H1. These are exact native PID `proc_pid_rusage` samples; lifetime
high-water sampling can miss a new peak after the final live sample.

An initial ordinary baseline had a 174.078-second sampling gap and roughly
174.844 seconds of first-turn wall time outside measured GPU execution,
consistent with system suspension. The entire run is excluded from speed
comparison, including its follow-up. It was preserved and replaced after
another full cooldown; no time was subtracted to manufacture a corrected rate.
A temporary `caffeinate -i` assertion protected the remaining runs and ended
with their driver. Maximum sampler gaps in all four retained runs were below
1.05 seconds. `pmset` reported no recorded thermal warnings; this does not
measure GPU clocks, temperature or throttling. System-wide swap was already
present and varied across runs; those counters do not attribute swapping to
this engine. This is one independently cooled pair per mode, not a statistical
confidence interval or a broad workload suite.

Valid receipts are `awake16-baseline-ordinary`, `native16-candidate-ordinary`,
`native16-baseline-h1-full` and `native16-candidate-h1-full` in the result
directory below. `native16-baseline-ordinary` is the excluded interrupted run.
Each valid result retains raw counts, durations, token hashes, exact binary
identity, cooldown protocol and native memory/thermal samples.

### Current baton and preserved artifacts

The user approved promoting the retained setup to the standard configuration.
Automatic target KV now resolves to F16 before admission or allocation, and
standalone execution defaults to H1/full. Omitted speculative sampling resolves
at the engine session boundary to lossless when a drafter is supplied and to
exact-match for target-only execution. The CLI and canonical benchmark preserve
that omission instead of requiring redundant flags. Explicit F32, wider horizons
and conditional verification remain available; incompatible explicit requests
still fail. Prepared sessions compare resolved precision and execution settings.
Existing sessions cannot switch numerical targets during reset or regeneration.

The standard paired command is:

```sh
./build/release/local-gguf-generate \
  models/gemma-3-12b-it-Q4_K_M.gguf --chat \
  --draft-model models/gemma-3-270m-it-Q8_0.gguf
```

Omit `--draft-model` for ordinary Gemma 3 chat. Model paths remain explicit;
there is no model discovery or implicit loading of a drafter. Context capacity
remains 16K by default, and 16K was actually filled and checked rather than
merely reserved. Packed weights, tied Q6 head, fused Q4 sums, F32 attention math
and F32 drafter execution are unchanged. The default promotion selects the
previously measured paths; it is not another kernel optimization or a claim of
universal optimality. Stop after defaults validation; no further kernel,
training or architecture experiment is active.

Defaults verification: full native release `bash scripts/ci.sh` passed **493
C++ tests**, with both existing F32 and F16 12B reference directories enabled.
The F16 checkpoint test now exercises automatic selection; explicit F32 keeps
its independent reference. Engine and CLI tests compare omitted paired controls
against explicit F16/H1/lossless/full, including token selection, sampler state,
reset/regeneration, allocation receipts and prepared-policy compatibility.
The Python suite passed 64 tests with 15 optional skips. A separate sandbox run
passed 493 entries through unavailable-device guards; that run is not GPU parity
evidence. Native compilation produced no warnings.

Four bounded actual-12B chat processes compared the previous binary's explicit
F16 ordinary/H1 configurations with the new binary's omitted controls. Each ran
the same two prompts, seed 42 and a 12-token response cap at default 16K capacity.
Within each mode, both turns matched selected-token hashes, token counts, cache
positions, allocation receipts and execution policy; H1 also matched proposal
and acceptance counts. This verifies default routing, not a new speed estimate.
The cooled performance evidence above remains the basis for retaining these
paths; it was not replaced by a short uncool comparison.

Defaults source snapshots, native/unavailable-device logs, bounded chat outputs
and `chat-equivalence.json` are ignored under
`build/gemma3-defaults-2026-09-29/`. The rebuilt standard executable SHA256 is
`5f569e982b29298ed25e081ea9fed67bd2d67e8d068ff58e33432ba0107b2fc0`.

The next substantial memory opportunity is to grow global KV capacity as
needed while preserving the logical context limit. Starting global storage
at 4K instead of 16K would avoid another 0.75 GiB of target half-cache capacity
at short contexts. Physical evidence now supports investigating that change;
growth must account for old/new allocation overlap, in-flight buffer ownership,
verifier capacity, rollback, context shifting and unchanged numerical state.
It remains a separate task. No paging framework, draft-cache precision change,
scratch-aliasing rewrite, token eviction or lower-bit KV was added here.

Task-local source snapshots, hashes, native/CPU logs, independent numerical
diagnostics and raw filled-context outputs are ignored under
`build/gemma3-f16-kv-2026-09-29/`. Raw cooled chat results and native PID memory
samples are ignored under `benchmarks/results/gemma3-f16-kv-2026-09-29/`.
The accepted baseline SHA256 is
`b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab`;
the measured explicit-F16 candidate is
`1432247a6eb8edf94a93390e591f2efd43a5ec3bb0401055a89cd3ad08c42216`.
No model, binary or result artifact should be committed. No model download,
training or external production inference was performed.

## Approved F16 KV and memory design (before implementation)

The pre-implementation investigation recommended the following. **Approved
implementation: native F16 target KV throughout ordinary Gemma 3 and H1,
preserving the accepted producer-fused K/8 Q4 sums.** The next substantial allocation hypothesis is
growing global KV capacity as needed without reducing the logical context limit.
It needs physical-memory evidence before investing in a cache ownership change.
These are separate mechanisms with separate numerical and lifecycle contracts.

This historical proposal superseded the closed native-group experiment below.
The completion report above owns the current state. The investigation itself
changed only this document; no implementation, model execution, build, GPU
benchmark, training or model download was performed during that investigation.
Source, existing receipts, independent scalar half conversions and primary
external implementations were examined. At that point production used F32
Gemma 3 target and drafter KV, H1 full lossless speculation, packed body weights
and the retained fused sums. The accepted binary and its evidence are identified
below.

### What actually occupies memory

The accepted 16K H1 chat receipt is
`benchmarks/results/gemma3-q4-native-groups-2026-09-28/native32-baseline-h1-full.json`.
Both turns report the same allocation ledgers:

| Owner | Bytes | MiB |
| --- | ---: | ---: |
| Target resident weights | 7,437,057,024 | 7,092.53 |
| Durable target F32 KV | 2,818,572,288 | 2,688.00 |
| Target ordinary/legacy scratch | 298,034,400 | 284.23 |
| Standalone owner: drafter, H1 arena and host state | 524,021,696 | 499.75 |
| Sum of these ledgers | 11,077,685,408 | 10,564.50 |

The separate sampled native lifetime-high-water physical footprint was
11,173,179,008 bytes; a live sampler can miss the final peak. Allocation ledgers,
virtual mappings, RSS and physical footprint are different measurements. Source
mapping, external tokenizer and driver costs are not all represented in this
sum. The H1 arena is already included in the standalone owner: do not count it
twice. There is no second full KV checkpoint; regeneration saves histories,
positions, sampler state and execution segments, then replays them.

For this actual 48-layer target, 40 layers have 1,024-row local rings and eight
have global history. There are eight KV heads of width 256. With global capacity
`C`, durable KV bytes are `2 * 8 * 256 * element_bytes * (40*1024 + 8*C)`.

| Global capacity | F32 durable KV | F16 durable KV |
| --- | ---: | ---: |
| 4,096 | 1.125 GiB | 0.5625 GiB |
| 8,192 | 1.625 GiB | 0.8125 GiB |
| 16,384 | 2.625 GiB | 1.3125 GiB |
| 32,768 | 4.625 GiB | 2.3125 GiB |
| 131,072 | 16.625 GiB | 8.3125 GiB |

The design does not impose an 8K context limit. Model limits and whole-process
memory admission still apply; the last row does not imply that 128K fits this
16GB machine alongside weights and the operating system.

Consistent F16 also halves target KV-shaped temporary storage. At current
capacities it saves **61.75 MiB ordinary scratch** (all-layer private rows,
undo, legacy chunk and width-two backups) and **1.5 MiB H1 private/undo storage**.
Together with the durable cache, the expected H1 owned-byte reduction is exactly
**1,475,608,576 bytes / 1.3742676 GiB**. This is descriptor arithmetic for the
existing layout, not a measured F16 process-footprint reduction.

### F16 implementation: one target policy, one shared block path

F16 is a storage precision change, not an all-half transformer. Keep projections,
Q/K normalization, RoPE, queries, attention dot products, online softmax,
accumulators, partition partials, output activations and Q4 receipts in F32.
The shared `encode_gemma_target_blocks()` already connects ordinary prefill,
scalar continuation and graph verification. Preserve that foundation.

The exact order is F32 projection and Q/K normalization, F32 RoPE, then half
storage of rotated K and projected V. Every attention use must read those stored
values, widening them to F32. **This includes the current token and private
verification rows before commit.** Computing attention with F32 private rows
and rounding only the accepted cache would define different ordinary and
speculative target conditionals. Commit and undo copy the already rounded bits.
Do not round K before RoPE or introduce a full F32 cache mirror/dequantization
pass.

Resolve an explicit engine request (`automatic`, `f32`, `f16`) once during
preparation. Recommended narrow ownership: carry the resolved precision in the
prepared runtime copy of `Gemma2ModelConfig`, which already carries the resolved
context capacity; keep loaded GGUF metadata and source identity unchanged. All
factory overloads and estimators must consume the same resolved copy. Automatic
initially preserves the current defaults, Gemma 3 F32 and Gemma 2 F16. Preserve
the existing Gemma 2 supported policy instead of adding an unvalidated mode.
Do not change precision midway through a session or silently retry F32 after an
F16 allocation/numerical failure. The first implementation leaves drafter F32.

| Boundary and source | Concrete change |
| --- | --- |
| `include/local_gguf_engine/engine/model.hpp`, `src/engine/model.cpp` | Replace the architecture-only `gemma_kv_dtype()` decision with resolved runtime precision; reject unsupported policy combinations. |
| `gemma2_metal_plan.hpp/.cpp`, `prepared_gemma2_plan.hpp/.cpp` | Thread policy through ordinary and standalone factories, resolve before admission/upload, allocate durable and ordinary temporary KV consistently. |
| `gemma2_chat_session.hpp/.cpp` | Carry requested/resolved precision through prepared-plan compatibility, reset, context rebuilding and regeneration. |
| `src/engine/target_verifier.cpp` | Keep private/backup allocation derived from the resident cache descriptor; make admission estimates use the same precision. |
| `metal_backend.mm`, `metal_context.hpp/.mm`, `metal_command_context.mm` | Add the missing precise half-cache attention specialization with fused Q4 sums; select it by actual cache dtype. |
| `gemma_target_runtime.cpp`, `target_verifier_runtime.cpp`, `gemma2_runtime.cpp` | Audit store/scatter/copy/undo extents, byte offsets and lifecycle; remove F32 byte assumptions in copy traffic metrics. |
| `src/frontends/cli/generate.cpp`, benchmark receipt plumbing | Expose the engine policy and report the actual resolved dtype; current architecture-based `f32` reporting is insufficient. |

The kernel gap is specific. `verifier_kv_store_kernel<half>` and half-cache
attention already exist, as do dtype-aware copy/scatter operations. However,
the retained Q4-sums route rejects non-F32 caches and only instantiates
`verifier_attention_cache_kernel<float4, true>`. Add `<half4, true>` under the
Gemma 3 precise compiler policy and wire its pipeline through the existing
context. Keep the shared F32 multipartition reduction and its K/8 receipts.
This must preserve both single-partition and reduced-output receipt production,
without an extra dispatch, separate sum-preparation pass or lost H1 weight
sharing. Unfused precise half attention must remain available for meaningful
parity checks. Gemma 2's compilation policy stays unchanged.

`ResidentTargetPayloadReceipt` authenticates uploaded weights, not KV state.
F16 must not fabricate a new model identity or change its weight hashes. Cache
precision belongs in plan/session compatibility and run provenance. Shape,
dtype, byte-count, alignment and alias checks must agree at every buffer boundary.
The verifier workspace currently inherits resident dtype correctly; retain that
mechanism rather than adding a parallel F16 allocator.

### Numerical evidence and the validation trap

`build/gemma3-target-f16-comparison.json` is a **four-token** independent CPU
study, not a native F16 validation. Maximum logit change against the F32 target
was 0.00888085 and maximum full-vocabulary T=1 total variation was 0.00038717;
argmax and top-40 membership matched for those prefixes. Maximum observed
absolute K/V were about 203.44/1,230.93, far below F16's 65,504 maximum finite
value. This supports investigation, not long-context quality or speed claims.
Of 786,432 stored values, 264,870 differed from simply rounding the old F32-run
caches because rounding changes subsequent computation. Generate a new F16
target oracle; casting old F32 goldens is wrong.

There is a concrete reference hazard in `src/tensor/f16.cpp`:
`f32_to_f16_bits()` adds a fixed rounding bias, giving ties away from zero instead
of nearest-even. For `1 + 2^-11`, it returns `0x3c01`; independent nearest-even
returns `0x3c00`. NumPy/PyTorch in the existing reference environment and Python
`struct` agree on tested normal/subnormal ties. Apple's current
[narrowing-conversion documentation](https://developer.apple.com/documentation/metal/mtlcompileoptions/floatingpointconversionroundingmode)
documents nearest-even as the default. The installed SDK does not expose that
new option, so an independent native cast fixture must establish this build's
actual rounding and subnormal behavior. Do not use the engine helper as its own
oracle, silently change every caller, or loosen tolerances at half boundaries.
If the shared helper needs correction, audit its callers and preserve existing
Gemma 2/quantization evidence as part of that necessary change.

F16 changes the numerical target. Lossless speculation must preserve the
**declared F16 target's sampling distribution**; it cannot claim exact identity
with the old F32 distribution. Test actual sampler probabilities, including
support changes, in addition to logits. Ordinary/verifier conditional parity
from the same prepared prompt cache remains required. Matrix prefill and scalar
continuation already have different F32 reductions; half rounding can magnify
midpoint differences. Preserve recorded execution segments during replay and
validate both entry paths against their independent equations.

Range tests must include maximum finite values, ties, subnormals and overflow.
There is no observed overflow in the tiny existing study, and no basis for
premature per-layer formats or rescaling. Silent clamp/NaN propagation is not a
valid fallback. If runtime overflow detection is necessary, integrate status
with the existing completion/readback boundary, without per-layer host waits.
Ordinary execution can already have published KV before completion; a late
failure must follow the poisoned-session contract, not claim untouched state.
This needs an explicit numerical-failure poison transition: after a successful
wait, `execute_contiguous_chunk()` clears `completion_unknown`, so its existing
catch does not automatically poison a subsequent validation failure.

### Other opportunities, ranked by useful scope

**1. Grow global cache allocation without shortening context.**
`Gemma2MetalKVCacheLayer` already separates `row_capacity`/`row_mask` from the
configured logical context. Attention uses that mask and schedules derive
visibility from logical positions. Cache-copy encoding permits different
source/destination masks. This makes bounded contiguous growth plausible
without a paged-attention rewrite. Keep all 40 local rings fixed at 1,024 rows;
grow only the eight global layer pairs in powers of two.

At the recorded final context of 2,403 tokens, a 4,096-row global allocation
would save another **0.75 GiB after F16**, making durable target KV 0.5625 GiB
while retaining the logical 16K limit and every visible token. Combined with
consistent F16, that is about **2.12 GiB less owned allocation** than current H1.
It is not evidence of 2.12 GiB less physical RAM or faster attention.

Before implementing, measure the current target at allocation, warmup and filled
prefixes. Shared Metal buffers are created without a full host memset; warmup
does not clear the complete cache, and attention visits populated rows only.
Untouched full-capacity pages may already be physically demand-backed. If so,
reducing nominal allocation alone may not justify growth copies and ownership
complexity. Measure physical footprint and Metal allocated bytes separately.

If the physical-memory benefit is substantial, reserve all potentially written
rows before encoding an ordinary chunk or opening a verification transaction.
Use a sensible initial floor, then geometric growth; preserve populated global
prefixes bit-for-bit. Finish prior commands, allocate/copy all required layer
pairs, and publish replacement owners only after successful completion. No
growth during a pending ticket or provisional callback. Allocation failure must
leave the old cache usable; GPU failure follows the existing poisoning contract.
Admission must account for old-plus-new overlap, and distinguish current
allocation from the budget needed to reach the advertised maximum. Never lower
the logical limit to disguise an allocation saving. Preserve warmup identity
checks with the initial floor, and add explicit growth/lease-generation tests.
Partition selection must continue to depend on visible rows, not spare capacity.
Test 4K/8K/16K crossings, prefill chunks, rollback and latency spikes; avoid
shrinking/reallocating on every turn.

[MLX's cache implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
separates logical offset from growing allocation, supporting the design principle;
its tensor concatenation is not the proposed production mechanism.
[PagedAttention](https://arxiv.org/abs/2309.06180) addresses serving fragmentation
and sharing, while [vAttention](https://arxiv.org/abs/2405.04437) uses CUDA virtual
memory to grow physical backing behind contiguous addresses. Neither establishes
a benefit or an equivalent Metal API for this single local session. The proposed
bounded contiguous growth is an adaptation to this engine, not a novel-algorithm
claim.

**2. Remove genuinely unused scratch without repeating the rejected rewrite.**
The 284.23 MiB ordinary scratch includes a 241.03 MiB generic 128-row verifier
arena. Its principal components are 96 MiB private KV, 24 MiB accepted backups,
64.5 MiB attention partials, 41.75 MiB activations and 14 MiB graph head/selection
storage. Ordinary execution uses a separate selector and legacy width-two
backups. Deleting unused legacy intermediates (32.69 MiB) plus those unused
generic selectors/backups could remove about **70.69 MiB at F32** while retaining
command order. After F16 the incremental saving is about **57.69 MiB**, since
part of that storage has already halved; do not add the full F32 figure again.

The previous resource-only refactor already removed 178.49 MiB, passed native
checks, and regressed cooled generation 2.95% overall / 7.03% on the follow-up,
with first-text latency 5.58% worse. It changed activation aliasing and per-layer
ordinary private-KV publication. That is rejected evidence, not an untried easy
win. A narrowly justified allocation-only cleanup is possible, but secondary;
keep the existing execution and private-row lifetimes, and verify actual chat
performance independently. Do not begin another general workspace framework.

**3. Weights and loading have limited duplicate-removal headroom.**
`upload_model_weights()` already shares the canonical tied Q6 embedding/output
allocation. Packed body weights have one resident copy. Their source GGUF is a
read-only, reclaimable file mapping; mapping its full size does not mean another
full private resident weight copy. Packing creates one tensor-sized host staging
buffer at a time, followed by a Metal copy. Repacking directly into final shared
Metal storage could remove that staging/copy and improve startup peak/latency,
but cannot eliminate multiple GiB of steady-state duplicate weights.

Apple's [no-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
and [llama.cpp's Metal allocator](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)
show page-aligned host ownership and Metal views. Applying this here requires
correct lifetime, alignment and buffer-size handling. Canonical GGUF Q4 bytes do
not match the accepted columns8 packed layout. A persistent packed mapping would
also need authenticated source identity, versioned layout and immutable lifetime;
do not build a disk-cache subsystem without startup evidence. Private storage,
residency sets or a higher wired-memory limit do not themselves compress memory.

**4. Drafter and host ownership are secondary.** The current drafter is already
sized to the 16K session, not the stale 32K estimate in older sections below. It
owns 271.82 MiB weights, 111 MiB durable F32 KV, 2.25 MiB private KV, 22.40 MiB
scratch and a 1 MiB reverse map. Independently growing its three global caches
from 4K instead of reserving 16K could save 72 MiB; unlike the target, its
allocator explicitly initializes full allocations. Keep that a separate change.
Draft F16 would save only 55.5 MiB durable KV and reopens its earlier rounding
contract. H1 selector capacity could save roughly 7 MiB, and immutable tokenizer
sharing could remove some host copies, but those copies currently protect mapping
pointer lifetimes. Neither merits displacing target KV work. Saved token/RNG
histories are small and do not duplicate full caches.

**5. Q8 KV is a later precision option, not part of F16 completion.** A
[Q8_0 block](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
stores 32 signed bytes plus a two-byte scale. Applied to all target KV at 16K,
that would require 0.69727 GiB instead of F16's 1.3125 GiB: another **0.61523 GiB**
before alignment, scratch or residual regions. It needs native dequantization
inside online attention, explicit scale/rounding semantics and a new numerical
target. Do not materialize a whole F16/F32 cache or full attention score matrix
to obtain compressed storage. [MLX's quantized attention path](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py)
illustrates how changing storage can change the execution algorithm.
[KIVI](https://arxiv.org/abs/2402.02750) provides useful asymmetric key/value
grouping and recent full-precision residual ideas, but grouping/flushing adds
rollback and replay state, and its CUDA serving results do not forecast this
single-stream M4 Air. Aggressive low-bit KV, token eviction and smaller weight
quantization all change target numerics or context semantics; none is a free
lossless memory optimization.

### Speed expectations and current engineering evidence

F16 halves logical bytes per KV entry, not F32 attention arithmetic or the
dominant weight projection traffic. Older roughly 2K profiles put a hypothetical halving of
attention runtime at approximately **5–6% whole-generation throughput gain**;
that is a sensitivity calculation across prior measurements, not a prediction
that half storage halves attention time. Short contexts have less opportunity.
Longer contexts and memory pressure may benefit more, but native speed can also
be neutral or regress. The eight global layers still read growing history; F16
does not make exact full-context attention constant-cost per token.

Current [llama.cpp F16/quantized attention tuning](https://github.com/ggml-org/llama.cpp/pull/25750)
and its [master integration](https://github.com/ggml-org/llama.cpp/pull/26570)
reinforce that device, dtype, row count and context shape matter. Their M4 Max
and serving results are not M4 Air chat forecasts. In
[vLLM Metal's shared-KV verification work](https://github.com/vllm-project/vllm-metal/pull/534),
preserving each row's accumulation order was insufficient until partition-family
selection also matched; a separate ordinary specialization avoided a regression.
Its reported single-stream benefit was mixed. These are reasons to preserve
current numerical routing and test threshold crossings, not to reopen H3 or
replace the accepted H1 attention kernel as part of F16. External moving branches
were examined on 2026-09-29; record a concrete commit before importing any future
implementation detail.

### Complete implementation and evidence path

1. **Settle the numerical contract and implement the entire F16 target path.**
   Add independent conversion fixtures and engine precision plumbing, resolve
   before allocation, complete precise fused/unfused half attention, and route
   all ordinary/verifier/cache lifecycle and reporting through that policy.
   Preserve weights, F32 activation math, K/8 receipts, default H1 schedule and
   drafter precision. A half resident allocation while attention-visible private
   K/V remains F32, or while fusion is disabled, is not a completed implementation.
   Raw projected/normalized Q/K/V staging intentionally remains F32.
2. **Prove fidelity through existing test owners.** Extend
   `tests/tensor/gemma3_metal_tests.cpp` with precise, zero-softcap F16 attention:
   nonzero queries, all GQA heads, extreme finite scores, mixed row lengths,
   ancestors/sibling isolation, local wraps, partition boundaries, offsets,
   canaries and exact fused-sum receipts. Current half tests use zero Q,
   representable V and Gemma 2-style softcap/non-precise compilation; they are
   insufficient. Replace the blanket half-plus-sums rejection test while keeping
   mismatched cache dtypes, half receipts and invalid math-policy cases rejected.
   Use `scripts/gemma3_target_reference.py --cache-dtype f16` and an authenticated
   distinct manifest; make runtime test reads dtype-aware and compare all layers'
   raw half cache bytes where equality is required. Keep independent equation
   tolerances separate from exact ordinary/verifier same-conditional checks.
   Cover full/conditional and wider supported graphs, accept-zero/partial/all,
   residual/bonus/carry and KV-only endpoints, callback stop/throw, RNG rollback,
   reset, regeneration and context shifting. Retain probability-law tests and
   existing F32/Gemma 2/drafter bounds. Verify admission equals actual allocation,
   invalid policy, CLI success/failure and unavailable-device behavior. Run full
   native CI with authenticated checkpoint evidence after implementation.
3. **Measure the actual memory and speed tradeoff.** Validate representative
   fixed prefixes around the 1,024-row wrap, 2K, 8K and 16K, including filled
   history, finite cache/logits and numerical/probability drift. Use frozen
   representative chat/code prefixes to compare teacher-forced token likelihoods
   and sampler-support changes; a few matching argmax tokens do not establish
   preserved answer quality. Then compare
   ordinary F32/F16 and H1 F32/F16 through actual chat, preserving raw counts,
   durations, hashes, acceptance and first-text latency. Use fresh processes and
   at least 240 seconds idle before each measured chat, without builds or GPU
   work during cooldown. Differing natural-stop text is allowed but not an
   exact-work comparison; use fixed-work diagnostics for causal attribution.
   Record allocation bytes and native footprint at load/warmup/prefill/decode,
   plus compression/swap/page-in and thermal evidence. Measure growth candidates
   only if the existing physical backing justifies them. Do not advertise memory
   capacity reserved but never filled as long-context execution evidence.
4. **Make the retention decision explicitly.** The predicted allocation saving
   and passing numerical tests justify evaluating F16 as a memory policy, not
   calling it faster. A speed claim/default performance promotion still needs
   the repository's comparable-chat gate: at least 2% aggregate generation gain,
   both turns winning and no greater-than-2% other headline regression. If F16
   chiefly saves memory, report measured footprint and latency tradeoffs and
   retain it as an explicit option only if its numerical and lifecycle contract
   passes; keep automatic F32 until the default decision is supported. Evaluate
   growth or allocation-only cleanup separately, so a rejected memory ownership
   change cannot obscure a sound F16 result.

**Investigation stop condition:** findings and implementation boundaries are now
recorded; no production change is claimed. The proposed implementation ends at
a fully validated ordinary/H1 F16 policy and a measured retention decision.
Global capacity growth is the next conditional memory task, not an implicit
requirement to build paging, lower-bit KV, a new drafter or another kernel search.

## Completed: native-group Q4 rejected after full implementation (2026-09-29)

The user authorized the 32-value-group proposal below. It was implemented end
to end, passed native correctness checks, and lost in both projection streams
and cooled H1 chats. **Do not retain this kernel. The accepted producer-fused
Q4 sums remain the production baseline.** This result supersedes the proposed
experiment in the reassessment below. Implementation, comparison, restoration
and final native CI are complete; no further kernel variants,
attention work, F16 KV, H3 work or training are part of this baton.

### What was actually implemented

The compact Q4 R1/R2 core assigned complete native 32-value scale/minimum groups
to workers. It retained 128 threads, four independent block partitions,
two-column reuse, paired-row weight sharing and the existing final partial
transpose. Adjacent low/high groups exchanged alternating packed words with an
xor-4 SIMD shuffle, loading each encoded word once. One float4 accumulator per
row/column replaced the separate low/high accumulators, and each native group
received one scale/minimum correction. This was the proposed computation,
not the previously rejected eight-worker approximation.

RMS, GELU and attention producers emitted contiguous K/32 receipts without an
extra pass. Each receipt used four sequential eight-value sums followed by
`((p0+p1)+p2)+p3`; prepared and unprepared consumers matched that order. Host
capacity/alias validation and verifier allocation/admission accounting all used
the new size. Scratch ownership is in `src/engine/target_verifier.cpp`.
Canonical Gemma 2 arithmetic, weight layout, output head and F32 policy stayed
unchanged. Source audits covered SIMD participation, padded columns, inactive
block workers, uneven K partitions and producer head/row boundaries.

Native `MODE=release LOCAL_GGUF_GEMMA3_TARGET_GOLDENS=build/gemma3-target-reference
bash scripts/ci.sh` passed **478 C++ tests**, including the authenticated 12B
checkpoint, exact ordinary/verifier logits and all-layer KV comparisons,
rollback/stopping/callback behavior, independent decoded-weight checks and the
existing severe cancellation tolerances. The 60 Python checks passed with 14
optional skips. No numerical tolerance was weakened. For the recorded final
packed checkpoint distribution check, maximum logit error versus the CPU golden
was `6.2108e-5`, total variation `1.7478e-6`, with unchanged selected support.

### Projection evidence and bounded follow-ups

The actual-weight diagnostic compiled the frozen baseline and candidate shader
literals with production math and dispatch settings. It checked independent
F64 decoded dots and all-output finiteness. The stream retained all **288 actual
Q4 tensors / 5,292,195,840 packed bytes**, dispatching each once in model order.
It used synthetic F32 activations and precomputed receipts, excluded Q6/head,
attention and model dependencies, and is not a chat or physical-bandwidth claim.
One warmup pair and eight alternating timed pairs were recorded per row count.

| Prepared Q4 stream | Baseline | Native32 | Candidate time change |
| --- | ---: | ---: | ---: |
| R1 | 52.084 ms | 60.848 ms | +16.82% |
| R2 | 54.462 ms | 73.194 ms | +34.39% |

Two bounded controls did not rescue it. Declaring the actual 128-thread pipeline
maximum still lost the isolated prepared R1/R2 screen by 23.85%/29.22%.
Postponing nibble normalization, using masked encoded values like the accepted
kernel, still lost the resident stream by 11.91%/31.56%. That final decode
control was a source-literal kernel diagnostic with independent numerical
checks, not another fully integrated/chat-tested implementation. It did not
justify replacing the already tested candidate or starting another geometry
search. Its complete source is preserved alongside the main candidate.

The useful negative result is specific: fewer source accumulators and fewer
scale corrections did not improve M4 execution. Paired-row sharing was present,
but its measured amortization was worse. The new dot chains have eight updates
instead of four, and sharing quant bytes adds shuffles. These are source facts;
their individual performance costs, physical registers, spills and occupancy
were not measured. Do not invent a hardware explanation or conclude that all
future projection work is impossible.

### Cooled actual H1 chats: reject

One fresh process per executable, each after 240 seconds of idle cooldown, used
the canonical two-turn natural-stop chat, H1, lossless sampling, full schedule,
16K F32 KV and the same Q8_0 drafter. No builds or other GPU tasks ran during
either cooldown/chat. Candidate ran first; the saved accepted baseline followed.

| Complete generation | Accepted baseline | Native32 candidate | Throughput change |
| --- | ---: | ---: | ---: |
| First, 1,344 tokens | 77.5958 s / 17.321 tok/s | 90.6274 s / 14.830 tok/s | -14.38% |
| Follow-up, 1,024 tokens | 79.8270 s / 12.828 tok/s | 86.2939 s / 11.866 tok/s | -7.49% |
| Aggregate | 15.042 tok/s | 13.384 tok/s | **-11.02%** |

Both runs produced hashes `3a9063eeaaadedd9` / `ecc10d1344ecd5e7`, 805/644
cycles, 537/378 accepted edges, and matching context/work counts. This pair
therefore has matching executed chat workloads despite the allowed change in
floating reduction order. Target GPU time per cycle increased from
85.917/109.778 ms to 102.262/121.442 ms. User-visible first-text latency rose
from 374.460/711.332 ms to 385.033/745.934 ms. The separate CLI prompt/TTFT
field was 293.147/611.204 ms versus 292.677/637.268 ms; do not confuse it with
first-text latency.

Sampled native lifetime-high-water physical footprints were 11,173,179,008
baseline and 11,165,576,856 candidate bytes; the sampler may miss a later peak
after its last live query. System swap was already active and changed across the runs;
this was not a pristine empty-machine experiment. Native disk-read increases
after roughly 30 seconds were only 2.88 MB / 1.51 MB respectively, giving no
evidence of repeated multi-GB weight paging during generation. `pmset` recorded
no thermal/performance warnings; this does not establish constant GPU clocks.
The matching-work chat loss and independent stream regressions are sufficient
to reject, not a reason to repeat thermal bursts or tune smaller details.

### Final state, receipts and stop condition

All five changed production files were restored byte-for-byte from the
start-of-task snapshot. Only baseline-compatible test improvements remain:
distinct codes across all eight Q4 groups and four-byte-misaligned output/sum
views with 17-float guards on each side. Existing K/8 receipts, canonical bit
checks and tolerances remain intact. The restored release chat, benchmark,
inspector and test binaries rebuilt without warnings. Final native CI with the
authenticated target goldens passed all 478 C++ tests and the 60 Python checks
(14 optional skips), recorded in `ci-restored-native.log`. The rebuilt chat
executable is byte-identical to the accepted saved executable below. The only
source-controlled changes from this task are this document and
`tests/tensor/gemma3_metal_tests.cpp`; earlier user work was preserved.

Ignored artifacts live in `build/gemma3-q4-native-groups-2026-09-28/`:
`before-source/`, `candidate-source/`, the complete candidate executable/test
binary, both shader controls, independent probes, raw timings,
`projection-summary.json`, `chat-summary.json` and CI logs. Canonical chats,
native PID memory/I/O and thermal receipts are under
`benchmarks/results/gemma3-q4-native-groups-2026-09-28/` with the prefix
`native32-{baseline,candidate}-h1-full`. These are local evidence, not committed
benchmark output. Executable SHA256 identities:

- Accepted: `b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab`.
- Full native32 candidate: `ef31cd852a56d803fc25773fb147fce9235f766f853e6cff0ff480b2c1dea3af`.

This baton is closed. Preserve the useful H1 fused-sums baseline.
Further work needs a new scoped decision; this result removes the immediate
case for more native-group geometry tuning, not the case for useful speculation.

## H=1 optimization reassessment (2026-09-28)

The user requested another objective investigation after the implementation
audit. No implementation, training, model download or GPU benchmark was run.
The accepted producer-fused Q4 baseline remains unchanged. This section
supersedes the audit's broad attention-first recommendation: **target projection
work is the main general H=1 cost; the measured attention opportunity is mainly
at longer contexts.** H=1 already provides useful speculation. The remaining
problem is cost per useful emitted token, not a missing speculative mechanism.

### Current cost and realistic leverage

Recalculated from
`benchmarks/results/gemma3-h1-optimization-2026-09-28/fused-baseline-h1-full.json`:

| Per H=1 cycle | First turn | Follow-up |
| --- | ---: | ---: |
| Complete cycle | 96.534 ms | 129.340 ms |
| Target GPU, including publication | 85.879 ms | 114.507 ms |
| Draft wall | 8.735 ms | 12.551 ms |
| Remaining cycle wall | 1.919 ms | 2.282 ms |
| Accepted proposals | 66.71% | 58.70% |

Target GPU work is about 89% of the cycle. At unchanged workload and acceptance,
cutting target time 10% gives about 9.7% more throughput; halving all draft wall
time gives only 4.7–5.1%. Eliminating the remaining cycle wall gives 1.8–2.0%.
That last number excludes the draft's own host gap and publication GPU kernels;
it is not an upper bound on every conceivable scheduling change.

Current instrumented target R2 profiles, saved in
`build/gemma3-h1-optimization-2026-09-28/baseline-stage-summary.json`, put
projections plus head at 78.841 / 76.510 ms for 32 / 2,090 cached tokens.
Attention is 1.426 / 12.622 ms. Encoder instrumentation and sums of medians
prevent treating these as exact chat wall fractions, but establish the dominant
category. In particular, FFN gate/up plus down account for about 53–56 ms.

Scheduled quantized body/head weights total 7.434 GB. At Apple's advertised
[120 GB/s M4 Air bandwidth](https://support.apple.com/en-gb/122209), an ideal
single read takes about 61.95 ms. This is a conditional streaming estimate,
not measured physical traffic or proof of bandwidth saturation. Compared with
the projection profile, it argues against expecting a twofold reduction from
kernel tuning alone. Cache behavior, duplicated traffic, decode instructions,
latency hiding and sustained clocks remain unmeasured at the hardware level.

### Recommended general-H=1 experiment: native 32-value Q4 groups

There is one concrete projection hypothesis that differs from the failed
experiments. Keep the current 128 threads, two output columns per worker,
eight columns per group, four block-worker partitions and R1/R2 weight reuse.
Change what each worker computes: a whole native Q4_K 32-value scale/minimum
group instead of four eight-value fragments from different groups.

For one group, the supplied quantized weights obey the real-arithmetic identity:

```text
sum_i x_i * (d * s_g * q_i - dmin * m_g)
    = d * s_g * sum_i(x_i * q_i) - dmin * m_g * sum_i(x_i)
```

The accepted kernel already factors its arithmetic, but repeats scale/minimum
corrections across four fragments of each group to retain the old reduction
layout. A whole-group worker could reduce four corrections to one and retain
one float4 dot accumulator per row/column instead of separate low/high float4
accumulators: 16 rather than 32 source-level partial components for R2. Retain
producer-fused sums, adapted to one sum per 32 values (K/32 rather than K/8
floats per row). No extra preparation dispatch should be introduced.

Unlike the rejected eight-worker prototype, this keeps four-way block
parallelism and does not serialize all blocks onto eight physical workers.
The existing final partial transpose can initially remain; removing it is not
a prerequisite for testing the intended group arithmetic. This is an
engine-specific design hypothesis, not a claim of worldwide novelty or speed.

The important obstacle is real: adjacent low/high groups share encoded quant
bytes. Naively assigning complete groups can duplicate loads. With the
existing column-pair lane mapping, adjacent groups occupy lanes differing by
four; one packed load can be shared with `simd_shuffle_xor(..., 4)`. That trades
loads for shuffles and requires a uniform, tail-safe schedule. Compiler
register allocation and instruction issue can still make it slower. The
previous failures rule out assuming that fewer source accumulators imply a win.

The existing `kquant_columns8` packing already contains the required group
codes and scale/minimum pairs; no new resident weight representation is needed.
A smaller independent compiler check is to declare the actual 128-thread
maximum for compact projection pipeline compilation. Their current convenience
pipeline creation does not supply that bound. Apple's
[Metal compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/)
describes this resource hint. Scope it to those fixed-geometry pipelines, not
256-thread producers or other kernels. It may help, do nothing or regress;
it is not evidence that the baseline spills, or a substitute for the group
arithmetic experiment.

Relevant implementation boundaries:

- `src/tensor/backends/metal_backend.mm`: Q4 core at
  `gemma3_q4_stream_packed_rows`, packed wrappers, fused RMS/GELU/attention sum
  producers and their common group definition.
- `src/tensor/backends/metal/metal_command_context.mm`: prepared-sum capacity
  and dispatch validation; `src/engine/gemma_target_runtime.cpp` and
  `src/engine/gemma2_metal_plan.cpp`: sum routing and scratch ownership.
- `tests/tensor/gemma3_metal_tests.cpp`: independent decoded-weight parity,
  all scale/minimum regions, cancellation, uneven K workers and N tails,
  producer receipts and R1/R2 equality. Actual-model logits/KV and lifecycle
  checks must still pass through the public target/verifier paths.

The new floating reduction order must be shared by ordinary and verifier rows,
including prepared/unprepared and shortened/wider execution paths. Preserve
the supplied weights and F32 policy. Old output-bit identity is not the oracle;
independent numerical accuracy, consistent target evaluation and the lossless
sampling law are. Do not weaken a tolerance to accept a failed implementation.

Before attributing any gain to registers, use physical evidence if available.
The machine currently selects `/Library/Developer/CommandLineTools`, and the
existing device probe exposes timestamp counters only. Full Xcode's
[GPU profiler](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance)
offers occupancy, limiter and bandwidth inspection; no full Xcode installation
was found under `/Applications` during this investigation.
Do not build a new profiling framework or claim those counters were captured.
An honest alternative is a bounded diagnostic of the actual candidate on real
projection shapes, followed by a full model projection stream and cooled chat
only if the mechanism has plausible whole-path benefit. Cache-hot isolated
tensor results cannot establish DRAM behavior or a retained chat improvement.

### Attention is a separate long-context opportunity

Gemma 3 12B has 40 local-window layers and eight global layers. Reweighting the
saved actual-head-shape synthetic attention probes changes the interpretation:

| Context regime | Old attention, 40 local + 8 global | Paired blocked candidate | Estimated saving |
| --- | ---: | ---: | ---: |
| About 2K | 13.034–13.537 ms | 12.516–12.741 ms | 0.294–1.021 ms |
| 8K | 30.977–31.480 ms | 21.121–21.346 ms | 9.632–10.358 ms |

These are sums of saved synthetic layer medians for aligned, shifted and
wrapped local windows, not real-chat predictions. Source:
`attention-reduced-probe.jsonl` and `attention-probe.mm` under the experiment
build directory. At 2,090 global keys the paired candidate is actually slower
than old R2. Its combined chat follow-up win cannot be causally explained by
these attention measurements alone.

For 8K+ conversation speed, use blocked sharing only for long global rows,
leaving local and shorter global attention on the existing algorithm. Global
layers alone contribute 9.147 ms of the synthetic 8K saving; the corresponding
blocked scalar probe is approximately neutral there. Select arithmetic by
each row's visible history, identically in ordinary and verifier execution.
At a threshold crossing, evaluate the two rows separately if their policies
differ. This solves the previous global scalar-rollout problem without changing
the numerical target according to whether speculation is enabled.

Close the audit's paired sibling/extreme-score and long full-model logits/KV
gaps, and test an actual long conversation before claiming a gain. Dense global
attention still scales with history; this does not make per-token work constant.

Fresh engineering evidence reinforces both the mechanism and its limits:
[vLLM Metal PR #534](https://github.com/vllm-project/vllm-metal/pull/534), merged
July 24, shares K/V across two-row windows with preserved per-row accumulation,
and corrected a partition-policy parity problem. It also needed a separate
ordinary specialization after a shared body regressed decode. Single-stream
results were mixed across chips; concurrency wins are not evidence for this
Air. Its streamed K fragments and threadgroup-resident Q are useful alternatives
if register pressure is established, not a reason to restart a broad kernel sweep.

F16 KV remains compatible with this architecture and saves about 1.3125 GiB
at 16K target capacity. It is a useful memory-headroom option, not a fix for
weight-projection cost. It changes target rounding, requires matching
prefill/ordinary/private-row/commit behavior, and must retain F32 accumulation
and fused-sum production. It is not merely an allocation-type switch or
distribution identity with the existing F32 target.

### What to optimize next if target kernels do not earn retention

Prefer improving the existing drafter's agreement with the target over another
controller/sampler rewrite. For H=1, useful progress is approximately `1 + A`
per cycle. A same-cost ten-percentage-point acceptance gain means about 6.0–6.3%
more throughput. Perfect acceptance at unchanged cost would give 20.0–26.0%;
neither improvement is promised, and higher acceptance also changes the mix
of one/two-row draft ingestion. This reduces expensive target passes per output
token even though each individual verification remains expensive.

[LK Losses](https://arxiv.org/abs/2602.23881) and the September 21 preprint
[Acceptance-Aware Draft Model Training](https://arxiv.org/abs/2609.24150) study
objectives matched to speculative acceptance. They support examining alignment,
not assuming a gain on this Gemma pair. A future pilot should keep the complete
270M architecture, use the actual quantized target's sampling distributions,
preserve tokenizer mapping and evaluate the exported Q8 draft's actual integer
law on held-out conversations. Compare a conventional distillation control with
an overlap-aware objective; avoid assuming a new loss is universally better.
Do not restart the earlier low-rank/shortlist architectures or launch substantial
training automatically. Training remains user-run.

There is also a real draft runtime opportunity: `gd_q8` in
`src/engine/gemma3_draft.mm` processes the known pending-token/root rows
independently. R2 weight/quant-decode sharing could help. However, the entire
drafter is only 9–10% of cycle time, the head already projects just one row,
and no current draft-stage profile establishes the addressable body time.
Its weight-byte fractions must not be presented as time fractions.

The code and measurements deprioritize:

- **Deleting a hidden forward or accepted-prefix replay:** neither is present.
  Pending/root ingestion is already batched, final proposals remain pending,
  accepted target K/V is published, and bonus candidates are reused.
- **Delaying the bonus head:** scalar/joint head diagnostics are 8.0104/8.1046 ms.
  Conditional separate heads cost about 12.7–13.4 ms at current acceptance,
  before extra synchronization. The second head row is already very cheap.
- **Sampler launches and broad CPU scheduling work:** limited exposed cost;
  the actual launch-removal experiment did not establish retained improvement.
- **Generic asynchronous drafting:** next-root dependencies and shared GPU
  bandwidth remain; multiple-device paper results do not remove them.
- **More horizons or speculative trees:** prior H3 work did not earn its cost;
  do not reopen it without a materially changed cost/acceptance balance.

For any cheaper proposal mechanism, the break-even condition is
`A_new > (1 + A_old) * C_new / C_old - 1`. Even halving all current draft wall
time permits only about 7.5–7.7 percentage points of acceptance loss. A cheap
proposal must report its actual q; vocabulary restriction or a stale proposal
distribution cannot be credited without this tradeoff.

**Recommendation:** the next general-speed implementation, if chosen, is the
whole-group Q4 hypothesis above, with fused sums retained and actual chat as
the decision point. Long-global attention is the stronger measured option
specifically for long conversations. If the projection experiment fails,
prioritize a same-architecture drafter-alignment pilot over further geometry
tuning. This investigation is complete; no implementation or training is active.

## H=1 implementation fidelity audit (2026-09-28)

The subsequent audit compared the original four proposals with preserved
source, production dispatch, tests, frozen executables and raw measurements.
**The experiments were real, but the earlier claim that all designs were
complete end-to-end implementations was too broad.** In particular, the second
Q4 experiment did not implement the proposed new reduction topology. Failed
retention screens do not establish that these mechanisms were fully explored,
optimally implemented or inherently unsuccessful.

### Findings that qualify the experiment report below

1. **Q4: one faithful implementation and one narrower shader prototype.**
   The one-column/256-thread variant implements the first proposed choice,
   including its predicted extra threads and repeated loads. The eight-K-worker
   variant instead retains four virtual old block-worker sums per row, the old
   low/high float4 accumulators, and the old 32-partial reduction. See
   `build/gemma3-h1-optimization-2026-09-28/q4-eight-lane-core.metal:29`, `:43`,
   `:83` and `:97`. It serializes block work while preserving the old arithmetic;
   it is not the proposed shorter-fragment/new-reduction design. That proposal
   explicitly allowed a floating reduction-order change shared by R1/R2,
   subject to independent numerical validation. Requiring old output bits was
   an unnecessary additional restriction. The eight-worker candidate ran only
   in the isolated shader probe, without production 64-thread dispatch
   integration or its own inactive-worker/column-tail coverage. Its real
   slowdown rejects that exact-order prototype, not the unimplemented design.
   Both Q4 timings are shape-weighted isolated tensor probes, not a complete
   model projection stream. Neither measured physical registers, spills or
   occupancy, so the proposed bottleneck was never established at that level.

2. **Attention: faithful sharing, with an expensive scalar rollout and gaps in
   paired validation.** The candidate really shares committed K/V across two
   rows and two GQA heads, uses stable blocked F32 softmax, and preserves each
   query's visibility and partition ownership. No concrete arithmetic or
   cache-source defect was found. However, it replaces all F32 scalar attention
   with the blocked algorithm; the >=1,024-key threshold gates pairing only.
   At 2,090 keys, the scalar probe takes 1.227 times the old scalar time. This
   gives a concrete reason to investigate the ordinary regression; it does not
   isolate its whole-chat cause. The paired implementation is faster than
   unpaired blocked execution, and at 8K takes 0.566 times old R2 time.
   No actual 8K conversation was measured: candidate H=1 ends at 2,403 tokens.
   The final sibling/extreme-score test uses only 1 or 34 visible keys and
   therefore bypasses pairing after the threshold change. Long synthetic
   paired tests do cover chain ancestry, ring wrap, windows, mixed partitions,
   scalar equality and an independent double reference. Actual-model goldens
   use positions 0–3 and do not exercise paired execution. Matching chat token
   hashes are useful evidence, but do not replace a long-context full-logit
   parity check. The saved 32/2,090-token full-model profiles are baseline
   profiles, not candidate parity results.

3. **Packed Q6 head: faithful to the proposed layout substitution.** It packs
   directly into the final allocation, preserves tied embedding/head ownership,
   implements the packed token gather and reaches the existing compact head
   kernel. No duplicate resident head or simplified arithmetic was substituted.
   Independent embedding parity, ownership/admission tests and actual-model
   goldens passed. The whole-head diagnostic found no useful gain for this
   layout/kernel combination. It did not measure startup or embedding/prefill
   performance, although the original proposal requested those checks. The
   small R2 timing difference is not proof that packed heads inherently lose,
   or that the reused compact kernel is optimal.

4. **Deferred sampler: faithful normal-path launch removal.** Source tracing
   and real-chat counters agree: 18 fewer dispatches/barriers per proposal,
   four extra readback bytes, unchanged common-path command/commit/wait counts.
   Overflow preserves the same random word, sorted tiles and integer-CDF law,
   completes the second submission before publication, and invalidates on
   failure. No new functional defect was found. Lower-level tests cover
   256/257-support boundaries, actual proposal/overflow execution, transactions
   and continuation. There is no dedicated public sampled-H=1 overflow plus
   callback-cancellation/continuation test; existing callback tests take other
   paths. The injected second-command failure is an allocation seam, not a GPU
   completion failure. Its stale-lease assertion uses an already invalidated
   lease and does not independently prove failure-specific invalidation. These
   are coverage limitations, not discovered runtime failures. The failed chat
   screen also includes a 3.56% increase in unchanged target follow-up GPU time;
   it cannot establish that launch removal caused the slowdown.

### Audit evidence and boundary

All 242 files in each of the before/combined/sampler source manifests were
rehashed without mismatch, as were the three corresponding frozen executables.
Packed-head files match their narrower manifest. All six chat runs agree across
expected, launch and reported executable SHA, exit successfully and record the
240-second cooldown. Host dispatch code and counters support actual candidate
activation. Reported throughput calculations reproduce from raw token counts
and complete-generation durations. No evidence of a wrong executable or
invented timing was found; thermal, page-in and causal-attribution limitations
remain as recorded below. No GPU performance counters established optimality.

The accepted fused-sums production source and release executable are unchanged.
This audit made documentation corrections only; no new build, GPU test or
benchmark was run. Detailed attention and sampler audits are preserved as
`audit-attention.md` and `audit-sampler.md` under the experiment build directory.

For a subsequent implementation decision, attention remains the clearest
measured opportunity: close the paired validation gaps and evaluate a selective
policy that does not impose the slower scalar algorithm everywhere. Each row
must choose the same numerical policy in ordinary and verifier execution,
including threshold crossings. The true Q4 shorter-fragment reduction remains
unimplemented; its register/issue bottleneck needs evidence before another
design is selected. The sampler has a much smaller addressable cost. These are
recommendations, not newly started implementation work. **The requested audit
is complete; stop here.**

## H=1 optimization experiment results (2026-09-28)

All four authorized areas were explored against the accepted producer-fused
Q4 baseline, with the implementation and validation limits above.
**None established all performance retention gates.**
The runtime source is restored byte for byte to the start-of-task snapshot;
fused sums remain the production baseline. The useful Q4 worker/column-tail
regression test is retained. Final release CI passed **478 C++ and 60 Python
tests**, including native Metal and the independent model goldens. The normal
release executable again matches the accepted baseline SHA below. This baton
ended with the limitations identified by the subsequent audit above. Further
kernel search, H=3 performance work, precision
changes, training and model downloads are outside the completed scope.

### Decisions

| Implemented design | Evidence | Decision |
| --- | --- | --- |
| Blocked precise F32 attention, with shared H=1 prefix reads | Combined with deferred sampling: H=1 aggregate throughput +4.34%, but ordinary follow-up throughput -4.30% | Archived and removed: shared scalar path fails the ordinary regression guard |
| Sole compact Q6 tied head/embedding, packed directly into final allocation | Actual complete head: R1 8.0104 → 8.1778 ms; R2 8.1046 → 8.1443 ms; exact logits/embedding parity | Removed: no useful head gain or persistent memory saving |
| Q4 one-column implementation and exact-order eight-worker probe | Both preserve exact arithmetic but synthesized shape-weighted R2 fused-receipt times regress 32.10% / 27.58% | Both removed; the proposed shorter-fragment/new-reduction design remains unimplemented |
| Deferred H=1 compact-sampler fallback | Isolated cooled chat: first +1.13%, follow-up -3.32%, aggregate -1.23% throughput versus fresh baseline | Removed: fewer dispatches did not establish a retained chat gain |

The implemented candidates and shader prototypes are preserved as reviewable
local artifacts. Their scope differs as documented in the audit above.
No correctness tolerance was weakened to make them pass.
Do not mistake the attention candidate's H=1 win for an accepted production
change, or the sampler screen for proof that its mechanism inherently slows
all workloads. In its isolated follow-up, unchanged target GPU time also rose
3.56%; system/thermal variation contributes uncertainty.

### Cooled real-chat evidence

Every run uses a fresh native process after at least 240 seconds of idle, no
concurrent builds/GPU probes, the same two naturally stopping sampled prompts,
seed 42, 16K F32 KV, and no retrieval. H=1 uses the existing 270M Q8_0 drafter,
full verification and lossless sampling. The clock below is complete generation
(`generation_completion_ns`), not the CLI's shorter generation interval.

| Run | First tokens / seconds | Follow-up tokens / seconds | Aggregate tokens/s |
| --- | ---: | ---: | ---: |
| Initial fused H=1 baseline | 1,344 / 78.023646 | 1,024 / 83.622179 | 14.649311 |
| Blocked attention + sampler, H=1 | 1,344 / 77.273028 | 1,024 / 77.649886 | 15.285021 |
| Fused ordinary baseline | 1,506 / 125.413716 | 1,026 / 111.794008 | 10.674189 |
| Blocked attention, ordinary | 1,506 / 125.424184 | 1,026 / 116.816661 | 10.452407 |
| Fresh fused H=1 confirmation | 1,344 / 79.079836 | 1,024 / 85.199355 | 14.414485 |
| Sampler-only H=1 | 1,344 / 78.196631 | 1,024 / 88.124769 | 14.237494 |

Within each mode, counts and selected-token hashes match across every run.
H=1 hashes are `3a9063eeaaadedd9` / `ecc10d1344ecd5e7`, with 805 / 644 cycles
and 537 / 378 accepted proposals throughout. Ordinary hashes are
`260ae35592bb6060` / `dd4aadff1d4f599f`. H=1 and ordinary have different
trajectories; do not call their cross-mode throughput an exact-work comparison.
Identical sampled prose is supporting provenance, not the lossless law's proof.

The combined H=1 candidate wins +0.971% / +7.691%; ordinary changes -0.008% /
-4.300% (-2.078% aggregate), failing the <=2% regression requirement. The
isolated sampler changes +1.129% / -3.320% (-1.228% aggregate) versus the fresh
baseline and -2.811% aggregate versus the original baseline. It neither wins
both turns nor reaches the required 2% aggregate improvement.

Latency does not rescue these generation results. First visible text for the
initial H=1 pair is 376.082 / 754.726 ms versus 382.417 / 753.800 ms (+1.685% /
-0.123%). For the fresh sampler pair it is 374.080 / 722.048 ms versus
374.195 / 697.550 ms (+0.031% / -3.393%). Corresponding reported first-selection
TTFT is 293.823 / 654.034 versus 296.097 / 654.236 ms, and 294.563 / 621.155
versus 295.216 / 595.665 ms. Ordinary first-visible latency improves while its
follow-up generation regresses. Preserve both latency definitions.

Measured native lifetime physical-footprint peaks are 11,163,053,720 /
11,166,609,024 bytes for the initial H=1 pair, 10,604,489,720 / 10,634,701,840
for ordinary, and 11,178,847,872 / 11,235,290,800 for the isolated sampler pair.
These are native `proc_pid_rusage` values, not wrapper RSS or the outer Python
process's `/usr/bin/time` footprint. Startup page-ins, compression and system
swap conditions differ. The initial H=1 baseline reads 14.683 GB versus the
combined candidate's 7.482 GB, concentrated near startup; the fresh baseline
reads 8.872 GB and sampler-only 9.008 GB. There are no absolute model-ready/turn
boundary timestamps to prove those I/O windows precede all generation.
`pmset` records no warning, but does not measure temperature or GPU clocks;
none of this establishes equal thermal states or absence of throttling. The
confirmation did not erase the initial H=1 win, but it is not a repeatability
study or an isolated attribution of that win to either mechanism.

### What the implementations established

Blocked attention uses absolute 32-key tiles, one stable output correction per
tile, independent row masks/partitions, and shared committed K/V reads. Q/K/V,
scores, probabilities and accumulation remain F32 with precise math. Scalar
and paired per-query arithmetic is identical, including mixed partitions,
shifted windows, ring wrap and private ancestors. Retaining SIMD0 state in
registers reduces static threadgroup staging to 6,192 bytes for R1 and 12,384
for R2. Paired execution requires both row counts >=1,024; shorter rows use
separate groups without another dispatch. Synthetic 8K paired attention takes
0.566 times the old kernel's time, but scalar and shorter-context results are
mixed. That is a kernel diagnostic, not a proven long-conversation chat gain.
A future selective policy would need consistent per-row arithmetic across
ordinary/verifier execution and threshold crossings; do not apply a batch-wide
cutoff that silently changes an individual row's numerical policy.

The head already amortizes its second row: canonical R2 is only about 1.2%
slower than R1 in the complete-head probe. Packed execution adds a transpose
and barrier without a measured benefit. Each representation is 825,753,600
bytes; both baseline and candidate own one in production. Direct-packing APIs,
packing-only tests and the unmeasured body-upload adjunct were removed too.

The Q4 candidates reduce source-level per-thread live state while preserving
block streams, factored scales/minima and final reduction order. Both produce
zero bit mismatches in all 28 actual-shape/row/receipt probe combinations.
Their timing regressions do not identify actual spills, registers or occupancy;
those physical counters were not measured. The eight-worker probe preserves
the old reduction rather than implementing the proposed new one. No third
geometry was attempted.

Deferred sampling removes exactly 18 dispatches/barriers per H=1 proposal,
with unchanged common-path command counts and four additional readback bytes.
Overflowing cutoff ties reuse the same word and sorted tiles in a second
submission after the existing completion boundary. No unresolved pending
history or verifier request is published. Independent integer-CDF checks,
256/257-support boundaries, mapped controls, pending/root batching, commit,
abort, sampled H2/H3, and injected second-submission failure all passed.
The common-path dispatch reduction is real; it did not establish a whole-chat
benefit worth retaining its additional exceptional-path machinery.

Current fused-baseline instrumented R2 profiling at 32 / 2,090 cached tokens
measures attention 1.426 / 12.622 ms, FFN gate/up 34.845 / 32.867 ms, FFN down
20.862 / 20.545 ms, Q/K/V 10.192 / 9.999 ms, attention output projection
5.050 / 4.974 ms and head 7.892 / 8.124 ms. This replaces the older ordinary
profile as current attribution. Instrumentation changes encoder boundaries;
sums of category medians are not uninstrumented chat durations. Most target
cost remains projections; this pass did not reveal a redundant forward pass
or a cheap layout change that removes it.

### Verification, preservation and stop condition

The combined attention/sampler source passed 41 focused native tests and full
release CI: 481 C++ / 60 Python tests, including authenticated independent 12B
target and 270M drafter goldens. Those short-position target goldens do not
cover paired attention; see the audit's test-dispatch and lifecycle gaps above.
The packed head and first Q4 candidate passed relevant native parity before
performance rejection; the second Q4 candidate has isolated shader parity.
Final restored-tree CI
passed **478 C++ / 60 Python tests** with the exact accepted release executable;
see `ci-final-accepted-executable.log`. The one retained test covers inactive/uneven K partitions and
N=1/7/8/9 tails, canonical/compact/prepared R1/R2 equality, and independently
constructed fused-sum receipts in `tests/tensor/gemma3_metal_tests.cpp`.

All evidence is ignored under `build/gemma3-h1-optimization-2026-09-28/` and
`benchmarks/results/gemma3-h1-optimization-2026-09-28/`. The latter contains raw
chat JSON, resource samples, launch/SHA manifests and logs for prefixes
`fused-baseline-h1-full`, `attention-sampler-candidate-h1-full`,
`fused-baseline-ordinary`, `attention-sampler-candidate-ordinary`,
`confirmation-baseline-h1-full`, and `sampler-only-candidate-h1-full`.

The build directory preserves `before-source/` and `before-identity.json`,
`attention-sampler-source/` and its identity manifest, `sampler-only-source/`
and its identity manifest, `packed-head-candidate-source/`, both Q4 shader
variants, all probes/profile tools, and decision/restoration receipts.
`kernel-decisions.md`, `h1-chat-comparison.md` and
`sampler-chat-comparison.md` contain detailed artifact-level analysis.
Frozen chat executables are `baseline-generate` (SHA-256
`b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab`),
`attention-sampler-generate`
(`0b34de1d6a3339f02838e50dd83c6980485b1500e0883457defc03099346be39`), and
`sampler-only-generate`
(`cd55aac9d2dcc978aa293c9a5104ecd0a63113ad2135fdd8fac74c9a0c0da265`).
No rejected runtime code or feature-only tests remain in the working source.

The final source/executable restoration and full CI are complete. Stop here.
The next decision belongs to the user: the preserved H=1 attention candidate
has a concrete ordinary-decoding tradeoff; further long-context specialization
would be new implementation work.

## Current state: accepted fused-sums H=1 baseline (2026-09-28)

Producer-fused Q4 sums are now **retained in production**, following the user's
explicit decision after personal testing. This supersedes the historical
non-retention decision below. The normal release executable contains the exact
candidate the user tested; no experimental executable or fused-sums flag is
required. Ordinary decoding remains the CLI default, and the existing explicit
270M / H=1 / full / lossless options select speculation. H=3 remains set aside.

Restored the candidate's eight production files and its deterministic tensor
tests from `build/gemma3-block-and-h1-2026-09-28/h1-source/`, checking source
identities before replacement. Expanded callback tests and all other existing
development were preserved. Full release CI passed **477 C++ and 60 Python
tests**, including native Metal and the available independent 12B/270M goldens.
The rebuilt `build/release/local-gguf-generate` matches the tested candidate:
SHA-256 `b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab`.
Adoption receipts and CI output are in
`build/gemma3-h1-deep-dive-2026-09-28/`.

RMS normalization, GELU and attention output now produce reusable Q4 minimum
correction sums while their values are already live. R1/R2 compact projections
consume them without another dispatch, barrier or wait. The sums preserve the
original consumer's addition order; they are not approximate quantization.
Scratch is bounded by two rows and the largest input width, about 15 KiB per
workspace for this target. Q6 and matrix prefill do not consume these receipts.

### Additional evidence from the user's chat

The attached transcript runs the fused candidate before the previous baseline.
The first three prompts, output text and token counts match. The fourth prompt
has different spelling and produces different token counts, so exclude it from
the matched comparison. These are the CLI's generation intervals, not the
benchmark harness's complete-generation durations.

| Matched turn | Tokens | Fused generation s | Previous generation s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| Greeting | 17 | 1.141149 | 1.159207 | +1.58% |
| Gothenburg | 1,048 | 60.711351 | 60.258080 | -0.75% |
| Fish/shrimp population | 900 | 55.712209 | 60.633577 | +8.83% |

Aggregate matched generation improved **3.82%**. Printed TTFT was
895.154 / 434.915 / 588.652 ms versus 2,023.757 / 2,288.027 / 575.792 ms.
The transcript does not establish cooldown, thermal state or run variance.
It supports the user's baseline decision without establishing a universal
kernel gain or validating the earlier 36.45% observation as causal.

## Preserved pre-experiment H=1 design rationale

The code, preserved experiments, current upstream implementations and papers
were examined. The recommendation is to improve the **target's actual two-row
GPU execution**, starting with blocked F32 attention for longer conversations.
Packed execution of the tied Q6 head is the next concrete layout opportunity;
Q4 live-state reduction has a larger addressable body but more uncertain payoff.
An H=1-specific sampling fallback change was proposed to remove redundant work
as a secondary improvement. The following is the original design rationale;
the experiment results and subsequent fidelity audit above supersede its
proposed ordering and distinguish measured work from unfinished proposals.
No training or model downloads were performed.

### Actual cost and existing work reuse

Preserved `fused-sums-h1.json` in
`benchmarks/results/gemma3-block-and-h1-2026-09-28/` gives:

| Per-cycle quantity | First turn | Follow-up |
| --- | ---: | ---: |
| Complete cycle | 95.099 ms | 124.819 ms |
| Target GPU | 84.941 ms | 110.779 ms |
| Draft wall | 8.744 ms | 12.514 ms |
| Remaining cycle time | 1.414 ms | 1.526 ms |
| Accepted proposals | 66.71% | 58.70% |

About **89% of cycle time is target GPU work**. With unchanged acceptance,
halving all drafting would improve throughput about 4.8–5.3%; cutting target
time 10% would improve it about 9.8%. Removing all remaining overhead has only
about a 1.2–1.5% ceiling. These are fixed-trajectory sensitivity calculations,
not performance forecasts. The recorded `selection_ns` excludes the complete
standalone rejection callback; do not present it as total acceptance cost.

The older ordinary stage profile put roughly 74 ms in projections/head, versus
about 1 ms of short-context attention and 10 ms near 2K context. It predates
compact packing and fused sums, and its encoder boundaries affect timings.
It identifies categories worth inspecting, not current H=1 stage percentages.
Current resident Q4 body weights occupy 5,292,195,840 bytes, Q6 body weights
1,316,044,800 bytes, and the tied Q6 head 825,753,600 bytes. Byte shares are not
execution-time shares.

The source audit found no hidden full forward pass to delete:

- `Gemma3DraftPlan::propose` batches a pending draft token with the known root
  where necessary and leaves its final proposal pending. H=1 does not forward
  that final proposal unnecessarily.
- `encode_gemma2_metal_target_verifier` executes the body once for root + proposal and
  projects both head rows together. The canonical Q6 head already selects
  `gemma3_q6_shared_rows<2>`; it is not two independent scalar head passes.
- Accepted target rows are published from private K/V, not replayed. The bonus
  distribution is reused. Draft q gathering reads the completed shared integer
  CDF and adds no GPU submission or wait.
- The approximately four waits per cycle correspond to draft proposal, target
  verification, target-prefix publication and draft-prefix publication.
- Two GQA query heads already share each K/V load. The missing attention reuse
  is between H=1's two positions, not between those existing GQA heads.

For fixed p and actual q, H=1 acceptance is already maximal:
`A = sum_v min(p(v), q(v)) = 1 - TV(p,q)`. A different lossless acceptance test
cannot raise it without changing q. This follows from the original
[speculative sampling construction](https://proceedings.mlr.press/v202/leviathan23a.html).
Perfect acceptance at unchanged recorded cost would improve throughput about
20% / 26%; a trained better drafter could matter, but it would not remove the
dominant target cost.

### 1. Blocked F32 attention with shared H=1 prefix reads

**Verified implementation issue:** `verifier_attention_cache_kernel` assigns a
separate threadgroup dimension to each verification row. Within its nominal
32-key block it still updates the online softmax and rescales the full output
accumulator after every individual key. Adjacent H=1 rows separately read the
same committed K/V prefix. This is real repeated work that fused Q4 sums does
not address.

**Source-backed mechanism:** current llama.cpp vector attention computes a
tile maximum and exponential sum, then rescales accumulated output once per
tile. Inspect the pinned
[normalization and value loop](https://github.com/ggml-org/llama.cpp/blob/f1ea206218210afb913ae2f5d2c51faed35915da/ggml/src/ggml-metal/kernels/fa_vec_common.metal#L399).
Its [F32-cache specialization](https://github.com/ggml-org/llama.cpp/blob/f1ea206218210afb913ae2f5d2c51faed35915da/ggml/src/ggml-metal/kernels/fa_vec_f32.metal#L14)
still stages Q as half, and its dimension-256 geometry differs from smaller
heads. Copy the data flow, not its precision shortcuts or every launch shape.
[Metal Flash Attention](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel%2BSoftmax.swift#L270)
independently uses blockwise output correction. Its
[precision policy](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionDescriptor/AttentionDescriptor%2BPrecisions.swift#L107)
supports all-F32 execution, but its matrix configurations do not prove a
four-effective-query decode speedup here.

The original [MTPLX packed GQA kernel](https://github.com/youssofal/MTPLX/blob/1de2b1c049136ed117af0c6712baaadd81820b51/mtplx/kernels/sdpa_gqa_packed.py#L135)
also shares K/V across query positions, separates the common prefix from the
causal tail, and removes per-row visibility branches from that prefix loop.
It supplies useful implementation evidence for the sharing mechanism. Its
BF16/F16 inputs, fast exponentials and per-token softmax are not the proposed
F32 blocked implementation; local window boundaries still need engine-specific
handling.

For one query and a score tile S, retain running maximum M, denominator L and
unnormalized output O. Compute:

```text
M' = max(M, max_i S_i)
c  = L == 0 ? 0 : exp(M - M')
e_i = exp(S_i - M')
L' = c * L + sum_i e_i
O' = c * O + sum_i e_i * V_i
output = O / L
```

Handle entirely masked tiles explicitly. Keep Q, K, V, scores, probabilities
and accumulation F32 under the existing precise math policy. This reduces
normalization dependencies and whole-output rescaling, not the required dense
attention interactions. It is different from the already rejected `exp(0)`
guard experiment.

**Proposed engine adaptation:** organize the two positions and their two GQA
heads as four effective queries over a shared committed prefix. Share K/V
loads while keeping four independent scores, masks and normalization states.
For global layers the committed prefix coincides. For local layers, windows
can differ by one position: share the intersection, and handle the leading
boundary and private causal tails explicitly. Row 0 sees the root; row 1 sees
root + proposal. Never apply one row's schedule to both.

The ordinary R1 and speculative R2 paths must use the same logical tiling,
per-query arithmetic and partition/reduction policy, including at thresholds
where adjacent rows have different partition counts. Fixed absolute-position
tile alignment is one candidate; validate it against ring wrap and private
overlays. Do not preserve an inefficient ownership layout merely because it
is old, or introduce a verifier-only numerical policy to claim parity.

Resource design matters on M4. A full 32-by-256 F32 K tile alone occupies
32 KiB; staging both K and V would require 64 KiB before other state. Avoid
that design. A scalar SIMD tile can distribute score/probability slots across
lanes and shuffle probabilities into the value loop, without materializing
full K/V tiles or padding four queries into an eight-row matrix tile. Four
queries' Q and output fragments already represent roughly 64 F32 components
per thread before temporary state in the straightforward 32-lane mapping.
Register pressure may require splitting query groups or the head dimension.
Fewer loads do not automatically mean higher occupancy or speed.

This is the strongest structural candidate for long-context H=1, not a claim
that attention dominates every prompt. If attention is fraction f of complete
time, even doubling its speed gives only `1 / (1 - f/2)` overall speedup.
F16 KV can remain a later explicit storage policy; this architecture need not
depend on changing precision. Dense global attention still grows with history.

### 2. Replace the tied Q6 head's resident layout

**Verified gap:** all body Q4/Q6 projections use `kquant_columns8`, but the
825.8 MB tied embedding/head remains canonical. Its R2 kernel already shares
decoding, so the opportunity is coalesced physical layout, not missing row
batching.

Reuse `repack_kquant_weight_columns8`, replace the head's GPU representation
at admission, preserve the tied alias, and add an embedding gather that reads
one token column from the packed allocation. Existing descriptor-based head
dispatch then reaches the compact Q6 kernel. Do not retain a second 826 MB
GPU allocation. Adjust admission accounting, payload receipts, readback
validation and startup staging together; an approximately 826 MB temporary
packing buffer still matters on this 16 GB machine even without duplicate
persistent residency.

This preserves the supplied Q6 values and reuses a proven engine mechanism.
It is a concrete hypothesis, not evidence that the canonical head is currently
bandwidth-bound. Packed token gathers may be worse for small embedding reads;
prefill and startup must be checked as well as generation. Test every Q6 code
region/scale, token and column tails, packed/canonical embedding parity, tied
ownership and independent real-model output goldens. The head accounts for
about 11.1% of scheduled weight bytes; its measured stage cost bounds benefit.

### 3. Reduce Q4 R2 live state without undoing fused sums

Current R2 compact Q4 workers retain two columns times two rows of low/high
float4 accumulators: 32 floating components before final sums and temporaries.
Current MLX [qmv_wide](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/quantized.h#L984)
uses eight K lanes per output, one output column per lane and short decoded
fragments reused across input rows. The relevant difference is live-state
organization; this engine already shares R2 weight decoding. Its affine
per-weight reconstruction cannot be copied without losing our factored
integer dot and precomputed Q4 minimum corrections. The
[scale-after-dot review](https://github.com/ml-explore/mlx/pull/3764)
supports arithmetic factoring, which our Q6 core already does.

Two distinct engineering choices should be evaluated from real compiler/GPU
evidence, not an unbounded tile sweep: one column per lane with the existing
logical K partition, or a shorter-fragment reduction shared by R1/R2. The
former can halve per-thread partial accumulator state but require twice as
many threads, extra repeated activation/metadata loads, and the same total
group state. The latter changes floating reduction order and may serialize
more K work per lane. Neither is a demonstrated occupancy improvement.

Q4 is about 80.1% of body weight bytes, making the addressable category large.
However, physical spills and issue limits in our current fused kernel remain
unmeasured. The earlier unsuccessful H=3 one-column/wider-row experiments
must inform this work; changing the row count is not by itself a new idea.
MLX's [small-M engineering discussion](https://github.com/ml-explore/mlx/issues/3852)
also reports unsuccessful custom kernels and separates cache-hot diagnostics
from DRAM-backed model execution. Do not turn another kernel sweep into the
product.

### 4. Remove H=1's normal-path full-sampler fallback launches

`Gemma3DraftPlan::Impl::sample` encodes 18 guarded fallback dispatches, totaling
15,364 threadgroups, even when its compact top-k <=64 support succeeds. Most
return immediately. The previously rejected attempt used indirect dispatch.

**New H=1-specific adaptation:** there is no subsequent draft forward dependent
on the sampled proposal in this command. Encode the compact path and guarded
pending/history publication, then use the proposal-completion wait that already
exists. If support fits 256 entries, finish normally. Only overflow submits
the existing full fallback and completes again. This removes the normal-path
launches without a new common-path wait or indirect argument buffer.

The current pending/history kernels assume a sampled token exists. They must
not publish uninitialized output on overflow. Preserve the previewed random
word, exact integer-CDF receipt and transaction ownership across both paths;
the support-size decision must precede random selection. Leave general top-k
and multi-proposal scheduling intact. Validate 256/257-way cutoff ties,
top-p/min-p boundaries, full 2^32 mass, pending-root batching, submission failure,
callback cancellation and continuation. Common compact success should still
use one proposal completion; rare overflow may use two.

This is a useful removal of known work, bounded by the drafter's small share
of cycle time. It does not replace the target optimization. Likewise, reusing
the target distribution already prepared before q gathering would remove a
second CPU preparation, but its likely gain does not justify a new broad
sampling infrastructure.

### Latency and proposals not selected

The chat CLI's printed TTFT is prompt-processing start to first **selected**
token. `first_text_ns` already exists separately in diagnostics and measures
the first nonempty output callback. Console generation spans first selection
to last selection; complete benchmark generation includes remaining work.
The user's pasted output does not contain first-visible-text measurements.
Report both latency boundaries in subsequent investigations.

Initial drafter context synchronization is already deferred until after the
first ordinary target token is published. Shader pipelines are constructed
during preparation. Do not diagnose the occasional two-second TTFT as shader
compilation from elapsed time alone or propose deferred draft prefill as new.
Publishing a known root before computing its target K/V would require explicit
pending-target recovery semantics: the existing contract makes K/V recoverable
before irreversible callbacks. It would not remove target computation.

Other designs were considered and are not the next priority:

- Delaying the bonus head loses joint Q6 weight reuse. It wins only if
  `joint_head > (1 + A) * scalar_head + synchronization`, a demanding condition
  at A around 0.59–0.67. No supporting measurement establishes it here.
- GPU lossless sampling would move a compact CPU decision into a new execution
  boundary. [FlashInfer's sorting-free work](https://flashinfer.ai/2025/03/10/sampling.html)
  and [current speculative sampler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh)
  are useful references, but their full-vocabulary GPU probability/RNG flow
  differs from our compact target support and producer-owned integer CDF.
- [SpecVocab, February 2026](https://arxiv.org/abs/2602.13836), has an actual
  learned vocabulary predictor, not a free feature of our checkpoint. Its
  [model integration](https://github.com/SamsungLabs/specvocab-sglang/blob/594c5c6/python/sglang/srt/models/llama_eagle3.py)
  and [indexed head](https://github.com/SamsungLabs/specvocab-sglang/blob/594c5c6/python/sglang/srt/layers/logits_processor.py)
  make the required training and execution changes concrete.
  [FR-Spec](https://github.com/thunlp/FR-Spec) and
  [VocabTrim](https://arxiv.org/abs/2506.22694) also support reducing draft
  vocabulary, but previous shortlist experiments and lost acceptance matter.
  A proposal change saving fraction s of cycle time helps only when
  `A_new > A_old - s * (1 + A_old)`: a 3% saving permits only about 4.8–5.0
  percentage points of acceptance loss here. This is our cost derivation,
  not a novel published algorithm.
- [nano-PEARL](https://github.com/smart-lty/nano-PEARL) explicitly separates
  draft and target GPU groups. Its overlap does not transfer directly to one
  M4 sharing bandwidth and thermals.
  [Windowed-MTP, July 2026](https://arxiv.org/abs/2607.21535) changes draft
  attention for very long contexts; it is not permission to truncate our
  existing drafter's global history silently.
- [llama.cpp PR #23114](https://github.com/ggml-org/llama.cpp/pull/23114) is a
  closed, unmerged two-query K/V-reuse experiment with encouraging M4 Max
  F16 microbenchmarks and mixed cross-device results. It supports examining
  reuse, not a promised gain for M4 F32. Fewer source loads, measured DRAM
  bytes, cache traffic and occupancy are different quantities.
- [Apple's WWDC26 TensorOps material](https://developer.apple.com/videos/play/wwdc2026/330/)
  includes useful cooperative tensor data flow, but M5 neural-accelerator
  claims are not M4 capabilities. Native int4 support is not direct support
  for GGUF Q4_K's scale/min structure. An eight-row matrix rewrite, F16
  intermediate probabilities, or arbitrary global-KV eviction would change
  the problem or the numerical contract.
- [BaseRT's public repository](https://github.com/basecompute/baseRT/blob/128bdd7348c6d0d271f47b5aca5b2fb16a5e1df6/README.md)
  exposes tooling and bindings, while its native engine is proprietary. Its
  [July 2026 paper](https://arxiv.org/html/2607.00501v1#S3) describes useful
  general techniques but provides no auditable tiny-batch Metal kernel that
  changes this ranking. Headline results are not evidence for our exact setup.

Upstream code was pinned to MLX `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`
and llama.cpp `f1ea206218210afb913ae2f5d2c51faed35915da`, both checked on
2026-09-28. Metal Flash Attention was inspected at
`8671cddc38f19a6eadb804dee6a3ca2954b8bf32`. The row/window adaptation and H=1
fallback scheduling above are proposed engine designs. No claim of worldwide
novelty or proven speedup is made.

### Proposed implementation baton and verification

Start from the accepted fused-sums binary. Attribute current H=1 target time
at representative short and longer prefixes using existing diagnostics; an
old ordinary stage profile cannot select current H=1 geometry. Keep diagnostic
work bounded and distinguish instrumentation overhead from real chat timing.
Proceed with blocked F32 attention and H=1 prefix sharing as one complete
attention design, retaining fused output sums. Implement the deferred compact
fallback as a separate attributable scheduling change. Packed tied-head
execution is the next concrete target project; change priority if fresh stage
evidence shows a larger realizable gain there. Q4 live-state redesign requires
evidence of a suitable execution bottleneck before another geometry experiment.

| Change | Main implementation locations | Existing verification homes |
| --- | --- | --- |
| Blocked R1/R2 attention, shared prefix, fused output receipt | `metal_backend.mm`, `metal_context.*`, `metal_command_context.*`, `target_verifier.cpp`; shared target schedule only where required | `gemma3_metal_tests.cpp`, `gemma3_target_runtime_tests.cpp`, `gemma2_metal_plan_tests.cpp` |
| Sole packed tied head plus embedding gather | `gemma2_metal_plan.cpp`, `kquant_packing.*`, Metal embedding/dispatch kernels, model/readback validators | `tensor_tests.cpp`, `gemma3_model_plan_tests.cpp`, `gemma3_metal_tests.cpp`, target goldens |
| Q4 shorter live-state execution | compact Q4 kernels and pipeline/dispatch validation | existing packed R1/R2 parity, tails, fused sums and target goldens |
| Deferred H=1 sampler overflow | `gemma3_draft.mm` compact sampling and pending/history publication | `gemma3_draft_tests.cpp`, `standalone_chat_tests.cpp`, lossless sampling tests |
| Latency interpretation/exposure if needed | `gemma2_chat_session.cpp`, CLI diagnostics, benchmark parser | existing CLI and quick-benchmark script tests |

For attention, cover individual and mixed row schedules around 256/257,
512/513, 1024/1025 and partition thresholds through the existing long-context
fixtures; exercise ring wrap, local/global layers, private ancestors, masked
tails and extreme finite scores. Require independent F32/CPU evidence,
ordinary/verifier numerical parity and final fused-sum parity. A new reduction
may change floating results, but cannot be justified by loosening goldens or
changing precision only for verification. Preserve rollback, stopping,
callback cancellation, regeneration and context behavior. Losslessness means
preserving the target sampling distribution, not identical seeded prose.

Run complete repository CI after implementation. Retention must use actual
H=1 chats, including first selection, first text, complete generation, token
counts, output provenance, acceptance, GPU cost, memory pressure and device
conditions. Compare against **fused H=1**, not the obsolete unfused baseline;
check the ordinary path when shared numerics change. Follow the established
thermal isolation and retention rules, respecting any explicit user limit on
benchmark runs. Preserve the baseline until an improvement is demonstrated.
Do not restart H=3, training, lower-precision KV, approximate target attention
or an open-ended kernel search. Finish the chosen mechanism, report measured
benefit or rejection, and remove rejected production variants.

## Historical screen: joint H=3 block verification and fused H=1 Q4 sums (2026-09-28)

**Status note:** the following records the original screening decision. Its
fused-H=1 non-retention and restored-source statements were superseded by the
user's later baseline adoption documented above. H=3 remains unretained.

Both authorized mechanisms were implemented end to end, independently reviewed,
checked for correctness, and measured in actual chats. **Neither change is
retained in production.** Joint H=3 did not demonstrate an improvement in this
screen. Fused H=1 is **promising but unvalidated**: its matched-work chat was
substantially faster, but the source of the gain is unresolved and follow-up
TTFT missed the written retention gate. Do not describe these as equivalent
negative results or claim that the H=1 candidate failed to accelerate the chat.

The production implementation is restored byte-for-byte to this task's starting
snapshot. Expanded deterministic callback/cancellation coverage remains in
`tests/engine/standalone_chat_tests.cpp`. All preexisting uncommitted development
is preserved. The experimental source, binaries and measurements remain in
ignored local artifacts. No further kernel variants, benchmarks, training or
F16 KV implementation are active. A later controlled H=1 comparison would need
a new authorization that includes a contemporaneous baseline; it is not an
automatic next task. The authorized implementation-and-screening scope is complete.

### Actual chat results and retention decisions

The user's continuing candidate-only instruction was followed: **one new joint
H=3 chat and one new fused H=1 chat**, in that order. No ordinary, retained H=1,
retained H=3 or llama.cpp performance run was repeated. Each candidate used a
fresh native process after a four-minute idle cooldown; builds and correctness
tests were excluded from cooldown and measurement. Both used the canonical
two-turn natural-stop conversation, seed 42, 16K context capacity, F32 KV,
Gemma 3 12B Q4_K_M, the existing 270M Q8 drafter, lossless sampling, full
verification and no retrieval. Frozen executable hashes were checked again
immediately before native launch.

Rates use selected tokens divided by complete generation duration. Aggregate
is total tokens divided by total duration. These are single observations
against recorded results, without measured variance or contemporaneous control.

| Execution | First tokens/s | Follow-up tokens/s | Aggregate tokens/s | First / follow-up TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Recorded ordinary | 11.337 | 7.148 | 9.162 | 293.401 / 609.708 |
| Recorded H=1 | 16.571 | 7.652 | 11.017 | 291.438 / 596.758 |
| Recorded H=3 | 13.586 | 7.753 | 10.185 | 291.558 / 611.236 |
| Joint H=3 candidate | 12.339 | 7.775 | 9.865 | 287.583 / 612.207 |
| Fused Q4 sums H=1 candidate | 17.494 | 12.691 | 15.033 | 291.159 / 618.484 |

Joint H=3 aggregate throughput is 3.15% below recorded H=3 and 10.46% below
recorded H=1. It generated 1,355 / 1,011 tokens in 109.810440 / 130.034983
seconds, with hashes `2bc53639d90c62a7` / `a01e08a803b5ae97`. The changed sampler
consumes random draws differently, so these differ from the recorded H=3's
1,381 / 1,102 tokens and conversation history. This is **not an equal-work
timing comparison** and cannot establish that joint verification intrinsically
reduces acceptance or contradicts its fixed-distribution expected-progress law.
It establishes that this implementation did not demonstrate a useful improvement
in the authorized chat screen. Its complexity was removed from production.

Fused H=1 improves observed aggregate throughput by **36.45%**, with first and
follow-up wins of **5.57% / 65.86%**. It exactly matches recorded H=1's 1,344 /
1,024 tokens, hashes `3a9063eeaaadedd9` / `ecc10d1344ecd5e7`, 805 / 644 cycles,
537 / 378 accepted proposals, attention work, dispatches, barriers and waits.
Complete generation took 76.828407 / 80.688961 seconds, versus recorded
81.104993 / 133.827465. This is a matched-work observation, unlike the joint
H=3 comparison. However, follow-up TTFT rose 21.726 ms (3.64%), exceeding the
repository's 2% allowance for another headline metric. That 19-row prefill
bypasses the new non-prefill R1/R2 path: the observed gate miss is **not proof
that fused sums slowed prefill**.

### Cost attribution and resource limits

| Execution | Target GPU ms/cycle, first / follow-up | Draft GPU ms/cycle, first / follow-up | Draft wall ms/cycle, first / follow-up |
| --- | ---: | ---: | ---: |
| Recorded H=3 | 158.381 / 241.070 | 23.454 / 36.026 | 24.290 / 37.039 |
| Joint H=3 | 159.301 / 245.642 | 23.286 / 36.059 | 24.202 / 37.278 |
| Recorded H=1 | 89.435 / 185.086 | 8.924 / 19.116 | 9.593 / 20.172 |
| Fused H=1 | 84.941 / 110.779 | 8.113 / 11.721 | 8.744 / 12.514 |

Joint H=3 first-turn useful progress fell from 2.503 to 2.285 tokens/cycle on
its changed trajectory; follow-up progress rose from 2.170 to 2.218. Target GPU
cost per cycle changed only +0.58% / +1.90%. The implementation still evaluated
all four target rows and gathered all three proposal laws; it did not silently
fall back or stop verification at the first failed prefix test. Its 592 / 455
cycles accepted 761 / 554 of 1,776 / 1,365 proposals, with zero numerical
refusals. Final accepted-prefix histograms were `[230,117,91,154]` and
`[174,111,67,103]`. These histograms describe the final published prefix and
correction location; they do not expose intermediate failed tests or rescues.

Fused H=1 target GPU cost fell 5.02% / 40.15%, but the **unchanged drafter's**
GPU cost also fell 9.09% / 38.68%. Target/draft GPU cost ratios changed +4.47% /
-2.39%. This prevents attributing the full 36.45% chat gain to the new kernels.
It is not a calibrated correction for device speed and does not prove zero
kernel benefit. Different device conditions could explain much of the change;
less target work could also reduce heat and accelerate the drafter on the same
GPU. Neither explanation is established. The candidate is preserved for a
later controlled decision, with no claimed causal or universal speedup.

Native lifetime peak physical footprints were 11,184,189,056 bytes for joint
H=3 and 11,179,454,080 for fused H=1, respectively 0.09% / 0.12% above their
recorded modes. These are PID-owned macOS physical-footprint high-water values,
not outer Python RSS. Native page-ins increased by only 4 / 1 after the first
30 seconds. System swap was already in use and changed during both runs; it is
not engine-owned memory. `pmset` recorded no thermal/performance warnings, but
GPU clocks, temperatures, power, spills and cache traffic were not measured.
Absence of a warning does not establish equal thermal conditions.

Both chats evaluated 16 / 19 prompt tokens and reported zero numerical
refusals. Joint H=3 reused 0 / 1,371 tokens and ended at cache positions
1,371 / 2,401. Fused H=1 reused 0 / 1,360 and ended at 1,360 / 2,403, matching
recorded H=1. Fused sums added no dispatch, barrier, wait or transient allocation.

### Implementations and correctness evidence

Joint H=3 implemented the actual joint prefix acceptance and weighted residual
algorithm from [Sun et al., Block Verification Accelerates Speculative Decoding](https://arxiv.org/html/2403.10444v3),
not a product-ratio-only substitute. With `a_0=1`, proposal recurrence was
`a_(i+1)=min(1,a_i*p_i(x_i)/q_i(x_i))`; correction mass was
`R_i(v)=max(a_i*p_i(v)-q_i(v),0)`. Intermediate prefix thresholds used
`sum(R_i)/(1-a_i+sum(R_i))`; the final threshold was `a_H`. Independent tests
selected the **longest** accepted prefix even after an earlier failed test.
Target distributions included each hypothetical prefix's repetition history;
draft probabilities were the actual integer-CDF proposal bins, including the
full `2^32` mass case. No extra target pass was introduced.

The first published block-derived token owned all proposal and decision draws
that influenced it; later accepted tokens advanced history only. Partial
callbacks, stops, cancellation, target K/V publication and drafter rollback
reused the engine's transaction/controller boundaries. Rejected suffix random
words were never reused after influencing an already published prefix.

Joint CI passed **476 C++ tests and all 60 Python tests**, including actual
12B/270M independent references and all-layer target parity. The subsequently
strengthened isolated sampling suite passed **18 tests**: exhaustive complete
autoregressive sequence probabilities for horizons 1/2/3, history-dependent
laws, zeros/equal/disjoint supports, strict later-prefix recovery, tiny binary64
residuals, one-word/full-mass bins, stale leases and RNG publication ownership.
An initial callback fixture failure was corrected by explicitly retaining its
deterministic drafter while making target sampling stochastic; no production
algorithm or tolerance was weakened to pass it.

Fused H=1 computed the original ordered eight-value Q4 minimum-correction sums
inside RMSNorm, GELU and the final attention output, then reused them across
compact Q4 consumers. RMS retained its original reduction tree; GELU retained
the scalar precise-tanh expression; attention emitted sums only after final
division, through either the single-partition or reduction path. No F16
activation, approximation, precision shortcut or separate preparation pass.
The producer's contiguous chunks were permuted into the consumer's
`[row][256-value block][worker][component]` layout without changing addition
order. Persistent scratch was bounded to two rows: 15 KiB per actual-model
workspace, potentially 30 KiB across ordinary/private workspaces, with matching
admission accounting. Existing barriers guarded producer/consumer reuse.

Application was Gemma 3 compact Q4, non-prefill, R1/R2, including ordinary
scalar decode and shortened speculative cycles. H=3 body tiles, Q6 consumers,
prefill and the drafter were unchanged. Validation guarded canonical F32
storage, dimensions, capacity, overlap, address bounds and precise contexts.
The ordinary runtime's existing vector-alignment assumptions were preserved;
this work did not introduce a general arbitrary-alignment buffer API.

Fused H=1 CI passed **477 C++ tests and all 60 Python tests**. New tests verified
ordered sums against the original consumer's independent index equations,
exact output/projection parity at K=3840/4096/15360 for one/two rows, column
tails, nonzero buffer views and canaries, single/mixed/multipartition attention,
and invalid storage/alias/precision contracts. Actual 12B scalar/two-row
verification and all-48-layer K/V retained authenticated reference parity;
existing unavailable-device and Gemma 2 checks also passed. The CI shell's
first Python invocation used system Python and skipped optional-library cases;
a separate run in the existing `.stage5a-venv` passed all 60 without skips.

Final forced-rebuild CI passed **474 C++ tests and all 60 Python tests**,
including native Metal and authenticated model references. The rebuilt
production executable exactly reproduces the starting SHA-256. The only
retained changes from this task are this document and the expanded callback
test. Log: `build/gemma3-block-and-h1-2026-09-28/ci-restored-final.log`;
restoration receipt: `build/gemma3-block-and-h1-2026-09-28/final-restoration.json`.

### Local evidence and preserved candidates

- Measurements: `benchmarks/results/gemma3-block-and-h1-2026-09-28/summary.json`,
  `joint-block-h3.*` and `fused-sums-h1.*` contain raw timing/counts, output hashes,
  frozen binary manifests, PID-owned memory and resource samples.
- Source/binaries: `build/gemma3-block-and-h1-2026-09-28/before-source/`,
  `block-source/`, `block-final-source/`, `h1-source/`, `block-identity.json`
  and `h1-identity.json`. The final block snapshot contains the later CPU
  boundary tests; the measured production binary's source is unchanged.
- Starting executable SHA-256:
  `f224066ddf385fbe0b6e49d108da43ef834bccbf6f8327ece87596b3a0662554`.
- Joint H=3 executable `block-local-gguf-generate`:
  `91b083d2cb5372b3afc1b6b4bb61701b76407d2b167ed6fb58626c6a4124b465`.
- Fused H=1 executable `h1-local-gguf-generate`:
  `b1dafea102bfc55809dad3a7b62bae95c01d0e6c767e1f54da1043c82f6f61ab`.
- Validation: `ci-block-final.log`, `sampling-final-boundaries.log`, `ci-h1.log`,
  `ci-h1-python-full.log` and `ci-restored-final.log` in the same build directory.
- Recorded H=3: `benchmarks/results/gemma3-four-row-2026-09-28/cooled-baseline-h3-full.*`.
  Recorded H=1/ordinary: `benchmarks/results/gemma3-repacked-projections-2026-09-28/cooled-candidate-h1-full.*`
  and `cooled-candidate-ordinary.*`.

## Completed task: grouped R2 and prepared Q4 sums screened (2026-09-28)

Both proposed mechanisms were implemented, checked and tested in actual H=3
chats, individually and together. **No performance change is retained.** All
production source was restored byte-for-byte to the snapshot from the start of
this task, preserving the existing H=1 path and prior uncommitted development.
The additional actual-12B H=3 numerical/publication coverage is retained in
`tests/engine/gemma3_target_runtime_tests.cpp`. Experimental source snapshots,
binaries, test logs and measurements remain in ignored local artifacts.

The authorized experiment is complete; no further kernel variants, training,
F16 KV work or benchmarks are active. The existing H=1 implementation remains
the retained speculative option. The research below is evidence for a future
decision, not an implementation queue.

### Actual-chat results and decision

Per the user's explicit instruction, **only the three new H=3 candidates were
benchmarked**. No ordinary, H=1 or retained-H=3 performance run was repeated.
Each candidate used a fresh process after a four-minute idle cooldown, with
builds and correctness tests outside the measurement window. The canonical
two-turn chat used seed 42, natural stopping, a 16K context capacity, F32 KV,
the 270M drafter, lossless sampling, full verification and no retrieval.
Run order was combined, grouping only, then sums only. There is one observation
per candidate, so there is no measured variance or contemporaneous baseline.

Throughput uses selected tokens divided by complete generation duration;
aggregate is total tokens divided by total duration, not the mean of rates.

| Execution | First tokens/s | Follow-up tokens/s | Aggregate tokens/s | First / follow-up TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Recorded ordinary | 11.337 | 7.148 | 9.162 | 293.401 / 609.708 |
| Recorded H=1 | 16.571 | 7.652 | 11.017 | 291.438 / 596.758 |
| Recorded H=3 | 13.586 | 7.753 | 10.185 | 291.558 / 611.236 |
| H=3 grouped R2 only | 14.087 | 8.349 | 10.794 | 290.892 / 598.439 |
| H=3 prepared Q4 sums only | 14.101 | 8.404 | 10.839 | 291.419 / 598.579 |
| H=3 both mechanisms | 14.386 | 8.691 | 11.145 | 288.591 / 2323.881 |

All new H=3 runs match the recorded H=3 workload: 1,381 / 1,102 selected
tokens, output hashes `54838a5ee694ee33` / `7098999366605b67`, 551 / 507 cycles,
828 / 593 accepted of 1,653 / 1,521 proposals, and zero numerical refusals.
They evaluate 16 / 19 prompt tokens, reuse 0 / 1,397 tokens, and finish at cache
positions 1,397 / 2,518. Complete generation seconds were respectively
98.032813 / 131.996510 (grouped), 97.936874 / 131.133843 (sums), and
95.997540 / 126.790687 (combined), versus 101.651302 / 142.133466 recorded H=3.
Recorded H=1 and ordinary generated different token counts and histories;
cross-mode comparisons are useful chat observations, not equal-work timings.

The single mechanisms improve recorded H=3 aggregate throughput by 5.98% and
6.42%, with wins on both turns and no headline regression. They therefore pass
the numerical screen against that historical H=3 result; they are **not**
rejected for failing its 2% threshold. However, both remain below recorded H=1
aggregate throughput and about 15% behind on its first turn. The combined
candidate is 9.42% above recorded H=3 aggregate, but only 1.16% above recorded
H=1, still 13.19% behind on the first turn, and has a large follow-up TTFT
regression. None establishes the useful H=3 result required by the agreed
decision boundary. This is insufficient evidence to retain the complexity,
not proof that these mechanisms can never help another workload.

| Execution | First / follow-up target GPU ms per cycle | First / follow-up draft wall ms per cycle |
| --- | ---: | ---: |
| Recorded H=3 | 158.381 / 241.070 | 24.290 / 37.039 |
| Grouped R2 only | 152.797 / 223.845 | 23.340 / 34.306 |
| Prepared Q4 sums only | 152.391 / 222.127 | 23.380 / 34.252 |
| Both mechanisms | 149.711 / 214.682 | 22.786 / 32.793 |

The best first-turn verifier reduction is 5.47%, well short of the earlier
approximately 21% planning requirement to match H=1 at unchanged drafting and
progress. Crucially, the unchanged drafter also became faster: its GPU time
fell 4.13% / 7.60%, 4.04% / 7.76%, and 6.10% / 12.00%, respectively. Target/draft
GPU time ratios changed by only approximately -0.1% to +1.2%. This limits
attribution of the apparent gains to the new target kernels. The drafter is
not a calibrated thermal correction: both models share the same device.

Native peak physical footprints were 11,182,927,488 / 11,183,074,992 /
11,175,177,856 bytes for grouped / sums / combined, within 0.09% of recorded
H=3's 11,173,981,800 bytes. Native process page-ins after the first 30 seconds
did not increase. System swap was already in use and varied across runs;
system-wide swap is not engine-owned memory. `pmset` reported no thermal or
performance warnings, but GPU clocks, temperatures, spills and cache traffic
were not measured. Absence of a warning does not establish equal device speed.

The combined follow-up TTFT spike was investigated without another performance
run. Its sampled native CPU/instruction burst was substantially larger than
recorded H=3 and the subsequent grouped-only run; page-ins did not explain it.
The three candidates have identical Metal source and pipeline construction.
Target prefill does not rebuild the model/prefix or explicitly construct a new
pipeline at the turn boundary; draft rebuild time was zero. A driver/cache or
other run-state effect is plausible, but unproved. Existing telemetry does not
split prefill GPU time from host/driver delay, so the cause remains unknown.
Even ignoring that spike, the combined result misses the intended first-turn
H=1 comparison and aggregate margin.

### Implementation and correctness evidence

- Grouping put two unchanged R2 consumers in one 256-thread group, with
  disjoint reductions and a uniform barrier. Compact Q4/Q6 body and native Q6
  head used the grouping only for non-prefill batches of 3-8 rows. R1/R2 and
  prefill dispatch/arithmetic were unchanged; no change to the drafter.
- Prepared Q4 sums preserved the original F32 addition order and reused each
  input preparation across its Q4 consumers. Persistent bounded scratch lived
  in the target verifier workspace, not the model plan. Explicit buffer size,
  packing, overlap and synchronization checks guarded reuse. No F16 activation,
  requantization, approximation, changed sampling rule or relaxed tolerance.
- Combined full repository CI passed **475 C++ tests and all 60 Python tests**,
  including native Metal tests, authenticated independent 12B/270M references,
  all-48-layer KV/logit parity, tails, cancellation and lifecycle coverage.
  Each single-mechanism build also passed the focused native suite; its
  optional actual-12B reference case was unset, so that evidence belongs to
  the combined full CI rather than the individual focused runs.
- The retained actual-model test checks every compact H=3 row's candidates
  against authenticated scalar execution and exact all-48-layer KV, then
  checks four-token publication and transaction closure. Its forced accepted
  prefix isolates numerical/publication behavior; it does not test proposal
  quality or replace the separate lossless probability-law tests.
- Final restored-source CI passed **474 C++ tests and all 60 Python tests**,
  including the new actual-12B H=3 check and existing independent references.
  A forced rebuild reproduced the original production executable SHA-256
  exactly. The earlier non-forced restoration check was cancelled after
  finding stale build output and is not counted as final validation. Final log:
  `build/gemma3-grouped-r2-2026-09-28/ci-restored-final.log`.

Local evidence is in `benchmarks/results/gemma3-grouped-r2-2026-09-28/`:
`summary.json` and `grouped-only-h3.*`, `sums-only-h3.*`, `combined-h3.*` retain
raw times/counts, hashes, binary manifests, PID-owned memory and resource logs.
The historical H=3 reference is
`benchmarks/results/gemma3-four-row-2026-09-28/cooled-baseline-h3-full.*`;
H=1 and ordinary references are `cooled-candidate-h1-full.*` and
`cooled-candidate-ordinary.*` under
`benchmarks/results/gemma3-repacked-projections-2026-09-28/`.
Source snapshots, binaries and CI logs are under
`build/gemma3-grouped-r2-2026-09-28/`.

| Binary | SHA-256 |
| --- | --- |
| Retained before experiment | `f224066ddf385fbe0b6e49d108da43ef834bccbf6f8327ece87596b3a0662554` |
| Grouped only | `2b7338153c93b741cc1e5b1b045b55abad6ee7007002343f477ae13d52e00c6e` |
| Sums only | `92facbacb69a155f8162295bdd6c5c9871d3d996404f505fedf81b943a706ef2` |
| Combined | `0bbce1bdb738b1d24725d672e65fad847b8e5df7de9c1f29e286a02ba4bb61d8` |

## Completed design assessment before the experiment

The user requested a deeper assessment of the fastest credible verifier for
this M4 after the rejected four-row experiments. Research and read-only code /
artifact inspection are complete. The preceding research turn changed only this
document and ran no build, GPU execution, benchmark, model download or training.
The retained production implementation and its previous correctness evidence
below were the starting point for the completed experiment above. This section
preserves the reasoning available before those measurements; descriptions of
untested mechanisms and proposed next steps are historical.

**Recommendation:** retain the efficient R2 arithmetic, bring both row pairs
onto the same weight tile together, and remove Q4 activation sums that are
recomputed across output columns. Compare that with a genuinely different
row-distributed SIMD design only if the measured instruction/cache tradeoff
warrants it. This is the strongest untested execution route found, not a claim
to have established the globally fastest kernel or guaranteed useful H=3.
Q4 body execution is the primary investment; Q6-only tuning, controller work
and draft overlap cannot plausibly supply the entire required improvement.

The completed decision and user-directed candidate-only comparison protocol
above supersede this proposal where they conflict. None of the remaining
alternatives is authorized as an active follow-on task.

### The actual target is cost per useful token

For horizon H, let `L_H = 1 + sum_i P(first i proposals accepted)` away from
stopping/context boundaries. The objective is
`(draft time + verifier time + other cycle time) / E[L_H]`, not acceptance
percentage or verifier time divided by drafter time.

The retained H=3 first turn had 551 cycles, 828 accepted proposals and
`E[L_3] = 2.50272`. Its mean complete cycle was 184.210 ms: 24.290 ms drafting,
158.381 ms target verification/publication, and about 1.540 ms remaining work.
The earlier retained H=1 first turn had `E[L_1] = 1.66708` and a 100.399 ms cycle.
Using those as planning inputs gives:

| Desired H=3 throughput relative to H=1 | H=3 verifier budget | Reduction from 158.381 ms |
| --- | ---: | ---: |
| Break even | 124.90 ms | 21.14% |
| 10% faster | 111.19 ms | 29.79% |
| 20% faster | 99.77 ms | 37.00% |

These modes generated different histories and were measured in different
screens. The table is sensitivity analysis, **not a matched-state prediction**.
Recalculate from fresh comparable-state R1/R2/R4 timings and actual chat
progress before a retention decision. At unchanged H=3 cycle cost, progress
would instead need to rise to 3.059 tokens/cycle (+22.22%) just to match that H=1
example. Making drafting entirely free gives only an optimistic 15.2%
throughput increase there. Removing the remaining host/other cycle overhead
cannot recover the needed 30-47 ms of verifier time.

The follow-up remains independently decisive: retained H=3 target time was
241.070 ms/cycle, and the rejected Q6 candidate did not improve it. The old
H=1 follow-up uses another prefix/trajectory; do not turn its numbers into an
exact H=3 comparison. Long-context attention and sustained device state must
be represented in the final decision.

### What the code and failed experiments establish

- The target body executes once for the requested verifier rows. There is no
  hidden full accepted-prefix replay to remove. Four-row head/selection tiles,
  private KV, candidate reuse and rollback already exist.
- Compact Q4 uses eight output columns and R2 per 128-thread group; compact Q6
  uses sixteen columns. Dispatch is `X = column tiles, Y = row tiles`. Four
  rows therefore create two row tiles in one projection dispatch. Metal's
  actual group scheduling and repeated DRAM traffic have **not** been measured;
  source grid order does not prove two physical full-matrix reads.
- Prior direct R4, one-column and phased Q4 experiments changed each thread's
  arithmetic/state. They do not test cooperative row-pair locality. The
  256-thread/two-R2 proposal was documented earlier but remains unimplemented,
  including in the inspected recent compact-layout experiment snapshots.
- Direct Q4 R4 increased nominal accumulator components from 44 to 88 per
  thread. That is source-level state, not a measured register allocation or
  spill diagnosis. Reducing state in the other tested designs did not suffice.
- The actual projection inventory contains 288 Q4 body matrices totaling
  5,292,195,840 compact bytes, 48 Q6 body matrices totaling 1,316,044,800 bytes,
  and the native tied Q6 head at 825,753,600 bytes. Q4 is about 80.1% of body
  weight bytes and 71.2% of total resident weight bytes. These are byte shares,
  not measured time shares. Q4 must be addressed for a substantial body gain.

The M4 Air has advertised 120 GB/s memory bandwidth. Reading the resident
7.437 GB once would take about 62 ms at that ideal rate, before attention,
activation traffic, decoding and arithmetic. This is a weights-only lower
bound under that streaming assumption, **not** a predicted verifier time or
proof that this workload is solely bandwidth-bound.
[Apple M4 Air specifications](https://support.apple.com/en-gb/122210)

### Lead design: cooperate without widening each thread

**A. Put two retained R2 consumers in one 256-thread group.** Eight SIMDgroups
process the same N tile, with the following logical mapping into the existing
compact helper:

```text
row_pair       = simd_group >> 2
logical_simd   = simd_group & 3
logical_group.y = 2 * group.y + row_pair
partial_sums   = worker_sums + 512 * row_pair
```

Keep each format's original K-worker partition, dot arithmetic and final reduction.
Q4 still has approximately 44 accumulator components per thread, rather than
88. Both helpers use separate 512-F32 scratch slices, totaling 4 KiB. All 256
threads must call the helper unconditionally and encounter the same final
threadgroup barrier; invalid tail rows/columns must not return early. The
canonical native Q6 head can apply the same row-pair mapping without that
transpose scratch. Give new pipelines their own 256-thread capability check;
the current 128-thread pipelines and requirements remain valid.

This design **still decodes weights twice**. It offers cache locality between
efficient consumers without decoded-weight staging, extra inner-loop barriers
or a doubled per-thread row state. Grouping doubles total group state and can
reduce scheduling flexibility; it does not guarantee a cache hit or lockstep
K progress. An eight-column Q4 tile spans 17,760 bytes at K3840 and 71,040 bytes
at K15360. Actual cache behavior matters.

As a locality control, interleave row pairs in a one-dimensional dispatch while
retaining 128 threads: `column_tile = id / row_tile_count`,
`row_tile = id % row_tile_count`. This changes logical adjacency but does not
guarantee common-core execution. Prefer the fastest measured mechanism; there
is no requirement that each weight be decoded exactly once if cached repeated
decoding is cheaper than explicit sharing.

Apple's Family 9 guidance specifically cautions that software-managed
threadgroup copies can lose to direct buffer access through the shared cache
hierarchy. Its MPP guide recommends locality-aware group traversal and also
warns that SIMDgroups can drift along K. These support testing this design,
not a claim that the present grid is demonstrably broken. Periodic barriers
are a possible response to observed drift, not a default addition.
[Shader memory guidance](https://developer.apple.com/videos/play/tech-talks/111373/),
[MPP guide, sections 2.3.3-2.3.4](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

**B. Compute the Q4 activation sums once per input.** The existing
`gemma3_q4_stream_packed_rows` accumulates four activation sums per row and
fragment worker inside every output-column-pair dot loop. Those sums depend
only on the input, K block and fragment worker. Recomputing them across N is
unnecessary mathematical work, independently of the unproven cache hypothesis.

Prepare `S[row, block, fragment_worker, component]`: eight workers times four
F32 components per 256 input values. Preserve the current four-item traversal
and its alternating `.x` then `.y` additions exactly. This needs `R*K/8` F32
values: 7.5 KiB for R4/K3840, or 30 KiB for R4/K15360. Load the four sums only
when applying the existing minimum correction, after the dot loop.

This removes up to 32 scalar additions per block/row/column-pair worker and
four live F32 sum components per row during the dot loop. Nominal Q4 R2 state
falls from 44 to 36; actual compiler allocation still needs measurement. Q/K/V
share one prepared input; gate/up share another. Attention output and gated
FFN output need their own preparations. Reuse one appropriately sized scratch
allocation across sequential consumers. Prepare only when a Q4 consumer needs
it; Q6 does not use the minimum correction.

Start with explicit GPU preparation so arithmetic and lifetime are auditable.
Fusing preparation into normalization/GELU is a later implementation choice
only if it removes measured overhead without changing the input or summation
sequence. Extra dispatch and scratch reads can offset saved arithmetic. There
is no established percentage gain. This can help R1/R2 as well: refresh
ordinary and H=1 comparisons rather than crediting their shared benefit only
to H=3. No current or inspected historical implementation has this preparation.

### Higher-risk competitor: distribute rows across SIMD lanes

A distinct original adaptation keeps one row and two columns per thread, and
shares decoded quant fragments through SIMD shuffles. One complete mapping
for four SIMDgroups and an R4/N8 tile is:

```text
row             = (lane >> 2) & 3
column_pair     = lane & 3
fragment_worker = 2 * simd_group + (lane >> 4)
producer_lane   = lane & ~12
```

The row-zero lane loads/decomposes the packed fragment; four row owners use
it. Preserve four explicit accumulators for the original `block % 4` K
partitions, then scatter to the original 32-worker reduction through 4 KiB
of scratch. Source-level accumulator state is about 28 F32 components/thread,
or 24 with the prepared activation sums, versus the rejected direct R4's 88.
CPU enumeration checked complete, unique row/column/fragment ownership and
four consumers per producer; it is not a compiled kernel or numerical proof.

The important counterargument is instruction issue. One-quarter-active
producer lanes do not make the decoder instruction stream four times cheaper.
Changing the K partition this way approximately doubles issued SIMD block-loop
iterations relative to the two retained R2 groups and adds shuffles. Sending
decoded values saves unpack work but needs more exchanges; sending packed bits
saves exchanges but duplicates unpacking. Explicitly named K accumulators or
constant-index unrolling avoid dynamically indexed thread arrays that can
spill. Uniform shuffle participation and padded-column initialization are
correctness requirements. This design is credible but ranks after grouped R2;
lower nominal state alone is not enough to predict a win.

An alternative shared decoded-ushort tile would preserve both R2 consumers
but add about 16 KiB of writes, 32 KiB of reads and two barriers per four-block
round. M4 on-chip storage is a shared resource, not free dedicated staging RAM.
Do not implement that extra mechanism without evidence that duplicated decode
cost exceeds staging/synchronization cost.
[Apple Family 9 architecture](https://developer.apple.com/videos/play/tech-talks/111375/),
[SIMD data exchange](https://developer.apple.com/videos/play/tech-talks/608/)

### Current original projects and competing routes

| Primary source inspected | Finding and relevance |
| --- | --- |
| [MLX small-batch qmv_wide, merged June 26, 2026](https://github.com/ml-explore/mlx/pull/3764) | Shares short decoded fragments across rows instead of invoking one matvec per row. Final posted M4 Pro R4 affine-int4 result is about 1.5x its old matvec path for a 15360x3840 matrix with BF16 activations. That is neither GGUF Q4_K nor F32 M4 Air evidence; our retained R2 already shares rows. Inspected merge: `548dd80e87454f6e4c1c7736ce09551d145c11d5`. |
| [llama.cpp Metal mul_mv at inspected revision](https://github.com/ggml-org/llama.cpp/blob/6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17/ggml/src/ggml-metal/kernels/mul_mv.metal) | Existing K-quant small-row execution provides a real reference for fragment reuse and row accumulators. It does not establish the proposed cross-row lane sharing as a proven fast path. |
| [llama.cpp September small-batch PR](https://github.com/ggml-org/llama.cpp/pull/29110) | Targets Q4_0/Q8_0; its M3 Ultra evidence is not a new solution for this Q4_K/Q6_K target. |
| [WWDC26 cooperative tensor inputs](https://developer.apple.com/videos/play/wwdc2026/330/) | Custom dequantization into cooperative matrix inputs can avoid the old decoded threadgroup staging. A direct-input F32 native shader variant is a distinct remaining matrix candidate. Reusing our existing M8 matrix tile for R4 wastes half its rows; a new implementation needs its actual tile/type support checked. Exact API availability on installed 26.6.2 needs a feature/compile check. New 27-only features must not be assumed available. |
| [Apple M5 TensorOps engineering](https://developer.apple.com/videos/play/tech-talks/111432/) | M4 uses older-device optimized shader paths, not M5 GPU Neural Accelerators. New API branding is not a new M4 hardware unit or an assured speedup. |
| [BaseRT original repository](https://github.com/basecompute/baseRT) | Public tools/API wrap a proprietary prebuilt engine; its performance claims do not expose kernels to inspect or reuse here. It is not a production dependency candidate. |

Matrix and fully expanded F32-fragment approaches must not be dismissed solely
because one earlier tile lost. However, changed accumulation/scaling order
needs an explicit common target numerical policy for ordinary and verifier
execution, independent accuracy evidence and normal-speed baselines. Do not
silently narrow F32 activations, requantize GGUF weights, weaken tolerances or
slow ordinary decoding to manufacture a speculative gain. Any native shader
primitive remains inside the engine-owned runtime; no external tensor or
inference executor is proposed.

The most credible algorithmic complement is **joint block verification**:
the [ICLR 2025 paper](https://arxiv.org/html/2403.10444v3) preserves the target
law while sometimes accepting a longer prefix from the same evaluated block.
[vLLM's merged June 2026 implementation](https://github.com/vllm-project/vllm/pull/46781)
reports 3.35 to 3.43 acceptance length and 251.25 to 256.85 tokens/s for one
MiMo/DFlash experiment (about 2.23%). It improves use of computed rows; it does
not make their target body cheaper. Our histograms lack full-block p/q ratios,
so they cannot predict its benefit or establish the needed 22% progress gain.

For a later engine adaptation, retain actual integer-CDF q masses and prepare
every target row with its hypothetical proposal-prefix repetition history.
Positive residual support lies inside the target's retained candidates, so a
small exact residual calculation is possible. Use a separately proven block
sampler; simply removing the current first-rejection break is incorrect.
Conditional early termination cannot assume tokenwise rejection remains the
block algorithm's decision rule. This is secondary to the execution task.

Conditional H=3 already exists as two R2 tiles with no prefix replay. Its
second-tile reach probabilities were 0.474 / 0.345 in the recent histories,
giving approximate verifier cost `(1 + reach_probability) * V_R2`. With old
first-turn H=1 timing as an illustrative input it still loses H=1 after eager
drafting. Reordering work alone is not an established large repair.

Asynchronous drafting in the [released SSD/Saguaro project](https://github.com/tanishqkumar/ssd)
uses a separate drafter GPU; on this M4 both models contend for shared GPU
resources, and even ideal overlap is insufficient in the first-turn budget.
[LayerSkip](https://github.com/facebookresearch/LayerSkip) reuses a target's
own trained early layers; our independent 270M cannot provide 12B hidden/KV
states. Neither is a missing drop-in cache optimization.

Shared-query blocked attention remains a useful complementary long-context
route. Forty local layers are window-bounded; eight global layers still need
growing dense history. Exact global attention cannot become constant-work per
token for arbitrary unchanged-model queries. F16 KV would save 1.3125 GiB at
16K, but defines a separately validated numerical storage policy and does not
remove Q4 weight work. Neither approximate KV verification nor heuristic head
pruning preserves the present target law. Certified head pruning could, but
has no demonstrated useful bound/cost here and is not the lead proposal.

### Proposed implementation, verification and decision boundary

1. **Establish the comparable cost curve and attribution.** Reuse existing
   fixed-state replay and stage hooks for R1/R2/R4 at common short and long
   prefixes, including the expensive follow-up range. Separate Q4/Q6 body,
   attention, head and CPU gaps. Record instrumentation perturbation. Enumerate
   available device counters instead of assuming cache/spill counters exist.
   This installation has CommandLineTools and no available `xctrace`; current
   timestamps do not measure register spills, occupancy or physical DRAM
   traffic. Lack of rich counters is not a blocker to controlled A/B timings,
   but limits causal claims.
2. **Implement grouped R2 and shared Q4 input sums end to end.** Compare
   locality-only, prepared-sum-only and combined execution so failures can be
   attributed. Retain plain R2/grid interleaving as controls. Only advance a
   row-distributed or cooperative-matrix competitor when evidence addresses
   its identified costs; do not accumulate all designs in production.
3. **Prove the unchanged target and lifecycle.** Use existing actual-weight
   projection parity, F32 cancellation tests, row/column tails, nonzero offsets,
   output guards and unavailable-device cases. Check R3/R4/R5/R7, all actual K
   widths, complete 48-layer KV/logits, ordinary/verifier consistency, rejection
   rollback, context boundaries, stopping and callback publication. Keep
   independent 12B/270M references. No golden replacement to hide an error.
   Run full `bash scripts/ci.sh` after implementation changes.
4. **Decide on complete chat performance.** After builds/tests, use fresh
   thermally isolated ordinary/H=1/H=3 processes with the established cooldown
   protocol, fixed sampling settings and raw workload/resource records. Check
   absolute R4 improvement, complete tokens/s on both turns, prompt/TTFT,
   memory pressure and numerical refusals. Recompute the required cost budget
   if ordinary/H=1 also improve. Require the repository retention gate and an
   actual useful H=3 result relative to the best retained mode; a faster
   projection or faster H=3 that still loses H=1 does not establish success.
5. **Stop at the architectural decision.** Retain only proven useful code;
   remove rejected dispatches/kernels and temporary probes. If these mechanisms
   cannot approach the recalculated whole-verifier budget, report that result
   and keep H=1. Do not silently pivot into drafter training, F16 KV, approximate
   verification or another ungated sequence of small kernel variants.

Implementation ownership is narrow despite the execution redesign:

| Owner | Proposed responsibility |
| --- | --- |
| `src/tensor/backends/metal_backend.mm` | Grouped compact/native-head kernels, Q4 input-sum preparation and consumers; optional evidence-driven competitor. |
| `src/tensor/backends/metal/metal_context.{hpp,mm}` | Explicit new pipelines, 32-lane SIMD and 256-thread validation; retained pipeline limits unchanged. |
| `src/tensor/backends/metal/metal_command_context.{hpp,mm}` | Dispatch mapping and explicit prepared-input binding/capacity checks. |
| `include/local_gguf_engine/engine/target_verifier.hpp`, `src/engine/target_verifier.cpp` | Persistent bounded scratch ownership and allocation; corrected during implementation from the originally proposed model-plan owner. |
| `src/engine/gemma_target_runtime.cpp` | Prepare each distinct input once and reuse across its projections; no frontend policy. |
| `tests/tensor/gemma3_metal_tests.cpp`, `tests/engine/gemma3_target_runtime_tests.cpp` | Extend existing independent numerical/tail/lifecycle coverage for the new boundaries. |

`target_verifier_runtime.cpp`, `chain_chat.mm` and the sampler retain their
semantics for this execution proposal. Changes there require a specific
demonstrated integration need; joint block sampling would be a later distinct
behavior change. Current packed weight residency, canonical tied head, F32
activation/KV policy and lossless sampling contract are reusable foundations.

## Completed experiment: four-row kernels rejected; retained kernels restored

The requested true four-row investigation is complete. **No four-row performance
change is retained.** The existing compact Q4/Q6 two-row execution and H=1 path
are restored byte-for-byte to the production source present at the start of this
task. The experiment does not establish a worthwhile H=3 speedup. Its
implementation scope stopped here; the subsequent research-only reassessment
is recorded above. This is evidence about the tested designs, not proof that a better
four-row design cannot exist.

### What was implemented and checked

- Direct four-row compact Q4/Q6 specializations shared decoded weights across
  all four rows. Q6 was promising; Q4 was slower. Q4's nominal live accumulator
  components grew from 44 to 88 per thread. Register spills/occupancy were not
  measured, so the source-level count is not proof of the hardware cause.
- A Q4 four-row/one-column-per-thread design reduced that count to 52, retaining
  the original K partition and reduction. It still lost to retained execution.
- A four-phase Q4 design reduced live state further. Its first version changed
  rounding by a few ULPs. Preserving the original first pair of scaled products
  together and explicitly matching subsequent FMA forms restored exact Metal
  parity without relaxing tests. Both inner-loop variants remained too slow.
  The exact phased full-target replay cost about 169.9 ms versus the prior
  retained 154.2 ms diagnostic at prefix 1536; these are uncooled separate
  processes, not a controlled chat comparison. The one-column replay was
  180.3 ms. All completed replay candidates preserved hidden values, selected
  candidates and physical K/V restoration exactly.
- The final candidate kept Q4 unchanged and used true four-row Q6 for compact
  body weights and the canonical tied output head. It added no weight copy,
  precision change, cache change or sampler/controller change. R1/R2 and
  prefill execution were preserved. Its fixed-state replay was 150.0 ms, but
  ordinary GPU time drifted between the two pairs; this only justified the
  actual-chat screen below, not retention.

The final Q6 candidate passed full CI: **474 C++ and all 60 Python tests**, with
Metal, authenticated 12B target goldens and existing 270M reference checks. The
independent projection, cancellation/F32 information, row tails, allocation
guards, unavailable-device and lifecycle requirements remained intact. The
final restored source also passed **474 C++ and all 60 Python tests** with the
same references enabled (`final-ci.log` in the artifact directory below), with
no compiler warnings. The rebuilt release executable exactly matches the
retained baseline SHA-256. `git diff --check` passed.
Only two useful test extensions remain: guarded row counts 4/5/7 and multirow
head projections at nonzero input-row offsets. Rejected kernels and diagnostic
code are absent from production; their ignored snapshots remain available.

### Cooled actual-chat rejection screen (2026-09-28)

One fresh process per mode, 240 seconds idle before each, the canonical natural
seed-42 two-turn conversation, 16K capacity, F32 KV, H=3/full/lossless and no
retrieval. No builds or other GPU tests ran during cooling or measurement.

| Metric, first / follow-up | Retained two-row baseline | Four-row Q6 candidate |
| --- | ---: | ---: |
| Selected tokens | 1381 / 1102 | 1381 / 1102 |
| Complete generation seconds | 101.651302 / 142.133466 | 97.414710 / 143.612703 |
| Complete generation tokens/s | 13.586 / 7.753 | 14.177 / 7.673 |
| Aggregate generation tokens/s | 10.185 | 10.302 |
| TTFT milliseconds | 291.558 / 611.236 | 291.431 / 2628.935 |
| First visible text milliseconds | 370.584 / 712.214 | 368.037 / 2733.367 |
| Verification/publication GPU milliseconds/cycle | 158.381 / 241.070 | 151.387 / 242.399 |
| Draft wall milliseconds/cycle | 24.290 / 37.039 | 23.740 / 38.559 |
| Native lifetime peak footprint bytes | 11,173,981,800 | 11,174,342,272 |

The candidate gained **4.35% first-turn throughput, lost 1.03% on the follow-up,
and gained only 1.14% aggregate**. It fails the 2% aggregate/both-turns retention
criteria even without the large follow-up TTFT regression. The cause of that
TTFT spike was not established. Unchanged drafter code also ran slower on the
follow-up, so do not attribute every timing difference to the Q6 kernel.

This before/after comparison is exact-work: token hashes
`54838a5ee694ee33` / `7098999366605b67`, counts and cache positions match. Both
runs had 551 / 507 cycles, 828 / 593 accepted proposals out of 1653 / 1521, and
zero numerical refusals. Mean accepted draft tokens/cycle were 1.503 / 1.170;
full three-token acceptance happened in 194 / 121 cycles. More drafted tokens
therefore did not translate into four useful output tokens every cycle.

Native footprint changed only +0.0032%; sampled RSS was 10,002,071,552 versus
10,151,854,080 bytes. Native page-ins after 30 seconds were 5 versus 0. No
thermal/performance warning was recorded, but GPU clocks and temperature were
not captured. One cooled pair does not establish variance or explain the TTFT
spike. The planned fresh H=1 comparison was deliberately cancelled during its
cooldown once H=3 failed retention; no native H=1 process launched. Do not claim
a new measured H=3/H=1 comparison from this task. The earlier retained H=1
results and the user's own tests remain separate evidence.

Artifacts and source/binary identities are under
`build/gemma3-four-row-2026-09-28/`; raw results, native memory samples,
`cooled-summary.json`, `replay-summary.json` and the rejection decision are under
`benchmarks/results/gemma3-four-row-2026-09-28/`. Nothing there is committed.

- Retained baseline executable SHA-256:
  `f224066ddf385fbe0b6e49d108da43ef834bccbf6f8327ece87596b3a0662554`.
- Rejected measured Q6 candidate executable SHA-256:
  `66c20a45ee7ea5e29404f8a589487a94352e8f5c9e01f9e53a726f80c23edfe8`.

## Retained baseline: compact projections

The compact lossless weight layout and matching Metal execution are implemented,
verified and retained as a **speculative-execution improvement**. In the cooled
actual-chat screen, H=1/full lossless decoding improved **18.86% aggregate over
the previous speculative path**, with wins of **28.05% first / 13.29% follow-up**
and identical tokens. Against the new ordinary path it improved **20.26%
aggregate**, winning **46.17% / 7.04%**. That cross-mode comparison has different
natural output lengths and histories; it is not exact-work.

Ordinary generation was **0.82% slower** in aggregate, within the 2% regression
guard, while TTFT improved on both turns. Do not advertise this as an ordinary
throughput optimization. The retention decision is for the requested verifier
improvement, with the ordinary regression checked separately. Ordinary decoding
remains the default; the existing explicit 270M/H=1/lossless options select the
measured speculative path. This implementation, validation and retention scope
is complete. It remains the production implementation after the completed
four-row investigation above. No further optimization task is active.

### Implementation and architectural decisions

Gemma 3 body Q4_K/Q6_K matrices use explicit `kquant_columns8` physical packing.
Shapes and GGUF quantization values are unchanged. Q4 expands its scale/min
metadata by four bytes per 256-weight block; Q6 rearranges its code bits without
expansion, apart from padded column tails. The actual target gains 143,032,320
resident weight bytes, from 7,294,024,704 to 7,437,057,024 bytes. Conversion holds
one tensor's CPU staging at a time. The tied native Q6 embedding/output head
continues sharing one allocation; there is no second resident model.

The initial compact-layout kernel retained the old lane assignment and regressed
verification. It was replaced before retention. Adjacent lanes now own adjacent
column pairs, so the loads consume the interleaved layout directly. A final
threadgroup transpose restores the original logical K workers and F32 reduction
order. R1/R2 share this mechanism, with two-row tiles for larger verifier batches
and 2 KiB of shared partial-sum storage per group. Invalid columns skip reads
but participate in the barrier. Prefill retains its F32 matrix arithmetic with
a layout-aware decoder. No decoded floating-point weight copy is retained.

`TensorDescriptor` distinguishes the execution layout from native GGUF bytes.
Loading, physical memory admission/accounting, resident payload framing,
readback and dispatch use that distinction. Source model validators and native
CPU/Metal consumers reject execution-packed inputs; conversion is explicit.
Pipeline validation requires 32-lane SIMD and 128-thread support, and packed
shader addressing rejects payloads beyond its 32-bit range. Gemma 2 keeps its
native layout. The drafter, probability law, sampling, callbacks, cache format,
rollback and scheduling are unchanged. Target activations, accumulation and KV
remain F32.

New conversion code is in `tensor/kquant_packing.*`; the integration is in
`tensor/tensor.*`, native consumer validators, `gemma2_model.cpp`,
`gemma2_metal_plan.cpp` and the Metal buffer/context/command/kernel files.
Coverage extends the existing tensor, Gemma 3 Metal and model-plan suites.
README and GRAPH describe the current surface. All earlier uncommitted work is
preserved; no model weights, binaries or benchmark results are added to Git.

### Correctness and diagnostic evidence

- `MODE=release MAKEFLAGS=-j2 LOCAL_GGUF_GEMMA3_TARGET_GOLDENS=build/gemma3-target-reference bash scripts/ci.sh`
  passed **474 C++ tests**, including Metal, authenticated 12B CPU-reference
  logits/all48-layer K/V, actual 270M checkpoint goldens and lifecycle tests.
  The system Python skipped 14 optional-reference checks; rerunning the Python
  suite with `.stage5a-venv/bin/python3` passed **all 60 tests**, without skips.
- CPU tests prove byte-exact conversion/restoration over varied shapes and raw
  half bits, independently inspect code/metadata fields, check zero padding,
  and reject invalid layouts and overflow. Metal tests cover actual K widths,
  odd column/row tails, nonzero allocation/row offsets, output guards,
  cancellation, F32 operand information and unavailable-device behavior.
- All **30,018,560** checked outputs in the actual-checkpoint projection screen
  were bit-exact and finite. No tolerance was relaxed or golden regenerated.
  Full-target CPU-reference error/probability bounds remain unchanged, and
  ordinary/verifier continuation and cache restoration are exact.
- At a shared synthetic 1536-token prefix, exact-state R2 replay measured
  83.158 ms ordinary GPU/token versus 86.871 ms verification/publication
  (**1.045 ordinary passes**). R4 measured 84.755 versus 154.154 ms (**1.819
  passes**). Candidates, hidden values and physical K/V restoration were exact.
  These are two-pair mechanism diagnostics without the chat cooldown protocol.
- Isolated inventory-weighted body costs relative to native were R1 0.983,
  R2 0.761, R4 0.678, R8 0.593 and R128 prefill 0.940. The initial old-lane
  compact candidate instead regressed R2 by 7.6%; it is not a production option.
  Those isolated costs are not whole-model throughput estimates.
- A separate baseline R2 stage diagnostic attributed 82.646 ms to body
  projections, 12.761 ms to attention and 10.312 ms to the head. Encoder
  instrumentation inflated verifier GPU time by 12.4%, so these fractions are
  diagnostic evidence of the main subsystem, not an exact uninstrumented split.

### Cooled actual-chat retention screen (2026-09-28)

Each mode used one fresh process, 240 seconds idle before launch, no simultaneous
build/test/GPU work, the canonical naturally stopping seed-42 two-turn chat,
16K capacity, F32 KV and retrieval disabled. Source and frozen executable hashes
were checked immediately before launch. Generation uses selected-token counts
divided by complete generation time, including final cache/callback work.

| Metric | Previous ordinary | New ordinary | Previous H=1/full | New H=1/full |
| --- | ---: | ---: | ---: | ---: |
| First / follow-up selected tokens | 1506 / 1026 | 1506 / 1026 | 1344 / 1024 | 1344 / 1024 |
| Complete generation seconds | 131.567500 / 142.525940 | 132.837467 / 143.533882 | 103.852180 / 151.614898 | 81.104993 / 133.827465 |
| Complete generation tokens/s | 11.447 / 7.199 | 11.337 / 7.148 | 12.941 / 6.754 | 16.571 / 7.652 |
| Aggregate complete generation tokens/s | 9.238 | 9.162 | 9.269 | 11.017 |
| TTFT milliseconds | 305.711 / 648.445 | 293.401 / 609.708 | 305.577 / 635.289 | 291.438 / 596.758 |
| First visible text milliseconds | 398.739 / 764.539 | 395.199 / 728.233 | 383.304 / 737.552 | 371.401 / 697.849 |
| Native lifetime peak footprint bytes | 10,444,532,584 | 10,646,711,288 | 11,020,283,448 | 11,166,248,576 |

Within each mode, counts, cache positions and hashes match: ordinary hashes are
`260ae35592bb6060` / `dd4aadff1d4f599f`; speculative hashes are
`3a9063eeaaadedd9` / `ecc10d1344ecd5e7`. Those before/after comparisons are
exact-work. Ordinary and speculative modes differ, including follow-up starting
cache positions 1522 versus 1360, so their comparison is actual-chat throughput
with different natural trajectories, not matched token work.

The speculative change clears the aggregate/both-turns gate and improves TTFT,
prompt throughput and first-visible-text latency. Ordinary generation changes
by -0.956% / -0.702%, with aggregate -0.824%; this is a small measured cost, not
a claimed benefit. The old speculative path in this same screen was only 0.34%
ahead in aggregate and lost the follow-up by 6.18% against old ordinary. The
new path wins the follow-up by 7.04% against new ordinary. This establishes a
useful retained result for the measured conversation, not universal superiority.

| H=1 cycle evidence, first / follow-up | Previous | New |
| --- | ---: | ---: |
| Cycles / proposed tokens | 805 / 644 | 805 / 644 |
| Accepted proposals | 537 / 378 | 537 / 378 |
| Accepted proposals per cycle | 0.667 / 0.587 | 0.667 / 0.587 |
| Draft wall milliseconds/cycle | 10.270 / 18.267 | 9.593 / 20.172 |
| Target verification/publication GPU milliseconds/cycle | 116.866 / 214.486 | 89.435 / 185.086 |
| Complete cycle milliseconds | 128.626 / 234.925 | 100.399 / 207.342 |
| Numerical refusals | 0 / 0 | 0 / 0 |

Acceptance did not improve. Verification/publication GPU cost fell 23.47% first
and 13.71% follow-up. Draft wall time rose on the follow-up despite unchanged
drafter code; do not attribute every timing difference solely to the new kernel.
The follow-up still costs substantially more than the first turn, and its 7%
advantage over ordinary is narrower. Dense global attention still grows with
history; this change does not make attention constant-work.

Native physical footprint is sampled from authenticated child PIDs, separately
from RSS. The speculative candidate adds 145,965,128 peak bytes (+1.32%); the
ordinary candidate adds 202,178,704 (+1.94%). Page-in counts after the first
30 seconds changed by 0 / 0 / 1 / 0 across the four modes. System swap was
already about 2.5–2.8 GiB and is not attributable to this engine from these
snapshots. Sampled peak RSS was 7,341,441,024 / 10,251,026,432 bytes for old/new
ordinary and 9,228,075,008 / 9,746,644,992 bytes for old/new speculation
(+39.63% / +5.62%). These increases have not been separately attributed; RSS
is a different accounting metric from the native Metal-inclusive physical
footprint. Do not claim every memory counter remained within 2%.
No thermal/performance warning was recorded, but GPU clocks and
temperatures were not measured. One cooled run per mode does not establish
variance or separate context growth from thermal effects. Model-loading and
repacking time were not separately timed; reported TTFT follows the canonical
startup-warmup protocol.

Artifacts, rejected-source snapshots and source/binary identities are under
`build/gemma3-repacked-projections-2026-09-28/`; raw diagnostics, cooled results,
resource samples and `cooled-summary.json` are under
`benchmarks/results/gemma3-repacked-projections-2026-09-28/`.

- Baseline executable SHA-256:
  `860f737ea5d98ebff27d079bb558b566087267b8b8c8c56f975cde7f29733abc`.
- Final tested/measured executable SHA-256:
  `f224066ddf385fbe0b6e49d108da43ef834bccbf6f8327ece87596b3a0662554`.

## Historical baseline before compact packing (2026-09-27)

The authorized projection implementation, correctness checks and cooled actual
chat comparison are complete. **Retain the new Metal projection execution:**
ordinary Gemma 3 chat improved **5.45% aggregate**, with wins on both turns and
identical token counts/output hashes. H=1 lossless speculation improved **8.83%
aggregate over that new ordinary path**, but its follow-up was **0.87% slower**.
It therefore does not meet the requirement to win both turns and remains an
explicit experimental option. Ordinary decoding remains the default. No claim
of a fully optimized pipeline or uniform speculative speedup is made.

The September 28 architectural reassessment below identifies the next substantial
opportunity and its rejection criteria. It is an assessed proposal, not an
implemented successor to the retained projection work.

### Retained implementation

- Q4 ordinary decoding keeps the existing precise packed kernel. Verification
  streams short activation fragments and shares integer decoding across two
  rows and two output columns. It preserves the ordinary kernel's worker
  assignment, dot order, scale/min expression and SIMD reduction. Native-shape
  and full-checkpoint tests confirm exact ordinary/verifier results.
- Q6 ordinary decoding and verification use the same new factored core. Each
  16-lane cluster handles two output columns; each native scale follows its
  16-value integer dot. Both paths share packed decoding and activation reads.
  Arithmetic differs from the old Q6 kernel; matching historical seeded text
  is not its correctness contract, even though these ordinary runs did match.
- Larger verifier batches use two-row tiles to bound live accumulators. Only
  three new pipeline states remain. Matrix prompt prefill, Gemma 2, sampling,
  draft scheduling, callbacks, cache publication and rollback are unchanged.
  No engine-resident buffers or runtime experiment switches were added.

The changes are confined to `src/tensor/backends/metal_backend.mm`,
`src/tensor/backends/metal/metal_context.*`,
`src/tensor/backends/metal/metal_command_context.*`,
`tests/tensor/tensor_backend_tests.cpp` and
`tests/tensor/gemma3_metal_tests.cpp`. All earlier uncommitted work is preserved.
Production remains engine-owned, native-packed, local and Metal-first, with
F32 activations, accumulation and target KV.

### Correctness evidence

Final `bash scripts/ci.sh` passed **467 C++ and 60 Python tests**, with Metal,
authenticated 12B target CPU-reference logits/all48-layer K/V, and actual 270M
checkpoint goldens enabled. `git diff --check` passed. No tolerance was relaxed
and no golden output was regenerated to fit a candidate.

Expanded existing tests cover R1–5/R8, the actual K=3840/4096/15360 dimensions,
odd output columns, nonzero head offsets, output guards and F32 information
preservation. An independently packed Q4 fixture makes its dense weights the
small difference of scale/min terms roughly 1,365 times larger; it checks
cancellation accuracy and exact ordinary/verifier equality. The inherited
strict projection envelope remains unchanged and is no longer described as a
general proof for every possible scale/min cancellation.

The final target test's packed-vs-CPU maximum logit error was **6.00814819e-05**;
the prepared distribution's total variation was **2.39206441e-06**, with no
support difference for the recorded policies. Matrix-prompt/packed continuation
had total variation **4.85341061e-07** against the CPU reference. New ordinary and
verifier continuation from the same cache is exact, including all-layer KV,
partial publication, rollback and stopping/callback behavior.

### Cooled actual-chat comparison

Each mode used one fresh native process, **240 seconds idle cooldown**, no
simultaneous build/test/GPU work, the canonical seed-42 naturally stopping
two-turn conversation, 16K context capacity, F32 KV and retrieval disabled.
Generation rates below use selected-token counts divided by complete generation
time, including final callback/cache work. Binary identities were checked before
launch and raw output hashes were retained.

| Metric | Previous ordinary | New ordinary | New H=1/full lossless |
| --- | ---: | ---: | ---: |
| First / follow-up selected tokens | 1,506 / 1,026 | 1,506 / 1,026 | 1,344 / 1,024 |
| Complete generation seconds | 131.130043 / 130.617360 | 128.332359 / 119.884186 | 92.601660 / 120.698637 |
| Complete generation tokens/s | 11.485 / 7.855 | 11.735 / 8.558 | 14.514 / 8.484 |
| Aggregate complete generation tokens/s | 9.673 | 10.201 | 11.102 |
| Reported TTFT milliseconds | 305.384 / 645.160 | 303.483 / 639.326 | 305.160 / 629.456 |
| First visible text milliseconds | 402.263 / 762.336 | 400.239 / 756.912 | 385.186 / 730.821 |
| Native lifetime peak physical footprint bytes | 10,455,624,576 | 10,450,447,184 | 11,019,841,008 |

New ordinary improved **2.18% / 8.95% per turn, 5.45% aggregate**. Both ordinary
runs produced hashes `260ae35592bb6060` / `dd4aadff1d4f599f` at the same cache
positions, so this is an exact-work comparison. TTFT, prompt throughput and
first-visible-text latency did not regress. This satisfies the projection
implementation's ordinary-path retention gate in this cooled comparison.

Speculation improved **23.68% / -0.87% per turn, 8.83% aggregate** against new
ordinary. Against previous ordinary it improved **26.37% / 8.01%, 14.76%
aggregate**, but the correct decision also compares against the improved target.
Its hashes were `3a9063eeaaadedd9` / `ecc10d1344ecd5e7`; follow-up cache position
was 1,360 versus ordinary's 1,522. **Different natural stopping lengths and
preceding histories mean this speculative comparison is not exact-work.** It
misses the both-turns-win gate, despite the positive aggregate result. Do not
enable speculation by default or present this as a uniform retained speedup.

| H=1 cycle evidence | First | Follow-up |
| --- | ---: | ---: |
| Cycles / proposed tokens | 805 | 644 |
| Accepted proposals | 537 | 378 |
| Accepted proposals per cycle | 0.667 | 0.587 |
| Draft wall milliseconds/cycle | 9.062 | 14.721 |
| Target verification/publication GPU milliseconds/cycle | 104.283 | 170.558 |
| Complete cycle milliseconds | 114.670 | 186.904 |
| Numerical refusals | 0 | 0 |

The follow-up's roughly 1.587 tokens of progress per cycle nearly repay its
186.9 ms complete cycle, leaving it near ordinary's 116.8 ms/token. Raising
acceptance alone or quoting the first turn hides this sustained cost limit.

Native memory was sampled from each authenticated child PID using
`proc_pid_rusage`, not inferred from the outer Python process's RSS. System
swap was already about 2.1 GiB before these runs. Native page-in counters were
essentially flat after the first 30 seconds (baseline/speculation unchanged,
ordinary +1 page); system swap/VM snapshots are preserved separately. These
observations do not identify GPU cache behavior or prove absence of memory
pressure. No thermal/performance warning was recorded, but GPU clocks and
physical temperatures were not captured. One cooled run per mode does not
establish run-to-run variance or isolate context effects from throttling.

### Mechanism checks and rejected alternatives

Exact-state replay uses one real prefix prefill, restores and byte-checks every
physical KV slot replay can overwrite, then times identical tokens in AB/BA
order. At prefix 1,536, the final complete two-row verifier plus publication
cost **101.93-104.68 ms**, versus **117.05-117.21 ms** for the preserved original
(about **11.8% lower mean cost**). Both implementations matched their own
ordinary candidates/hidden states exactly and remained finite. These short
diagnostics are not the cooled chat benchmark.

The four-row result did **not** establish an absolute speedup: old verification
cost **232.28-234.60 ms**, final **235.35-248.83 ms**. Ordinary-token GPU costs in
those runs also differed (**84.21-85.20 ms** versus **91.52-91.84 ms**), so do not
attribute the difference to one kernel or claim controlled thermal equivalence.
Four-row correctness passed, but H=1 is the measured chat configuration; larger
horizons have no new retention result.

Implemented and rejected alternatives included expanded F32 weight fragments,
8/16/32-lane layouts, virtual K workers, vector FMA accumulators, wider column
tiles and padded F32 matrix execution. Weight sharing alone was insufficient:
the old packed kernel already factored quantization scales and reused
activations across columns. The selected implementation preserves those useful
properties while adding row reuse. Register spilling and physical DRAM traffic
were not measured. Rejected runtime machinery was removed; its source/binaries
remain only in ignored diagnostic artifacts. Shared-KV attention was not added.

### Boundary and next decision

This implementation and validation scope is complete. No model was downloaded,
training started, external production runtime introduced or commit created.

F16 target KV is still a separate, unimplemented storage-policy task. At 16K it
would save **1,409,286,144 bytes (1.3125 GiB)**. It need not redesign the retained
projection kernels or sampling controller. It does require consistent rounding
of durable, current-chunk and verifier-private KV before attention, dtype-aware
copy/scatter/rollback handling, and independent numerical and real-chat checks.
Its memory reduction is arithmetic; its performance gain is not yet measured.

If larger-horizon verification is pursued, a bounded unimplemented candidate is
to place two existing R2 subtiles in one 256-thread workgroup: pass
`logical_group.y = 2*group.y + simd_group/4` and
`local_simd_group = simd_group%4` to the R2 helper. This keeps per-thread row
accumulators and arithmetic unchanged while offering cache locality between
four rows. It still duplicates decoding across subtiles and may reduce
occupancy; verify pipeline thread limits, tails, exact parity and real-path
cost before retaining it. It is a design candidate, not a performance claim.

Starting sources/binaries, rejected shaders and the final source manifest are
under `build/gemma3-shared-projection-2026-09-27/`. Raw diagnostics, cooled
results, resource samples and `cooled-final-summary.json` are under
`benchmarks/results/gemma3-shared-projection-2026-09-27/`.

- Previous ordinary executable SHA-256:
  `7b321ed3780ca378f04acd706cd01c99a578db203434d46a22b8ff4417c23dd4`.
- Final tested/measured executable SHA-256:
  `860f737ea5d98ebff27d079bb558b566087267b8b8c8c56f975cde7f29733abc`.

## Architectural reassessment: block efficiency and proposal quality (2026-09-28)

The user requested a fundamental reassessment before further incremental tuning.
Read-only code/artifact inspection and current primary-source research are
complete. No new inference runs, benchmarks, implementation, model downloads or
training were performed for this assessment. Only this document was changed.

**Recommendation after reviewing the experiment history:** among the three
execution proposals, compact lossless weight repacking with a corresponding
small-row projection core is the first architectural candidate for a substantial
verifier gain. Shared blocked attention ranks second as a complementary
improvement; superblock integer-factored matrix execution ranks third because
of its numerical and small-batch cost risks. This ordering revises the earlier
attention emphasis using the historical Gemma 3 stage profile below. It is an
engineering judgment with moderate-to-low confidence, not a measured winner.
Keep the 270M and correct sampler/cache foundations. Choose the execution
investment or a bounded distillation pilot using sustained matched-state costs
and useful progress, not the verifier/drafter latency ratio. Stop H=1 controller
polishing, broad tree expansion and ungated kernel searches.

This proposal is complete; implementation is not started by this document.
The additional investigation included independent algorithm, residency and
Metal audits, primary-source counterexamples, and a small exact-integer algebra
check. It did not run a model or a GPU performance experiment. Original ideas
below are locally developed hypotheses, not claims of global novelty or speed.

### Ranking against the actual experiment history

The earlier Gemma 3 pipeline audit already measured ordinary one-row stages at
approximately 2,090 cached tokens. Its preserved
`benchmarks/results/gemma3-pipeline-optimization-2026-09-27/stage-profile-summary.json`
reports median attention 10.016 ms and projection sums 73.625 ms, within a
92.960 ms instrumented GPU command. FFN gate/up and down account for 51.789 ms
of those projection stages. About 79% of that command was attributed to
projections and 11% to attention; stage medians do not sum to whole-command time.

Provenance matters: `build/prepare-gemma3-stage-profile.py` instruments only
`row_count == 1U`. The summary's six/five "rows" denote sampled commands, not
verifier batch width. The raw stdout confirms Gemma 3, F32 KV, 16K capacity and
no drafter. Encoder splitting perturbs execution; this was neither sustained
chat nor a current R2/R4 verifier profile. Projection kernels later changed.
However, the attention template/instantiations were verified byte-identical in
the baseline, instrumented snapshot and current production (SHA-256
`09ad4a314cd6a43c08e253bba1539a2320c0f23186b880fd95d6dffa047dfedc`).
This is relevant evidence against an attention-only rescue, not a hard current
verifier time bound.

| Rank for substantial verifier improvement | Reason and strongest limitation |
| --- | --- |
| 1. Compact lossless layout plus small-row projection execution | Targets the dominant measured subsystem and changes an execution representation retained by previous lane/register experiments. Can preserve existing float arithmetic. Its removable unpacking/address/stall share is unmeasured; dominance of projections does not prove dominance of packing overhead. A byte permutation alone is not the proposed performance mechanism. |
| 2. Shared blocked attention | Removes concrete repeated history work and per-key rescaling; no prior shared-query blocked implementation was found. More credible as a bounded gain or longer-context improvement than the primary remedy at these measured contexts. |
| 3. Integer-factored matrix projection | A distinct algebraic candidate, but keeps underfilled matrix tiles and changes reduction/cancellation behavior. Earlier expanded F32 MMA flattened row scaling while being far slower at R1/R2. The new formulation must overcome those costs, not merely exhibit a flatter ratio. |

The rejected attention `exp(0)` guards and Q/K norm/RoPE fusion did not test
blocked softmax or query sharing. Conversely, repeated four-row projection
failures mean that simply widening the old register tile is not a fresh design.
The earlier padded matrix and current cold R2 evidence also argue against
committing to a wholesale rewrite without a bounded numerical/cost screen.

Under the old ordinary profile, halving attention alone would improve that
instrumented command by about 5.7%, not tens of percent. More importantly,
ordinary decoding also benefits from common kernel changes. At unchanged
accepted progress, the relative speculative speedup changes by
`S_new/S_old = (T_new/T_old) / (C_new/C_old)`. The fractional cycle saving must
exceed the fractional ordinary-token saving. With the recorded follow-up's
`C/T ~= 1.60`, saving 5 ms ordinary and 10 ms per speculative cycle gives only
about 1.1% additional speculative advantage. These are illustrative calculations,
not new measurements; cross-row sharing must earn the additional benefit.

If the objective is improving speculation specifically rather than general
target throughput, distillation of the existing 270M remains a serious competing
investment. It improves accepted progress without also accelerating the ordinary
baseline, but its attainable overlap is unknown. The current fixed-cost H=1
sensitivity and cold R2 replay do not justify either promising a large training
gain or insisting that more target-kernel work must precede every training pilot.

### What is fundamental, and what the measurements do not establish

For a query q, dense softmax attention is
`A(q) = sum_i exp(q dot k_i) v_i / sum_i exp(q dot k_i)`.
A new query changes every historical weight. The previous query's output and
normalizer therefore cannot be updated by adding only the newest K/V row.
There is no known practical, general, exact constant-work replacement for this
checkpoint's growing global attention. Fixed windows, compressed target memory
and linear-attention substitutions change the operator or discard information.
The paper titled [Softmax Attention with Constant Cost per Token](https://arxiv.org/abs/2404.05843)
explicitly modifies the similarity function; it is not a drop-in Gemma kernel.

Gemma 3 already bounds local attention: this target has 40 local layers with
1,024-token windows and eight global layers. Exact kernel improvements can
reduce repeated work and amortize history reads across several queries, without
making global attention independent of history length. A bounded or compressed
**draft-side** memory is compatible with lossless verification, provided the
actual proposal law is recorded; its effect on acceptance must be measured.

The follow-up increased average attended positions per target query from about
648 to 1,168. However, **we have not measured attention's share of the final
sustained verifier time**. The 170.6 ms figure is the whole target verification
and publication GPU time, not attention time. Both drafting and verification
became about 63% slower, with only a five-second inter-turn pause and no GPU
clock/temperature capture. Do not describe attention as the established cause
of the complete slowdown.

The target is not accidentally run twice: `target_verifier_runtime.cpp` invokes
the transformer once for the requested rows and heads each row once.
`chain_chat.mm` reuses final-row candidates and advances conditional tiles
without replaying their accepted prefix. The exact-state R2 diagnostic charged
only 0.052-0.054 ms GPU time to publication, versus 101.9-104.7 ms verification.
Controller and cache-commit cleanup cannot remove tens of milliseconds here.

### The decision equation and current limits

For H drafted tokens, let L be useful emitted progress including the correction
or bonus token, D drafting cost, V target verification cost, O other cost, and T
ordinary time per token at a comparable state:

`speedup = T * E[L] / (D_H + V_(H+1) + O_H)`

Ignoring end-of-response truncation,
`E[L] = 1 + sum_(i=1..H) P(first i proposals accepted) <= H+1`.

The recorded H=1 follow-up has E[L] approximately 1.587, cycle 186.904 ms and
ordinary 116.846 ms/token. Holding those observed costs fixed:

- Perfect acceptance reaches only about **1.25x ordinary throughput**.
- Making everything outside target verification free reaches only about
  **1.087x ordinary throughput** at the observed acceptance.
- A 20% gain at that acceptance needs a cycle at most **154.5 ms**; a 50% gain
  needs at most **123.6 ms**. The current cycle is 186.9 ms.

These are conditional cost budgets, not matched-work predictions or universal
limits on speculative decoding. Improving ordinary execution changes T too.
Two-row exact-state replay already gives V/T of 1.13-1.20, whereas the separate
follow-up chat averages imply roughly 1.46. The latter uses different natural
histories and thermal trajectories, so these numbers do not establish a causal
regression. Resolving this difference is more informative than assuming every
verifier invocation is intrinsically inefficient.

For the same fixed-cost sensitivity calculation, 80% acceptance would give
about 1.125x ordinary throughput; reducing verifier time by 20% at current
acceptance would give about 1.214x; both changes together would give about
1.377x. Neither the acceptance gain nor the time reduction has been demonstrated.
At the recorded H=1 cost, a 20% gain from acceptance alone would require about
92% acceptance. These thresholds bound a training proposal's payoff without
pretending to predict whether that training can reach them.

For a fixed p and q, H=1 rejection sampling already attains overlap
`sum_x min(p(x), q(x)) = 1 - TV(p,q)`; a different acceptance implementation
cannot freely increase it while preserving the same distributions.

R4 currently costs **2.56-2.72 ordinary-token GPU times** in exact-state replay.
With the explicitly hypothetical constant conditional acceptance 0.587, H=3
would produce only `1 + a + a^2 + a^3 = 2.134` tokens per cycle. It could not
win at that cost even with a free drafter. Actual depth-wise survival must be
measured; the H=1 average is not evidence of deeper-position acceptance.

### Fresh findings that change the design choice

The GPU cannot generally turn four target rows into the cost of one merely by
reading the same weight address. Each row still needs its own dense products,
nonlinearities, query-dependent attention and cache state. Source-level sharing
may save decoding or cache traffic without reducing physical DRAM traffic;
larger register state can reduce occupancy enough to lose the saving. Our
existing failures have relevant external counterexamples:

- [MLX issue 3839](https://github.com/ml-explore/mlx/issues/3839) reports that
  `qmv_wide` improved small batches substantially, but four-row FFN projections
  still cost about 1.47-1.52 single-row calls on M4 Pro; eight rows cost about
  2.46-2.69 calls. These are affine Q4/BF16 microbenchmarks, not GGUF/F32 Air
  results. They show both opportunity and residual per-row cost.
- [mlx-node's negative results](https://github.com/mlx-node/mlx-node/blob/main/docs/perf.md#verify-kernel-investigation--negative-results)
  document failed matrix, multi-row and asynchronous variants. Its separate
  same-checkpoint audit found a duplicate target forward that explained a large
  gap. Our verifier/chain inspection does not find that duplicate: one shared
  body call and reuse of final candidates already exist. Copying that scheduling
  fix would not remove a corresponding pass here.
- [Rigel](https://arxiv.org/html/2606.12765v1), a June 2026 single-M4-Max preprint,
  reports evidence consistent with Metal 4.1 tensor operations executing on
  shader cores rather than a separate matrix accelerator. It does not establish
  a controlled F16-versus-F32 matrix
  speed ratio for this Air. Apple's newer M5-specific accelerator results are
  not an M4 performance promise. F32 matrix instructions already exist and are
  used by this engine's prefill; precision alone is not the missing feature.

These are reported measurements and implementation evidence from other
configurations, not proof that our kernels are optimal. The current R4 dispatch
is two independent R2 tiles, and attention repeats its K/V loop per query row.
Those are actual structural opportunities. Their net benefit still depends on
resource use and the cost of the common R1 path.

### New execution proposal: separate GGUF storage from Metal consumption

Keep GGUF as the authenticated model source, but allow a different compact
immutable representation for resident projection weights. The file byte layout
need not be the runtime's internal memory layout.

Reorder packed codes by the output-column/K-fragment order the Metal tile
consumes. Put codes and scale/min metadata in separately aligned planes. Q6's
210-byte record currently forces safe `ushort` loads; plane separation removes
that record-alignment obstacle. Q4 can retain its metadata or expand 12 packed
scale/min bytes into 16 plain bytes: 144 becomes 148 bytes/block, 2.78% larger.
This trades some traffic for fewer bit extracts and must earn its cost; a pure
permutation saves no logical weight bytes. [FLUTE](https://arxiv.org/abs/2407.10960)
supplies prior art for execution-oriented rearrangement, under different
CUDA/codebook assumptions with no transferable Air speedup.

Conversion must preserve every code, scale, minimum and tensor meaning. Stream
into a replacement allocation; do not keep a second complete resident model.
Keep source identity and resident-layout identity distinct, account for
conversion peak memory/startup, and preserve tied embedding/head ownership.
Starting with transformer projections avoids forcing an embedding-layout change
just to evaluate the mechanism.

Do not permanently expand codes to int8 or float weights. Q4 int8 payloads with
the original metadata grow 144 to 272 bytes/block (+88.9%); Q6 grows 210 to 274
(+30.5%). Fully folded int16 weights are larger still. Keep compressed weights
resident and create only bounded on-chip fragments during execution.

A corresponding R4 vector candidate assigns query rows to separate SIMD groups
with roughly R1-sized per-thread accumulators, sharing a bounded integer-code
tile and native metadata. This differs from increasing per-thread row arrays.
Preserve each row's K-worker assignment, partial-dot order and final reduction.
Staging, barriers and additional reads may still outweigh decoding savings.
Repacking can preserve current float arithmetic exactly; it is useful only if
it reduces meaningful instruction cost or enables useful row reuse without
excessive live state. Neither layout nor staging alone proves a large gain.

### Original algebraic candidate: fold integer scales before the matrix product

The current vector kernels already factor native scales after short integer
dots. Proposing only "factor the scales" repeats existing work. This distinct
candidate changes the matrix operands so floating scale epilogues can occur per
256-weight superblock instead of per 16/32-weight local group.

For Q4_K, within superblock b and local group g:

```
w[b,g,i] = d[b] * s[b,g] * q[b,g,i] - dmin[b] * m[b,g]
P[b,g,i] = s[b,g] * q[b,g,i]                 # integer 0..945
u[r,b,g] = sum_i x[r,b,g,i]
y[r,n] += d[b,n] * sum_(g,i) x[r,b,g,i] * P[b,n,g,i]
          - dmin[b,n] * sum_g u[r,b,g] * m[b,n,g]
```

For Q6_K, use `P = s * (q - 32)`, an integer in `[-4064,4096]`, and
`y[r,n] += d[b,n] * sum_(g,i) x[r,b,g,i] * P[b,n,g,i]`.

Decode/fold P once into a tile consumed by multiple rows; retain F32 X and F32
accumulation. Q4's correction is an eight-term dot per superblock/output, with
activation group sums shared across output columns. The old rejected matrix
path constructed fully scaled float weights before multiplying; this formulation
uses folded integer-valued operands instead. An F32 implementation still uses
32-bit operands, so on-chip staging volume is not automatically reduced.
A pure-integer check covered all folded Q4/Q6 code ranges
and 64 synthetic full blocks (17,472 checks total); both algebraic identities
held exactly. This checks derivation, not floating-point parity or performance.

There are substantial reasons this could still fail:

- Folding uses one integer multiply per weight. At R4, Q6 folding introduces
  256 integer products per block/output versus roughly 80 scale-related float
  products in the retained formulation. The gain must come from execution
  efficiency, not a claim that this necessarily reduces operation count.
- An eight-row matrix tile is only 12.5%, 25% or 50% occupied at R1/R2/R4.
  Historical isolated padded F32 matrix costs were about 154/152/162 ms versus
  approximately 77/103/204 ms for the old vector family. Its better R4 ratio
  partly came from worse R1. Do not repeat that mistake or compare ratios alone.
- Q4 folded integers fit exactly in half, but arbitrary F32 activations do not.
  Q6 folded integers above 2048 are not all exactly representable in half either.
  Multiple half components require extra products, and no inspected M4 evidence
  establishes a compensating throughput multiplier. Evaluate F32 first.
- Algebraically identical real expressions are not identical float programs.
  Superblock factoring changes rounding, cancellation and reduction. An ordinary
  vector implementation is not automatically equal to a matrix verifier. A
  viable design needs a common per-row arithmetic contract, exact ordinary/
  verifier equality and the existing independent numerical bounds, without
  degrading R1. Failure here rejects the candidate.

This is the most distinct locally derived projection hypothesis, **not the
most proven fix**. It is a bounded competitor to compact-layout vector execution.
Do not build a new runtime around it before numerical and native-shape cost
tests pass. In particular, the cancellation fixture cannot be replaced by a
tolerance based only on the small final dequantized weights.

### Algorithmic redesigns: which work actually disappears?

| Proposal | Exactness/work test and decision |
| --- | --- |
| Replace H=1 rejection with common randomness or Gumbel coupling | Cannot exceed `1-TV(p,q)` for the same marginals. The current rule already attains maximal overlap. |
| Remove or delay the bonus row | H proposals without a bonus give the same progress and target row count as H-1 proposals with a bonus, with extra drafting. Executing the second H=1 row only after acceptance costs `(1+a)T + D` for `1+a` tokens and loses batching. |
| Wider candidate fanout | A four-row depth-one tree emits at most two tokens, less than its measured 2.56-2.72 target-step cost even with perfect coverage. |
| Overlap all drafting | Making the measured 14.721 ms draft free improves current cycle throughput only 8.55%, about 1.077x ordinary at unchanged acceptance. Real overlap competes for the same GPU/DRAM and unknown corrections. |
| Eliminate all controller overhead | Only about 1.625 ms/cycle remains outside draft and verifier: a 0.88% cycle-throughput ceiling. This cannot supply tens of milliseconds. |
| Certify a token after approximate target attention | A correct token does not certify the hidden states/KV needed next. Exact accepted-state repair can reinstate body work. Changing target attention changes the target law. |
| Reuse rejected suffix KV after a correction | Those states have the wrong causal prefix and are invalid under the corrected prefix. |

[Block verification](https://arxiv.org/abs/2403.10444) and
[Sequoia](https://arxiv.org/abs/2402.12374) improve use of computed chains/trees;
neither removes their body cost or rescues these inequalities.
[Drafter-invariant decoding](https://arxiv.org/abs/2408.07978) changes coupling,
not the maximal-overlap bound. Approximate body certificates must cover future
state; [runtime-certified attention](https://arxiv.org/abs/2605.20868) bounds a
local attention result, not the complete model's sampling distribution.

Two additional original/adapted ideas are technically plausible but secondary:

1. **Recycle target distributions as proposals.** Cache compact distributions
   from evaluated rows by their hypothetical suffix. Choose a cached law before
   drawing, record its actual proposal masses, then use the full target verifier
   and correction. Stale matches reduce acceptance, not correctness; never reuse
   their KV as target state. This extends
   [token recycling](https://arxiv.org/abs/2408.08696) to stochastic probability
   receipts. A 1,024-by-256 token/mass cache is roughly 2 MiB before metadata.
   With current costs, a free proposer needs about 50.3% overlap for a 2% gain
   over ordinary; unchanged 58.7% overlap predicts only 7.7%. Cold starts and
   stale context may defeat it. This is not the core redesign or a claimed first.
2. **Certify sparse target-head evaluation after exact body/KV execution.**
   Bound omitted head-weight blocks to prove the actual top-k support before
   ordinary sampling. This avoids body-cache debt, but high-dimensional bounds
   may be loose, search/index costs may defeat streaming, and the head is only
   11.3% of target weight bytes. Certificates must cover actual F32 arithmetic,
   sampling transforms and tie order. Without finite top-k, further exact
   normalization/sampling machinery is required. An unproved shortlist is not
   a certificate. [Exact MIPS research](https://arxiv.org/abs/1706.01449) also
   finds cases where optimized brute-force execution beats pruning.

A simpler established supporting mechanism is reducing only the **draft**
vocabulary ([FR-Spec](https://arxiv.org/abs/2502.14856),
[VocabTrim](https://arxiv.org/abs/2506.22694)). Its 262,144-by-640 Q8 tied head
contains 178.26 MB, 62.5% of draft weight bytes. A 32K indexed shortlist reads
22.28 MB/head; arbitrary input/correction IDs still require the full tied table
resident. Record actual renormalized q, zero outside the shortlist; full target
residual sampling preserves p. Saving 4 ms at current cost permits only a
3.4-point acceptance drop to break even. This can support a viable block but
cannot make the 170.6 ms verifier cheap.

### Substantial execution change: shared blocked attention

Make a useful block share work throughout the target, particularly its FFN
projections and attention. `packed_row_dispatch` in
`metal_command_context.mm` currently divides R4 into independent R2 tiles.
Those tiles each decode the same packed weights; caches may already avoid
some repeated DRAM traffic, but integer decoding and row work still recur.
Earlier isolated projection diagnostics put roughly 70% of projection time in
the FFN. They justify investigating block projection efficiency, not claiming
an exact stage breakdown of the final chat run.

The corresponding attention change is a common blocked attention core:

- Load a logical K/V tile for several verification queries and both GQA heads.
  The existing kernel already shares K/V between the two heads, but launches
  separately per query row and rescales its output accumulators after every key.
- Compute a tile of scores, normalize by its maximum and update the accumulated
  output once per tile. Four rows times two GQA heads naturally fill an
  eight-query Metal matrix tile. Splitting the 256-wide head across SIMD groups
  is a concrete way to limit live matrix state.
- Give each row its own causal, local-window and private-suffix mask. Fixed
  logical-position tile boundaries and consistent reductions must make an
  ordinary row agree with the same row in a verifier batch, including partition
  transitions and ring wrap. Do not reuse one query's attention output for another.

This is an adaptation of existing mechanisms, not a claimed new invention.
The September 27 [mlx-dspark multirow attention implementation](https://github.com/ARahim3/mlx-dspark/blob/e918d401c6a9c5374b9108ee0649c003b78fb8da/src/mlx_dspark/multirow_attn.py)
uses this query/GQA matrix packing, blocked softmax and split head dimension.
Its source reports attention and whole-verifier gains on M4 Pro, but uses
BF16/F16 operands and rounds probabilities before P-times-V. It does not prove
our F32 ordinary/verifier contract or performance on M4 Air.
[Current llama.cpp Metal attention](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/ggml/src/ggml-metal/kernels/fa_vec_common.metal)
also reuses K fragments and updates softmax in blocks. The earlier
[two-query PR](https://github.com/ggml-org/llama.cpp/pull/23114) reports lower
L1 traffic without corresponding DRAM reductions in its counter example:
source-level sharing does not imply proportional memory-bandwidth savings.

The execution design must keep ordinary decoding fast. Padding its one row to
a wide matrix can erase the gain; a previous padded F32 projection experiment
already demonstrated this problem. A larger speculative/ordinary ratio obtained
by slowing ordinary decoding is unacceptable. Any coordinated arithmetic change
needs the existing independent numerical bounds and new ordinary/verifier parity.

F16 KV remains a useful independent storage policy: 1.3125 GiB saved at 16K.
It halves KV storage/loads, not FFN weights or the whole verifier's cost. F16 KV
does not authorize narrowing activations or attention probabilities. If a future
compute-precision policy is proposed, define and validate the ordinary target
under that policy as well; do not silently narrow only verification.

### Substantial proposal change

A target-conditioned parallel drafter can reuse hidden features that the 12B
target has already computed, and propose several positions with one small
backbone pass. This addresses limited multi-step overlap and sequential drafting
together. [DFlash](https://arxiv.org/abs/2602.06036) is the established foundation;
[DFlash 2](https://inco.ai/blog/dflash2/) adds cheap neighboring-position coupling
and candidate-path selection. Its [original implementation](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)
contains non-greedy p/q rejection and residual sampling. These mechanisms need
target-specific trained weights; the inspected Gemma releases target Gemma 4,
not this Gemma 3 12B checkpoint. Published speedups are not predictions for our pair.

Reuse `feature_bank.*` and target feature taps, `chain_chat.mm`, target verifier
transactions, the sampling laws and rollback infrastructure. Implement the new
drafter's native feature projection, block execution, cache and probability
receipts at the draft boundary. Preserve the working 270M path for comparison.
Pin the target checkpoint, quantization, feature definition and sampling policy
in the training/evaluation artifacts; full-precision teacher agreement alone
does not establish overlap with this engine's actual target.
Budget the feature cache, vocabulary head and drafter weights before training;
a five-layer target-width drafter is not automatically cheaper than 270M. A
bounded feature-memory variant is a training adaptation to evaluate, not an
assumed loss-free compression of the target's own cache.
Training or distilling the existing 270M is another possible way to improve q,
but it does not by itself remove its sequential drafting cost. Substantial
training remains user-run; do not start it as a runtime optimization experiment.
[DistillSpec](https://research.google/pubs/distillspec-improving-speculative-decoding-via-knowledge-distillation/)
supports distillation of a standalone autoregressive drafter without changing
its serving architecture. For our H=1 path, the proposal is conditioned on a
valid target prefix, so representative target-prefix distribution matching is
a sound starting point; an elaborate rejected-suffix training system is not a
prerequisite. Judge checkpoints by held-out overlap and cost under the deployed
target/sampling policy, not the drafter's standalone answer quality. The actual
achievable improvement and training resource cost for this pair remain unknown.

[DSpark](https://arxiv.org/html/2607.05147v1) provides a useful confidence/cost
scheduling design and a crucial exactness warning: choosing draft length after
looking at later sampled candidates can introduce selection bias. Decide each
position's admission before drawing that position, or derive the appropriate
correction. The paper's Appendix A contains a counterexample. Fixed horizons
avoid that new issue. [Block verification](https://arxiv.org/abs/2403.10444) can
improve progress from already-computed multi-token blocks without extra target
queries; it offers nothing new at H=1 and does not make target execution cheaper.

### Proposed next phase and stop condition

The next substantial execution task should be judged by a cost curve and useful
progress, not a collection of isolated kernel wins:

1. Attribute the existing complete verifier's projection/head, attention and
   other costs at fixed real cache states, including sustained conditions. Use
   perfect-proposal replay to establish the architecture's optimistic ceiling;
   it is a diagnostic, never a production speedup result.
   Separate cold from sustained cost at the same prefix: do not infer an
   attention bottleneck from a longer second conversation turn. Capture GPU
   timing and available clock/thermal/resource evidence without claiming a
   temperature or bandwidth measurement that the tooling did not obtain.
2. If attribution identifies substantial removable verification work, design
   native block execution around that cost and evaluate the complete change
   with identical states and unchanged precision contracts. If the verifier
   already remains close to one ordinary pass for two rows under sustained
   conditions, prioritize a bounded distillation pilot for the existing 270M
   over an open-ended kernel rewrite. Use the measured payoff thresholds to
   choose; do not require a kernel improvement merely to justify training.
   For projections, compare compact-layout vector execution and the folded F32
   matrix formulation on actual FFN shapes, including R1. For attention, use
   fixed logical tiles and shared queries with unchanged F32 probabilities.
   Reject a candidate before full integration if its canonical numerical path
   fails, it makes ordinary execution materially worse, or its attributable
   saving is too small to address the chosen throughput objective. This is one
   feasibility phase with a decision, not an indefinite sweep of variants.
3. Pair the cost curve with held-out accepted-prefix survival and actual draft
   expense. A new target-conditioned parallel drafter is justified only if
   it offers enough additional return over the existing architecture to cover
   its implementation, training and memory costs. Validate a retained change
   in cooled real chat, including follow-up, against the current ordinary path.

Reuse existing replay tools rather than build a general profiling product.

For scale, the recorded follow-up needs cycle cost at most 154.5 ms for a
20% ordinary-throughput improvement at unchanged acceptance. Holding D/O fixed
means reducing V from 170.6 to about 138.2 ms, approximately 19%. If attention
were only 20% of V, even halving it would not meet that objective: the planning
speedup would be about 1.09x. If it were 40%, halving it would yield about 1.21x.
Those fractions are deliberately hypothetical until attribution exists. The
same Amdahl calculation applies to unpacking/repacking. A target of 20% is an
opportunity-size example, not a prediction or replacement for the repository's
actual retention gate. Updating ordinary execution also updates the denominator.

The implementation boundary, if that next phase is authorized, is concrete:

| Area | Relevant code/tests and responsibility |
| --- | --- |
| Resident execution layout | `src/engine/gemma2_metal_plan.cpp` upload/admission/identity and Metal buffer weight metadata; preserve source authentication, tied ownership and bounded conversion memory. Do not disguise a new layout as unchanged raw GGUF bytes. |
| Projection and attention kernels | `src/tensor/backends/metal_backend.mm`, `metal/metal_context.*`, `metal/metal_command_context.*`; common numerical core, layout-aware dispatch, bounded scratch and row-specific masks. |
| Shared target integration | `src/engine/gemma_target_runtime.*` and `target_verifier_runtime.cpp`; one body execution, full target distributions, existing private-row publication. |
| Numerical proof | `tests/tensor/tensor_backend_tests.cpp`, `tests/tensor/gemma3_metal_tests.cpp`; reversible packing, real native shapes, cancellation, tails, exact R1/R2/R4 candidates and all-layer KV, independent CPU-reference bounds. |
| Attention/lifecycle proof | Existing target runtime/verifier tests: logical partition boundaries, local-ring wrap, mixed accepted/rejected suffixes, partial publication, stops, callbacks, context capacity and unavailable Metal behavior. |
| End-to-end decision | Existing exact-state replay plus cooled actual chat, source/binary identities, complete generation time and native memory; `bash scripts/ci.sh` after implementation. No performance claims from isolated kernels alone. |

Do not replace the sampler or controller to accommodate a numerically different
verifier. Change ordinary and verifier arithmetic together only when the common
new evaluator remains within the independent target contract. Keep probability
receipts and rollback infrastructure. F16 KV can be added independently later;
neither integer folding nor row-shared attention requires an F32-only KV layout.

Reject a block when its target cost alone exceeds ordinary cost times its
expected useful progress; drafting cannot rescue it. If even perfect proposals
cannot offer a substantial margin, stop that design before training. Do not
expand into additional controller tweaks, larger trees, sparse target attention
or tiny sampler experiments to avoid that conclusion. The existing repository
retention gate still applies; no new performance result is claimed here.

## Previous state: speculative performance audit completed

The user's follow-up requested an audit of the slower pairing and correction of
bad assumptions or avoidable work. The audit found a real first-token scheduling
problem, fixed it with cache/lifecycle coverage, and rejected two unproven kernel
changes. **A retained speculative generation speedup is still not established.**
Ordinary decoding remains the default; the complete native pair remains an
explicit experimental option. No model download, training or external production
runtime was introduced. Earlier uncommitted implementation work is preserved.

### Retained implementation

- `gemma2_runtime.cpp` now validates direct-ID context/capacity before mutation,
  caches and journals the first ordinary target token, invokes its callback,
  then synchronizes the drafter before another cycle or any return. Previously
  draft synchronization and the initial speculative cycle delayed publication.
  Deferred work remains charged to complete generation; this is not a timer
  exclusion. A first selected token need not contain visible text bytes.
- `standalone_chat.*` supports checked pending observation. EOS, one-token
  budgets, callback stop and callback exceptions leave a coherent target/draft
  prefix. A valid cached continuation can already end in a pending draft token;
  that case materializes the known rows once, records only the new target token,
  and poisons the owner on an uncertain operational failure.
- `standalone_chat_tests.cpp` checks publication order, independent target/draft
  cache parity, checkpoints, every exit, exact-prefix continuation and subsequent
  reuse after a throwing callback. `tensor_backend_tests.cpp` retains native
  15/16/60-block projection/head coverage with scalar bit parity, independent
  dequantization, source-row offsets, odd tails and row-boundary guards.

Target transformer arithmetic, target sampling transforms and the proposal q
law are unchanged. Different seeded text remains allowed; speculation must
preserve the conditional target distribution, not merely sound plausible.

### What the audit established

The earlier **−3.56%** cooled result was a valid failed retention result for that
configuration. It did not establish an optimized limit. Natural stopping lengths
and follow-up cache positions differed; clocks/temperature were not captured.
Later-turn GPU costs rose about 45% in both modes, so those numbers do not isolate
thermal effects from context and workload. Aggregate speculative GPU timing also
includes the drafter; `learned_target_gpu_ns` is the target-only subtotal.

No duplicate accepted-row target replay, accidental extra final draft forward or
extra host wait between autoregressive draft depths within a batch was found.
Each draft batch still submits and waits once. Existing pending/root batching
and the callback publication boundary are sound. The measured repetition penalty was **1**, so
bounded versus full repetition history cannot explain that run's overlap/cost.

A bounded ignored diagnostic now restores and byte-checks the exact physical KV
slots that replay overwrites, including local-ring rows. It uses one real prefix
prefill, identical tokens/selection requests, warm ordinary and verifier paths,
and AB/BA ordering. Extra parity reads/restores are outside timing. At prefix
positions 16 and 1,536, the original two-row verifier plus publication cost
**1.37–1.43 ordinary-token GPU times**; four-row cost was **2.67–3.65**. Candidate
and hidden values were finite and ordinary/verifier candidates matched exactly.
This isolates compute shape; it omits drafter residency/authentication and is
not a sustained chat throughput result or proof of a universally optimal H.

### Tested and removed

1. A streaming packed-weight-sharing kernel kept exact scalar arithmetic but
   made two-row verification cost **3.06–3.19 ordinary-token GPU times** and
   four-row verification **8.24–10.45**. It reloads activations and serializes
   row arithmetic; register spilling was not measured. Repeated unpacking in
   the existing parallel kernel is therefore not, by itself, proof of a bad
   implementation. The rejected shader/pipelines were removed completely.
2. Fixed 15/16/60-block specializations of the existing parallel verifier
   preserved exact outputs but added ten pipeline states. Short-context cost
   overlapped the original range; longer-context measurements varied. No
   isolated retained benefit was established, so those states/routing were
   removed after the cooled candidate also failed.
3. Compact draft sampling still encodes 18 guarded full-fallback dispatches,
   totaling **15,364 threadgroups** per compact head that immediately return.
   A candidate wrote packed indirect grid arguments on the GPU and used zero
   groups on compact success, retaining the original grids on overflow. It
   preserved exact q receipts for compact cutoff ties, 257-way overflow and
   general top-512 sampling. Eight-call timings were noisy, including material
   variation in the unchanged general path; no standalone speedup was proven.
   This additional buffer/dispatch machinery was removed. Full fallback storage
   and guarded dispatches therefore remain in the final runtime. The API layout
   was checked against [Apple's indirect-dispatch documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(indirectbuffer:indirectbufferoffset:threadsperthreadgroup:)?language=objc).

The matrix-prefill kernel was not substituted only for verification: the
current ordinary/verifier conditional contract requires exact continuation
from the same cache. Merely retaining F32 operands does not prove identical
finite arithmetic after a different reduction order. No tolerance was relaxed.

### Cooled actual-chat comparison

One fresh ordinary baseline and one candidate each had **240 seconds of idle
cooldown**, with no concurrent build/test/GPU work. Both used the canonical
natural-stop, seed-42 two-turn chat, 16K capacity and retrieval disabled. The
candidate contained the retained publication fix **plus the subsequently removed
fixed-block and indirect-dispatch experiments**. These are measurements of that
candidate, not a throughput claim for the final smaller change.

| Metric | Ordinary baseline | Measured H=1/full candidate |
| --- | ---: | ---: |
| First / follow-up selected tokens | 1,506 / 1,026 | 1,344 / 1,024 |
| First / follow-up complete generation seconds | 124.917568 / 119.527071 | 108.124577 / 130.950095 |
| First / follow-up complete generation tokens/s | 12.056 / 8.584 | 12.430 / 7.820 |
| Aggregate complete generation tokens/s | 10.358 | 9.905 |
| First / follow-up reported TTFT seconds | 0.303433 / 0.642646 | 0.302998 / 0.628943 |
| First / follow-up first visible text seconds | 0.402299 / 0.760319 | 0.380548 / 0.726980 |
| Observed native lifetime maximum physical footprint bytes | 10,447,776,616 | 11,032,653,296 |

Generation changed **+3.10% / −8.90% per turn, −4.38% aggregate**. First visible
text improved **5.41% / 4.38%**. The first ordinary publication precedes the
candidate's speculative kernels/sampler; its ordering fix is retained, while
unsupported throughput candidates are removed. This did not meet the generation
retention gate and does not justify enabling speculation by default.

Candidate first/follow-up cycles were 805 / 644, with 537 / 378 accepted proposals
(**0.667 / 0.587 per cycle**). Draft wall cost was **9.321 / 13.896 ms per cycle**;
target verification/publication GPU cost **123.408 / 187.368 ms**; whole cycles
**133.968 / 202.891 ms**. On follow-up, baseline complete generation averaged
116.5 ms/token and speculative progress averaged about 1.587 tokens/cycle:
target verification/publication alone nearly used that time allowance before
adding drafting and other cycle work. This is a cost/overlap problem in the measured sustained run, not
proof that eliminating small draft overhead or raising an acceptance percentage
alone will deliver useful speed.

Baseline hashes remained `260ae35592bb6060` / `dd4aadff1d4f599f`; candidate hashes
were `3a9063eeaaadedd9` / `ecc10d1344ecd5e7`. Follow-up cache positions were
1,522 versus 1,360. **Different natural-stop work and preceding histories mean
this is not an exact-work comparison**; one pair does not establish variance or
separate context from thermal behavior. Both native footprints were captured
with `proc_pid_rusage`; the outer Python driver's `/usr/bin/time` footprint is
not the engine's. System swap/VM snapshots and thermal-warning status are retained;
absence of a warning does not prove equal GPU clocks or no throttling.

### Verification, evidence and boundary

Final CI after removing the experiments passed **467 C++ and 60 Python tests**,
with Metal, authenticated 12B target goldens and actual 270M checkpoint goldens
enabled. `git diff --check` passed. The final executable also completed a native
two-turn, 96-token-per-turn paired smoke check: 61 / 64 speculative cycles,
cache frontiers 112 / 228, and zero numerical refusals. Its hashes were
`2328915662618d86` / `014e8fa3d926796f`. This bounded smoke is not a new cooled
throughput comparison of the final binary.

All starting source files and executable hashes were preserved under
`build/gemma3-speculative-audit-2026-09-27/`. Rejected candidates survive only as
ignored patches/binaries for provenance, not runtime switches. Raw diagnostics,
cooldown protocol, native resource samples and the comparison JSON are under
`benchmarks/results/gemma3-speculative-audit-2026-09-27/`.
The measured candidate SHA-256 was
`8dea3f60661738a5bd920d3c72543f8fca6050b420819bba7a4521a8597b34cd`;
the frozen ordinary baseline remains
`ecd342d8c41d6ba23ddd7737cb77df7c6c8a96910544272997e514c3e6f83f59`.
The final retained executable SHA-256 is
`7b321ed3780ca378f04acd706cd01c99a578db203434d46a22b8ff4417c23dd4`.

This audit's implementation, verification and reporting boundary is complete.
Further work should be
chosen from measured needs: actual p/q overlap on representative durable chat
prefixes, sustained verifier/ordinary cost under comparable context and resource
conditions, and consistent KV precision if pursuing memory savings. F16 target
KV at 16K would save **1,409,286,144 bytes (1.3125 GiB)**, but requires identical
rounding boundaries for durable, current-chunk and verifier-private KV before
attention, plus independent numerical/rollback evidence. It is not implemented
here. No claim that this pipeline is fully optimized, or that the pair can never
be useful, follows from this audit.

## Historical completion record: native Gemma 3 speculative integration


The authorized implementation now connects the actual Gemma 3 12B Q4_K_M
target to the complete existing Gemma 3 270M Q8_0 drafter through native Metal
chat. Full CI and the cooled real-chat comparison are complete. **The measured
pair did not earn a retained chat speedup:** H=1 regressed aggregate generation
3.56% and also regressed latency. The functional pairing remains an explicit
experimental option; ordinary target decoding remains the default. No model was
downloaded and no training was started. Prior uncommitted target work is retained.

### Implemented ownership and mechanisms

| Owner | Implemented behavior |
| --- | --- |
| `standalone_draft_artifact.*` | Authenticated direct-ID compatibility checks all 262,144 token strings, score bits, types and consumed special-token semantics. Only the two verified image-marker type differences are allowed. Total history/correction conditioning remains separate from proposal eligibility; the generic Gemma 2 bridge remains supported. |
| `standalone_chat.*`, `tokenizer.*` | Direct-ID prompt/correction synchronization removes the generic text workspace. The drafter inherits the target's bounded repetition window in direct mode, including a zero window; generic cross-family mode retains full native history. Native saved checkpoints preserve pending-token state. |
| `gemma3_draft.*` | Global cache/history capacity follows the prepared session, with independent F32 512-row local rings. Pending/root ingestion shares the proposal command transaction. Independent projection/norm work joins at actual dependencies; one-partition attention writes directly using the established reduction order. |
| `gemma3_draft.*` sampling | For top-k 1–64, bounded candidate merges find the cutoff and compact all ties when support fits 256 entries. Selection and q lookup use the same integer CDF intervals totaling 2^32. Larger support, top-k zero/large k and excessive ties select the full fallback before sampling. Invalid finite arithmetic poisons the transaction. |
| `chain_chat.*`, `gemma2_runtime.*` | Actual horizon is bounded by prepared capacity, remaining output and both contexts before random draws. Full/conditional schedules reuse the prepared arena, including shorter horizons with awkward tile boundaries. Zero horizon uses coherent ordinary continuation; callbacks, stopping, residuals, bonus ownership, rollback and epochs stay under the existing controller. |
| `prepared_gemma2_plan.*`, target/draft/chain estimators | Combined admission runs before target resident upload. Checked estimates include tied weights once, target/draft caches and scratch, private/undo rings, selected host ledgers and the lazy ordinary 128-row workspace. Actual standalone owned payload is checked against its estimate. |
| `runtime/sha256.*` | macOS CommonCrypto replaces scalar SHA-256 while retaining every whole-file, canonical source and resident receipt. Updates are chunked for the 32-bit CommonCrypto length API. No external inference library or new package is used. |
| CLI and `quick_benchmark.py` | Diagnostic schema 6 reports actual horizon/proposal/exposure counts and the effective repetition policy/window. The reader still validates older schema 3–5 results; bounded/censored outcomes are distinguished from rejection. |

The target transformer and sampling law were not approximated for speculation.
The optimized draft's finite-grid law is explicitly versioned as
`explicit_or_target_scalars_target_window_v2` for direct Gemma 3 pairing and
`explicit_or_target_scalars_native_history_v2` for generic TLI. Different seeded
answers are permitted; incorrect q receipts or a different conditional target
distribution are not.

The full fallback sampler buffers remain allocated, and its guarded dispatches
remain encoded. Compact sampling reduces executed sorting/filtering work; it
does not eliminate all full-vocabulary storage or launch overhead. No unmeasured
learned router, lazy draft tiles, Q8 prefill matrix kernel, head fusion, joint
block-acceptance algorithm or trained parallel drafter was added. Active horizon
currently adapts to capacity/budget, not an uncalibrated learned cost forecast.

### Verified resources and correctness

- At 16K, target weights/KV/ordinary scratch remain respectively
  **7,294,024,704 / 2,818,572,288 / 298,019,040 bytes**. The H=1 standalone
  owner reports **524,006,312 bytes**, including the draft and chain storage.
- Session-sized draft KV is **116,391,936 bytes** at 16K, exactly 96 MiB less
  than a fixed 32K draft cache. Direct H=3 transaction KV/norm storage is
  **2,523,136 bytes**, versus the former 11,511,808-byte bound.
- Actual 32K H=1 paired preparation refused **13,256,915,192 accounted bytes**
  against **12,713,033,728 available budget bytes**, before target upload. This
  does not remove target-only 32K support or impose an 8K context cap.
- Admission is an accounted payload/selected-host-storage estimate, not exact
  RSS. Parsed GGUF containers, some verifier host objects and allocator overhead
  are outside that ledger; mapped weight pages are reclaimable and are not
  falsely counted as duplicate resident device weights.
- Full release CI: **464 C++ tests and 60 Python tests passed**, with Metal
  and the authenticated target goldens enabled. The actual 7.3 GB whole-GGUF
  digest test also exercises the CommonCrypto update-length boundary. Published
  SHA vectors, streaming boundaries and state lifecycle passed.
- Coverage includes all-ID bridge refusals, exact compact/full proposal masses
  and ties, independent residual/sequence laws, complete draft checkpoint
  logits/KV, configured context endpoints, local-ring restoration, prepared
  schedule bounds, callback/failure/stop paths, direct repetition-window
  behavior, public paired chat, regeneration and follow-up. Existing target
  goldens and ordinary/verifier parity bounds were unchanged.
- Actual paired chat also completed a **1,516-token prompt** with both local
  rings wrapped and a natural 12-token answer. This is execution evidence, not
  a long-context throughput retention claim.

Verification command:

```sh
PATH="$PWD/.stage5a-venv/bin:$PATH" MODE=release MAKEFLAGS=-j2 \
  LOCAL_GGUF_GEMMA3_TARGET_GOLDENS=build/gemma3-target-reference bash scripts/ci.sh
```

### Measured diagnostics and provenance

Bounded 96-token real-chat screens, without thermal retention claims:

| Configuration | Accepted proposals/cycle | Draft ms/cycle | Target GPU ms/cycle | Complete generation seconds |
| --- | ---: | ---: | ---: | ---: |
| H=3 full | 0.979 | 19.690 | 203.769 | 10.874 |
| H=3 conditional | 0.979 | 19.812 | 135.260 | 7.629 |
| H=1 full, final binary | 0.548 | 7.779 | 106.449 | 7.195 |

The two H=3 schedules produced identical diagnostic token hashes. H=1 was the
candidate selected for cooled measurement; this does not prove a universal
optimal horizon. With the same H=1 output hash, total process time changed from
78.444 to 27.993 seconds after native hashing; time outside the measured turn
changed from 70.866 to 20.466 seconds. This includes preparation and process
overhead, not an isolated hash microbenchmark or time to first text after
process launch. All authentication streams remain intact.

Source/binary provenance and CI logs are under the ignored
`build/gemma3-speculative-integration-2026-09-27/`. Diagnostic/measurement records
are under `benchmarks/results/gemma3-speculative-integration-2026-09-27/`.
The baseline executable SHA-256 is
`ecd342d8c41d6ba23ddd7737cb77df7c6c8a96910544272997e514c3e6f83f59`;
the final candidate is
`0deb37ee7ee2b7a91ac9e97d1c19fa8e0c36ecf1dc455e359d0ab7251d49101b`.

### Cooled chat result and retention decision

One fresh baseline and one fresh H=1/full candidate each received **240 seconds
of idle cooldown**, with no concurrent build, test or GPU work. Both ran the
canonical two-turn seed-42 chat to natural stop with 16K context, identical
target sampler settings and retrieval disabled. Complete-generation clocks
include final cache work, callbacks and result assembly.

| Metric | Target-only baseline | H=1 lossless pair |
| --- | ---: | ---: |
| First / follow-up generated tokens | 1,506 / 1,026 | 1,450 / 1,076 |
| First / follow-up generation seconds | 125.425742 / 124.502266 | 121.761368 / 136.786410 |
| First / follow-up complete generation tokens/s | 12.007 / 8.241 | 11.909 / 7.866 |
| Aggregate complete generation tokens/s | 10.131 | 9.770 |
| First / follow-up prompt tokens/s | 52.765 / 29.767 | 48.498 / 27.674 |
| First / follow-up reported TTFT seconds | 0.303234 / 0.638285 | 0.329911 / 0.686563 |
| First / follow-up first visible text seconds | 0.388637 / 0.744855 | 0.447373 / 0.833323 |

Generation changed **−0.82% / −4.54% per turn, −3.56% aggregate**. First visible
text regressed **15.11% / 11.88%**; prompt throughput also regressed. This fails
the repository gate. The promising short-screen rate did not translate into a
win over ordinary decoding on the complete chat, so no default speedup or
separately retained kernel-speed claim is made.

Both evaluated 16 then 19 prompt tokens. Follow-up retained contexts differed:
1,522 baseline versus 1,466 candidate tokens. Selected-token hashes were
`260ae35592bb6060` / `dd4aadff1d4f599f` for baseline and
`e71edc6518706d78` / `bfd3725b3a34a270` for candidate. Lossless answers and
natural stopping lengths may differ; **this is not an exact-work comparison**
and one pair does not establish variance or universal behavior.

Actual first/follow-up draft cycles were 861 / 671, with 588 / 405 accepted
proposals (0.683 / 0.604 per H=1 cycle). Draft wall cost averaged
9.996 / 14.173 ms per cycle; target verification GPU cost averaged
129.342 / 187.323 ms; complete cycles averaged 141.179 / 203.757 ms. Target
verification therefore accounts for about 92% of measured cycle time. The
observed accepted progress did not repay combined verification/draft/publication
cost. Draft prompt synchronization additionally cost 23.903 / 87.734 ms,
with no rebuild. Acceptance alone would have hidden this result.

The candidate's native lifetime peak physical footprint observed through macOS
`proc_pid_rusage` was **11,019,496,920 bytes** (about 10.26 GiB). Its native RSS
sampling peak was 8,088,862,720 bytes; RSS is not GPU physical footprint. The
baseline native footprint was not captured in this comparison; the outer
Python driver's `/usr/bin/time` footprint must not be presented as the engine's.
System swap usage changed 2,924.62 → 2,429.62 MiB across baseline, then
2,429.62 → 2,656.31 MiB across candidate. These are system-wide observations,
not proof of process-specific swapping. Passive `pmset` observations reported
no recorded thermal/performance warning; they do not establish equal GPU clocks
or absence of fanless throttling.

Evidence: `cooled-baseline.json`, `cooled-candidate-h1.json`, their `.log` and
`.resources.json` files, `cooled-comparison.json`, and
`candidate-native-rusage.json` in the measurement directory above. The earlier
empty `baseline.json` / failed `baseline.log` came from a sandbox without a Metal
device and are not benchmark evidence. GPU checks and valid measurements ran
with actual native Metal access.

### Stop boundary and the next decision

The approved integration, optimization mechanisms, correctness checks and real
chat characterization are complete. Keep ordinary decoding as the production
default and the faithful complete 270M pairing available explicitly. This
result does not condemn speculative decoding or substitute a toy model for the
released drafter; it establishes the current pair's measured cost/overlap limit.

Further work should target useful progress per target-verification cost and
first-text latency. A cost/survival policy needs measured context/shape costs;
the current H=1 result does not justify extrapolating unseen deeper survival.
Initial draft synchronization and delayed first publication are concrete latency
owners to examine. A faithful joint block algorithm or a trained feature/parallel
drafter remains a separate architecture decision with its own probability proof,
native execution and training requirements. Do not start another broad
temperature/horizon sweep or substantial training as an automatic continuation.

The research below is preserved as the pre-implementation design basis; the
verified state above supersedes its descriptions of then-missing features.

### Completed speculative integration research, 2026-09-27

The research turn changed documentation only. At that handoff the production
Gemma 3 12B target and verifier worked, but the existing 270M drafter was not yet
connected. No implementation, training, model download, performance benchmark
or speedup claim belonged to that research turn. The subsequent user-approved
implementation is recorded above; future-tense descriptions below are historical
design recommendations, not the current completion state.

This section owns the speculative integration plan and supersedes the earlier
short following-baton notes and section 8's preliminary readiness discussion.
The completed pipeline review, measurements and original target design remain
below as evidence. Repository inspection used the current working tree, including
the uncommitted target integration and retained verifier optimizations, rather
than treating the old `dff16f0` research snapshot as current code.

### Recommendation and scope

Implement the **complete existing 270M autoregressive drafter with lossless
chain verification**, using the existing controller and target runtime. Make
direct token synchronization, bounded exact proposal sampling, session-sized
draft storage and efficient command scheduling part of that implementation.
Prepare reusable capacity once; choose actual proposal/verification work for
the current context and remaining output budget. Optimize useful committed
tokens per elapsed millisecond, including draft work, publication, rejection,
callbacks, prefill and memory pressure.

The largest concrete avoidable work found in this audit is in draft sampling
and synchronization, not an absent GPU drafting path: proposals already execute
on Metal as one batch of commands without a host wait per proposed token.
Target verification already shares weights across small row batches and owns
rollback correctly. Replacing those foundations would discard working value.

There is no evidence yet of a retained speedup for this model pair. Same-family
membership does not establish token semantics, distribution overlap or speed.
The earlier perfect-proposal target episodes imply only about 1.40–1.46x for
their particular H=3 workloads **before charging drafting and rejection**.
They are not an acceptance forecast, a context-wide cost model or a guarantee.
The integration should expose the actual limits rather than hide them behind a
toy drafter, greedy token matching or acceptance-only reporting.

| Priority | Proposed work | Reason and limitation |
| --- | --- | --- |
| Required integration | Authenticated direct-ID bridge, total correction/history mapping, shared lifecycle | The current bridge explicitly rejects a Gemma 3 target. Removing that guard alone is insufficient. |
| High-value draft work | Small-top-k exact support extraction; compact probability receipts with full-support fallback | Current sampled drafting sorts all 262,144 entries at every depth. Ties and actual finite-grid probabilities must remain correct. |
| Required resource work | Configurable draft context, checked combined admission, remove identity-mode text workspace | Saves known unnecessary storage without changing target context or numerical precision. |
| High-value scheduling work | Batch pending/root ingestion with proposal commands; join only actual dependencies; specialize one-partition draft attention | Removes known waits/passes. Retain only mechanisms whose whole-path evidence supports them. |
| Required verification work | Actual horizon, budget/context-tail handling, full/conditional cost accounting and coherent ordinary fallback | Fixed H can produce unused work; conditional target verification currently still drafts all H tokens eagerly. |
| Profile-dependent work | Q8 prefill row reuse, head/candidate fusion, lazy draft tiles, extra projection shapes | These are real mechanisms, but their cost/benefit on this GPU is not established. |
| Separate algorithmic candidates | Joint block verification; trained feature/parallel drafters | Potentially substantial, but require different probability/controller proofs or trained weights. They are not configuration switches for the current drafter. |

### What was inspected and what the actual pair permits

Read the operating contract, engine principles, current baton, retrospective
and graph locators; then followed artifact authentication, tokenizer mapping,
prepared-plan admission, the complete draft execution/sampling/transaction path,
standalone ledgers, chain verification, sampler/controller ownership, target
selection and relevant tests. Public primary literature and released code were
checked through 2026-09-27; file revisions as recent as 2026-09-26 are pinned
below. Research copies and the local metadata audit are ignored artifacts under
`build/speculative-integration-research-2026-09-27/`, not runtime dependencies.

The downloaded files were inspected again, without evaluating the models:

| Property | Target | Existing drafter |
| --- | --- | --- |
| File | `models/gemma-3-12b-it-Q4_K_M.gguf` | `models/gemma-3-270m-it-Q8_0.gguf` |
| Transformer | 48 layers, width 3,840, FFN 15,360 | 18 layers, width 640, FFN 2,048 |
| Attention | 16 query / 8 KV heads, head width 256 | 4 query / 1 KV head, head width 256 |
| Local/global pattern | Five local then one global; local window 1,024 | Five local then one global; local window 512 |
| Vocabulary | 262,144 | 262,144 |
| Context | Advertised 128K; runtime configurable, default 16K | Model limit 32K; current runtime allocation fixed at 32K |
| Resident weights | 7,294,024,704 bytes | 285,018,624 bytes |

All token strings and scores match by ID. Token types differ at exactly
255999 (`<start_of_image>`) and 256000 (`<end_of_image>`): target type 3,
draft type 4. The additional scalar tokenizer metadata difference is absent
versus false `tokenizer.ggml.add_sep_token`. These findings support a narrowly
validated direct-ID mode; they do not justify globally weakening tokenizer
identity or assuming every control token is an ordinary proposal.

The target has 6,251 type-3 entries. The current Gemma 3 tokenizer recognizes
eight named special-token spellings, while `gemma_mapping_key` excludes other
type-3 entries from ordinary proposal mapping. Therefore audit **all IDs and
their conditioning behavior**, including reserved controls, unknown/BOS/EOS,
turn markers, byte fragments and the two image markers. Keep two contracts
separate: how every target history/correction token conditions the drafter, and
which tokens the drafter may propose. It is valid for q to have narrower
support; it is not valid to narrow the target's p to improve acceptance.

No KV or embedding sharing follows from the family name. The weights, hidden
widths, KV-head counts, local windows and positional configurations differ.
Both models must keep their own correctly positioned caches.

### Existing execution path and ownership to reuse

The current ordinary session already accepts Gemma 3 architecture and explicit
standalone options. `PreparedGemma2Plan` owns authenticated preparation; its
historical name does not require a parallel Gemma 3 plan/controller hierarchy.
The first pairing blocker is `StandaloneTliMapping`'s target-profile check.
Target receipt validation remains mandatory.

For a chain of H proposals, the existing path is:

1. The target selects/observes the root using its ordinary sampling law.
   `StandaloneChatPlan` preflights capacity, starts a draft transaction and
   ingests that root, materializing a previously pending token when necessary.
2. `Gemma3DraftPlan::propose` evaluates H heads and samples H tokens, with H−1
   intervening causal transformer forwards. Its final proposed token stays
   pending, avoiding a needless last forward. All these operations use one
   command buffer and one completion wait.
3. The GPU also writes mapped target IDs and per-row target-history requests.
   `ChainChatVerifier` borrows those device buffers with an epoch-bound binding.
   It does not upload the full proposal stream and histories again from CPU.
4. Each target tile evaluates causal rows, runs the existing compact selector,
   and exposes candidates. Draft `gather` reads the selected token's actual q
   and q on the target support from completed shared storage; it does not
   dispatch a new GPU gather. A sampler fork previews acceptance/correction.
5. Target rows are provisionally published. The controller adopts only the
   authoritative sampler/callback prefix; cancellation, exceptions and stopping
   can restore the unexposed suffix. Draft publication retains the corresponding
   native prefix. The correction or all-accepted bonus is carried into the next
   cycle rather than sampled twice.

Root ingestion, proposal production, target tiles, target publication and draft
publication still have distinct submissions/waits. Combine compatible work,
but do not merge away the point where target decisions become authoritative.
The existing saved draft checkpoint preserves native token emissions and the
pending-token state; reconstructing it by retokenizing text would be wrong.

| Owner | Existing foundation / intended change |
| --- | --- |
| `src/engine/standalone_draft_artifact.cpp` and its public header | Retain pinned artifact authentication; add explicit Gemma 3 direct-ID compatibility proof and separate conditioning/proposal maps. |
| `src/engine/prepared_gemma2_plan.cpp` | Validate compatibility and combined resource requirements before expensive resident upload; retain atomic preparation and receipts. |
| `src/engine/standalone_chat.cpp` and its public header | Own direct-ID prompt/correction ingestion, native ledgers, per-cycle bounds, draft options and joint failure/callback behavior. |
| `src/engine/gemma3_draft.mm` and its public header | Session-sized caches, exact fast sampler/receipts, reusable active horizon, draft command scheduling and measured Q8/attention improvements. |
| `src/engine/chain_chat.mm` and its public header | Active horizon and tile schedule; preserve fork adoption, borrowed device epochs, provisional publication and correction/bonus ownership. |
| `src/engine/sampling.cpp`, `src/engine/speculative_controller.cpp` and their headers | Reuse p/q acceptance, residual sampling and probability leases; extend only where compact receipts or actual horizon need it. A joint block law would be a separate explicit contract. |
| `src/engine/gemma2_runtime.cpp` and `src/engine/gemma2_chat_session.cpp` | Choose actual work before random draws, handle tail/ordinary fallback, attribute complete costs, preserve conversation and regeneration behavior. |
| `src/engine/target_verifier.cpp`, `src/engine/target_verifier_runtime.cpp`, `src/engine/gemma_target_runtime.cpp` | Reuse private rows, hybrid attention, exact continuation and undo; change only measured verification scheduling/storage needs. |
| `src/tensor/backends/metal/metal_command_context.mm` | Existing packed target row projections and compact target selection; no speculative-only approximation of target arithmetic. |
| `src/frontends/cli/generate.cpp`, `scripts/quick_benchmark.py` | Report actual horizon exposure, complete draft cost and resource/mode identity; keep scheduling policy in the engine. |

### Probability contract: what “lossless” requires

Let p be the target distribution **after its actual** temperature, top-k,
top-p, min-p and bounded repetition transform at the current authoritative
prefix. Let q be the actual proposal distribution at that same prefix,
including masking, token mapping and finite-random-grid rounding. For a proposed
token y, tokenwise speculative sampling uses:

    acceptance(y) = min(1, p(y) / q(y))
    correction(x) = max(p(x) - q(x), 0) / sum_z max(p(z) - q(z), 0)

After all proposals are accepted, sample the bonus from the appropriate next p.
This is the standard lossless construction, with mean one-step acceptance
`sum_x min(p(x), q(x)) = 1 - TV(p,q)`.
[Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)

Here the draft CDF uses integer boundaries totaling 2^32. Its bin widths,
not an ideal softmax or only the selected logit's normalized score, define q.
The residual needs q for every token in p's retained support. Never renormalize
q over that support: q may assign probability elsewhere. Preserve the selected
interval/lease, zero-mass cases, mapped support and invalidation epochs.
Existing target support is bounded at 256 candidates; a dense CPU vocabulary
copy or dense GPU residual is not inherently necessary.

Changing draft arithmetic, draft filtering or draft repetition policy can change
q without changing the final p law, provided the actual new q is correctly
recorded and used. Such changes still need independent numerical and lifecycle
evidence. Changing verification arithmetic so it represents a different p is a
different matter: do not use a faster, lower-precision target only for
verification and call that lossless relative to ordinary execution.

Choose adaptive horizon/schedule from the durable prefix, remaining capacity
and **previous authoritative decisions**, before previewing this cycle's random
words. Existing forks discard unused proposal draws. If a new policy inspects
an uncommitted draft suffix and then rewinds/reuses its random words, the future
proposal law can become conditioned on those peeked values. Avoid that design
unless random consumption and its probability proof are explicitly replaced.
Do not count censored depths as rejections or choose fallbacks from acceptance
uniforms. Token confidence alone is not calibrated acceptance probability.

### Direct-ID synchronization and resource admission

Add a checked identity mode to the existing mapping/standalone owners. Validate
the complete token arrays, semantic flags, special IDs, normalization and
template/control behavior. Normalize only the documented compatible metadata
differences at this boundary; do not bypass authenticated target identity.
Use the target's already-rendered token history directly, render the conversation
once, and map corrections by ID. Preserve the existing cross-generation TLI
fallback for Gemma 2.

The identity path must prove that **every permitted authoritative token append**
needs at most one native draft token before advertising a one-token bound.
Proposal eligibility may still exclude controls independently. Include negative
tests for permuted IDs, mismatched pieces/types/flags and missing control
semantics. Do not silently repair the target tokenizer or change ordinary p as
part of the bridge.

The current generic TLI bound uses three times the longest piece length:
48-byte pieces produce a 144-token append bound. H=3 reserves
`2*144 + 3 + 1 = 292` private rows. A proven one-ID bound reduces that logical
requirement to six, with the current 64-row prefill floor retained. Across the
18 layers plus private normalized states this changes **11,511,808 to
2,523,136 bytes**, saving **8,988,672 bytes** without changing the model.

Direct IDs also avoid the generic context text and tokenizer workspace. At 16K,
the current text bound is 802,816 bytes. Its tokenizer workspace reserves
escaped text, symbols and bigram candidates proportional to that bound; using
the current 40-byte symbol and 32-byte candidate ABI gives approximately
111.6 MB of additional reserved host storage. This is a derived capacity
estimate, not measured RSS. Make the generic workspace optional in identity mode
and report actual owned capacity rather than retaining it invisibly.

Parameterize the draft's global-cache and history capacity through its prepared
plan. Keep local rings at 512 rows and F32 KV. With 15 local and three global
layers:

    draft_KV_bytes(C) = (15*512 + 3*C) * 256 * 2 * sizeof(float)

At 32K this is 217,055,232 bytes; at 16K it is 116,391,936 bytes, saving exactly
96 MiB for a 16K session. Propagate capacity through append/preflight, position
checks, history buffers, reset/rebuild, attention bounds and transaction tests;
do not change the fixed constant only at allocation.

| Combined-memory fact | Consequence |
| --- | --- |
| Current target at 16K: 10,410,616,032 bytes for weights, durable KV and ordinary scratch | A useful starting capacity, subject to actual paired allocation and memory-pressure checks. |
| Current target at 32K: 12,558,165,216 bytes | Target-only short execution does not establish room for a drafter. |
| Add current 32K draft weights/KV and a two-row verifier: at least 13,080,011,980 bytes | Already above the previously recorded 12,713,115,648-byte recommended working set, before draft scratch and host allocations. |

Use checked aggregate estimates **before** resident preparation and an actual
unique-allocation ledger afterward: target, draft, graph, probability receipts,
tokenizers/ledgers, private/undo rings and startup peaks. Account for tied
embeddings once per model; do not count shared aliases twice. Recommended working
set is an admission input, not free RAM or a guarantee against swap.

Keep context configurable. There is no proposed 8K cap. On this machine, support
only capacities which pass combined admission and execution checks; explain
larger-capacity refusal. Above the drafter's 32K model limit, use an explicitly
defined target-only continuation or reject the paired configuration. Do not
silently drop global draft history, reset positions, or extrapolate its RoPE.
Target KV compression/offload would be separate numerical/resource work, not
an assumed way to make this pair fit.

### Draft sampler and transformer optimization design

**Replace unnecessary full-vocabulary sorting while keeping the complete
drafter.** The sampled default uses top-k 40, top-p .95, min-p .05. In
`Gemma3DraftPlan::Impl::sample`, filtered sampling currently performs a 512-entry
tile sort and nine full-vocabulary merge passes, then full-vocabulary filtering,
prefix/CDF construction and selection. The two pair-sort buffers alone use
4 MiB. H=3 retains 6,291,480 bytes of CDFs, before the other vocabulary buffers.

A suitable Metal fast path finds the top-k cutoff with bounded tile candidates,
counts/compacts all eligible entries at or above it, and sorts/filters only that
support. Apply the existing ascending-tail top-p and min-p semantics, then
construct the actual integer sampling intervals. Keep reusable capacity and a
correct full-support path for top-k zero, large k or excessive cutoff ties.
The fallback must depend on logits/support before the draw, not on the chosen
random word or a failed acceptance outcome.

**All cutoff ties currently survive.** Existing tests intentionally give every
vocabulary entry the same logit with top-k one, yielding full-vocabulary support.
The target's fixed-size selector cannot simply replace this sampler: truncating
those ties would change q's declared semantics, and reusing old q receipts would
break correctness. Reuse proven reduction/comparison components, not incompatible
filtering assumptions.

A compact receipt can retain token IDs, cumulative integer boundaries and the
selected interval for each depth. Lookup q on p's at-most-256 IDs by bounded
merge/search, returning zero outside q support without renormalization.
Maintain a full-CDF representation for the general case under the same lease
contract. Sampling and lookup must use the **same** rounded masses. Either prove
the optimized path preserves the old intervals, or explicitly define and test
its changed draft law; no identical seeded text requirement justifies lying
about q. Support-overflow fallback should stay device-side where practical,
without a per-depth host readback.

The Q8 tied embedding/head is **178,257,920 bytes (170 MiB)**, approximately
62.5% of draft resident weights. It is read for every full-vocabulary head.
Head-local candidate extraction may combine logit transforms/top-k reduction
with projection output and remove subsequent passes, but it does not by itself
remove that weight traffic. Preserve a path for full logits/ties and numerical
checking. Measure head versus sorting cost before choosing complex fusion.

The current Q8 projection evaluates rows independently, including 64-row
prefill. Batched prefill can reuse decoded weights across rows; an appropriate
native Q8 matrix kernel is a credible TTFT/follow-up candidate. Unknown causal
draft tokens cannot be evaluated as parallel rows just because the target
verifier can batch known proposals. The root and a known pending token are
different: both are already known and may share a causal ingestion batch.

The draft encoder places a buffer barrier after every dispatch. Independent
Q/K/V projections can join once before their consumers; independent Q/K norms
and gate/up projections likewise have shared dependency points. Layer KV
publication copies can join before their first dependent use. Express the real
read/write dependencies and status-buffer requirements before removing barriers.
Do not transfer a target-kernel failure or win mechanically to this different
640-wide, one-KV-head model.

Draft local attention already dispatches only one 512-token partition; global
partition counts are bounded by context. It still launches the partition
reduction for the one-partition case. A direct-output specialization can remove
that pass while preserving the existing softmax reduction/order. Query-head
grouping that shares the one KV head is another measured candidate; register
pressure may offset the saved reads. Keep the F32 cache and independent goldens;
the old half-cache parity/replay failures are not a reason to relax tolerances.

**Draft repetition is currently different from the target's.** The draft marks
its entire native prefix at each depth, despite inheriting scalar penalty
options; the target uses its bounded repetition window (up to 64). The existing
`native_history_v1` draft law is explicit and tested, so this is not an
undiscovered losslessness bug. For direct-ID pairing, evaluate an explicit
bounded-window draft policy: it removes full-history marking and may improve
overlap with p. A short-window scan is simpler than a new incremental history
index. Preserve the old mode where needed and update diagnostics/probability
tests when q changes. Equal temperatures or penalties alone do not maximize
overlap, and a broad parameter sweep is not the next task.

### Submission, verification and adaptive work

Provide an internal operation that encodes pending/root ingestion and proposal
production into one draft command transaction, with one status/receipt completion.
The current proposal loop already has no host wait between draft depths; retain
that property. Preserve pending-last-token elision. Keep publication completion
at the point required to make delivered callbacks and subsequent recovery true.
Do not claim a submission reduction by moving uncharged work into the next cycle.

The target's Gemma 3 verifier uses the shared transformer, hybrid masks and
private KV. Two-row projections use the packed two-row specialization; larger
requests use the four-row family. Compact selection batches are four rows.
Larger horizons may require additional tiles and weight work. Prompt matrix
kernels intentionally do not define canonical scalar continuation arithmetic;
enabling them for verification requires proving the same conditional target,
not merely accepting a looser logit tolerance.

Full H=3 verifies four input rows in one invocation. Current conditional H=3
uses two tiles of two input rows, testing two draft edges and then one; the
second tile is reached only if the first two survive. The all-accepted final
row supplies the bonus distribution without another head. Other registered
schedules include H=4 as input rows (2,3), H=7 as (4,4), and H=8 as (4,5).
Preserve already-verified-prefix reuse and the existing small undo arenas.
The old Gemma 2 conditional result does not select the Gemma 3 schedule.

For cost accounting, let s_i be the probability that at least i proposed edges
survive, with s_0=1. For an uncensored cycle the standard useful-progress model is:

    G(H) = 1 + sum_{i=1..H} s_i
    C_full(H,L) = D(H,L) + V(H+1,L) + U(H,L)
    C_cond(H,L) = D(H,L) + sum_j s_bj * V(r_j,L+b_j) + U(H,L)

Here b_j counts edges before tile j, r_j is its input-row count, D includes
draft ingestion/heads/sampling, and U includes publication, correction/endpoints
and exposed host work. Use consistent carry/root accounting so the already
selected token is not counted twice. The conditional equation is for today's
**eager** H-token drafting. Use empirical prefix survival, not an unqualified
constant-acceptance approximation s_i=alpha^i. Expected U may depend on rejection.

Compare C/G against ordinary cost at the same context and actual workload.
An extra depth is worthwhile only when its marginal useful progress per marginal
cost improves the existing ratio; kernel shape changes make that cost irregular.
The 2% chat retention rule remains the final decision, not the cost estimate.

Prepare maximum supported capacity once, then pass actual H through standalone,
proposal bindings, chain schedules and metrics. Bound it by the output budget,
target context, draft context and correction/pending slack. H=0 is a coherent
ordinary path. Keep the existing supported horizon range through 23; do not
hard-code H=3 as the architecture. Near output/context tails, avoid generating
an H-token suffix which cannot be emitted.

For adaptive policy, a small table of measured tile/draft costs by context
range plus authoritative per-depth exposure/survival is sufficient initially.
Include full and conditional schedules, update slowly and use hysteresis.
Do not add a learned router, server-wide token budget or CUDA-graph abstraction.
Report active horizons and opportunities so rejected, censored and
horizon-limited outcomes remain distinguishable. Current fixed-horizon benchmark
validation must be updated coherently with those metrics.

Ordinary fallback still has a draft coherence cost. Either synchronize committed
ordinary tokens in bounded batches or mark the draft stale and explicitly
rebuild before resuming. Include that cost and preserve saved regeneration
state; never resume from silently stale KV. Avoid enabling/disabling every token.

Lazy draft tiles could save late draft work when most chains fail early, but
today conditional verification does **not** do that. A real lazy implementation
would retain per-tile q leases, cumulative random ownership, private draft
frontiers and verifier epochs across resumptions. Charge its extra submissions,
waits and ingestion before retaining it. It is justified only if saved unused
draft work exceeds that overhead.

Prompt synchronization currently prefills the drafter before target prefill,
charging draft prefill to first text latency. Profile this separately from
decode. Better Q8 prefill and direct IDs remove real work. Deferring draft
initialization until after a first ordinary token is another scheduling option,
but it moves cost into later latency and needs honest cache/callback accounting.
On one M4 GPU, concurrent target/draft work competes for bandwidth; CPU encoding
overlap and multi-GPU parallel drafting are different claims.

### Current literature and released engineering: what transfers

These sources inform mechanisms, not performance promises for this machine.
CUDA/H100 speedups, server concurrency results and trained-drafter acceptance
cannot be imported as expected M4 gains.

| Primary source, version inspected | Finding and decision for this engine |
| --- | --- |
| [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | Tokenwise acceptance/residual law and useful-progress cost basis. Reuse the existing implementation; verify actual q and canonical p independently. |
| [Block Verification Accelerates Speculative Decoding, ICLR 2025, v3](https://arxiv.org/html/2403.10444v3) | A different joint acceptance algorithm can improve expected progress for a fully evaluated block. This is more than batching ordinary tokenwise verification; see the explicit candidate boundary below. |
| [EAGLE-3, March 2025](https://arxiv.org/html/2503.01840v1), [released implementation](https://github.com/SafeAILab/EAGLE) | Multi-level target features, direct token prediction and training-time rollout address a different drafter architecture. Existing target feature ownership is useful, but a complete trained model/native forward would be required. The old toy R experiment did not test this architecture. |
| [Efficient Speculative Decoding for Llama at Scale, August 2025](https://arxiv.org/html/2508.08192v1) | Whole-pipeline work includes launch/synchronization, tree kernels, training and serving costs. Apply the dependency/host-gap reasoning; do not import its hardware gains or replace the chain with an unproved tree sampler. |
| [DFlash, February 2026](https://arxiv.org/html/2602.06036v1), [released implementation](https://github.com/z-lab/dflash) | Parallel block drafting conditioned on target features changes causal draft work fundamentally. Requires compatible trained weights and a native parallel draft architecture; it cannot batch the current 270M model's unknown autoregressive tokens. |
| [DFlash 2 engineering release](https://inco.ai/blog/dflash2/) (accessed 2026-09-27) | Adds local learned convolutions and a small candidate-path selector to improve block coherence. Released examples concern other target models. It is a trained architectural option, not evidence that the selected Gemma 3 pair has those capabilities. |
| [DFlare, June 2026](https://arxiv.org/html/2606.02091v1) | Layerwise target-feature fusion is another trained-drafter direction. Record it for a future faithful architecture decision rather than introducing unused feature plumbing now. |
| [DSpark, July 2026](https://arxiv.org/html/2607.05147v1) | A parallel backbone with lightweight sequential correction and calibrated confidence motivates survival-aware verification. Its confidence head and serving assumptions do not exist in the current 270M artifact. |
| [vLLM adaptive-verification documentation, August 2026 revision](https://github.com/vllm-project/vllm/blob/d5cadcee8641a9fcec15facb5a9157d157daa207/docs/features/speculative_decoding/adaptive_verification.md) | Uses confidence-derived survival and profiled hardware costs; documented support requires DSpark confidence heads and constrained execution backends. Adapt the objective to one local request, not its multi-request scheduler or default profiling context. |
| [Speculative Speculative Decoding / Saguaro, March 2026](https://arxiv.org/html/2603.03251v1), [code](https://github.com/tanishqkumar/ssd) | Anticipates verifier outcomes to overlap draft work on separate GPU resources. A single shared-memory M4 has no free second GPU; speculative future-branch work needs a demonstrated overlap/cancellation benefit before adoption. |
| [FlashInfer sorting-free sampling, March 2025](https://flashinfer.ai/2025/03/10/sampling.html) | Shows that full vocabulary sorting is avoidable. Its sampling approach is not by itself a receipt for q on residual-support tokens; use the insight with an exact bounded-support Metal design. |
| [FlashSampling, March 2026](https://arxiv.org/html/2603.15854v1) | Fuses LM-head work and sampling with Gumbel selection. The inspected paper leaves implemented top-k/top-p extensions as future work. A sampled winner alone does not supply the current p/q residual contract; head/candidate fusion here is an adaptation. |
| [Vocabulary trimming coverage/latency study, March 2026](https://arxiv.org/html/2603.05210v1) | Head size and response-domain coverage trade off. Relevant because this draft head is 170 MiB, but a shortlist changes q and acceptance. Preserve the faithful full-vocabulary baseline before considering an explicitly measured workload-specific optimization. |
| [Dynamic speculation lookahead, Hugging Face/Intel, October 2024](https://huggingface.co/blog/dynamic_speculation_lookahead) | Useful dynamic-horizon engineering precedent. A fixed confidence threshold is neither a calibrated acceptance estimate nor a hardware-optimal policy for this engine. |
| [Sequoia, NeurIPS 2024](https://arxiv.org/abs/2402.12374) | Hardware-aware tree design is a legitimate future alternative. Generic graph storage here does not imply that the current lossless chain controller implements a lossless tree algorithm. |

Released source was read, not just project summaries:

| Pinned file revision | Code-level observation |
| --- | --- |
| [MLX-LM generate.py, c2047f8, 2026-09-09](https://github.com/ml-explore/mlx-lm/blob/c2047f83edcd8abe9071286a82225306854c9a78/mlx_lm/generate.py) | The inspected speculative function asynchronously schedules draft work, batches target evaluation and trims caches, but accepts by sampled-token equality. It is not a reference implementation of this engine's p/q residual algorithm. Token matching and maximal p/q acceptance have different efficiency. |
| [llama.cpp common/speculative.cpp, 2b70583, 2026-09-24](https://github.com/ggml-org/llama.cpp/blob/2b70583997bbacd39702f98bbb1c355c02382383/common/speculative.cpp) | The simple drafter path chooses the best candidate, applies confidence/budget limits and avoids forwarding the unused final proposal; other classes serve trained architectures. For a greedy proposal q is a point mass, not its displayed softmax confidence. |
| [vLLM rejection_sampler.py, 8c1557a, 2026-09-16](https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/sample/rejection_sampler.py) | Contains explicit p/q acceptance, positive-part residuals and separate point-mass handling. Useful independent algorithm reference; its dense GPU implementation is not automatically better than our bounded target-support lookup. |
| [vLLM adaptive_verification.py, a4eb3f2, 2026-09-26](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py) | Builds prefix-survival/cost decisions and accounts for execution-shape buckets. Transfer the measured-cost principle; do not interpolate blindly across our two-/four-row kernel boundaries. |

These are individual file revisions, not a claim that all files came from one
common upstream release. URLs, download hashes and snapshot sizes are recorded
in the ignored `sources.json`. External engines remain research/reference tools
and must never execute the production inference path.

**Joint block verification deserves a real evaluation, with its own proof.**
The ICLR paper uses information from the whole block to alter acceptance and
correction jointly. One of its recurrences is
`a_0=1; a_i=min(1, a_(i-1)*p_i(y_i)/q_i(y_i))`, with residuals involving
`[a_i*p_(i+1)-q_(i+1)]_+`. These expressions alone are not the full algorithm:
the block-prefix selection probabilities and all boundary cases are essential.
Its expected-length result under the paper's information assumptions does not
prove lower wall time than conditional early verification on this M4.

The current per-depth `ChainSampleCallback`, scalar rejection controller and
sampler checkpoints do not implement that joint law. A faithful candidate needs
a block decision/receipt, independently enumerated joint output probabilities,
correct RNG adoption, and publication/callback stopping at each delivered prefix.
It would normally pay for full verification before that decision, losing some
conditional early-exit savings. Evaluate it against the best measured tokenwise
chain after complete pairing; do not label today's batched verification “block
verification” in the paper's algorithmic sense or paste the illustrative
appendix into production.

Trained EAGLE/parallel-draft work is a subsequent architecture choice if actual
270M cost/overlap limits the result. No compatible trained Gemma 3 12B artifact
for these architectures was validated by this audit; that is not a claim that
none can exist. Keep the real architecture, conditioning, causal/joint proposal
law and training-time rollout together. Parallel position marginals cannot
replace the conditional q required by rejection sampling when the proposal
process couples positions. Substantial training remains for the user to run.

Approximate target attention/KV selection with an admitted quality tradeoff,
such as the stated tradeoff in [Dustin](https://arxiv.org/abs/2606.24957), is
outside this lossless target contract. Smaller draft support can remain lossless
with honest q and residuals, but is not a substitute for implementing the
selected complete 270M model. No speculative speed claim may rely on degrading p.

### Implementation path for the next authorized effort

These are dependency-ordered parts of one complete integration, not artificial
approval gates or a proposal to stop after a small compatibility patch.

1. **Close compatibility and resource ownership.** Prove the direct-ID mode,
   wire the existing session/prepared/standalone path, parameterize draft context
   and private bounds, remove identity-mode text storage, and admit the combined
   footprint before resident upload. Retain authenticated target/draft receipts.
2. **Complete the efficient faithful draft path.** Implement exact bounded
   support extraction/receipts with general fallback; combine root/pending
   ingestion and proposal commands; remove proven unnecessary dependencies and
   the single-partition attention reduction. Preserve full-model numerical
   reference checks and all probability/lifecycle contracts. Profile the
   remaining head, transformer and prefill costs before adding larger kernels.
3. **Complete verification scheduling and tails.** Thread actual horizon
   through the existing chain, reuse prepared maximum storage, preserve
   target arithmetic/compact selection, implement coherent ordinary fallback
   and account for full versus conditional work. Add bounded survival/cost
   adaptation only with trustworthy authoritative metrics and random ownership.
4. **Prove the connected product path.** Run the focused deterministic
   compatibility, probability, independent target/draft, and failure/lifecycle
   checks below, then full repository CI. Remove rejected implementations and
   temporary profiling paths.
5. **Measure actual chat and make the retention decision.** Attribute all
   costs, use fresh cooled target-only/candidate runs and representative longer
   contexts, then retain only demonstrated wins. Report a correct but slower
   pair honestly; do not silently enable it as the default. If the measured
   limiting factor motivates joint block verification or a trained drafter,
   document that concrete next choice rather than starting training.

### Verification, measurement and stop condition

Extend the existing tests at the owning boundary rather than building a second
test framework:

| Contract | Existing test owners and required evidence |
| --- | --- |
| Authentication/bridge | `tests/engine/standalone_draft_artifact_tests.cpp`: full token-ID semantics, exceptional controls, unsupported metadata, correction append bound, rejected permutations and preserved Gemma 2 mapping. |
| Draft numerical execution | `tests/engine/gemma3_draft_tests.cpp` and existing independent CPU goldens: full checkpoint layers/head, batch versus scalar, local/global windows, high positions, configured capacities, unavailable Metal and poisoned failures. Keep established tolerances. |
| Proposal probability | `tests/engine/gemma3_draft_tests.cpp` and `tests/engine/sampling_tests.cpp`: compact/full q, finite-grid bins, zero support, full cutoff ties, top-k zero, greedy point masses, p support outside q, malformed/expired receipts and residual normalization. Use independent tiny-vocabulary marginal and joint-sequence calculations. |
| Target conditional law | `tests/engine/gemma3_target_runtime_tests.cpp` and `tests/engine/gemma2_metal_plan_tests.cpp`: ordinary versus every used verifier shape/history, authenticated target goldens, local 1,024 wrap, global context and physical KV restoration. Do not weaken bounds for a faster kernel. |
| Joint lifecycle | `tests/engine/standalone_chat_tests.cpp` and `tests/engine/speculative_controller_tests.cpp`: rejection at every depth, full acceptance/bonus, pending-last token, correction, callbacks stopping/throwing after each prefix, EOS, remaining-budget/context tails, draft 512 wrap, follow-up, reset/regeneration/rebuild, and failure injection before/after publication. |
| Public path/resource reporting | `tests/frontends/cli/generate_cli_tests.cpp` and `tests/scripts/quick_benchmark_script_tests.py`: actual Gemma 3 pair success, incompatible/capacity failure, effective q policy, actual horizon/exposure metrics and consistent resource/identity attribution. |

For adaptive choices, test a policy transition without future random inspection,
censored outcomes, horizon zero, all supported bounds and reuse of prepared
storage. For changed storage/commands, verify stable allocations across cycles
and restored **physical** local rings, not just logical token counts. An
unavailable device must fail explicitly rather than execute a hidden CPU path.

Use a bounded native diagnostic to attribute draft prefill/rebuild,
root/pending ingestion, transformer, head, filtering/selection, proposal
publication, target transformer/head, commits/rollback, submissions and host
gaps. Report CPU wall and GPU time without double-counting overlap. Retain
per-depth reached/accepted/rejected/censored counts, tile shapes and a few
context ranges. Such diagnostics explain cost; they do not establish a retained
chat gain. There is no reason to add an always-on per-token trace framework.

The real performance decision uses the repository's actual two-turn sampled
chat executable, with identical model files, sampler settings, context capacity
and retrieval setting. Keep raw counts/durations, hashes, TTFT, prompt throughput,
generation throughput, peak footprint and memory-pressure/thermal observations.
Different lossless seeded answers and natural stopping lengths are allowed;
report that those runs are not an exact-work comparison. Avoid attributing an
apparent gain solely to a shorter generated workload.

Each candidate and baseline needs a fresh process and at least four minutes of
idle cooldown, with no overlapping GPU/build/test activity. Require at least 2%
aggregate generation improvement, wins on both turns, and no greater-than-2%
regression in another headline metric under the applicable correctness contract.
Do not substitute a back-to-back H/temperature sweep. Check useful longer
contexts after the initial screen, including both local-ring wraps and increasing
global attention; 32K paired execution is conditional on aggregate admission.

For orientation, the existing explicit-H benchmark interface is:

    make MODE=release benchmark
    scripts/quick_benchmark.py --model models/gemma-3-12b-it-Q4_K_M.gguf --context-size 16384 --no-speculative --skip-kernels --cooldown 240
    scripts/quick_benchmark.py --model models/gemma-3-12b-it-Q4_K_M.gguf --context-size 16384 --draft-model models/gemma-3-270m-it-Q8_0.gguf --draft-horizon 3 --speculative-sampling lossless --speculative-schedule full --skip-kernels --cooldown 240

These commands were **not run for this research**, and the paired command is
not expected to succeed until the bridge/integration is implemented. H=3/full
is a reproducible starting measurement, not the selected optimum. Keep
benchmark parsing in step with any explicit adaptive-mode interface.

Future implementation is complete when the selected pair executes through the
real native chat path with correct distribution/cache/context/callback behavior,
combined resource accounting, efficient draft/verification ownership, full CI,
and honest cooled performance evidence. “Fully optimized” is not provable by
inspection; no acceptance threshold or theoretical ceiling substitutes for that
evidence. This research baton stops at the documented plan.

## Completed baton: native Gemma 3 pipeline optimization

On 2026-09-27 the user requested a further deep pipeline review and implementation
of worthwhile optimizations before connecting the drafter. That review is
complete. Verifier rollback storage and backup synchronization were improved;
ordinary kernel/workspace candidates failed the cooled chat gate and were
removed. The native F32 target, ordinary/verifier continuation contract and
prior integration are preserved. Full CI and actual-checkpoint verification
passed. Stop here: drafter pairing and training have not started, and no
additional ordinary-chat speedup or global optimality is claimed.

The frozen baseline is the completed integration binary with SHA-256
`326fcd150716e39838f835c4ab2ce85ca4e8bb779d8d646fa41adabf905d5d75`, copied to
`build/gemma3-pipeline-baseline-local-gguf-generate`; source and identities are
preserved under `build/gemma3-pipeline-baseline-source` and its adjacent manifest.
Both the combined and resource-only ordinary candidates passed correctness but
failed the cooled real-chat gate. All ordinary changes have been restored to
the frozen integration baseline. Only verifier-scoped undo sizing and backup
synchronization remain. Three other kernel candidates were rejected after
bounded native diagnostics.

### Retained implementation and final verification

- `src/engine/target_verifier.cpp` sizes each graph undo ring to the next power
  of two covering its maximum accepted path, bounded by arena rows and the
  existing 24-row path limit. One/three/24-row arenas allocate one/four/32 undo
  rows. The ordinary 128-row workspace still allocates its original 32 rows.
- `src/engine/target_verifier_runtime.cpp` derives copy/restore masks from owned
  storage, rejects paths longer than the encoded request before mutation, and
  joins all independent layer backups at one barrier before publication. On
  the 48-layer target this removes 47 barriers per graph provisional commit.
- `src/engine/gemma_target_runtime.cpp` stages width-two backups ahead of the
  block encoder and joins them at its existing initial embedding barrier,
  removing 48 backup barriers. Ordinary block encoding, private storage,
  publication order, allocation and numerical kernels remain at the baseline.
- Existing tests in `tests/engine/gemma2_metal_plan_tests.cpp` and
  `tests/engine/gemma3_target_runtime_tests.cpp` cover small capacities, undo-ring
  wrap, context-tail rollback, oversized-path rejection, full physical K/V
  restoration, stable allocations and the synchronization counts. No new
  production abstraction, public API or instrumentation remains from the
  rejected candidates.

| Allocation | Frozen integration baseline | Retained implementation |
| --- | ---: | ---: |
| Ordinary scratch | 298,019,040 bytes | 298,019,040 bytes |
| Full four-row verifier arena | 46,689,156 bytes | 24,668,948 bytes |
| Conditional two-row arena | 43,365,988 bytes | 19,772,908 bytes |

The respective graph savings are **22,020,208 bytes (21 MiB)** and
**23,593,080 bytes (22.5 MiB)**, with unchanged weights, F32 KV and context
capacity. These are measured allocation reductions, not a claimed chat speedup.

Final release CI passed **455 C++ tests with Metal and all 59 Python tests**,
without warnings or optional-reference skips. It used the existing reference
environment and authenticated 12B goldens. Original logit/KV tolerances and exact
ordinary/width-two continuation checks are unchanged. Log:
`build/gemma3-pipeline-final-ci.log`. Independent read-only review found no
actionable issues and confirmed that all numerical kernels, public headers and
ordinary execution/storage match the frozen baseline.

Both final actual-checkpoint verifier diagnostics passed:

| Diagnostic | Full four-row, H=3 | Conditional two-row arena, H=3 |
| --- | ---: | ---: |
| Ordinary four-row wall/GPU | 309.081 / 305.570 ms | 305.262 / 302.242 ms |
| Verification wall/GPU | 210.120 / 207.377 ms | 216.537 / 214.515 ms |
| Commit wall/GPU | 1.444 / 0.113 ms | 1.001 / 0.106 ms |
| Complete perfect episode | 211.963 ms | 217.666 ms |
| Perfect-progress ceiling | 1.458x | 1.402x |
| Controller and authoritative token path | exact | exact |

The full invocation also matched candidate order/probabilities and final hidden
states exactly. Conditional candidate identity was not separately measured.
These are bounded native diagnostics with perfect proposals, excluding drafting
and rejection, not cooled performance comparisons or speculative product gains.
No additional ordinary chat pair was run after restoring the ordinary path:
the final scope changes only verifier storage/backup scheduling, whose end-to-end
value must be measured after connecting the drafter.

Evidence directory: `benchmarks/results/gemma3-pipeline-optimization-2026-09-27/`.
Final logs are `verifier-final-full-4.log` and `verifier-final-conditional-3.log`;
unprefixed verifier logs in that directory belong to the rejected combined
candidate. `final-binary-identity.json` pins the final executables. Generate
SHA-256: `ecd342d8c41d6ba23ddd7737cb77df7c6c8a96910544272997e514c3e6f83f59`;
benchmark SHA-256: `92657cc429c59e3b89569879e9aca36d94c0ad817af0c0748a08ab04eac00e1e`.
Source snapshot/manifest and the three-production-file/two-test diff are under
`build/gemma3-pipeline-final-*`. Prior uncommitted integration work is preserved;
nothing was committed or pushed.

### Pipeline review and candidate evaluation (2026-09-27)

The review followed loading/residency, prompt prefill, scalar continuation,
attention, native projections/head, compact sampling, command submission,
publication and rollback through actual ordinary/full/conditional execution.
The prior retained chat put approximately 99% of decode wall time on the GPU;
removing all remaining host time would not meet the generation retention gate.

An isolated native Metal counter diagnostic attributed stages at a short prefix
and about 2,090 tokens. Median projection totals were 73.80/73.63 ms, including
gate/up 32.37/32.03 ms, down 19.52/19.76 ms, QKV 9.32/9.47 ms and the tied head
8.09/7.85 ms. Attention grew from 1.01 to 10.02 ms. The diagnostic split encoder
boundaries to use supported stage-boundary counters: GPU totals were 85.13/92.96
ms, and both boundaries and instrumentation perturb the production schedule.
These numbers identify costs, not retained speedups or measured physical memory
bandwidth. The device exposed stage-boundary timestamp sampling, not dispatch
sampling; full Xcode/xctrace was unavailable. No profiling path was added to
production. Counter interpretation follows Apple's
[sampling documentation](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)
and [timestamp conversion](https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/converting_gpu_timestamps_into_cpu_time).

Mechanisms in the rejected combined candidate (experimental, not current):

- One GPU-generated local/global RoPE coefficient table served every layer in
  an invocation. A fused kernel preserved the original 32-lane Q/K head RMS
  reduction, both F32 normalization products, NeoX rotation order and private V
  copy. It replaced four Q/K preparation dispatches per layer with one. Including
  table setup, a 48-layer invocation removed 143 dispatches and 48 barriers.
  Table storage was 2 KiB per capacity row; Gemma 2 needed no table allocation.
- `GemmaTargetBlockScratch` contained shared transformer storage. The ordinary
  workspace owned only its logits, selector interface, width-two backups and
  execution state. Its private K/V pair was published after each layer's
  attention and reused after the existing dependency barrier. Graph arenas
  retained separate all-layer private K/V until controller commit.
  Disjoint hidden intermediates shared one allocation; GELU replaced its dead
  gate input. Legacy Gemma 2 scratch and graph-only selectors/undo were removed
  from ordinary Gemma 3 execution. These were actual ownership changes.
- It also included the verifier backup/undo improvements described above;
  only those mechanisms survived the subsequent resource-only comparison.

That rejected candidate's ordinary scratch ledger was **111,125,008 bytes**,
down from 298,019,040: **186,894,032 bytes / 178.236 MiB saved**. Its scratch no
longer scaled with durable context capacity. Full-four and conditional-two allocations were
24,247,060 and 19,561,964 bytes, previously 46,689,156 and 43,365,988.
Native weights, F32 KV, configured context, output head and sampling were unchanged.

Rejected candidates (all remained in ignored diagnostic snapshots):

- Reusing the existing parallel-row shader for scalar projection removed a
  nonexistent-row loop but produced no consistent gain in dominant projections.
- Four output columns per SIMD group preserved exact results on 144 tail/shape/
  row combinations and seven actual checkpoint tensors. It did not establish a
  scalar gain and regressed stable four-row FFN-up by about 9.6%. An apparent
  four-row gate improvement was a warmup/clock transition; no selective head
  variant was retained from those noisy measurements.
- Skipping attention `exp(0)` preserved finite output bits and exceptional
  behavior but produced one-row ratios of 0.961/0.975/0.998 at 1,024/2,090/8,192
  tokens. It was rejected. Naive gate/up/GELU fusion was not implemented: the
  existing two-output shader already shares activation work, and that fusion
  would save only about 11.8 MB of intermediates against 7.294 GB of logical
  weights without removing dominant weight work. More normalization fusion was
  similarly not justified by the measured cost/complexity bound.

Larger FP32 prompt tiles could reuse decoded weights across more prompt rows;
this remains an unmeasured prefill hypothesis, not evidence of a generation
gain. Neither these screens nor the selected mechanisms establish global
optimality. Drafter cost, acceptance and sustained chat remain later empirical
questions; no speculative pairing or training was started.

Combined-candidate verification: release CI passed **457 C++ tests with Metal and all 59 Python
tests**, without warnings or optional-reference skips. It used the existing
reference environment and explicitly enabled authenticated 12B goldens; log
`build/gemma3-pipeline-ci.log`. The original logit/KV bounds and exact ordinary/
width-two continuation checks are unchanged. Added checks cover fused versus
standalone Q/K output bits at high positions and odd rows, aliases/views/tails,
actual allocation ownership and unique-byte ledgers, full physical cache
restoration across wrap, small graph undo rings, context-tail rollback, rejected
oversized paths and backup synchronization. Read-only independent code review
found no actionable issues.

That candidate's actual-checkpoint full-four and conditional-H3 diagnostics also passed
exact hidden/candidate comparison where measured, authoritative token paths and
controller outcomes. These are perfect-proposal target diagnostics, excluding
drafting and rejection, and are not a speculative product speedup. Evidence is
under `benchmarks/results/gemma3-pipeline-optimization-2026-09-27/`; isolated
kernel screens and source manifests are under `build/gemma3-pipeline-*` and
`build/gemma3-attention-exp-*`.

The combined candidate **failed** the prescribed cooled chat gate. Each fresh
process received 240 seconds of idle cooldown with no concurrent builds/tests/
GPU work. Both retained exact counts and hashes: 16/0/1,506 and
`260ae35592bb6060`, then 19/1,522/1,026 and `dd4aadff1d4f599f`.

| Metric | Frozen integration baseline | Combined candidate |
| --- | ---: | ---: |
| First-turn generation | 12.046 tokens/s | 12.029 tokens/s |
| Follow-up generation | 10.634 tokens/s | 9.235 tokens/s |
| Aggregate generation | 11.431 tokens/s | 10.715 tokens/s |
| First/follow-up prompt throughput | 52.185 / 29.100 tokens/s | 52.642 / 30.202 tokens/s |
| First/follow-up time to first text | 0.386 / 0.751 s | 0.407 / 0.752 s |
| Native peak footprint | 10,450,824,016 bytes | 10,330,794,808 bytes |

Aggregate generation regressed **6.26%**, with -0.14%/-13.16% per turn, and first
text latency regressed 5.35% on the first turn. This is not retained as a speed
improvement. The comparison does not isolate the cause or prove identical GPU
clocks; no thermal/performance warning was reported. Baseline and candidate
generation-completion durations were 125.026/96.482 and 125.196/111.098 seconds.
Files: `baseline.*`, `candidate.*`, `candidate-comparison.json` and
`candidate-binary-identity.json` in the pipeline evidence directory. Candidate
SHA-256: `d3ab5911e1345ca4479c81c4cbde715f29ff47a3f3631cd71c06aaebc4a8b51d`.
The fusion was removed before evaluating resource ownership separately.

The rejected resource-only source contained no numerical Metal kernel changes relative
to the frozen integration baseline. Release CI again passed **456 C++ tests
with Metal and all 59 Python tests**, including authenticated 12B goldens and
exact ordinary/width-two continuation. Build warnings: zero. Log:
`build/gemma3-pipeline-resource-only-ci.log`. The removed fusion's dedicated
primitive test was also removed; its experimental evidence remains in ignored
artifacts. Its ordinary scratch was **110,862,864 bytes**, saving **187,156,176 bytes /
178.486 MiB**. Its full-four and conditional-two arena descriptor ledgers were
24,238,868 and 19,557,868 bytes. Its executable SHA-256 was
`eea48927cb6887f7c6ca31d0db9880e05a01c1da5c28ef1294da7850c342cf26`.

The separate resource-only chat **also failed** the retention gate after its own
240-second idle cooldown, using the same frozen baseline and exact workload.
First/follow-up generation was 12.099/9.887 tokens/s (+0.45%/-7.03%); aggregate
was 11.093 tokens/s (**-2.95%**). Generation-completion durations were
124.469/103.774 seconds. Prompt throughput was 51.860/30.163 tokens/s; first text
was 0.408/0.743 seconds, including a 5.58% first-turn latency regression.
Native footprint fell to 10,337,594,216 bytes, but that did not justify keeping
an ordinary candidate which failed the speed contract. Raw evidence is
`resource-only.*`, `resource-only-comparison.json` and its binary identity file.
The comparison does not establish the physical cause of the slowdown.

The large ordinary workspace refactor, activation aliases, per-layer ordinary
publication and all new numerical kernels have therefore been removed from
production. The smaller final change retains only verifier-owned mechanisms:
bounded graph undo capacity and batched graph/width-two backups. The ordinary
128-row arena still gets its original 32-row undo allocation, and ordinary
prefill/decode retains its original storage and command order. Verification of
this final scope passed as documented above; no learned drafter is connected.

### Target-integration handoff (superseded by the speculative plan above)

**Native target integration completed on 2026-09-27.** Gemma 3 12B chat,
shared target verification, independent numerical/lifecycle checks, complete CI,
resource characterization and the cooled real-chat retention screen are complete.
The agreed target-integration stop boundary has been reached. The completed
2026-09-27 speculative research at the top of this document now specifies the
follow-on implementation, resource admission, optimization candidates and
verification. That pairing is not implemented and no learned speculative
default is enabled.

### Original target-integration research and authorization

Research date: 2026-09-26. Source tree inspected: clean `main` at `dff16f0`.
The user selected the downloaded Gemma 3 12B target and existing Gemma 3 270M
instruction-tuned drafter. This supersedes the completed cleanup baton.

**Implementation authorized on 2026-09-26 and completed as one effort.** The
scope covered target binding, native Metal chat, shared batched target
verification, independent correctness evidence, and real-path resource/performance
characterization.
The research below is the design basis, not an approval gate between phases.
Context is configurable, with a 16K default and measured 32K allocation/short
execution within this machine's memory limits. Neither is an architectural 8K cap.
The later learned-speculation tuning and substantial training remain outside
this target-integration handoff.

Completed target-integration objective: a correct, fast, engine-owned, resident Metal
Gemma 3 12B text-chat target, with its actual small-batch verification path ready
before tuning standalone speculative decoding. Lossless means preserving the
chosen target's sampling distribution, not identical seeded text. Native GGUF
quantization and documented numerical precision define the target; speculation
must not become an excuse for changing that target.

There is no evidence yet that this pair is faster, or that any proposed kernel is
optimal on this machine. Design away known unnecessary work now; use independent
numerical evidence and real chat measurements to choose between plausible
implementations. “Maximum optimization” is not a finish condition. The concrete
handoff in section 11 is.

## Implemented target boundary and current verification

The source now implements the downloaded **Gemma 3 12B Q4_K_M text target**.
Sections 1–10 below preserve the design research at `dff16f0`; descriptions of
missing functionality there describe that starting point. This section owns
the implemented state and takes precedence over those research observations.

### Runtime and ownership

- The existing GGUF loader, model binding, resident plan, sampler, chat session,
  topology verifier, publication/rollback controllers, and native Metal backend
  are reused. Existing `Gemma2*` public type names now also carry explicit
  Gemma 3 configuration where required; this avoids a parallel execution stack.
- `src/engine/gemma_target_runtime.cpp` owns the shared transformer block
  encoder for Gemma 3 ordinary prefill/decode, width-two verification, and the
  full/conditional graph verifier. The Gemma 2 ordinary product path remains
  intact. Its graph verifier uses the extracted shared block encoder.
- Native Q4_K/Q6_K values are retained. The Q6 embedding and tied output head
  share one allocation. Q/K head RMS normalization, the 5-local/1-global layer
  pattern, local/global RoPE bases and global `1/8` frequency scaling, four
  block norms, F32 embedding scaling, tanh GELU, and absent softcaps are explicit.
  Gemma 3 uses precise Metal math and F32 activations, accumulators, and KV.
- Prefill executes chunks of at most 128. Chunks of four or more use native
  FP32 SIMD-group matrix projections with F32 operands and accumulation; smaller
  chunks, all decode/verification projections and all output-head rows use
  packed Q4_K/Q6_K arithmetic. Parallel packed rows preserve the scalar block
  workers and reduction order. The shared transformer encoder carries one
  explicit prompt-prefill projection choice; there is no second transformer.
  Decode, prompt chunks and verification use the same online attention mechanism. A chunk's
  private KV stays separate from the durable ring until all its queries have
  consumed their causal windows. Publishing a chunk larger than a local ring
  writes only the newest occupant of each slot, avoiding competing GPU writes.
  KV-only prompt chunks omit the final norm and output head.
- Attention launch sizing reads the existing immutable, CPU-staged schedules.
  It launches only the maximum partition count required by active rows and
  omits reduction when every row has one partition. Per-row shader arithmetic
  and the 32-slot partial layout remain unchanged, including mixed row lengths.
- The plan owns its reusable ordinary execution workspace. Verification arenas
  retain bounded private KV, ancestor/schedule staging and rollback buffers.
  The existing horizon 23, 1,025-row and 64-layer limits remain. F32 cache copy,
  provisional publication, rejection, accepted-prefix rollback and pending
  correction handling use the existing controller semantics.
- Target context is configured separately from the advertised model maximum.
  The default is 16,384; supported capacities are powers of two, at least the
  local window, within metadata and the device budget. The early admission
  check counts unique weights plus durable KV against the device's recommended
  working set and existing Metal allocations. It is a lower-bound guard, not
  a promise that scratch, other apps or an attached drafter will fit.
- About 30.7 MiB of legacy prefill intermediates, plus smaller decode/verifier
  intermediates, remain alongside the Gemma 3 workspace and are included in its
  scratch ledger. The attempted ordinary ownership/lifetime refactor reduced
  allocation but failed cooled chat retention and was removed. Do not mistake
  its experimental smaller ledger for the retained ordinary implementation.
- CLI chat supports `--context-size`, reports actual capacity/cache precision
  and allocation counts, and uses the checkpoint's Gemma 3 text template.
  System text is folded into the first user turn with the exact template's
  whitespace behavior. Prefix reuse, regeneration, clearing/reset, context
  shifting, streaming and stopping retain the shared session lifecycle.
- Gemma 3 defaults to ordinary target decoding. `--retrieval` is an explicit
  experimental opt-in. The existing 270M drafter is **not yet connected to this
  target**; preparing that pair must still fail explicitly. No learned
  speculative default or speedup is claimed.
- GGUF block-quantization row validation now uses the contiguous first axis,
  including matrices whose output dimension is not block-aligned. This fixes
  an old assumption exposed by the smaller numerical fixtures.
- Metal command creation and discarded encoders now drain their temporary
  Objective-C ownership internally. Buffer/encoder accessors borrow owned ARC
  references, and CPU buffer-content access uses a precise receiver lifetime.
  C++ callers no longer need an outer autorelease pool to release discarded
  queue slots or destroyed model buffers. Full-suite checks exposed both
  lifetime failures; focused tests now exercise abandoned commands and actual
  allocation release after CPU access and completed GPU dispatches.
- Buffer upload/allocation obtains the Metal device directly instead of
  compiling an unused shader profile. Gemma 3 rollback copies use its already
  selected precise profile, avoiding a second profile's initialization during
  startup warmup.
- A validated verifier request now retains the vocabulary, layer count,
  context, window and layer period used to prove it. Encoding checks those
  values against the bound target in constant time before mutation, preventing
  a valid schedule for another capacity or attention pattern from being reused.

### Independent correctness evidence

- The bounded CPU reference in `scripts/gemma3_target_reference.py` decodes the
  actual checkpoint and executes independent F32 transformer equations. Its
  Q4_K and Q6_K decoders matched the pinned original C dequantizers exactly over
  32,512 random values per format. No external inference runtime executes the
  production path.
- Authenticated four-token goldens under `build/gemma3-target-reference/` pin
  the whole GGUF, reference identity and 149 output files. The native checkpoint
  test passed nine complete 262,144-entry logit comparisons across scalar
  execution, four-row matrix prefill, packed continuation from a matrix prompt,
  width-two verification, and decoding after rollback: 2,359,296 compared
  logits. All 48 layers' K/V were checked across those transitions:
  3,342,336 compared values.
  Bounds were fixed before execution: logits `abs(error) <= 0.002 +
  0.0003*abs(reference)`; KV `<= 0.0005 + 0.0003*abs(reference)`. These are
  assertion bounds, not measured maximum errors. Scalar argmax tokens matched.
- Reconstructed identical matrix-prepared prompt caches also produced exact
  ordinary/width-two logits, exact rollback continuation, and exact retained
  K/V across all 48 layers: three full-vocabulary comparisons and three logical
  KV rows. This tests the continuation arithmetic independently of prompt-path
  rounding differences.
- At the fourth authenticated prefix, default target transforms (temperature
  0.8, top-k 40, top-p 0.95, min-p 0.05) retained the same four-token support
  for repetition penalties 1.0 and 1.1. Matrix-prefill versus CPU-reference
  maximum logit error was `5.76973e-5`, with probability total variation
  `8.62939e-7`. Matrix-prefill versus packed-prefix total variation was
  `2.45026e-6`; packed continuation from the matrix prompt versus CPU was
  `8.92997e-7`. These are bounded prefix observations, not long-context quality
  claims or a proof of equal laws for different prompt evaluators.
- A separate complete six-layer fixture checks independent transformer
  equations, scalar versus batched prefill across an eight-slot local wrap,
  continued decode, and all-layer KV. Primitive checks cover Q6 embedding,
  head RMS, scaled RoPE, F32 attention, cache copy/scatter, row tails and packed
  scalar/batch equality, including actual 3,840/4,096/15,360 input widths.
- Sixteen scripted chain cases exercise full and conditional verification:
  all acceptance, first/middle rejection, callback stop/throw, stop tokens,
  accepted-prefix budgets, and correction materialization at the budget.
  Independent CPU-derived target tokens/KV drive these checks. Existing
  finite-grid lossless probability-law tests remain the sampling foundation.
- CLI fixtures cover multi-turn reuse, regeneration/reset, configurable context,
  context exhaustion/shifting, and explicit refusal of the unconnected drafter.
  Tokenizer tests compare the released template's system folding and Unicode
  trimming behavior. Unavailable-Metal paths remain covered.
- Focused evidence: `build/gemma3-target-checkpoint-parity.log`,
  `build/gemma3-target-focused-all.log`, `build/gemma3-target-chain.log`,
  `build/gemma3-target-bounds.log`, and `build/gemma3-python-full.log`.
  The optional reference environment ran all 59 Python checks without skips.
- Complete repository CI passed **455 C++ tests with Metal**, including both
  real checkpoint suites and the explicitly enabled 12B target goldens:
  `MODE=release LOCAL_GGUF_GEMMA3_TARGET_GOLDENS=build/gemma3-target-reference
  bash scripts/ci.sh`. Log: `build/gemma3-integration-ci-final.log`. Its system
  Python invocation passed with 14 optional-reference skips; the separate
  reference-environment run above executed all 59 Python checks. The final
  suite includes verifier-proof binding, queue-slot and allocation-lifetime
  regressions, the complete Gemma 3 CLI tokenizer fixture, FP32 matrix operand
  checks, and attention launch thresholds/mixed rows with both cache precisions
  and buffer offsets. The release test, CLI and benchmark builds were warning-free.

### Precision decision

Prompt matrix accumulation has a different F32 reduction and scale association
from packed GEMV. It retains native weights and F32 precision, and is checked
against the same independent transformer bounds. Ordinary and speculative
continuation start from the same prepared prompt cache and use the same packed
arithmetic for every verification tile size; a proposal cannot select a different
matrix target evaluator. Independent probability reports quantify prompt-path
rounding differences without treating close logits as proof of equal finite
sampling laws. The lossless acceptance/residual law still applies to the actual
transformed target probabilities supplied to it.

The independent F16-cache candidate is recorded in
`build/gemma3-target-f16-comparison.json`. Across four checkpoint positions,
maximum logit absolute errors versus F32 were approximately 0.000177, 0.002223,
0.008881 and 0.004092; T=1 full-softmax total variation reached 0.000387.
Argmax and top-40 membership matched these prefixes. Large individual V values
and propagated differences in later layers remain visible. This short-prefix
study does **not** establish long-context fidelity or native F16 performance.
F32 remains the production target definition; no F16-cache option was promoted.

### Prior integration capacity and verifier measurements

Ignored evidence lives under
`benchmarks/results/gemma3-target-integration-2026-09-26/`.

A fresh final native 32,768-capacity process completed a 15-token prompt and eight
generated tokens. Its allocation ledger was 7,294,024,704 bytes of unique
weights, 4,966,055,936 bytes of durable KV and 298,084,576 bytes of scratch:
12,558,165,216 bytes total. `/usr/bin/time -l` reported a peak process footprint
of 12,599,667,536 bytes (about 11.73 GiB), maximum RSS of 10,440,720,384 bytes and
zero process swaps. System-wide swap usage changed from 4,133.94 to 3,600.25 MiB
during this check; that is an observation, not attribution of every swapped
page to this process. No thermal/performance warning was available. This proves
32K allocation and short native execution, **not a 32K-filled context workload,
long-context quality, or a comfortable target-plus-drafter memory budget**.
Startup plus execution took about 24.94 seconds; its hot-run token rates are not
retention evidence. Files: `final-context-32768.*`. The earlier build's 37.35-second
smoke and its different memory-pressure observations remain in `context-32768.*`;
these were not a thermally controlled startup comparison. The default stays at 16K.

A 131,072-capacity request was refused before uploading weights: unique
weights plus F32 KV require at least 25,144,982,528 bytes before scratch, versus
about 12.713 GB of available recommended Metal budget. This is an explicit
resource decision, not an 8K/16K architectural context ceiling. The final binary
reconfirmed that refusal in about 0.31 seconds (`final-context-131072.*`).

The first real-checkpoint small-batch diagnostic used a 16-token prefix and
scripted perfect proposals. Full four-row verification matched ordinary
candidate tokens/logits and final hidden states exactly (measured maximum
absolute error zero); the controller and authoritative token path also matched.
It took 337.27 ms target wall time / 336.46 ms GPU, versus 314.54 / 311.18 ms
for four ordinary rows. Complete episode time was 338.91 ms, including a
1.50 ms commit and 0.033 ms controller work. Its additional arena was
46,689,156 bytes. Conditional H3 used two two-row tiles and took 349.59 ms
per complete episode, versus 326.77 ms for the ordinary rows; its arena was
43,365,988 bytes. These are diagnostic observations without the cooled
retention protocol. They correctly include the H+1 endpoint row, and establish
**no useful speculative speedup**, even before paying for the drafter.

The dominant measured remaining cost is GPU computation. The first packed
row-reuse implementation still performs almost all F32 arithmetic separately
per row, lacks the scalar path's 15/60-block specialization, and keeps large
per-thread temporary arrays. Actual register spilling was not measured.
Attention also launches its fixed 32 partitions even at a one-partition prefix;
unused groups return immediately, and the separate reduction is unnecessary
there. These findings motivate a bounded matrix-path comparison, not controller
micro-optimization or a conclusion that Gemma 3 speculation cannot work.

The first packed-row reuse plus 15/60-block specialization set **failed the
real-chat retention gate**. Both fresh processes received 240 seconds of idle
cooldown, with no concurrent builds/tests/GPU work. They used the canonical
two-turn conversation, seed 42, 16K F32 target-only decoding, and natural stops.
Counts and selected-token hashes matched exactly: 1,506 tokens / hash
`260ae35592bb6060`, then 1,026 / `dd4aadff1d4f599f`. Baseline complete-generation
rates were 11.839 and 7.682 tokens/s; the candidate produced 10.701 and 8.548.
Aggregate throughput was effectively unchanged (9.710046 versus 9.710131
tokens/s). Prompt throughput fell from 20.3/16.1 to 12.0/10.7 tokens/s; TTFT
rose from 0.790/1.182 to 1.333/1.781 seconds. This optimization set is rejected.
Raw evidence: `baseline.json`, `candidate.json`, their resource observations,
and `packed-candidate-comparison.json` in the directory above. The `.time.txt`
files wrap the Python driver: their peak-footprint field describes that driver,
not the native engine; do not confuse it with the directly measured 32K native
footprint. Baseline source and binary provenance are preserved in
`build/gemma3-kernel-baseline-manifest.json` and its adjacent diff/snapshot.

The subsequent actual-checkpoint diagnostics separate two useful mechanisms:

| Mechanism | Four ordinary rows | Full four-row episode | Conditional H3 episode |
| --- | ---: | ---: | ---: |
| FP32 matrix for every projection | 615.65 ms | 165.49 ms | 311.41 ms |
| Parallel packed rows, scalar arithmetic retained | 318.76 ms | 207.88 ms | 210.20 ms |

Each conditional diagnostic timed its own four-row control: 619.51 ms for the
matrix candidate and 310.66 ms for parallel packed. These are uncooled mechanism
diagnostics, not retained chat speedups. The matrix candidate's apparent 3.72x
full-episode ceiling uses its own much slower scalar control; it is not a 3.72x
product gain. Parallel packed achieves 1.53x/full and 1.48x/conditional optimistic
ceilings while preserving ordinary scalar cost. Full verification matched all
compared hidden/candidate values exactly; both schedules matched authoritative
tokens and controller outcomes. Draft computation and real acceptance are absent.

The selected implementation uses parallel packed arithmetic for **all decoding,
verification and output-head rows**, and FP32 matrix multiplication only for
ordinary prompt chunks of at least four rows. This preserves one continuation
evaluator while allowing matrix prefill. The earlier sequential row-reuse and
15/60-block specialization kernels/pipelines have been removed. Logs and binary
identity records are `verifier-parallel-rows-*`, `verifier-fp32-matrix-*`, and
`kernel-*-manifest.json` under the evidence directory. Final combined CI and
real-chat validation include the additional attention launch pruning and startup
fixes; those changes are not included in the diagnostic timings above.

The final binary's full-four diagnostic took 211.39 ms per complete episode
versus 315.61 ms for four ordinary rows (1.493x optimistic ceiling). Conditional
H3 took 215.61 ms versus 314.73 ms (1.460x). Full candidate/hidden values remained
exact; both controller and authoritative token paths matched. Its additional
arenas remain 46,689,156/full and 43,365,988/conditional bytes, with 1,280 bytes
of compact readback per episode. Final logs are `verifier-final-full-4.log` and
`verifier-final-conditional-3.log`. These still exclude real drafting and
acceptance losses; they establish a useful target batching mechanism, not a
retained speculative product gain.

### Prior integration retained real-chat result

The final release binary passed the prescribed retention screen against the
clean native packed baseline. Each fresh process received 240 seconds of idle
cooldown, with no concurrent builds/tests/GPU work. Both used the canonical
two-turn sampled conversation, seed 42, 16K F32 capacity, native Q6 head,
target-only decoding and natural stops. All evaluated/cached/generated counts
and selected-token hashes matched: first 16/0/1,506 and `260ae35592bb6060`;
follow-up 19/1,522/1,026 and `dd4aadff1d4f599f`.

| Metric | Clean native baseline | Final implementation |
| --- | ---: | ---: |
| First-turn generation, complete clock | 11.839 tokens/s | 11.986 tokens/s |
| Follow-up generation, complete clock | 7.682 tokens/s | 8.604 tokens/s |
| Aggregate generation | 9.710 tokens/s | 10.339 tokens/s |
| First/follow-up prompt throughput | 20.262 / 16.079 tokens/s | 51.340 / 29.128 tokens/s |
| First/follow-up reported TTFT | 0.790 / 1.182 s | 0.312 / 0.652 s |
| First/follow-up time to first text | 0.877 / 1.299 s | 0.410 / 0.770 s |

Aggregate generation improved **6.48%**, with wins on both turns (+1.24% and
+11.99%) and no headline regression. Generation-completion durations were
127.207/133.554 seconds before and 125.643/119.251 after. Summed GPU time fell
from 258.166 to 242.285 seconds. First-turn compute dispatches fell from
1,394,543 to 1,383,023; follow-up dispatches stayed at 950,063 while empty
attention threadgroups were removed. These measurements retain the combined
implementation; they do not isolate the contribution of each change or prove
optimality across prompts, context lengths or sustained thermal states.

The final native process took 271.93 seconds including startup/warmup and the
five-second inter-turn pause. Peak footprint was 10,449,857,360 bytes (about
9.73 GiB), maximum RSS 10,627,203,072 bytes, and reported process swaps zero.
Its allocation ledger was 7,294,024,704 weight + 2,818,572,288 KV + 298,019,040
scratch = 10,410,616,032 bytes. System swap usage changed from 2,783.12 to
2,965.19 MiB; raw page/compression observations are retained. No thermal or
performance warning was available, which does not establish equal clocks.

Evidence: `final.json`, `final-comparison.json`, `final.chat.stdout`,
`final.chat.stderr`, `final.native-time.txt`, `final.resources.json`, and
`final-binary-identity.json` alongside the earlier `baseline.*`. The final
executable SHA-256 is
`326fcd150716e39838f835c4ab2ce85ca4e8bb779d8d646fa41adabf905d5d75`;
the baseline is
`66b3b14521b2b7ce97a7c0fe9e170d99bdc49e0d412e70215bbd3ab6b72de187`.
This is one cooled comparison meeting the repository gate, not a statistical
variance study, a llama.cpp comparison, or a measured 270M speculative gain.

## 1. Verified starting point and preserved evidence

At the research starting point, the production chat target was Gemma 2 9B.
Foundations retained for the new target:

- Native GGUF/mmap loading, tensor layout validation, tokenizer, Q4_K/Q6_K/Q8_0
  and floating-point primitives, resident Metal buffers and command encoding.
- Gemma 2 prefill, decode, retrieval speculation, chat prefix reuse, context
  rebuilding, regeneration, streaming, stops, and diagnostics.
- Complete authenticated Gemma 3 270M Q8 transformer, F32 local/global K/V,
  private draft transactions, pending final-token handling, full-vocabulary
  draft selection, and native emission ledgers.
- Lossless acceptance/residual sampling with the actual finite-grid proposal
  law, checkpointed random streams, and target-authoritative sampler transforms.
- Validated proposal graphs, chain/full/conditional verification, private target
  K/V, provisional publication, undo of overwritten ring slots, and callback
  lifecycle tests. Existing horizon 23, verifier row limit 1,025 and layer limit
  64 need not be expanded for 48-layer Gemma 3 or shrunk to a preferred horizon.
- Independent 270M CPU equations and pinned real-checkpoint logits/all-layer K/V
  goldens. These are not 12B target goldens.

Historical cleanup CI passed 49 Python and 423 C++ tests with Metal. This was
reported at `dff16f0`, not rerun during this research. Evidence remains in
`build/gemma-foundation-cleanup-final-ci.log` and ignored
`benchmarks/results/gemma-foundation-cleanup-3iDPvN/`. The cleanup preserved
observed behavior; no learned speculative default earned promotion.

The complete research archive is
`archive/speculative-research-2026-09-25` at
`4579c7f5b903006e7db4f7c82e6e2dce9b6533da`.
The prior cleanup handoff, including its exact measurements, is recoverable with
`git show dff16f0:docs/NEXT_TASK.md`. See
[SPECULATIVE_DECODING_RETROSPECTIVE.md](../SPECULATIVE_DECODING_RETROSPECTIVE.md)
for the readable account. Do not resume archived training arms, compact heads,
shape registries, or broad experiments as prerequisites for this integration.

## 2. Actual artifacts and machine

### 2.1 Target and drafter identity

These values were read from the files, not inferred from their names. Whole-file
SHA-256 was computed from the same read-only mapping used by the independent
GGUF reader. The existing `build/release/gguf-inspect` also accepted the target's
payload layouts and all tensor types.

| Property | Target | Existing drafter |
| --- | --- | --- |
| File under `models/` | `gemma-3-12b-it-Q4_K_M.gguf` | `gemma-3-270m-it-Q8_0.gguf` |
| Whole file bytes | 7,300,574,976 | 291,545,600 |
| Tensor payload bytes | 7,294,024,704 (6.79309 GiB) | 285,018,624 (0.26544 GiB) |
| GGUF architecture / version | `gemma3` / 3 | `gemma3` / 3 |
| Metadata name / finetune | `Gemma 3 12b It` / `it` | `Gemma 3 270m It` / `it` |
| Layers / hidden / FFN | 48 / 3,840 / 15,360 | 18 / 640 / 2,048 |
| Query heads / KV heads / head width | 16 / 8 / 256 | 4 / 1 / 256 |
| Vocabulary | 262,144 | 262,144 |
| Advertised context / local window | 131,072 / 1,024 | 32,768 / 512 |
| Global RoPE base | 1,000,000 | 1,000,000 |
| RoPE scaling metadata | linear, factor 8 | absent |
| RMS epsilon | approximately 1e-6 | approximately 1e-6 |
| Tensor types | 288 Q4_K, 49 Q6_K, 289 F32 | 127 Q8_0, 109 F32 |
| Tensor count | 626 | 236 |
| Embedding / output | Q6_K embedding; no separate output tensor | Q8_0 embedding; no separate output tensor |

Target SHA-256:
`7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Drafter SHA-256:
`0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3`.
The latter matches the existing artifact pin.

Target quantization metadata is `general.file_type=15`, quantization version 2.
This is a mixed Q4_K_M checkpoint, not uniformly four-bit weights. Every layer
has Q4_K Q/K/O/gate/up matrices. V and down are each Q6_K in 24 layers and Q4_K
in the other 24. Norms are F32. The tied embedding/head is Q6_K and occupies
825,753,600 bytes, approximately 0.76904 GiB.

GGUF shapes are contiguous input dimension first:

| Tensor | GGUF dimensions |
| --- | --- |
| `token_embd.weight` | `[3840, 262144]` |
| `blk.i.attn_q.weight` | `[3840, 4096]` |
| `blk.i.attn_k.weight`, `attn_v.weight` | `[3840, 2048]` |
| `blk.i.attn_output.weight` | `[4096, 3840]` |
| `blk.i.ffn_gate.weight`, `ffn_up.weight` | `[3840, 15360]` |
| `blk.i.ffn_down.weight` | `[15360, 3840]` |
| `attn_q_norm.weight`, `attn_k_norm.weight` | `[256]`, shared across heads |
| Other norms | `[3840]` |

The directory contains only language-model tensors, including the two image
marker vocabulary entries. It contains no vision encoder/projector. Text-only
chat is the selected scope; a multimodal tag/template does not implement image
inference.

The hash identifies the local artifact. Its upstream quantizer, conversion
revision, and exact download repository were not authenticated. Metadata names
Google's pretrained base, not a complete conversion provenance chain. Do not
label it QAT, an unquantized reference, or an authenticated Google-distributed
GGUF without additional evidence. The official HF config endpoint returned HTTP
401 in this pass; public Google implementation and pinned GGUF sources provide
the architecture evidence below.

### 2.2 Tokenizer comparison

The files have identical token strings and scores for all 262,144 IDs, identical
chat templates, and identical BOS/EOS/unknown/padding IDs and common whitespace
flags. BOS=2, EOS=1, start-of-turn=105, end-of-turn=106. There are two token-type
differences: IDs 255999 (`<start_of_image>`) and 256000 (`<end_of_image>`) are
control type 3 in the target and user-defined type 4 in the drafter. The drafter
also explicitly stores `add_sep_token=false`, absent from the target.

The tokenizer's Gemma 3 profile already recognizes both image marker spellings
as special regardless of those types. Its semantic identity includes types, so
these tokenizer identities still differ. `StandaloneTliMapping` currently
rejects every target profile except Gemma 2; merely deleting that guard is not
a sufficient compatibility implementation.

Matching the tables also does not complete the special-token audit: the target
declares 6,251 type-3 entries, whereas `token_id_is_special` for Gemma 3 uses
eight recognized spellings. Check reserved/control entries and their encode,
decode, proposal and correction treatment against the independent tokenizer.
Do not equate that small whitelist with the complete GGUF control table.

For this pair, shared text IDs can avoid decode/re-encode work. Prove an
injective direct mapping for supported emissions and shared control roles;
handle the two image markers explicitly. It is valid to exclude unsupported
controls from the draft proposal support while preserving target support and
conditioning corrections appropriately. Do not suppress target tokens solely
to make the draft bridge easier. Keep the general correction/ledger machinery
for cases requiring it, and preserve literal-text versus control interpretation.

### 2.3 Measured hardware properties, not performance measurements

Read-only OS queries reported Mac16,12, Apple M4, 10 CPU cores, 10 GPU cores,
16 GiB physical memory, and Metal 4 support. A device-property query without
model buffers or GPU dispatches reported:

- `hasUnifiedMemory = true`.
- `recommendedMaxWorkingSetSize = 12,713,115,648` bytes (11.84 GiB).
- `maxBufferLength = 9,534,832,640` bytes (8.88 GiB).

The working-set recommendation is not free memory, a reservation, or a hard
allocation ceiling. Other apps, display resources, driver storage, clean file
pages, compression and swap still matter. Read these properties at preparation,
validate individual allocations, and report the plan's real allocation ledger.
Do not change system wired-memory limits to make a plan appear to fit.
[Apple's property documentation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
and [resource storage guidance](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
explain the device/storage boundary.

## 3. Source-backed numerical contract

Primary implementation sources inspected:

- [Google Gemma configuration](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py)
  and [Google model equations](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py),
  pinned at `014acb7ac4563a5f77c76d7ff98f31b568c16508`.
- [Gemma 3 GGUF execution](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/src/models/gemma3.cpp),
  [conversion](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/conversion/gemma.py),
  [position configuration](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/src/llama-model.cpp),
  and [window semantics](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/src/llama-hparams.h)
  at the repository's existing llama.cpp pin
  `d48a56effbba99944a677938d4beb63a0065ecdf`.
- [Transformers v4.55.0 Gemma 3 equations](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gemma3/modeling_gemma3.py),
  already used by the 270M independent reference.
- [Gemma 3 technical report, architecture section](https://arxiv.org/html/2503.19786v1#S2):
  design rationale for QK normalization and five local layers per global layer.

These are correctness references, not production dependencies. Source agreement
does not establish the provenance of the downloaded quantization.

### 3.1 Required forward computation

For this 12B text checkpoint, with mathematical matrices represented as
output-by-input and GGUF storage interpreted accordingly:

1. `x = sqrt(3840) * dequantize(E[token])`.
2. Each layer starts with `u = RMSNorm(x, attn_norm)`.
3. Project Q/K/V from `u`; normalize each 256-element Q and K head independently
   using the learned Q/K norm vectors. Do not normalize V or all heads together.
4. Apply half-split/NeoX RoPE to normalized Q/K. GQA maps query heads `2h,2h+1`
   to KV head `h`. Scale attention by `1/sqrt(256) = 1/16`, not by
   `1/sqrt(3840/16)`.
5. Apply causal local/global attention, project concatenated attention through O,
   then `a = x + RMSNorm(O(attention), post_attention_norm)`.
6. `u = RMSNorm(a, ffn_norm)`;
   `f = Wdown(GELU_tanh(Wgate(u)) * Wup(u))`;
   `x = a + RMSNorm(f, post_ffw_norm)`.
7. Final logits are `E * RMSNorm(x, output_norm)` using the same Q6_K values as
   embeddings. This file specifies no final softcap; attention also has none.

Use F32 accumulation/normalization as the initial numerical basis. The GGUF
conversion folds Gemma's learned `(1 + weight)` offset into every norm tensor,
including Q/K norms. Multiply by the stored norm weights; do not add one again.
Input embedding scale is the F32 square root for this GGUF path, not a
BF16-rounded source-framework scalar. An independent reference must describe
such dtype differences rather than demand bit equality to a different runtime.

Compiler math policy is part of this contract. The 270M library explicitly uses
safe math and precise floating-point functions; the shared Metal library is
currently compiled with `options:nil`. Choose and record the new target's math
policy and test it across scalar/batched paths. Do not silently inherit different
reassociation/transcendental behavior between ordinary decode and verification,
or change the Gemma 2 library globally as an incidental optimization.

### 3.2 Layer and position rules

Zero-based global layers are 5, 11, 17, 23, 29, 35, 41 and 47. The remaining
40 layers are local. For a query at logical position `t`, allowed local keys
are `max(0, t-1023) <= s <= t`; global keys are `0 <= s <= t`. The window
includes the current token. The pinned GGUF reference masks `t-s >= 1024`.

For each pair index `j` in `[0,127]`, pair coordinates `j` and `j+128`:

- Local angle: `t * 10000^(-2j/256)`.
- Global angle: `(t/8) * 1000000^(-2j/256)`.

The factor 8 applies to global RoPE only and from the first position, not only
beyond 32K. Local RoPE remains unscaled. A cache slot index is never a RoPE
position. Verification rows require explicit logical positions independent of
packed row indices or branch topology.

The file omits the local base and layer-pattern keys. Resolve those omissions
from the verified Gemma 3 architecture contract above, record the resolved
values in model identity, and validate overrides if later supported. Do not
silently guess policies for unknown Gemma-family geometries.

### 3.3 Chat and stopping

Implement the file's text-template behavior: one BOS; alternating user/model
turn markers; assistant-to-model role conversion; content trimming; optional
initial system text folded into the first user turn with the template's two
newlines; and the model generation prefix. Unsupported templates must fail
explicitly. Check Unicode whitespace and literal special-token behavior against
an independent rendering/tokenization reference, not merely the current
Gemma 2 formatter, which rejects system messages and trims ASCII whitespace.

Target EOS and end-of-turn are stop candidates. Retain existing handling for
reverse stop strings across token boundaries, UTF-8 streaming, interrupted or
throwing callbacks, and the selected/cached status of hidden stop tokens. The
target controls stopping; draft EOS or a later unaccepted draft stop does not.

## 4. Code audit: concrete changes and reuse

Paths below are current owners; new file names are suggestions, not a mandate
to create a large hierarchy. Consult `GRAPH.md` for test locations, and update
it only when actual modules/commands change.

| Area / current files | What exists or blocks Gemma 3 | Proposed change |
| --- | --- | --- |
| `include/local_gguf_engine/engine/model.hpp`, `src/engine/model.cpp` | Architecture enum and stored config only support Gemma 2; consumed-metadata digest is Gemma 2-specific | Add an explicit Gemma 3 config/architecture alternative, resolved local/global policy and scaling, native tensor validation, separate model context limit and runtime capacity; version identity for newly consumed semantics |
| `gemma2_model.*` | Sound tensor binding and tied-output handling; no Q/K norm fields | Add a compact `gemma3_model.*` binder with actual 12B shapes/types; reuse checked lookup/layout helpers where useful |
| `gemma2_metal_plan.*` | Resident weights/cache/scratch ownership is sound; layer pattern is Gemma 2; prefill workspace scales with full context; always allocates retrieval verifier scratch | Add a Gemma 3 resident plan with per-layer resolved policy, bounded capacity, native tied Q6 head, shared transient workspaces and mode-specific allocations |
| `gemma2_runtime.cpp` | Separate prefill, scalar decode and two-row retrieval schedules; fixed 10K RoPE and softcap assumptions | Build one Gemma 3 block encoder used by prefill, decode, endpoint materialization and verification; expose row/output/cache views explicitly |
| `src/tensor/backends/metal_backend.mm` | Actual shader source; K-quant GEMV/GEMM, online attention, RMSNorm+residual, GELU; embedding kernel is F16; attention and candidate kernels apply softcap | Add Q6 embedding gather, headwise Q/K RMSNorm/RoPE and explicit no-softcap paths; adapt existing attention to the actual cache dtype and masks |
| `metal/metal_context.*`, `metal/metal_command_context.*` | Pipeline specialization and resident dispatch; shape pins target 14/16/56 K-blocks; command header imports Gemma 2/verifier types | Add measured 15/16/60-block dispatch support; generalize only reusable operation descriptors; support Q6 head row tiles and explicit transform policy |
| `src/engine/target_verifier.cpp` | Topology attention builder calls `gemma2_layer_uses_sliding_window`; target plan requires a Q8 head; target K/V and backups are F16 | Make layer policy an explicit validated input; share topology validation; bind Gemma 3 transaction storage and native Q6 selection |
| `target_verifier_runtime.cpp`, `target_verifier_episode.cpp` | Private K/V, tickets, provisional commit and undo are valuable; forward loop is Gemma 2-specific | Reuse transaction lifecycle; execute Gemma 3 through its shared block encoder and correct positions/masks; preserve accepted-path publication |
| `chain_chat.mm`, `standalone_chat.*`, `prepared_gemma2_plan.*` | Draft-independent control but concrete Gemma 2 plan, callbacks, stats, target ticket calls and identities | Extract a narrow target execution boundary at preparation/episode level; retain one shared controller/lifecycle rather than duplicate it for each family |
| `standalone_draft_artifact.cpp` | Authenticated 270M artifact; bridge rejects Gemma 3 target; general expansion bounds size draft transactions | Add verified same-family mapping/control policy and an exact direct-ID capacity bound where proved; preserve conservative bounds elsewhere |
| `gemma3_draft.mm` | Complete 270M transformer and finite-grid q; geometry/Q8 kernels intentionally fixed; 32K draft capacity | Reuse as the drafter unchanged initially; do not scale its constants into a 12B runtime; profile selector/dispatch costs in the later speculative phase |
| `src/tokenizer/tokenizer.cpp` | Gemma 3 encoding exists; chat/generate format dispatch accepts only Gemma 2 | Add exact supported Gemma 3 text-template handling and metadata/special-token tests |
| `gemma2_chat_session.*`, `src/frontends/cli/generate.*` | Valuable turn/cache/callback logic, but session/prepare/metrics target Gemma 2 | Extract shared session behavior with explicit target dispatch, add runtime context capacity, preserve public success/failure behavior and diagnostic timing definitions |
| `sampling.*`, `speculative_controller.*` | Existing probability law and transitions are reusable | Keep authoritative; extend only actual interface needs, with analytic probability and history tests |
| `scripts/reference_gguf.py`, `gemma3_reference_golden.py` | Independent GGUF reading and 270M/Q8 equations | Add independent K-quant decoding and a bounded 12B reference/golden route; preserve the existing 270M pin and coverage |
| `scripts/quick_benchmark.py`, `src/benchmarks/*` | Real chat runner is reusable; standalone head map assumes Q8 target; kernel probes target Gemma 2 | Make production model/head/context identification accurate; explicitly route or omit unsupported diagnostic probes without bypassing chat; preserve historical result parsing |
| `Makefile`, tests, `docs/GRAPH.md` | Explicit build/test registry and locator | Register only added modules and meaningful coverage; no replacement harness |

Two easily missed points in current code:

- The retained Gemma 2 plan can requantize Q6 down matrices to Q4 and convert its
  F16 output head to Q8. Those are Gemma 2-specific product policies. Copying
  them into Gemma 3 would change this target's weights. The initial Gemma 3 path
  must consume the supplied Q4_K/Q6_K values without additional quantization.
- Existing verifier GQA geometry already matches 12B's two query heads per KV
  head and 256-wide heads. Reuse that work. Its softcap, F16 cache, layer pattern,
  RoPE, Q/K normalization and head restrictions still need explicit changes.

## 5. Recommended ownership and execution design

Prefer a typed Gemma 3 target plan plus a small shared execution boundary over
either renaming Gemma 2 or enlarging the fixed 270M drafter. An `EngineModel`
config variant can preserve architecture distinctions. The target owns weights,
layer descriptors, durable K/V, workspace arenas, and the state that tracks
pending verification/poisoning. A prepared session owns its target and optional
drafter; neither CLI nor benchmark code owns inference semantics.

A useful block-encoding boundary accepts explicit activation rows, logical
positions, layer policy, readable K/V spans, writable K/V destinations, and
required output rows. Ordinary decode, prefill and verification differ in
scheduling and cache publication, not their transformer equations. Make
normalization, projection, rotary convention, cache rounding, attention and
head selection the same numerical operations wherever the shapes permit.

Share the target-neutral proposal topology/controller and ticket lifecycle.
Resolve the concrete target once per preparation/episode using a small variant,
overload or equivalent narrow adapter. Do not introduce virtual dispatch per
primitive, a model graph compiler, a general tensor framework, or a registry of
hypothetical model families. Extract neutral status/selection/cache descriptors
when the second target actually needs them; avoid a repository-wide rename.
Gemma 2 compatibility wrappers can keep existing callers/tests stable.

Prepare resource capacities before generation. A target-only plan should not
allocate the learned verifier's maximum graph workspace or draft CDFs. An
explicit speculative plan allocates its configured row/horizon capacity once;
capacity constants remain supported independently of the chosen allocation.
Keep device input/probability leases epoch-checked and alive through decisions.

Immutable weights remain in native packed form. Alias the tied embedding/head
allocation and count it once in memory receipts. Q6 row gathering dequantizes
only selected embeddings; the output GEMV/GEMM reads the same matrix. Expanding
the matrix to F16 would take 1.875 GiB instead of 0.76904 GiB; retaining both
would be worse. A Q8 reconversion would add storage and alter values.

The existing shared-buffer copy path is a valid first implementation. It copies
mmap data into Metal storage; unified memory does not make that copy disappear.
Mapped clean pages can be evicted, so virtual mapped size is not additive RSS,
but startup can still page through both copies. Keep mmap/tensor-view lifetime
explicit and avoid full intermediate dequantization or duplicate uploads.

A later no-copy mapping candidate could use one page-aligned owning mapping
and tensor subviews, subject to Metal's allocation/alignment/lifetime rules,
file immutability and device buffer limits. Do not naively create a no-copy
buffer for each 32-byte-aligned GGUF tensor or assume a read-only mmap is an
acceptable Metal backing without validation. Compare startup, faults, RSS and
chat before choosing it. Private storage or a heap is likewise not automatically
faster on a unified-memory GPU. Keep one measured product policy, not a new
matrix of storage modes.

## 6. Memory budget, context and transactional caches

### 6.1 Allocate the requested capacity, not the advertised maximum

For runtime context capacity `C >= 1024`, resident target K/V bytes are:

`2 * 8 * 256 * bytes_per_element * (40*1024 + 8*C)`.

This uses 40 local rings and 8 full-prefix global caches, not 48 global caches.
The counts below exclude weights, temporary append rows, rollback backups,
activations, selection, tokenizers, the drafter and OS overhead.

| Runtime capacity | F16 target K/V | F32 target K/V |
| --- | ---: | ---: |
| 4,096 | 0.5625 GiB | 1.125 GiB |
| 8,192 | 0.8125 GiB | 1.625 GiB |
| 16,384 | 1.3125 GiB | 2.625 GiB |
| 32,768 | 2.3125 GiB | 4.625 GiB |
| 131,072 | 8.3125 GiB | 16.625 GiB |

Use a configurable 16,384-token initial default and assess 32,768 tokens with
the actual prepared allocation ledger and memory-pressure checks. The 8K
configuration is useful for bring-up, not a product ceiling. Preserve 131,072
as the model's position/context capability metadata; do not mutate it to mean
allocated storage. Add a validated runtime context option and report both.
Keeping powers of two initially fits existing ring masking and bounds
validation. Do not claim full 128K product support on this 16 GiB machine.

Use F32 target K/V for the initial correctness path. Evaluate F16 before freezing
the performance baseline, with independent F16-rounding reference evidence and
full-depth/window/verifier tests. The existing drafter already exposed amplified
F16 midpoint errors; that proves the issue is real, not that all 12B F16 cache
implementations must fail. Never select a lower precision silently because an
allocation fails, or relax parity tolerances to make it pass. Keep the drafter's
proven F32 contract.

The drafter currently allocates 207 MiB of durable F32 K/V (15 local rings of
512 rows plus 3 global caches of 32K), in addition to 0.26544 GiB of weights,
private rows, activations, tokenizers and selection. Its proposal CDF costs
`8*(262144+1)*H` bytes: approximately 2 MiB per horizon position. It also owns
sorting buffers. A 278 MiB file is not its resident total.

Both files' longest raw token piece is 48 UTF-8 bytes. Reusing the current
general bridge bound for this pair gives 144 append tokens and 292 private
draft rows at H3, about 10.27 MiB of draft private K/V. A verified direct-ID
route can tighten that bound as well as eliminate context retokenization.
The native weights of both models plus 16K F32 target K/V and
current durable draft K/V already total about 9.886 GiB, before these private
rows and all other workspaces. This is an allocation lower bound, not RSS.

### 6.2 Avoid context-sized attention intermediates

Current Gemma 2 prefill allocates one aliased F32 score/probability workspace
`[128, query_heads, context_length]`. Substituting 128K would allocate 1 GiB for
that workspace alone; even 8K is 64 MiB. Use an online, tiled attention path for
Gemma 3 so scratch scales with query tiles and reduction partitions, not all
query/key pairs. A small dense CPU/reference attention implementation remains
useful as an independent oracle.

Alias activation workspaces only after their last consumer. In particular, Q/K
head normalization and half-split rotation have cross-coordinate dependencies;
unsafe in-place aliases can race. Keep private K/V and rollback backups live
until the final callback receipt. Reuse compatible decode/prefill/verifier
activation arenas across mutually exclusive commands with explicit ownership,
not hidden overlapping borrows.

### 6.3 Publication and replay invariants

A chunk/transaction must read both the durable prefix and its private new rows.
Do not write an entire prompt chunk into a local ring before evaluating earlier
queries: wrapping can destroy keys that those queries still need. Stage current
rows, mask each query by logical position, then publish only the durable suffix.
This mechanism should serve both chunked prefill and speculative append.

Verification must never expose rejected K/V. Preserve target ticket ownership,
private rows, selected-path scatter, provisional commit, backup/restore of
replaced ring slots, and poisoning on device failures. Global truncation may
hide future rows by logical bounds; it cannot restore overwritten local slots.
A callback can make a prefix irreversible, including when it throws. Roll back
only the undelivered suffix and its sampler/draft state.

Each private target row across all 48 layers costs 0.375 MiB in F16 or 0.75 MiB
in F32. Four rows are inexpensive; allocating 1,025 rows unconditionally is not.
Current verifier backup capacity is 32 rows and costs 12/24 MiB for F16/F32 at
12B geometry. Size according to the actual commit contract while preserving
all supported accepted-path lengths; avoid arbitrary shrinkage to H3.

Regeneration must reproduce the saved numerical timeline. Current Gemma 2
session cache segments remember prefill versus decoded ranges because replacing
all replay with one large prefill can change rounding. Preserve this distinction
unless the new path proves equivalent replay. Context shifting rebuilds the
explicit retained conversation and both model states. It must not treat slot
rotation or subtracting positions as sufficient global-cache transformation.

The initial 16K target capacity is inside the drafter's 32K limit.
For later larger capacities, choose an explicit policy: a validated draft-only
window/rebuild or an explicit target-only continuation when drafting cannot fit.
A draft can condition on less context without invalidating lossless correction,
but q must be the law it actually used and acceptance may fall. Never silently
truncate target history, reuse stale draft global K/V, or discover insufficient
correction capacity after delivering a callback. The current code fails on
native draft overflow; larger-context handling is work, not an existing feature.

## 7. Optimization plan, ordered by whole-path relevance

### 7.1 Quantized projections and row reuse

The target's seven matrices per layer plus tied head are already supported
formats. Input widths are 3,840 (15 K-quant blocks), 4,096 (16), and 15,360 (60).
Existing function-constant specializations cover 14/16/56, so most new shapes
use generic block loops. Add only justified 15/60 specializations, including
both Q4 and Q6 down variants, after testing the 15-block worker tail. Output
widths are friendly multiples of 64; no broad arbitrary-shape framework is
needed.

Separate shape regimes:

- Single-row decode: packed GEMV, native quantized loads, F32 accumulation,
  compact target selection and one resident command schedule.
- Small verifier batches: begin with the real packed row kernels for 2–4 rows,
  including H+1 input/output bookkeeping. Share quantized weight loads across
  useful rows where register use/occupancy permit. A separate invocation of
  scalar decode for every proposal is a correctness oracle, not an adequate
  final verifier.
- Prompt prefill: retain the existing tiled SIMD-group matrix path as a
  candidate, with bounded chunks and correct tails; choose row tile/chunk sizes
  from actual prompt throughput and memory behavior.

Inspection found the current packed row-batch shader processes two activation
rows but scans packed weights inside each row's loop. Cache locality may help;
this is not guaranteed single-dequantization reuse. The ordinary dispatcher's
row-tile comment/grid and the shader's two-row tile also warrant an audit for
redundant threadgroups before copying the dispatch. The verifier explicitly
uses a two-row grid. Do not claim a new speedup or failure from this inspection.

Existing tiled GEMM stages weights and activations as half before F32 matrix
accumulation. That is a numerical choice, not only a faster scheduling choice.
Validate deep-network prefill and small-batch parity before sharing it with
verification. Preserve scalar accumulation order when possible; otherwise
justify error against independent equations, especially near top-k boundaries.
The old width-four failure does not reject every 12B row-sharing design, but it
is a reason to avoid retaining a large unmeasured kernel family.

From tensor payloads, FFN matrices are approximately 70.5% of target weight
bytes and the tied head is 11.3%. Those are traffic estimates, not measured time.
Prioritize useful weight reuse over cosmetic removal of tiny operations.
[The pinned llama.cpp Metal implementation](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal.metal)
provides independent examples of packed K-quant loads, quantized embedding
fetches and distinct vector/tiled attention paths; it does not prove its launch
configuration is best for our engine.

### 7.2 Attention, Q/K normalization and fusion

Keep the paired-query-head GQA mapping so one K/V load can serve two queries.
Use online softmax with F32 running maximum, denominator and weighted value
sum. For another score/value tile, rescale the old sums by `exp(m_old-m_new)`,
accumulate `exp(score-m_new)` and its value products, then divide once. This
avoids storing a full score matrix while preserving the mathematical attention.
It is an adaptation of the IO-aware mechanism described by
[FlashAttention](https://arxiv.org/html/2205.14135v2), not a reproduction of its
CUDA implementation or a promise of its published speedups.

Use a direct output when one partition suffices and merge partials otherwise.
Partition rules should depend on each query's logical key range, not on how many
proposal rows happen to share a dispatch. This helps stable prefill/decode/replay
parity. Tune local 1K and global longer-context work separately after attribution;
a fixed maximum partition launch can waste work on short prefixes.

Implement per-head Q/K RMSNorm with width 256 and explicit positions first.
A fused Q/K normalization + RoPE + private-K store is a strong candidate because
these are required new operations and fusion removes intermediate traffic and
barriers. Keep V's storage/rounding separate in the contract, and do not rotate
in place until the pair dependencies are safe. Similarly reuse proven
post-norm-plus-residual and GELU-times-up fusion.

Further candidates are grouped Q/K/V dispatch and gate/up projection scheduling.
Mixed Q4/Q6 V types and independent output buffers make a single concatenated
weight upload unnecessary and potentially wasteful. Fuse only when it reduces
measured work without increasing register pressure or breaking precision.

### 7.3 Head and sampling

Generalize the target head tile encoder to native Q6_K and explicit source row
offsets. The current shared verifier refuses anything except a Q8 head. All
ordinary and speculative paths must use the same native head and raw-logit
transform. Represent “no softcap” explicitly in host validation and shader
specialization. Passing zero, infinity or a huge fake softcap through the
existing positive-softcap code is numerically wrong or unnecessary work.

Project only required output rows. Prompt prefill usually needs only the last
row's logits. Verification needs distributions at each relevant proposal prefix
and the post-full endpoint when used. The ordinary last-prefill-row optimization
already exists and should be preserved. Selective last-layer FFN/output-row
work, as in the reference graph, is a later bounded candidate when intermediate
features are not requested; all necessary K/V must still be produced.

Retain the current compact target selection contract: target top-k is 1–256
(default chat top-k 40), with enough base candidates plus repetition-history
logits to reproduce the transformed top-k exactly. The target uses its full
262K output head before selecting that support. Do not mistake compact readback
for a compact learned vocabulary or restrict the target to the draft support.
Pure unrestricted full-vocabulary target sampling is not implemented by the
current bounded sampler; do not silently advertise it. Extending it would be a
separate explicit API/probability-law change.

The drafter's q is different: it evaluates its full vocabulary, applies its own
configured transforms, and samples from a 32-bit finite-grid CDF. `gather`
reads q only at the proposed token and target support after GPU completion.
Because residual probability outside target support is zero, this is sufficient
for exact correction under the supported target sampler. Keep its lease and
actual integer masses rather than recomputing an ideal softmax on CPU.

A fused head-plus-top-k kernel could avoid a 1 MiB logits write/read per row,
but cannot avoid evaluating the full head and needs exact repetition/tie rules.
It is a profiling candidate, not the first integration task. Likewise, the
270M selector currently sorts the vocabulary when filters require it and stores
full CDFs. Optimizing that is a later drafter cost question, not a reason to
replace its faithful probability law with a truncated approximation.

### 7.4 Command scheduling and residency

Keep weights, K/V, scratch, proposals and selection requests resident. Avoid
host reads of hidden states/full logits, transient tensor allocations, and
per-layer command submission. Reuse engine host command-context storage and
prepare required pipelines before timed generation. Metal command buffers are
single-use driver objects; do not describe their reuse as zero driver allocation.

Existing target verification performs a completion wait, publication work, and
possibly undo/finalization at callback boundaries. These waits have semantics.
Measure their contribution before batching them or moving decisions onto GPU.
An all-GPU acceptance path would also need exact target transforms, integer q,
checkpointed RNG and recoverable per-callback prefixes. It is not required to
make the target ready, and CPU correction over at most 256 probabilities is a
reasonable starting boundary.

Encode independent Q/K/V or gate/up operations without unnecessary intermediate
barriers; synchronize their actual consumers. Do not delete barriers wholesale.
Target and drafter share a bandwidth-constrained GPU, so simultaneous execution
is not automatically useful and does not remove autoregressive dependencies.
Metal 4 support is not a reason to replace the existing shader path with an
external encoded network or tensor runtime. Raw Metal scheduling changes need
measured CPU/GPU-gap evidence; SIMD-group primitives already serve native code.

## 8. Speculative readiness and the later integration

The lossless algorithm is established in
[Leviathan et al., Algorithm 1 and Appendix A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf):
for a proposal `y ~ q`, accept with `min(1,p(y)/q(y))`; on rejection sample from
normalized `max(p-q,0)`. If all proposals survive, sample the next target
conditional. Here p includes target temperature, repetition history and filters;
q is the actual transformed finite-grid draft law. Family membership is not
part of that proof.

Reuse `sampling.cpp`, its analytic marginal tests and the controller. The target
integration must establish these boundaries before drafter tuning begins:

1. Ordinary and verifier rows compute the same target conditional for identical
   retained history/positions, subject to the documented numerical contract.
2. Request construction uses the 5:1 layer policy in both topology creation and
   validation, including every conditional tile. Validation must not regenerate
   a Gemma 2 attention schedule and accidentally bless the wrong masks.
3. Candidate probabilities include the hypothetical accepted prefix in
   repetition history, with the same window eviction and stop semantics as
   ordinary generation. Verification does not resample a proposal from p and
   compare IDs in place of the acceptance/residual algorithm.
4. Private target rows can publish any accepted prefix and recover every
   callback-delivered prefix after rejection, interruption or exception.
5. Corrections, final accepted tokens and all-accepted bonus/carry state have
   explicit selected-versus-materialized positions. Preserve the existing
   pending-token optimization; do not count or forward a carried token twice.
6. The endpoint can materialize required K/V without an unnecessary output-head
   projection when its next distribution is not needed.
7. Full and conditional schedules use the same target weights/cache precision
   and preserve the target law. H3 conditional was the best existing learned
   configuration on Gemma 2, without earning default promotion. It is only a
   starting comparison, not the new target's chosen optimum.
8. The direct-ID bridge has tested control/correction semantics and a proved
   capacity bound. Reducing `max_append_tokens` to one is valid only when every
   supported target append has an exact one-ID draft representation; do not
   replace the existing conservative bound based only on matching vocab size.

Do not duplicate graph validators, reduce retained topology capacities, or add
new tree/training machinery. First exercise the actual new target verifier with
scripted proposals, then attach the existing 270M to that boundary. Scripted
perfect proposals establish numerical/lifecycle capability and optimistic target
cost, not real draft acceptance or a product speed claim.

For planning only, let `A` be accepted prefix length, `Tordinary` ordinary target
time per token, `D(H)` complete draft cost, `V(tile)` actual target tile cost, and
`U` selection/publication/correction cost, with all costs expressed as time.
For a full chain the rough throughput comparison is
`Tordinary * E[A+1] / (D(H) + V(H+1) + U)` for uncensored cycles with consistent
carry accounting. Conditional verification replaces V with the sum of tile
costs weighted by the probability that each tile is reached. Stops, budget
endpoints, prompt work and extra correction materialization must be accounted
for in the real execution counters. Do not assume `V(H+1)=Tordinary` or infer
speed from mean acceptance alone. This cost decomposition is an engine planning
adaptation, not an empirically fitted prediction.

Apple specifies 120 GB/s memory bandwidth for this M4 Air. Streaming the native
7.294 GB target payload once per scalar step gives an optimistic calculation of
about 60.8 ms (16.45 steps/s), before attention traffic, arithmetic, dispatch,
head selection, paging or thermal effects. This is not a measured limit or a
speed promise; multirow reuse changes the premise. It explains why amortizing
real weight passes is more promising than saving a few host instructions.
[Apple M4 Air specifications](https://support.apple.com/en-us/122209).

## 9. Independent correctness evidence and focused tests

Reuse the existing test runner and support code. Add deterministic coverage for
the new behavior, not a parallel testing product. Do not require all 12B weights
in every unit test, and do not call skipped local-checkpoint tests passing
checkpoint parity.

| Evidence | Minimum useful coverage |
| --- | --- |
| Binding/metadata | Real directory shapes/types, tied-head alias, Q/K norms, resolved omitted defaults, mixed V/down dtypes, malformed shapes/types/scaling/pattern, overflow and unsupported architectures |
| Independent quantization/math | Q4_K/Q6_K block scales/minima/signs and tail blocks; embedding/head orientation; stored norm offset; per-head Q/K RMSNorm; GELU; half-split rotation; factor 8 only on global layers; no softcaps |
| Attention/cache | Nonzero patterned K/V at local positions 1023/1024/1025; layer transition 4/5/6; paired-head GQA; causal chunk masks; multiple ring wraps; global retention; explicit high logical RoPE positions without allocating 128K |
| Native target parity | Small independent complete-transformer fixtures; actual 12B checkpoint selected hidden states/logits and representative/all-layer selected K/V slices; prefill, scalar continuation and small-batch verification |
| Precision/replay | F32 baseline; explicit F16 candidate with rounding-aware oracle; scalar versus batched/tiled numerical drift; regeneration across mixed prefill/decode segments; cached-prefix reuse |
| Target selection | 262K logits with extreme finite values, no-softcap identity, ties and repetition penalties crossing the top-k boundary; CPU full-logit transform versus compact candidates; identical transforms in ordinary/verifier endpoints |
| Transaction/lifecycle | First/middle/all acceptance, residual and bonus/carry, local wrap, every callback prefix, callback throw, stop/budget/context boundary, failure before and after provisional publication, stale lease, poisoning |
| Tokenizer/chat/API | Exact template and target/draft control cases, initial system folding, whitespace and multilingual/byte pieces; two turns, regenerate, context shift, EOS/end-of-turn/reverse stops; unavailable Metal and allocation/config failure |
| Lossless law | Retain exact analytic marginals, independent injected draws, support mismatch/zero masses, finite-grid boundaries, RNG checkpoint/rollback and target-history authority; connect these tests to the new target |

Extend the offline GGUF reference with independent Q4_K/Q6_K decoding rather
than importing the engine's optimized decoder as its oracle. Validate those
blocks against known layouts and the pinned external reference. Stream one
layer or bounded matrix/head blocks at a time: dequantizing the entire 12B to
F32 is approximately 47 GB and cannot be the reference plan on this machine.
Run heavyweight reference generation separately from the resident engine.

Use exact checkpoint IDs/token sequences and record reference revision, cache
precision, logits convention, and numerical tolerances with goldens. Compare
absolute/relative logit and K/V error, selected intermediate values, and target
probabilities on the same prefixes; cosine similarity or readable output alone
is insufficient. Set tolerances from independent operator error and accumulated
precision behavior before retaining an optimization. Batch-dependent divergences
must be investigated, not hidden by a tolerance increase.

The pinned llama.cpp can supply a second numerical check on this exact GGUF,
with cache dtype, context, RoPE and native weights matched. The existing stored
Gemma 2 performance numbers are not a Gemma 3 baseline. No external runtime may
be used by production chat, even as an automatic fallback.

## 10. Proposed implementation sequence

These phases describe one authorized target-integration effort and a later
speculative phase. Carry the target effort through its handoff without treating
the intermediate phases as separate permission gates.

### Phase 1 — Bind the actual target and establish its oracle

Add typed Gemma 3 model configuration/weights, architecture validation,
resolved layer/RoPE semantics, source identity, requested context capacity, and
native Q6 embedding/head contract. Implement the bounded independent reference
route and exact supported text-template behavior. Establish real-checkpoint
reference evidence without production execution through another library.

Exit: the actual file binds correctly, malformed inputs fail before expensive
device preparation, memory is calculable, and the numerical/chat contract has
independent executable evidence. No claim of working 12B chat yet.

### Phase 2 — Complete resident target inference end to end

Implement the Gemma 3 plan and shared block encoder, Q6 embedding gather,
headwise Q/K normalization, correct local/global RoPE, online attention,
private chunk staging and native Q6 head/compact selection. Connect real
prefill, decode and text chat with bounded initial context; use engine-owned
Metal throughout. Keep useful existing quantized kernels and F32 cache initially.

Exit: actual 12B chat, multi-turn reuse, stops, regeneration/context handling,
unavailable-device behavior and checkpoint numerical parity pass. Preserve
Gemma 2 tests while extracting only the shared session machinery required.

### Phase 3 — Make target verification a supported execution path

Adapt shared topology validation, target tickets, private K/V, native head row
tiles and chain execution to the new target. Use its same numerical core;
exercise full/conditional schedules and scripted proposals through the real
controller/publication/callback path. Allocate by configured capacity.

Exit: verifier versus ordinary target logits/K/V/selection parity, accepted-path
publication, all rollback/stop/callback endpoints and context limits pass. The
small-batch path executes actual batched rows, not H scalar calls hidden behind
a batching interface. Existing graph capacities and validators remain intact.

### Phase 4 — Profile and retain a bounded target optimization set

Establish an ordinary Gemma 3 real-chat baseline and attribution. Measure native
resident memory/startup behavior and determine whether F16 target K/V satisfies
the independent precision contract. Freeze the chosen target representation
before assessing lossless speculation.

Optimize the measured dominant costs: likely 15/60-block projections, useful
small-row weight reuse, online attention scheduling, required-operation fusion,
and unnecessary allocations/dispatches/readbacks. Use scripted verifier costs
as diagnostics alongside actual chat, not as a replacement for it. A larger
prefill matrix tile, no-copy weights, fused head selection or scheduling rewrite
requires evidence for that specific cost. Remove rejected candidates.

Exit: correctness and full CI pass; retained changes have comparable real-path
evidence; target-only and verifier costs/resources are known; stop at the
handoff in section 11. It is acceptable to finish with a clear measured
remaining bottleneck rather than promise an unattainable globally optimal
implementation.

This handoff establishes the target mechanism and its verification interface.
The real drafter may favor different batch sizes or expose another target cost;
later tuning can revisit kernels without rewriting target semantics or cache
ownership. Source inspection cannot guarantee that target performance work
will never be useful again.

### Later phase — Attach 270M and tune useful lossless speculation

Complete the validated direct-ID/control mapping and prepared-plan wiring,
then run the actual pair with the retained finite-grid q and target-authoritative
sampler. Start with a small full-chain and conditional comparison informed by
measured target cost; H3 is a reference configuration, not a requirement.
Measure accepted-prefix survival, draft/verification/commit/endpoint costs and
real natural-stop chat. Adjust horizon/tile strategy only when the saved target
work pays for it. Profile draft selection or synchronization if those become
dominant. No training is required to establish this pair's baseline; substantial
training remains a separate user-run decision.

## 11. Verification, performance protocol and stop conditions

For implementation changes run `bash scripts/ci.sh` after the relevant checks,
with actual Metal coverage or explicitly report unavailable-device limitations.
Doc-only work does not require CI. Tests using omitted local weights/goldens
must clearly report that omission.

Before calling the target ready for the speculative phase, require:

- Actual downloaded Gemma 3 12B text chat runs natively with its Q4_K/Q6_K values,
  correct target semantics and supported public lifecycle behavior.
- Runtime capacity, cache precision, unique resident weights, scratch and optional
  verifier/draft allocation ownership are explicit and fit the measured machine
  with assessed memory pressure. No metadata-limit-based 128K allocation.
- Ordinary and batched verifier paths use the same target definition and pass
  independent numerical, probability-transform and transactional tests.
- A small real batched verifier is measured; H+1 outputs, conditional tiles,
  corrections, accepted-path publication and callback rollback are already
  supported by the target boundary. Later work should not need another target
  transformer or cache-owner rewrite to attach the drafter.
- Target-only chat baseline, startup/TTFT, prompt and generation counts/times,
  memory metrics and remaining cost attribution are recorded in ignored output.
  No speculative speed claim is inferred from scripted proposals.

Use `make MODE=release benchmark` and the canonical real-chat driver with this
model once its target/head/probe routing is supported. Add only the needed
model/context options and correct diagnostics; keep frontend logic separate
from inference. For any retained optimization, use a fresh candidate and
baseline process and at least four minutes of idle cooldown before each as
required by `AGENTS.md`. Avoid concurrent builds, tests or reference generation.
Require at least 2% aggregate generation improvement, wins on both turns and
no greater-than-2% regression in another headline metric, with equivalent
workload definitions and the applicable correctness contract.

Record raw evaluated/cached/generated counts, durations, output hashes, seed,
model/resident identity, context/cache precision, head format, configuration,
process wall time, peak RSS, allocation ledger, page/swap/compression observations,
and available thermal observations. Distinguish TTFT/startup effects from token
throughput. Missing thermal warnings do not prove equal clocks. Use existing
metrics and scripts; do not build an evidence-management framework.

Different natural-stop output lengths produce different follow-up contexts.
For lossless runs, retain that fact explicitly and support attribution with
fixed-prefix diagnostics when necessary; do not call them exact-work pairs or
require identical selected-token hashes. For a refactor of the frozen Gemma 2
exact-match path, retain its established hash checks. The Gemma 3 speculative
control must be Gemma 3 ordinary decoding (and any independently retained
Gemma 3 retrieval mode), not Gemma 2 performance. Do not promote retrieval or
learned speculation as a Gemma 3 default without its own evidence.

Changing the reference model invalidates the stored external performance
comparison under `LLAMA_CPP_REFERENCE_BASELINE.md`. If a new llama.cpp comparison
is authorized during implementation, run it separately with this exact file and
comparable template/sampling/cache/timing settings, then update that document's
owned identity/result. Do not silently reuse Gemma 2 figures or replace the
pinned revision during routine development.

Remaining empirical questions: sustained target cost at longer contexts and
prompts, verification scaling beyond the measured two/four-row cases, long-context
F16-cache fidelity, combined target/drafter memory pressure, target/draft
distribution overlap and full-versus-conditional scheduling under actual
acceptance. These cannot be resolved by source inspection alone. Upstream GGUF
conversion provenance is also still unverified; local identity and architecture
are known.

**Completed target-integration stop condition:** the native target and batched verifier meet section
11's correctness, lifecycle, resource and measurement requirements; full CI is
complete and remaining limitations are recorded. Stop before learned-speculation
tuning or substantial training.

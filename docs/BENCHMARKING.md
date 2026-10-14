# Release chat benchmarking

This benchmark compares the installed CLIs as people use them: `guffin -m MODEL`
and `llama-cli -m MODEL`, with their normal chat defaults. It runs locally using
Python's standard library. No model download or evaluation service is involved.

## Workload and timing policy

The [frozen corpus](../benchmarks/chat/README.md) contains 32 two-turn MT-Bench
conversations across eight categories, plus six supplementary conversations for
short replies, multilingual text, document/code input and sustained discussion.
There are 38 conversations and 80 user turns per configuration. Supplementary
results are reported separately from the balanced MT-Bench core.

Each case runs **once per configuration**, in a fresh CLI process. Configuration
order rotates between cases. The runner waits **five minutes before each
conversation**, including the first one and the first pending case after resume.
It keeps the CLI and its chat cache alive during **two-minute pauses between
user messages**. After the final answer, it closes the process before the next
five-minute wait. These pauses are excluded from native engine timings.

Sampling, seed, context capacity, output limits, GPU placement, draft horizon
and warmup remain each CLI's defaults. Replies finish under the CLI's own
stopping policy. Follow-ups include that configuration's preceding answers;
the user messages match, but generated text, history and token counts can differ.
This measures ordinary chat experience, not identical token sequences or model
quality. Gemma 2's default Guffin path includes suffix retrieval and its existing
weight conversions; those remain part of the comparison.

## Run

Build and install the current release executable first:

```sh
make install
```

From the repository, inspect the schedule without starting inference:

```sh
python3 scripts/release_chat_benchmark.py \
  -m models/gemma-3-12b-it-Q4_K_M.gguf \
  -o benchmarks/results/gemma3-normal-local --plan
```

Start the default-chat comparison:

```sh
caffeinate -i python3 scripts/release_chat_benchmark.py \
  -m models/gemma-3-12b-it-Q4_K_M.gguf \
  -o benchmarks/results/gemma3-normal-local
```

`caffeinate -i` prevents idle system sleep during the long pauses. Keep power
mode and background workload consistent; cooling pauses cannot prove equal
GPU clocks or temperature. The runner records power, thermal-warning and swap
snapshots, rather than claiming those conditions are controlled perfectly.

Run the speculative comparison separately:

```sh
caffeinate -i python3 scripts/release_chat_benchmark.py \
  -m models/gemma-3-12b-it-Q4_K_M.gguf \
  --draft-model models/gemma-3-270m-it-Q8_0.gguf \
  -o benchmarks/results/gemma3-speculative-local
```

Without `--draft-model`, the campaign contains only ordinary Guffin and
ordinary llama.cpp. With it, the campaign contains only speculative Guffin and
speculative llama.cpp. Each has its own manifest, receipts, resume state and
report. Use different output directories and run the campaigns sequentially.

Guffin receives `--draft-model FILE`; the release llama-cli receives
`--model-draft FILE --spec-type draft-simple`. The mode flag is necessary:
llama.cpp 0.5.0/build 11146 otherwise loads the drafter without activating it.
Each implementation keeps its own default horizon and speculative algorithm.

For Gemma 2, substitute `models/gemma-2-9b-Q4.gguf`, omit the drafter and use
`benchmarks/results/gemma2-local`. These examples collect new local campaigns
into ignored directories. The three committed final campaigns are complete;
preserve their evidence and never use their directories for a fresh run.

Each complete campaign contains two configurations and its waits alone take
about **9.1 hours**; both Gemma 3 campaigns take about **18.3 hours** together.
Loading and inference add to that. `--suite core` selects the 32 balanced cases;
`--case ID` selects individual cases. Neither changes inference settings. A 15-minute
startup/turn deadline prevents indefinite hangs; a timeout is a failed case,
never a shortened successful answer. `--timeout SECONDS` changes this safety
deadline if necessary.

## What is measured

The runner uses a pseudoterminal so both CLIs execute their real interactive
chat paths. It drains stdout and stderr continuously, transports multiline
messages through each CLI's native continuation syntax, and waits for native
metrics plus the next input prompt before considering a turn complete.

Only measurement logging is added: Guffin's `--decode-diagnostics` and
llama-cli's `--log-verbosity 4`. The latter exposes server-slot counts and
phase durations behind the CLI. Logging has overhead, especially during
prompt processing; these are instrumented default-chat measurements.

| Metric | Boundary and convention |
| --- | --- |
| Internal TTFT | Post-tokenization prompt handling through first target-token selection. Uses each engine's native clock; excludes model loading and terminal input. |
| Prompt tokens/s | Newly evaluated prompt tokens divided by that prompt interval. Cached tokens are recorded separately. |
| Common generation tokens/s | `(selected tokens − 1) / generation duration`, clamped to zero steps for an empty/single-token response. Duration runs from first to last token selection. |
| Native generation tokens/s | Each CLI's printed rate, retained alongside the common rate. Guffin counts selected tokens; this llama-cli counts subsequent generation steps. |
| Guffin completion clocks | Its native full-turn, first-text and generation-completion durations, retained separately; llama-cli does not expose equivalent clocks here. |
| Process memory | macOS physical-footprint lifetime high-water mark from the live CLI process, including draft allocations; resident memory is a separate sampled maximum. |

Counters retain native token conventions, including any selected stopping
control token; they are not reconstructed by tokenizing visible answers.
Llama durations are rounded to hundredths of a millisecond. The parser checks
counts, durations, printed rates and task identity with the printed precision.
It rejects incomplete or inconsistent metrics.

Guffin diagnostics identify the stop reason. This llama-cli does not expose an
exact per-turn stop reason in its ordinary logs, so that field remains unknown.
Frontend completion must not be described as proven EOS termination. Known
capped/stopped-short and incomplete cases are excluded from headline results.
Context fit can also change through natural conversation growth; failures and
native counts remain visible rather than being silently truncated by the harness.

Memory is sampled once per second and at turn boundaries. Native lifetime peak
footprint survives a peak between samples; a peak after the last live sample
can be missed. RSS maxima can miss intervening peaks. Swap snapshots are
system-wide and cannot attribute pages to either engine.

## Results and recovery

The output directory holds a manifest, atomic conversation receipts and raw
stdout/stderr logs. Receipts retain user messages, displayed answers, answer
hashes, token counts, durations, memory and failure details. The manifest pins
model/binary hashes, CLI versions/help, linked-library identities, corpus and
measurement-script hashes, hardware, OS, relevant environment and the protocol.
The reporter writes JSON, CSV, Markdown and standalone SVG charts under
`report/`. The repository retains exactly the final `gemma2-release-v2`,
`gemma3-normal-release-v2` and `gemma3-speculative-release` campaign trees,
including receipts and raw logs. Other generated campaigns remain ignored.

Headline comparisons and charts use only cases completed by **both** engines
for the same target, mode and suite. Aggregates sum tokens and durations before
calculating throughput. Separate coverage, category, first-turn/follow-up and
unpaired results remain available. One pass does not support confidence
intervals, a stability claim or a universal percentage advantage.

Press Ctrl-C to stop, then rerun the same command to resume pending cases.
Every attempt is saved before launching the CLI. Completed, failed and
interrupted conversations are **never automatically repeated**. A failure
stops the current invocation; resuming proceeds to unattempted jobs and keeps
incomplete cases explicit. A campaign containing incomplete attempts returns
nonzero. Changed binaries, models, corpus, scripts, timing policy or recorded
execution environment require a new output directory. Switching between
ordinary and speculative mode also requires a separate directory. Existing
results remain available to the offline reporter; the updated collector does
not resume campaigns recorded with an earlier measurement-script hash.

Reports can also be regenerated offline:

```sh
python3 scripts/release_chat_report.py benchmarks/results/gemma3-normal-release-v2
```

The [final results](PERFORMANCE.md#final-release-results) link the three complete
campaigns. Their successful conversations are performance observations, not
independent numerical or answer-quality qualification. Earlier
[development findings](PERFORMANCE.md#historical-development-evidence) and the
[pinned llama.cpp reference](LLAMA_CPP_REFERENCE_BASELINE.md) remain historical.
Use a new directory for any later build or protocol; preserve final receipts.

# Next Task: Attribute the Resident Decode Step

## Verified Current State

The canonical quick benchmark now measures the actual local interactive product:

```sh
make MODE=release benchmark
scripts/quick_benchmark.py --model models/gemma-2-9b-Q4.gguf
```

It launches `build/release/local-gguf-generate ... --chat --backend metal`.
Each fresh process receives
`Hello, is AI getting smarter?` and then
`Okay, so will it always progress this fast?`, with normal interactive sampling
and seed 42. Both responses run to their natural stop. There is no 64-token cap,
synthetic input, custom libllama helper, or hidden local warm-up.

The headline rates use the pinned `llama-cli` `result_timings` boundaries in
both products. Prompt processing begins after chat rendering/tokenization,
before KV reconciliation, and ends when the first token is sampled; that same
duration is internal TTFT. Generation begins at the first sampled token, ends
at the final sampled token, and divides the full sampled-token count by that
duration, matching llama.cpp's convention exactly. Local retains resident
prefill and decode-step durations as diagnostics, not headline rates.
The runner verifies that raw counters reproduce every displayed rate. The
pinned llama.cpp benchmark and stored result are separate in
`docs/LLAMA_CPP_REFERENCE_BASELINE.md` and are not run during ordinary local
optimization work. When a terminal is attached, the child stream is rendered in its alternate
screen during inference and cleared before the compact report. This retains the
real terminal/compositor contention seen by manual chat without leaving the
generated responses in the benchmark output. The follow-up is submitted only
after turn one completes and a fixed five-second input pause; the pause is not
inside either engine timing interval, but avoids the unrealistic always-ready
input that kept Metal at decode clocks.

The quick screen is one local run with 30-second idle periods before chat and
before diagnostics. This avoids multiplying long natural responses on the
fanless M4; it is not the five-pair retention protocol. Local has no warm-up.
Five local projection probes launch afterward in a separate process and cannot
warm the headline chat.

The old capped in-process benchmark, custom libllama helper, conversation
benchmark, frontend-TTFT calibration, startup/memory reporting, scenario JSONL,
and old retention comparator have been removed.

## Objective

Attribute one warmed production resident decode step to its important operation
families, then select exactly one kernel family for the following optimization
baton.

At minimum distinguish:

- Q4_K projections by exact Gemma shape;
- the Q6_K down projection;
- the F16 output projection;
- attention;
- normalization, residual, RoPE, and KV-store work;
- compact candidate selection; and
- unaccounted command-buffer time.

## Constraints

- Measure the unchanged production decode schedule and buffers; do not create a
  second inference path.
- Keep the normal quick benchmark output at its compact chat and kernel tables.
  Detailed attribution belongs in machine-readable diagnostics.
- GPU intervals must not add synchronization, command buffers, barriers, or
  readbacks to normal inference when diagnostics are disabled.
- Report duration, percentage of complete decode GPU time, dispatch count,
  approximate bytes, effective bandwidth where meaningful, and the maximum
  plausible whole-token gain for each family.
- Preserve exact selected-token parity, F16 KV contents, cache position,
  resident allocation identities, and unavailable-Metal behavior.
- Do not optimize or replace a kernel in this task.

## Verification

- Add deterministic aggregation/schema tests and Metal attribution coverage.
- Prove attributed plus explicitly unaccounted time matches the production
  command-buffer GPU interval within the defined rounding tolerance.
- Run `bash scripts/ci.sh`.
- Run one real release quick benchmark and one diagnostic decode profile after
  a short thermal cooldown.

## Stop Condition

Stop when a warmed production decode step has trustworthy per-family GPU
attribution and one family has the largest evidence-backed whole-token
improvement ceiling. Record that family and its acceptance test as the next
implementation baton; do not optimize it yet.

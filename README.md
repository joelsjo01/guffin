# local-gguf-engine

A from-scratch local GGUF inference engine optimized for a MacBook Air M4.
The native Metal path runs Gemma 3 12B Q4_K_M text chat and retains Gemma 2 9B
Q4 support, with resident weights, KV caches, prefill, decode, and sampling. External inference engines
are used only as references and benchmarks.

Gemma 3 target execution includes batched verification, transactional cache
publication and optional lossless speculation with the complete native Gemma 3
270M drafter. Compact lossless weight packing and coalesced Metal projections
improve verification; H=1 passed the recorded cooled chat speedup gate on both
turns. Ordinary decoding remains the default, with speculation explicitly
selected. Read
[what we learned about speculation](docs/SPECULATIVE_DECODING_RETROSPECTIVE.md)
and [the current boundary and research archive](docs/NEXT_TASK.md) before
continuing that work.

## Requirements

- An Apple Silicon Mac. The optimized release build currently targets M4.
- macOS with Metal support.
- Apple Clang and the Xcode Command Line Tools.
- A local supported Gemma 2 GGUF or Gemma 3 12B Q4_K_M GGUF (native Q6_K tied head).

Install the command-line tools if `clang --version` is unavailable:

```bash
xcode-select --install
```

Model files are not included. Put them under `models/`; that directory is
ignored by Git. The commands below assume:

```text
models/gemma-2-9b-Q4.gguf
```

Model file size is not the runtime memory footprint. Resident weights, KV
cache and scratch buffers share unified memory with macOS and other apps.
Head and context settings affect these allocations; `--decode-diagnostics`
reports the actual prepared buffers.

## Quick Start

From the repository root:

```bash
make MODE=release generate

./build/release/local-gguf-generate \
  models/gemma-2-9b-Q4.gguf \
  "What is the capital of France?" \
  32 \
  --backend metal \
  --greedy
```

The final positional number is the maximum number of new tokens. One-shot
generation supports Gemma 2 and Gemma 3 on Metal with greedy selection.

Startup loads the model and prepares its Metal execution plan before prompt
processing begins.

## Interactive Chat

Build and start a persistent chat session:

```bash
make MODE=release generate

./build/release/local-gguf-generate \
  models/gemma-2-9b-Q4.gguf \
  --chat \
  --backend metal
```

Chat uses the same effective defaults as ordinary `llama-cli`: temperature
`0.8`, top-k `40`, top-p `0.95`, min-p `0.05`, repetition penalty `1.0`, a
64-token repetition window, and a random seed. A turn runs until Gemma emits
end-of-turn/EOS, the user interrupts it, or the context is exhausted.

Use explicit options when reproducible output or a response limit is useful:

```bash
./build/release/local-gguf-generate \
  models/gemma-2-9b-Q4.gguf \
  --chat \
  --max-new-tokens 128 \
  --backend metal \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --min-p 0.05 \
  --repeat-penalty 1.0 \
  --seed 42
```

`--max-new-tokens N` adds an optional per-turn cap. `--context-shift` opts into
dropping the oldest complete turns when the context fills; it is disabled by
default.

Available chat commands:

- `/exit` or `/quit`: leave chat;
- `/clear`: clear conversation state while keeping the loaded plan;
- `/reset`: reset the session;
- `/regen`: regenerate the last assistant response;
- `/tokens`: show cached and remaining context tokens;
- `/read <path>`: use a text file as the next prompt.

End an input line with `\` to continue a prompt on the next line. Successful
turns print llama.cpp-compatible prompt/generation throughput and internal
TTFT, plus the raw token counts and durations behind those rates.

## Gemma 3 12B Chat

```bash
make MODE=release generate
./build/release/local-gguf-generate \
  models/gemma-3-12b-it-Q4_K_M.gguf --chat --context-size 16384
```

The default capacity is 16K; `--context-size 32768` requests 32K. Capacities
must be powers of two, at least the local window, within model metadata and the
device memory budget. Weights retain their GGUF values. The default
`--kv-cache auto` selects native F16 target cache storage for ordinary and
speculative decoding; `--kv-cache f32` selects the previous numerical policy.
Attention arithmetic remains F32. The precision
is fixed for the session, including reset and regeneration; the drafter remains
F32. F16 defines a slightly different numerical target, and lossless speculation
preserves the distribution of that selected target policy.
`--decode-diagnostics` reports capacity, cache precision and resident allocations.
128K is the model's advertised limit, not the default allocation on a 16 GiB Mac.
This is text-only inference; no vision encoder or projector is loaded.

On macOS 15 and later with supported Apple GPUs, the engine requests that model
buffers stay resident for three minutes after inference, helping avoid memory
restoration delays between messages. This is automatic for ordinary and speculative decoding;
after that idle window, residency refreshes stop.

Gemma 3 starts with ordinary target decoding. `--retrieval` explicitly enables
the existing experimental retrieval proposer; `--no-speculative` disables all
speculation. `--draft-model` explicitly enables the supported 270M pairing
described below.

For independent checkpoint checks, generate ignored goldens with
`scripts/gemma3_target_reference.py` using the optional reference environment,
then run the test binary with `LOCAL_GGUF_GEMMA3_TARGET_GOLDENS` pointing at that
directory. For F16, generate a separate reference with `--cache-dtype f16` and
set `LOCAL_GGUF_GEMMA3_TARGET_F16_GOLDENS`; casting F32 reference caches does not
reproduce F16 inference. Production inference does not import Python or external
runtimes.

## Optional Gemma 3 Drafter

For the supported, pinned `gemma-3-270m-it-Q8_0.gguf` drafter with Gemma 3 12B:

```bash
make MODE=release generate
./build/release/local-gguf-generate \
  models/gemma-3-12b-it-Q4_K_M.gguf --chat \
  --draft-model models/gemma-3-270m-it-Q8_0.gguf
```

Supplying the drafter selects the retained setup by default: F16 target KV,
horizon 1, full verification and lossless sampling, with a 16K context capacity.
The fused Q4 sums and packed weights remain enabled. No tuning flags are needed.
See `docs/NEXT_TASK.md` for the measured memory and chat results.

This runs the complete native drafter, not an external inference engine. The
target sampling distribution is preserved, but the same seed can produce
different text from ordinary generation. Missing or incompatible draft files
fail explicitly. No Python environment is needed for inference.

The engine validates all token IDs before using direct history synchronization,
keeps independent model caches, and checks the combined memory requirement
before uploading the target. The drafter supports at most 32K context; actual
paired capacity also depends on the device budget. Full and conditional
verification support prepared horizons from 1 through 23, shortened near output
or context limits. `--draft-horizon` and `--speculative-schedule full|conditional`
remain explicit overrides; larger horizons do not necessarily run faster.
`--decode-diagnostics` reports actual proposals, reached depths, draft work and
the effective sampling policy. The existing Gemma 2 target pairing remains
available through the same explicit options.

Offline Gemma 3 reference checks use optional dependencies listed in
`scripts/reference_requirements.txt`; the production executable does not.

## Inspect A GGUF File

```bash
make MODE=release gguf-inspect
./build/release/gguf-inspect models/gemma-2-9b-Q4.gguf
```

This prints a concise model, metadata, and tensor summary without running
inference.

## Build And Test

The default build is an unoptimized debug build:

```bash
make generate
make gguf-inspect
```

Use release mode for inference and performance measurements:

```bash
make MODE=release generate
make MODE=release gguf-inspect
```

Run the debug test suite and repository checks:

```bash
make test
bash scripts/ci.sh
```

Run optimized tests or remove all build output:

```bash
make release-test
make clean
```

Build products are written to `build/debug/` or `build/release/`.

## Benchmark The Real Generation Path

Build and run the canonical local benchmark:

```bash
make MODE=release benchmark
scripts/quick_benchmark.py --model models/gemma-3-12b-it-Q4_K_M.gguf \
  --context-size 16384 --no-speculative --skip-kernels
```

The headline invokes the real `local-gguf-generate --chat` product with the
fixed two-turn conversation, seed 42, and normal sampled generation. Responses
run until their ordinary stop condition, without a short token cap. Session
preparation includes the documented 16-token prefill and decode/verifier warm-up. The script reports the chat product's rates plus evaluated,
cached, and decoded counts and raw timing durations.

When attached to a terminal, the runner renders both chat streams in the
terminal's alternate screen during measurement. This preserves the real
streaming/compositor load, then restores the screen before printing the compact
result tables. It submits the follow-up only after the first response finishes,
with a fixed five-second input pause; keyboard waiting is not timed, but the
normal between-turn Metal power state is preserved.

The quick screen uses one fresh process and four-minute idle periods before chat
and kernel diagnostics because repeated natural responses can heat this fanless
machine enough to bias results. Local kernel probes run only after the headline
chat in a separate process. Gemma 3 uses `--skip-kernels` because those standalone
probes describe the Gemma 2 kernel mix; actual chat remains measured. Use `--json` for
machine-readable output; benchmark result files are local artifacts and should
not be committed.

The pinned llama.cpp run is intentionally separate and stored in
`docs/LLAMA_CPP_REFERENCE_BASELINE.md`. Re-run it only under that document's
stated invalidation conditions. Retaining an optimization requires a comparable
candidate/baseline pair with separate cooldowns and the criteria in `AGENTS.md`.

## Current Scope

- Runnable inference: Gemma 2 and Gemma 3 text chat on resident Metal.
- Main target: Gemma 3 12B Q4_K_M with losslessly packed mixed Q4_K/Q6_K projections.
- CPU operators: correctness references and tests, not the product path.
- Runnable tools: `local-gguf-generate` and `gguf-inspect`.
- Current phase: measuring and removing end-to-end decode and prefill
  bottlenecks.

For contributors, start with `AGENTS.md`, then `docs/ENGINE_PRINCIPLES.md` and
`docs/NEXT_TASK.md`. Use `docs/GRAPH.md` only to locate relevant code and tests.

## Common Problems

`Metal backend unavailable`
: Run on a Metal-capable Apple Silicon Mac and avoid environments that block
  access to the system Metal device.

`local-gguf-generate supports Gemma 2 and Gemma 3 generation`
: The selected GGUF uses another architecture. Other model families are not a
  current production runtime target.

The process appears idle after loading the model
: Metal plan creation is currently slow. Allow roughly 12 seconds before
  treating startup as stuck.

The process exits because of memory pressure
: Reduce `--context-size` and inspect `--decode-diagnostics`. Model weights,
  cache, scratch, tokenizers and macOS share the same physical memory.

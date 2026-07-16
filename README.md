# local-gguf-engine

A from-scratch local GGUF inference engine optimized for a MacBook Air M4.
The current production path runs Gemma 2 9B Q4 with model-resident Metal
weights, KV cache, prefill, decode, and sampling. External inference engines
are used only as references and benchmarks.

## Requirements

- An Apple Silicon Mac. The optimized release build currently targets M4.
- macOS with Metal support.
- Apple Clang and the Xcode Command Line Tools.
- A local Gemma 2 GGUF model using the currently supported tensor layouts.

Install the command-line tools if `clang --version` is unavailable:

```bash
xcode-select --install
```

Model files are not included. Put them under `models/`; that directory is
ignored by Git. The commands below assume:

```text
models/gemma-2-9b-Q4.gguf
```

The current plan keeps about `6.84 GB` of weights and reserves about `5.64 GB`
of F32 KV cache, in addition to macOS and application memory. A low-memory Mac
may not be able to run this target model.

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
generation is currently Gemma 2, Metal, and greedy only.

The first run can spend roughly 12 seconds creating the Metal plan before
prompt processing begins. Startup performance is a known active problem.

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
turns print prompt and generation throughput.

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

Always benchmark a release build:

```bash
make MODE=release generate

./build/release/local-gguf-generate \
  models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." \
  8 \
  --backend metal \
  --greedy \
  --bench \
  --warmup 1 \
  --repetitions 5 \
  --output human
```

The benchmark runs cold, warm, cached-turn, and seeded-sampling cases, so it
can take several minutes on the current engine. Keep the machine's thermal
state stable when comparing changes.

For machine-readable output:

```bash
./build/release/local-gguf-generate \
  models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." \
  8 \
  --backend metal \
  --greedy \
  --bench \
  --warmup 1 \
  --repetitions 5 \
  --output jsonl > benchmarks/results/gemma2-local.jsonl
```

Benchmark result files are local artifacts and should not be committed.

## Current Scope

- Runnable inference: Gemma 2 on the model-resident Metal path.
- Main target: Gemma 2 9B Q4 with mixed Q4_K/Q6_K projection weights.
- CPU operators: correctness references and tests, not the product path.
- Runnable tools: `local-gguf-generate` and `gguf-inspect`.
- Current phase: measuring and removing end-to-end decode, prefill, startup,
  and memory bottlenecks.

For contributors, start with `AGENTS.md`, then `docs/ENGINE_PRINCIPLES.md` and
`docs/NEXT_TASK.md`. Use `docs/GRAPH.md` only to locate relevant code and tests.

## Common Problems

`Metal backend unavailable`
: Run on a Metal-capable Apple Silicon Mac and avoid environments that block
  access to the system Metal device.

`local-gguf-generate supports Gemma 2 generation only`
: The selected GGUF uses another architecture. Other model families are not a
  current production runtime target.

The process appears idle after loading the model
: Metal plan creation is currently slow. Allow roughly 12 seconds before
  treating startup as stuck.

The process exits because of memory pressure
: The current Gemma plan reserves more than 12 GB for resident weights and KV
  before accounting for macOS and temporary memory.

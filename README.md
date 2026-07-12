# local-gguf-engine

A local-first GGUF inference engine optimized for Apple Silicon M4.

The aim of this project is to build a from-scratch GGUF inference engine that
is tuned for one machine first: a MacBook Air M4. The performance target is to
beat `llama.cpp` on this computer for selected local models and quantization
formats.

The engine owns the inference path directly. External engines may be used for
learning, compatibility checks, and benchmarks, but not as the production
runtime.

## Goals

- Runtime must work locally without network access.
- GGUF model files should remain local and are not committed to the repository.
- Correctness must be tested, but the design should favor measured performance
  on the target MacBook Air M4.
- Hot paths should move toward C++ and Metal rather than broad portability.
- Tests should be added alongside new behavior.
- The engine should be built iteratively, starting from small verifiable pieces.

## Development

This project uses C++20 with Apple Clang and a small Make-based build scaffold.

Run local checks with:

```bash
bash scripts/ci.sh
```

Build the GGUF inspection CLI with:

```bash
make gguf-inspect
./build/debug/gguf-inspect models/gemma-2-9b-Q4.gguf
```

Build the generate CLI with:

```bash
make generate
./build/debug/local-gguf-generate models/gemma-2-9b-Q4.gguf "Hello" 8 --greedy
```

## Run A Model

Put local GGUF files under `models/`. Model files are ignored by Git.

For the current Gemma 2 Q4 target, build the release generator:

```bash
make MODE=release generate
```

Run deterministic greedy/argmax generation on the Metal path:

```bash
./build/release/local-gguf-generate models/gemma-2-9b-Q4.gguf "What is the capital of France?" 32 --greedy
```

Gemma generation is greedy by default and Metal-only. Persistent `--chat`
supports seeded sampling controls and prints prompt/generation throughput after
every successful response.

Benchmark the real generation path:

```bash
./build/release/local-gguf-generate models/gemma-2-9b-Q4.gguf \
  "Explain why the sky is blue." 128 --backend metal --greedy \
  --bench --warmup 1 --repetitions 5 --output human
```

Use `--output jsonl` and shell redirection for machine-readable results. The
standalone `local-gguf-bench` scaffold remains retired.

## Status

This project is in the early implementation stage. The GGUF reader, tokenizer,
Tensor Core, CPU reference primitives, and Metal backend primitives are in
place for the current Gemma 2 Q4 direction. The generator now exposes the
Gemma 2 greedy Metal path only; CPU/Llama generation, sampled CLI generation,
and the old standalone benchmark runner have been removed to keep the
architecture pointed at the model-resident Metal runtime.

The current runnable surfaces are `gguf-inspect` for concise GGUF summaries and
`local-gguf-generate` for Gemma 2 generation, persistent chat, and real-path
benchmarking.

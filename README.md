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
./build/debug/gguf-inspect models/SmolLM2-135M-Instruct-F16.gguf
```

Build the generate CLI with:

```bash
make generate
./build/debug/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "Hello" 8
```

## Run A Model

Put local GGUF files under `models/`. Model files are ignored by Git.

For the current SmolLM2 F16 target, build the release generator:

```bash
make MODE=release generate
```

Run with the CLI's default sampled settings:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "What is the capital of France?" 64
```

Run deterministic greedy/argmax generation:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "What is the capital of France?" 32 --greedy
```

Or use zero temperature for the same argmax path:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "What is the capital of France?" 32 --temperature 0
```

CPU is the default generation backend. To try the explicit Metal/GPU path on
Apple Silicon, pass `--backend metal`:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "What is the capital of France?" 32 --backend metal --greedy
```

Metal generation is currently an opt-in inspection path, not a speed guarantee.
Use `--backend cpu` to request the default CPU path explicitly.

Sampling controls are engine-owned and deterministic when `--seed` is set:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "Write a haiku about rain." 64 --temperature 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.1 --seed 1
```

To inspect prompt formatting, token IDs, and first-token logits:

```bash
./build/release/local-gguf-generate models/SmolLM2-135M-Instruct-F16.gguf "Hello" 1 --debug-first-token
```

Build the benchmark scaffold with:

```bash
make MODE=release benchmark
./build/release/local-gguf-bench --iterations 1 --output benchmarks/results/smoke.txt
```

Benchmark reports are written under `benchmarks/results`, which is ignored by
Git.

## Status

This project is in the early implementation stage. The GGUF reader and initial
Tensor Core are in place for F32, F16, and Q8_0 payloads. The engine can now
load a narrow Llama-family GGUF model structure, tokenize prompts from GGUF
metadata, bind F16 model weights, and run cached CPU generation for the first
SmolLM2 inference path, with opt-in Metal generation routing available through
`--backend metal`, including seeded sampling controls, a greedy/argmax mode,
tokenizer-EOS stopping, and narrow SmolLM2 chat-template prompt formatting.
The current runnable surfaces are
`gguf-inspect` for concise local GGUF summaries, `local-gguf-generate` for
minimal text generation, and `local-gguf-bench` for engine-native
tensor/load microbenchmarks.

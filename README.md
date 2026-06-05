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

Build the benchmark scaffold with:

```bash
make MODE=release benchmark
./build/release/local-gguf-bench --iterations 1 --output benchmarks/results/smoke.txt
```

Benchmark reports are written under `benchmarks/results`, which is ignored by
Git.

## Status

This project is in the early implementation stage. The GGUF reader and initial
Tensor Core are in place for F32, F16, and Q8_0 payloads. Real inference is not
implemented yet. The current runnable surfaces are `gguf-inspect` for concise
local GGUF summaries and `local-gguf-bench` for engine-native tensor/load
microbenchmarks.

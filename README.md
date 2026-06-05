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

## Status

This project is in the early implementation stage. Milestone 1, the GGUF
reader, can parse tiny synthetic GGUF fixtures through bounded raw tensor
payload views. The next work is Milestone 2: the internal tensor core.

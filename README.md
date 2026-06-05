# local-gguf-engine

A local-first GGUF inference engine for Apple Silicon.

The aim of this project is to explore and build the core pieces needed to run
GGUF language models locally on a Mac. It is focused on understanding model
loading, tensor execution, token generation, sampling, and hardware-aware
inference from the ground up.

The project starts with a simple CPU implementation and can later grow toward
Apple Silicon acceleration.

## Goals

- Runtime must work locally without network access.
- GGUF model files should remain local and are not committed to the repository.
- The first versions should favor correctness and clarity over raw speed.
- Tests should be added alongside new behavior.
- The engine should be built iteratively, starting from small verifiable pieces.

## Development

This project uses Swift and Swift Package Manager.

Run local checks with:

```bash
bash scripts/ci.sh
```

## Status

This project is in the early implementation stage. The current scaffold is a
minimal Swift library with a CPU-first test baseline.

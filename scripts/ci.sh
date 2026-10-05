#!/usr/bin/env bash
set -euo pipefail

echo "Running repository checks..."

python3 -m unittest discover -s tests/scripts -p '*_tests.py'

if [[ ! -s "README.md" ]]; then
  echo "Missing or empty README.md" >&2
  exit 1
fi

if git ls-files | grep -E '\.(gguf|ggml|safetensors|pt|pth|onnx|bin|ckpt|npy|npz|parquet|arrow|lgweights|lgspec|lgspecbundle|lgtrace)$' >/dev/null; then
  echo "Model, trace, artifact, or training-output files must not be committed." >&2
  exit 1
fi

make MODE=debug test

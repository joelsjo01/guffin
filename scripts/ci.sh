#!/usr/bin/env bash
set -euo pipefail

echo "Running repository checks..."

python3 -m unittest discover -s tests/scripts -p '*_tests.py'

if [[ ! -s "README.md" ]]; then
  echo "Missing or empty README.md" >&2
  exit 1
fi

if git ls-files | grep -E '\.(gguf|ggml|safetensors|pt|pth|onnx|bin)$' >/dev/null; then
  echo "Model or weight files must not be committed." >&2
  exit 1
fi

if [[ -f Makefile ]]; then
  make test
elif [[ -f Cargo.toml ]]; then
  cargo test
elif [[ -f CMakeLists.txt ]]; then
  cmake -S . -B build
  cmake --build build
  ctest --test-dir build --output-on-failure
elif [[ -f package.json ]]; then
  npm test
else
  echo "No engine toolchain detected yet. Repo hygiene checks passed."
fi

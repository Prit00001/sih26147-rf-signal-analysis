#!/usr/bin/env bash
# Builds the sigscope._native pybind11 extension and installs it into the
# package directory. Covers: FR-17, SR-05 (hardened compile flags).
set -euo pipefail
cd "$(dirname "$0")/.."

# Explicit .venv python, not bare `python` -- MEASURED real bug (found via
# CI, not hypothetical): a bare `python` on PATH can resolve to the system
# interpreter instead of this project's venv, and CMake/pybind11 would then
# build the extension against the WRONG Python ABI, producing a .so the
# venv's own pytest cannot import.
PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
PYBIND11_DIR=$("$PY" -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake -S native -B native/build -G Ninja \
  -DCMAKE_BUILD_TYPE="${SIGSCOPE_BUILD_TYPE:-Release}" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPYTHON_EXECUTABLE="$PY" \
  -DSIGSCOPE_SANITIZE="${SIGSCOPE_SANITIZE:-OFF}"
cmake --build native/build -j

built_so=$(find native/build -maxdepth 1 -name "_native*.so" | head -n1)
cp "$built_so" src/sigscope/
echo "installed: src/sigscope/$(basename "$built_so")"

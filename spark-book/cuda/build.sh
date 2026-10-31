#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ $(uname -m) == aarch64 ]] || { echo 'This build targets the ARM64 WSL2 lab'; exit 2; }
command -v nvcc >/dev/null || { echo 'Install a compatible toolkit compiler first'; exit 2; }
mkdir -p cuda/bin runs/cuda
nvcc --version > runs/cuda/nvcc-version.txt
nvcc -std=c++17 -arch=native -O2 -lineinfo cuda/cuda_lab.cu -ldl \
  -o cuda/bin/cuda-lab > runs/cuda/build-perf.txt 2>&1
nvcc -std=c++17 -arch=native -O0 -g -G cuda/cuda_lab.cu -ldl \
  -o cuda/bin/cuda-lab-debug > runs/cuda/build-debug.txt 2>&1
printf '%s\n' 'built=cuda/bin/cuda-lab' 'built=cuda/bin/cuda-lab-debug'

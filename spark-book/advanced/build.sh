#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ $(uname -m) == aarch64 ]] || { echo 'This build targets the ARM64 WSL2 lab'; exit 2; }
command -v nvcc >/dev/null || { echo 'A qualified CUDA toolkit compiler is required'; exit 2; }
mkdir -p advanced/bin runs/advanced
nvcc --version > runs/advanced/compiler.txt
nvcc -std=c++17 -arch=native -O2 -lineinfo advanced/collectives.cu \
  -lcublas -o advanced/bin/collectives > runs/advanced/build-collectives.txt 2>&1
nvcc -std=c++17 -arch=native -O2 -lineinfo advanced/async_lab.cu \
  -o advanced/bin/async-lab > runs/advanced/build-async.txt 2>&1
nvcc -std=c++17 -arch=native -O0 -g -G advanced/instruction.cu \
  -o advanced/bin/instruction-debug > runs/advanced/build-debug.txt 2>&1
nvcc -std=c++17 -arch=native -O2 -lineinfo -Xptxas=-v \
  advanced/instruction.cu -o advanced/bin/instruction-opt \
  > runs/advanced/build-opt.txt 2>&1
printf '%s\n' 'built=collectives,async-lab,instruction-debug,instruction-opt'

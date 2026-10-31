#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ $(uname -m) == aarch64 ]] || { echo 'This build targets the ARM64 WSL2 lab'; exit 2; }
command -v nvcc >/dev/null || { echo 'A qualified CUDA toolkit compiler is required'; exit 2; }
variant=${1:-reference}
flags=()
case "$variant" in
  reference) out=kernels/bin/kernel-lab ;;
  starter) out=kernels/bin/kernel-student; flags=(-DLAB_STARTER) ;;
  *) echo 'Choose reference or starter'; exit 2 ;;
esac
mkdir -p kernels/bin runs/kernels
nvcc --version > "runs/kernels/compiler-$variant.txt"
nvcc -std=c++17 -arch=native -O2 -lineinfo "${flags[@]}" \
  kernels/kernel_lab.cu -o "$out" > "runs/kernels/build-$variant.txt" 2>&1
printf '%s\n' "built=$out"

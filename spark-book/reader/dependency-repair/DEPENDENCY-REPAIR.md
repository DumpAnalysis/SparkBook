# Dependency repair and reproduction

The first installation used the book candidate torch 2.10.0+cu130 on existing CPython
3.14.4, whose official ARM64 wheel exists. Its small CUDA computation passed, but pip
check rejected nvidia-cusparselt-cu13 0.8.0 because the installed WHEEL declared
`py3-none-manylinux2014_sbsa`. The publisher wheel filename declared aarch64. Initial
NumPy absence and the architecture-range warning are retained in commands 039/040.

Official replacement candidates were inspected. Torch 2.11 still pins 0.8.0; later
candidates pin 0.8.1. Torch 2.14.1+cu130 and NumPy 2.5.4 were installed, retaining all
installer reports. The replacement cuSPARSELt 0.8.1 had the same invalid internal tag.
Commands 007 and 055, and the enclosing repair command 052, preserve these failures.

The local correction is **0.8.1+sparkbook.1**. It changes the WHEEL platform tag to
aarch64, marks the binary distribution non-pure, labels the local version, and regenerates
RECORD. The native ELF is AArch64 (e_machine=183), matching the publisher filename;
its SHA-256 and every nonmetadata payload byte were verified unchanged after installation.
This is a disclosed downstream packaging change, not an unmodified upstream wheel.
It does not certify manylinux ABI compliance or qualify sparse CUDA operations.

Reproduction on this project: source exercise-env.sh. requirements-exercises.lock contains
the exact public/local wheel URLs and SHA-256 values; the repaired wheel is retained under
runs/local-wheels/. To reproduce elsewhere, first install the recorded upstream 0.8.1
wheel and use the preserved repair script and metadata patch, then update the lock's local
file URL to that rebuilt artifact and confirm its bytes. The evidence export excludes
wheel binaries. The original URL/hash, library hash, source, patches and reconstruction
record remain available under artifacts/cusparselt-metadata-repair/.

Official references consulted 2026-10-30:
- PyTorch CUDA 13.0 wheel index: https://download.pytorch.org/whl/cu130/torch/
- NVIDIA package index through PyTorch: https://download.pytorch.org/whl/cu130/nvidia-cusparselt-cu13/
- PyPA wheel specification: https://packaging.python.org/en/latest/specifications/binary-distribution-format/
- PyPA version specifiers: https://packaging.python.org/en/latest/specifications/version-specifiers/

Final pip check passed. Final companion GPU probe and full matrix verification passed
without stderr warnings. The loaded framework libraries come from the venv; the CUDA
driver comes from the Windows-mapped /usr/lib/wsl paths. CUDA 13.0 framework libraries
do not replace the installed NVCC 13.4.92 toolkit or Windows driver.

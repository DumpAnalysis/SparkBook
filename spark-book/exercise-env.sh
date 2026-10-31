#!/usr/bin/env bash
# Source this file to use the project exercise environment and existing CUDA tools.
spark_book_env_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$spark_book_env_root/.venv-exercises/bin/activate"
export PATH="/usr/local/cuda-13.4/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
# Matching Python headers extracted locally after the recorded Triton failure.
export CPATH="$spark_book_env_root/runs/development/python-3.14.4-headers/extracted/usr/include/python3.14:$spark_book_env_root/runs/development/python-3.14.4-headers/extracted/usr/include${CPATH:+:$CPATH}"
export TRITON_CACHE_DIR="$spark_book_env_root/runs/development/triton-cache"
export HF_HOME="$spark_book_env_root/models/hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
unset spark_book_env_root

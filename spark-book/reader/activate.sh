#!/usr/bin/env bash
# Source from the lab root. This selects an existing environment and installs nothing.
spark_reader_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$spark_reader_root/.venv-exercises/bin/activate" ]]; then
  echo 'Create .venv-exercises first; see reader/FRESH_SETUP.txt.' >&2
  unset spark_reader_root
  return 2
fi
source "$spark_reader_root/.venv-exercises/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$spark_reader_root/models/hf-cache"
if [[ -d /usr/local/cuda-13.4/bin ]]; then export PATH="/usr/local/cuda-13.4/bin:$PATH"; fi
unset spark_reader_root

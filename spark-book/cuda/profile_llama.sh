#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ ! -e runs/linux-server-state.json ]] || { echo 'Stop the recorded server first'; exit 2; }
command -v nsys >/dev/null || { echo 'nsys is missing'; exit 2; }
mapfile -t bins < <(find linux-bin -type f -name llama-completion)
mapfile -t models < <(find models/q4_k_m -maxdepth 1 -name '*-00001-of-*.gguf')
[[ ${#bins[@]} == 1 && ${#models[@]} == 1 ]] || { echo 'Expected one llama-completion and first shard'; exit 2; }
mapfile -t libdirs < <(find linux-bin -type f -name '*.so*' -printf '%h\n' | sort -u)
libpath=$(IFS=:; echo "${libdirs[*]}")
export LD_LIBRARY_PATH="$libpath${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p runs/cuda
out=${1:-runs/cuda/llama-profile}
[[ ! -e "$out.nsys-rep" ]] || { echo 'Choose a new report prefix'; exit 2; }
exe=$(realpath "${bins[0]}")
chmod u+x "$exe"
"$exe" --help > "$out-help.txt" 2>&1
printf '%s\n' "$exe" > "$out-executable.txt"
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o "$out" \
  "$exe" -m "${models[0]}" -ngl 99 -c 2048 -n 64 \
  --seed 42 --temp 0 --no-conversation \
  -p 'Explain why a successful model load does not prove that every prompt fits.' \
  > "$out-console.txt" 2> "$out-errors.txt"
printf '%s\n' "report=$out.nsys-rep" "console=$out-console.txt"

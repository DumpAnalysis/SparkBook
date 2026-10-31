#!/usr/bin/env bash
# Exercise 13. Capture a bounded invalid-header test against the local binary.
set -euo pipefail
root=${1:?supply the new LAB_RUN directory}
[[ -d "$root" ]] || { echo 'Create LAB_RUN first' >&2; exit 2; }
mapfile -t binaries < <(find linux-bin -type f -name llama-server)
[[ ${#binaries[@]} == 1 ]] || { echo 'Expected one qualified llama-server under linux-bin' >&2; exit 2; }
mapfile -t libdirs < <(find linux-bin -type f -name '*.so*' -printf '%h\n' | sort -u)
libpath=$(IFS=:; echo "${libdirs[*]}")
export LD_LIBRARY_PATH="$libpath${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
out="$root/invalid-model"
mkdir -- "$out"
printf 'This is deliberately not a GGUF file.\n' > "$out/invalid.gguf"
set +e
timeout 20s "${binaries[0]}" -m "$out/invalid.gguf" --host 127.0.0.1 --port 8081 > "$out/stdout.txt" 2> "$out/stderr.txt"
status=$?
set -e
printf '%s\n' "$status" > "$out/exit.txt"
if [[ $status -ne 0 && $status -ne 124 ]] && grep -Eqi 'invalid magic|expected .GGUF.' "$out/stderr.txt"; then
  printf 'Expected format rejection. Read %s/stderr.txt and recheck the baseline.\n' "$out"
else
  printf 'Different outcome, exit %s. Inspect %s before interpreting it.\n' "$status" "$out" >&2
  exit 1
fi

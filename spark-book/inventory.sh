#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p runs
if [[ $# == 0 ]]; then
  # Preserve the original invocation and the collector's default input.
  out=runs/inventory.txt
  exec 3>"$out"
elif [[ $# == 2 && $1 == --out ]]; then
  out=$2
  # Creating a named snapshot must never overwrite an earlier one.
  set -o noclobber
  exec 3>"$out"
else
  echo 'Usage: bash inventory.sh [--out NEW_FILE]' >&2
  exit 2
fi
probe() {
  local status=0
  printf '\n$'; printf ' %q' "$@"; printf '\n'
  "$@" || status=$?
  printf '[exit %d]\n' "$status"
}
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  probe uname -a
  probe cat /etc/os-release
  probe python3 --version
  probe free -h
  probe df -h .
  if [[ -e /dev/dxg ]]; then echo 'dxg=present'; else echo 'dxg=absent'; fi
  smi=$(command -v nvidia-smi || true)
  [[ -n "$smi" ]] || smi=/usr/lib/wsl/lib/nvidia-smi
  if [[ -x "$smi" ]]; then
    probe "$smi"
    probe "$smi" --query-gpu=name,driver_version,memory.total,memory.used --format=csv
  else
    echo 'nvidia-smi unavailable; both GPU queries skipped'
  fi
} >&3 2>&1
exec 3>&-
echo "architecture=$(uname -m)"
if [[ -e /dev/dxg ]]; then echo 'dxg=present'; else echo 'dxg=absent'; fi
printf 'saved=%s\n' "$out"

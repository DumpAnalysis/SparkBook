#!/usr/bin/env bash
# Use the same Linux ARM64 Nsight Systems installation for capture and export.
set -euo pipefail
image=nvcr.io/nvidia/pytorch@sha256:307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e
if [[ ${1:-} == --image ]]; then echo "$image"; exit 0; fi
: "${LAB_RUN:?Source reader/new_run.sh first}"
[[ $(uname -sm) == "Linux aarch64" ]] || { echo 'This pinned build requires Linux aarch64.' >&2; exit 2; }
tools="$LAB_RUN/tools"
dist=NsightSystems-cli-2026.5.1
binary=target-linux-sbsa-armv8/nsys
mkdir -p "$tools"
(
  flock -x 9
  if [[ ! -d "$tools/$dist" ]]; then
    docker image inspect "$image" >/dev/null 2>&1 || docker pull --platform linux/arm64 "$image" >&2
    stage=$(mktemp -d "$tools/.nsight-XXXXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    docker run --rm --pull=never --network none --read-only --platform linux/arm64 \
      --entrypoint tar "$image" -C /usr/local/cuda-13.4 -cf - "$dist" | tar --no-same-owner -xf - -C "$stage"
    "$stage/$dist/$binary" --version | grep -F '2026.5.1.' >&2
    echo "$image" > "$stage/$dist/source-image.txt"
    mv -T "$stage/$dist" "$tools/$dist"
  fi
  [[ $(cat "$tools/$dist/source-image.txt") == "$image" ]] || { echo "Unexpected profiler source in $tools/$dist" >&2; exit 2; }
) 9<"$tools"
exec "$tools/$dist/$binary" "$@"

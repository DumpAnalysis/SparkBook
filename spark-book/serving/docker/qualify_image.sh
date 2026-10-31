#!/usr/bin/env bash
# Target only. Pull a candidate, verify native architecture, record immutable identity.
set -euo pipefail
candidate=${1:?usage: qualify_image.sh CANDIDATE NEW_OUTPUT_DIR}
out=${2:?new output directory required}
mkdir "$out"
exec > >(tee "$out/transcript.txt") 2>&1
set -x
docker context show
docker version
docker pull --platform linux/arm64 "$candidate"
docker image inspect "$candidate" > "$out/image.json"
arch=$(docker image inspect --format '{{.Architecture}}' "$candidate")
test "$arch" = arm64
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$candidate" > "$out/digests.txt"
test -s "$out/digests.txt"
set +x
printf '%s\n' 'ARCHITECTURE PASS; inspect digests.txt and choose the matching repository.'
printf '%s\n' 'This does not qualify the CUDA runtime, kernels, backend, or model.'

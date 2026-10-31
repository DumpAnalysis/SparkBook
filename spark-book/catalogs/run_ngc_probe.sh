#!/usr/bin/env bash
# WSL2 target only. Invoke through catalogs/capture.py to retain all output.
set -euo pipefail
record=${1:?usage: run_ngc_probe.sh IMAGE_JSON}
image=$(python3 - "$record" <<'PY'
import json,re,sys
r=json.load(open(sys.argv[1]));image=r['pinned_image']
if r.get('architecture')!='arm64' or r.get('os')!='linux' or not re.fullmatch(r'nvcr\.io/nvidia/pytorch@sha256:[0-9a-f]{64}',image):raise SystemExit('invalid pinned ARM64 PyTorch record')
print(image)
PY
)
[[ $(uname -m) == aarch64 ]] || { echo 'Requires native ARM64 WSL2'; exit 2; }
docker pull --platform linux/arm64 "$image"
test "$(docker image inspect --format '{{.Architecture}}' "$image")" = arm64
docker run --rm --gpus all --platform linux/arm64 \
  --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 512 \
  --memory 12g --cpus 4 --shm-size 1g \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD/catalogs:/lab:ro" \
  --entrypoint python "$image" /lab/gpu_probe.py

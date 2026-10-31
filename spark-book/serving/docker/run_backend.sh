#!/usr/bin/env bash
# One backend at a time. Requires a previously resolved ARM64 image digest.
set -euo pipefail
backend=${1:?usage: run_backend.sh llama|vllm|triton IMAGE@sha256:DIGEST MODEL_DIR [FIRST_SHARD]}
image=${2:?image required}
models=$(realpath "${3:?model directory required}")
[[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'Digest required' >&2; exit 2; }
test -d "$models"
pids=256
[[ "$backend" = vllm || "$backend" = triton ]] && pids=1024
tmpfs=rw,size=1g,mode=1777
if [[ "$backend" = vllm ]]; then tmpfs=rw,exec,nosuid,nodev,size=1g,mode=1777; fi
common=(--detach --name "spark-$backend" --platform linux/arm64
  --user 10001:10001 --read-only --cap-drop ALL
  --security-opt no-new-privileges --pids-limit "$pids" --cpus 4
  --memory 20g --stop-timeout 30 --tmpfs "/tmp:$tmpfs"
  --mount "type=bind,src=$models,dst=/models,readonly")
case "$backend" in
llama)
  shard=${4:?first split shard basename required}
  [[ "$shard" != */* && -f "$models/$shard" ]] || {
    echo "first shard must be a basename present in $models" >&2; exit 2;
  }
  docker run "${common[@]}" --gpus all -p 127.0.0.1:8080:8080 "$image" \
    -m "/models/$shard" --alias local-model --host 0.0.0.0 --port 8080 \
    -c 4096 -np 1 -ngl 99 --fit off --no-context-shift --metrics -lv 5 ;;
vllm)
  bootstrap=$(realpath "$(dirname "$0")/vllm_wsl_entrypoint.py")
  test -f "$bootstrap"
  docker run "${common[@]}" --gpus all --shm-size 1g \
    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -e USER=sparkbook \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e HF_HUB_OFFLINE=1 \
    -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
    -e TRITON_CACHE_DIR=/tmp/triton \
    -e HF_HOME=/tmp/hf -e VLLM_CACHE_ROOT=/tmp/vllm \
    -p 127.0.0.1:8000:8000 \
    --mount "type=bind,src=$bootstrap,dst=/bootstrap.py,readonly" \
    --entrypoint python3 "$image" /bootstrap.py vllm serve /models \
    --served-model-name local-model --host 0.0.0.0 --port 8000 \
    --max-model-len 2048 --max-num-seqs 2 --tensor-parallel-size 1 \
    --kv-cache-memory-bytes 1073741824 --enforce-eager --dtype auto --gpu-memory-utilization 0.12 ;;
triton)
  gpu=(); [[ "${TRITON_GPU:-0}" = 1 ]] && gpu=(--gpus all)
  docker run "${common[@]}" "${gpu[@]}" --shm-size 1g \
    -p 127.0.0.1:8000:8000 -p 127.0.0.1:8002:8002 \
    --entrypoint tritonserver "$image" --model-repository=/models \
    --model-control-mode=none --strict-readiness=true ;;
*) echo 'Unknown backend' >&2; exit 2 ;;
esac

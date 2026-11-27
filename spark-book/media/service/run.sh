#!/usr/bin/env bash
# Build or select the image before running this one-owner check.
set -euo pipefail
image=${1:?usage: bash media/service/run.sh IMAGE LOCAL_MODEL NEW_OUTPUT}
model=$(realpath "${2:?local MMS model directory required}")
out=${3:?new output directory required}
here=$(cd -- "$(dirname -- "$0")" && pwd)
test -f "$model/config.json"
python3 -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",8022)); s.close()'
mkdir -- "$out"
out=$(realpath "$out")
docker image inspect --format '{{.Id}}' "$image" > "$out/image-id.txt"
cid=$(docker run -d --init --network host --platform linux/arm64 --gpus all \
    --user "$(id -u):$(id -g)" --read-only --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,size=1g -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton \
    --mount "type=bind,src=$model,dst=/model,readonly" --mount "type=bind,src=$out,dst=/output" "$image")
cleanup() {
    docker stop --time 30 "$cid" >/dev/null
    docker logs "$cid" > "$out/service.log" 2>&1
    docker inspect --format '{{.State.ExitCode}}' "$cid" > "$out/exit.txt"
    docker rm "$cid" >/dev/null
}
trap cleanup EXIT
python3 "$here/check.py" --out "$out/requests.json"
cleanup
trap - EXIT
test "$(cat "$out/exit.txt")" = 0
grep -q 'server_drained' "$out/service.log"

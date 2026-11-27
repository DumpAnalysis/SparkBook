#!/usr/bin/env bash
# Probe the same read-only bind mount and numeric UID as the serving pod.
set -euo pipefail
image=${1:?usage: bash serving/docker/check_model_access.sh IMAGE@DIGEST MODEL_DIRECTORY}
models=$(realpath "${2:?model directory required}")
[[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'Supply a qualified image digest' >&2; exit 2; }
docker run --rm --network none --platform linux/arm64 --user 10001:10001 --read-only \
    --cap-drop ALL --security-opt no-new-privileges \
    --mount "type=bind,src=$models,dst=/models,readonly" --entrypoint /bin/sh "$image" -ec '
    test -x /models && test -r /models || { echo "UID 10001 cannot traverse /models" >&2; exit 1; }
    for shard in /models/*.gguf; do
        test -f "$shard" && test -r "$shard" || { echo "Missing or unreadable GGUF: $shard" >&2; exit 1; }
        test "$(head -c 4 "$shard")" = GGUF || { echo "Invalid GGUF signature: $shard" >&2; exit 1; }
        printf "Readable: %s\n" "$shard"
    done'

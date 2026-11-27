#!/usr/bin/env bash
# Run the same four held-out questions and eight serial requests for each native owner.
set -euo pipefail
: "${LAB_RUN:?Source reader/new_run.sh first}"
if [[ -f runs/linux-server-state.json ]]; then bash serve.sh stop; fi
for quant in q4_k_m q8_0; do
  (
    bash serve.sh start 4096 1 99 gpu "$quant"
    trap 'bash serve.sh stop' EXIT
    owner=$(python -c 'import json; print(json.load(open("runs/linux-server-state.json"))["directory"])')
    cp "$owner/command.sh" "$LAB_RUN/$quant-command.sh"
    grep -E 'offloaded [0-9]+/[0-9]+ layers|model buffer size|KV buffer size|compute buffer size' "$owner/stderr.txt" > "$LAB_RUN/$quant-placement.txt"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$LAB_RUN/$quant-memory.csv"
    python reader/heldout_answers.py --server http://127.0.0.1:8080 --out "$LAB_RUN/heldout-$quant"
    python serving/client.py chat-load --base http://127.0.0.1:8080 --model local-model --requests 8 --concurrency 1 --out "$LAB_RUN/$quant-load"
  )
done

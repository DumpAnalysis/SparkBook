#!/usr/bin/env bash
# Four fresh owners keep the model, prompt treatment and timing logs identifiable.
set -euo pipefail
: "${LAB_RUN:?Source reader/new_run.sh first}"
if [[ -f runs/linux-server-state.json ]]; then bash serve.sh stop; fi
for quant in q4_k_m q8_0; do
  for protocol in direct review; do
    (
      suffix=""
      if [[ "$protocol" == direct ]]; then suffix="-direct"; fi
      bash serve.sh start 4096 1 99 gpu "$quant"
      trap 'bash serve.sh stop' EXIT
      owner=$(python -c 'import json; print(json.load(open("runs/linux-server-state.json"))["directory"])')
      cp "$owner/command.sh" "$LAB_RUN/$quant$suffix-command.sh"
      grep -E 'offloaded [0-9]+/[0-9]+ layers|model buffer size|KV buffer size|compute buffer size' "$owner/stderr.txt" > "$LAB_RUN/$quant$suffix-placement.txt"
      nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$LAB_RUN/$quant$suffix-memory.csv"
      python reader/heldout_answers.py --server http://127.0.0.1:8080 --protocol "$protocol" --out "$LAB_RUN/heldout-$quant$suffix"
      python serving/client.py chat-load --base http://127.0.0.1:8080 --model local-model --requests 8 --concurrency 1 --out "$LAB_RUN/$quant$suffix-load"
    )
  done
done

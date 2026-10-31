#!/usr/bin/env bash
# Run from spark-book after qualifying the WSL2 torch build.
set -euo pipefail
mkdir -p embeddings/runs/gpu
python gpu_probe.py > embeddings/runs/gpu/probe.txt 2>&1
python embeddings/encode.py --model embeddings/models/minilm \
  --device cuda --batch 8 --out embeddings/runs/gpu/base \
  > embeddings/runs/gpu/encode.txt 2>&1
python embeddings/finetune.py --model embeddings/models/minilm \
  --device cuda --batch 4 --epochs 3 \
  --out embeddings/models/minilm-gpu-tuned \
  > embeddings/runs/gpu/finetune.txt 2>&1
python embeddings/encode.py --model embeddings/models/minilm-gpu-tuned \
  --device cuda --out embeddings/runs/gpu/tuned \
  > embeddings/runs/gpu/after.txt 2>&1
python embeddings/project.py --input embeddings/runs/gpu/tuned \
  --out embeddings/runs/gpu/maps > embeddings/runs/gpu/maps.txt 2>&1
printf '%s\n' 'Completed. Inspect embeddings/runs/gpu logs and JSON reports.'

#!/usr/bin/env bash
# Run from the lab root in the prepared SDXL training environment.
set -euo pipefail
model=$(realpath "${1:?usage: bash media/train_image_adapter.sh MODEL DATA NEW_OUTPUT [UPSTREAM_SCRIPT]}")
data=$(realpath "${2:?training data directory required}")
output=${3:?new output directory required}
media_dir=$(cd -- "$(dirname -- "$0")" && pwd)
test -f "$model/model_index.json" && test -f "$data/metadata.jsonl"
mkdir -- "$output"
output=$(realpath "$output")
upstream="$output/upstream.py"
if [[ -n "${4:-${SDXL_TRAINER:-}}" ]]; then
    cp -- "${4:-$SDXL_TRAINER}" "$upstream"
else
    curl --fail --location --silent --show-error https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/text_to_image/train_text_to_image_lora_sdxl.py --output "$upstream"
fi
printf '%s  %s\n' e733e9cce32fbe1802dc61426d75f3ab12ff6e824e3f2ee8fa258a60650bdfeb "$upstream" | sha256sum --check
patch --batch --fuzz=0 --output "$output/trainer.py" "$upstream" < "$media_dir/sdxl-v0351.patch"
printf '%s  %s\n' 63bb07101d42adf5ebfbcc06bfdf34b0a19d5a51367ed0a6979f06b867045e67 "$output/trainer.py" | sha256sum --check
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
python "$output/trainer.py" --pretrained_model_name_or_path "$model" --variant fp16 \
    --train_data_dir "$data" --caption_column text --resolution 512 --train_batch_size 1 \
    --gradient_accumulation_steps 1 --gradient_checkpointing --mixed_precision fp16 --rank 4 \
    --learning_rate 1e-4 --max_train_steps 72 --checkpointing_steps 72 --seed 1610 \
    --report_to none --output_dir "$output"

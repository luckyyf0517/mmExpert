#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh

python train_llm.py \
    --config config/llm/phi3.yaml \
    --data_root feature/20251105_090705_clip \
    --batch_size 8 \
    --num_workers 4 \
    --max_epochs 10 \
    --gradient_accumulation_steps 1 \
    --world_size 1 \
    --zero_stage 2 \
    --dtype bf16
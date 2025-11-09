#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh data_root

DATA_ROOT=$1

if [ -z "$DATA_ROOT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: data_root is required"
    echo "Usage: bash scripts/train_llm.sh data_root"
    exit 1
fi

deepspeed --include localhost:0,1 --master_port 1234 \
    train_llm.py \
    --config config/llm/phi3.yaml \
    --data_root ${DATA_ROOT} \
    --batch_size 12 \
    --num_workers 4 \
    --max_epochs 10 \
    --gradient_accumulation_steps 1 \
    --zero_stage 2 \
    --dtype bf16 \
    --split train
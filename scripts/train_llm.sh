#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh config data_root

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/train_llm.sh config data_root"
    echo "Example: bash scripts/train_llm.sh config.yaml /data/root"
    exit 1
fi

CONFIG=$1
DATA_ROOT=$2

deepspeed --include localhost:0,1 --master_port 1234 \
    train_llm.py \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 6 \
    --num_workers 4 \
    --max_epochs 3 \
    --gradient_accumulation_steps 2 \
    --zero_stage 2 \
    --dtype bf16 \
    --train_split "dataset/HumanML3D/_split/train.json" \
    --test_split "dataset/HumanML3D/_split/test.json" \
    --use_random_question_for_caption false

#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh config data_root

CONFIG=$1
DATA_ROOT=$2

deepspeed --include localhost:0,1 --master_port 1234 \
    train_llm.py \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 6 \
    --num_workers 4 \
    --max_epochs 2 \
    --gradient_accumulation_steps 2 \
    --zero_stage 2 \
    --dtype bf16 \
    --split train \
    --use_random_question false

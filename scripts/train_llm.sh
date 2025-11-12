#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh config data_root split_files...
# split_files: One or more split file paths (required)

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/train_llm.sh config data_root split_files..."
    echo "Example: bash scripts/train_llm.sh config.yaml /data/root /path/to/train.json"
    echo "Example: bash scripts/train_llm.sh config.yaml /data/root /path/to/train1.json /path/to/train2.json"
    exit 1
fi

CONFIG=$1
DATA_ROOT=$2
shift 2  # Remove first two arguments

# Join all remaining arguments with commas for the split parameter
SPLIT_FILES=$(IFS=,; echo "$*")

deepspeed --include localhost:0,1 --master_port 1234 \
    train_llm.py \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 6 \
    --num_workers 4 \
    --max_epochs 1 \
    --gradient_accumulation_steps 2 \
    --zero_stage 2 \
    --dtype bf16 \
    --split "${SPLIT_FILES}" \
    --use_random_question_for_caption true

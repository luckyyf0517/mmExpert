#!/bin/bash

# Evaluation script for fine-tuned Phi3 with mmwave features
# Usage: bash scripts/eval_llm.sh config data_root model_checkpoint split_files...
# split_files: One or more split file paths (required)

if [ $# -lt 4 ]; then
    echo "Usage: bash scripts/eval_llm.sh config data_root model_checkpoint split_files..."
    echo "Example: bash scripts/eval_llm.sh config.yaml /data/root checkpoint.pth /path/to/test.json"
    echo "Example: bash scripts/eval_llm.sh config.yaml /data/root checkpoint.pth /path/to/test1.json /path/to/test2.json"
    exit 1
fi

CONFIG=$1
DATA_ROOT=$2
MODEL_CHECKPOINT=$3
shift 3  # Remove first three arguments

# Join all remaining arguments with commas for the split parameter
SPLIT_FILES=$(IFS=,; echo "$*")

deepspeed --include localhost:0,1 --master_port 1234 \
    evaluate_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 4 \
    --split "${SPLIT_FILES}" \
    --use_random_question_for_caption true
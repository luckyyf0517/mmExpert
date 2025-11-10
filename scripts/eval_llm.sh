#!/bin/bash

# Evaluation script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/eval_llm.sh path/to/checkpoint.ckpt data_root

DATA_ROOT=$1
MODEL_CHECKPOINT=$2

if [ -z "$MODEL_CHECKPOINT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: Model checkpoint path is required"
    echo "Usage: bash scripts/eval_llm.sh path/to/checkpoint.ckpt data_root"
    exit 1
fi

if [ -z "$DATA_ROOT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: data_root is required"
    echo "Usage: bash scripts/eval_llm.sh path/to/checkpoint.ckpt data_root"
    exit 1
fi

python evaluate_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config config/llm/phi4.yaml \
    --data_root ${DATA_ROOT} \
    --split test \
    --batch_size 4 

#!/bin/bash

# Debug script to test checkpoint on training dataset
# This checks if the model has properly fitted the training data
# Mimics training flow with bf16-mixed precision
# Usage: bash scripts/debug_llm.sh config data_root model_checkpoint [max_steps]

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/debug_llm.sh config data_root model_checkpoint [max_steps]"
    echo "Example: bash scripts/debug_llm.sh config.yaml /data/root checkpoint.pth 10"
    exit 1
fi

CONFIG=$1
DATA_ROOT=$2
MODEL_CHECKPOINT=$3
MAX_STEPS=${4:-10}
BATCH_SIZE=1

echo -e "\033[32m[INFO]\033[0m Testing checkpoint on TRAINING dataset"
echo -e "\033[32m[INFO]\033[0m Config: ${CONFIG}"
echo -e "\033[32m[INFO]\033[0m Checkpoint: ${MODEL_CHECKPOINT}"
echo -e "\033[32m[INFO]\033[0m Data root: ${DATA_ROOT}"
echo -e "\033[32m[INFO]\033[0m Max steps: ${MAX_STEPS}"
echo -e "\033[32m[INFO]\033[0m Batch size: ${BATCH_SIZE}"
echo ""

# Run debug script (mimics training flow with bf16-mixed)
python debug_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --max_batches ${MAX_STEPS} \
    --split dataset/HumanML3D/_split/test.json \
    --batch_size ${BATCH_SIZE} 

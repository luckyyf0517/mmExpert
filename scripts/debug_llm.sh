#!/bin/bash

# Debug script to test checkpoint on training dataset
# This checks if the model has properly fitted the training data
# Mimics training flow with bf16-mixed precision
# Usage: bash tmp/debug.sh DATA_ROOT MODEL_CHECKPOINT [MAX_STEPS]

DATA_ROOT=$1
MODEL_CHECKPOINT=$2
MAX_STEPS=${3:-10}
BATCH_SIZE=1


echo -e "\033[32m[INFO]\033[0m Testing checkpoint on TRAINING dataset"
echo -e "\033[32m[INFO]\033[0m Checkpoint: ${MODEL_CHECKPOINT}"
echo -e "\033[32m[INFO]\033[0m Data root: ${DATA_ROOT}"
echo -e "\033[32m[INFO]\033[0m Max steps: ${MAX_STEPS}"
echo -e "\033[32m[INFO]\033[0m Batch size: ${BATCH_SIZE}"
echo ""

# Run debug script (mimics training flow with bf16-mixed)
python debug_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config config/llm/phi3.yaml \
    --data_root ${DATA_ROOT} \
    --max_batches ${MAX_STEPS} \
    --split train \
    --batch_size ${BATCH_SIZE} 

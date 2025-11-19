#!/bin/bash

# Evaluation script for fine-tuned Phi3 with mmwave features
# Usage: bash scripts/eval_llm.sh config data_root model_checkpoint

if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/eval_llm.sh config data_root model_checkpoint"
    echo "Example: bash scripts/eval_llm.sh config.yaml /data/root checkpoint.pth"
    exit 1
fi

CONFIG=$1
DATA_ROOT=$2
MODEL_CHECKPOINT=$3

deepspeed --include localhost:0,1 --master_port 1234 \
    evaluate_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 4 \
    --test_split "dataset/HumanML3D/_split/test_QAs.json" \
    --use_random_question_for_caption true

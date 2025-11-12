#!/bin/bash

CONFIG=$1
DATA_ROOT=$2
MODEL_CHECKPOINT=$3

deepspeed --include localhost:0,1 --master_port 1234 \
    evaluate_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --batch_size 4 \
    --split test \
    --use_random_question true
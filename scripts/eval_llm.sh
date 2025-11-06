#!/bin/bash

# Evaluation script for fine-tuned Phi3 with mmwave features
# Usage: bash scripts/eval_llm.sh --model_checkpoint path/to/checkpoint.ckpt

# Default parameters
MODEL_CHECKPOINT=""
DATA_ROOT="feature/20251105_090705_clip"
OUTPUT_FILE="evaluation_results.json"
SPLIT="test_QAs"
BATCH_SIZE=4
MAX_NEW_TOKENS=50
TEMPERATURE=0.7

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_checkpoint)
            MODEL_CHECKPOINT="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_CHECKPOINT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: --model_checkpoint is required"
    echo "Usage: bash scripts/eval_llm.sh --model_checkpoint path/to/checkpoint.ckpt"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: Model checkpoint not found: $MODEL_CHECKPOINT"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "\033[31m[ERROR]\033[0m Error: Data directory not found: $DATA_ROOT"
    exit 1
fi

# Build evaluation command
CMD="python eval_llm.py \
    --model_checkpoint ${MODEL_CHECKPOINT} \
    --data_root ${DATA_ROOT} \
    --output_file ${OUTPUT_FILE} \
    --split ${SPLIT} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}"

echo -e "\033[32m[INFO]\033[0m Starting evaluation..."
echo "Model: ${MODEL_CHECKPOINT}"
echo "Data: ${DATA_ROOT}"
echo "Output: ${OUTPUT_FILE}"
echo "Command: ${CMD}"
echo ""

# Run evaluation
${CMD}
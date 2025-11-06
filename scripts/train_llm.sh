#!/bin/bash

# Training script for fine-tuning Phi3 with mmwave features
# Usage: bash scripts/train_llm.sh

# Set the path to your configuration file
CONFIG_FILE="config/llm/phi3.yaml"

# Data root path (feature directory)
DATA_ROOT="feature/20251105_090705_clip"

# Basic training parameters
BATCH_SIZE=8
WORLD_SIZE=1
ZERO_STAGE=2
DTYPE="bf16"

# Optional: Enable CPU offloading for memory efficiency (set to 1 to enable)
OFFLOAD=0

# Build the command
CMD="python train_llm.py \
    --config ${CONFIG_FILE} \
    --data_root ${DATA_ROOT} \
    --batch_size ${BATCH_SIZE} \
    --world_size ${WORLD_SIZE} \
    --zero_stage ${ZERO_STAGE} \
    --dtype ${DTYPE}"

# Add offloading if enabled
if [ ${OFFLOAD} -eq 1 ]; then
    CMD="${CMD} --offload"
fi

echo "Starting training with command:"
echo ${CMD}
echo ""

# Run the training
${CMD}
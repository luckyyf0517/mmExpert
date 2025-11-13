#!/bin/bash
# Simple experiment runner
# Usage: ./run_experiments.sh <config_folder_path> [all|specific_configs]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_folder_path> [all|specific_configs]"
    exit 1
fi

CONFIG_DIR=$1
CONFIG_SELECTION=${2:-"all"}

# Get configs to run
if [ "$CONFIG_SELECTION" = "all" ]; then
    CONFIGS=($(ls "$CONFIG_DIR"/*.yaml | xargs -n 1 basename))
else
    # Get all remaining arguments as config names
    shift 1
    CONFIGS=("$@")
fi

# Run experiments
for config in "${CONFIGS[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo "Running: $config"
    
    torchrun --nproc_per_node=2 train_clip.py \
        --model-config "$CONFIG_DIR/$config"
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m[INFO]\033[0m $config completed"
    else
        echo -e "\033[31m[ERROR]\033[0m $config failed"
    fi
done

echo "All experiments completed"
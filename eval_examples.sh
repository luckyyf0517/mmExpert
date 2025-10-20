#!/bin/bash
# CLIP Model Evaluation Examples

# Set paths
MODEL_DIR="/root/autodl-tmp/mmExpert/logs"
CONFIG_DIR="/root/autodl-tmp/mmExpert/config"

echo "=== CLIP Model Evaluation Examples ==="
echo ""

# Example 1: Full evaluation on all-view model
echo "1. Full evaluation (All-view model):"
echo "python evaluate_clip.py \\"
echo "  --model_path ${MODEL_DIR}/clip_all_view.ckpt \\"
echo "  --config_path ${CONFIG_DIR}/clip.yaml \\"
echo "  --output_path results/clip_all_view_eval.json"
echo ""

# Example 2: Full evaluation on doppler-only model
echo "2. Full evaluation (Doppler-only model):"
echo "python evaluate_clip.py \\"
echo "  --model_path ${MODEL_DIR}/clip_doppler_only.ckpt \\"
echo "  --config_path ${CONFIG_DIR}/clip_doppler_only.yaml \\"
echo "  --output_path results/clip_doppler_only_eval.json"
echo ""

# Example 3: Quick evaluation for development
echo "3. Quick evaluation (for development):"
echo "python quick_eval.py ${MODEL_DIR}/clip_latest.ckpt ${CONFIG_DIR}/clip.yaml"
echo ""

# Example 4: Compare two models
echo "4. Compare two models:"
echo "echo '=== All-view Model ===' && python quick_eval.py ${MODEL_DIR}/clip_all_view.ckpt ${CONFIG_DIR}/clip.yaml"
echo "echo '=== Doppler-only Model ===' && python quick_eval.py ${MODEL_DIR}/clip_doppler_only.ckpt ${CONFIG_DIR}/clip_doppler_only.yaml"
echo ""

# Example 5: Batch evaluation
echo "5. Batch evaluation of multiple checkpoints:"
echo "for checkpoint in ${MODEL_DIR}/clip_*.ckpt; do"
echo "  name=\$(basename \$checkpoint .ckpt)"
echo "  echo \"Evaluating \$name...\""
echo "  python quick_eval.py \$checkpoint ${CONFIG_DIR}/clip.yaml > results/\${name}_quick.txt"
echo "done"
echo ""

echo "=== Usage Instructions ==="
echo "1. Make sure your model checkpoint exists"
echo "2. Choose the appropriate config file (clip.yaml or clip_doppler_only.yaml)"
echo "3. For quick testing during development, use quick_eval.py"
echo "4. For comprehensive evaluation, use evaluate_clip.py"
echo "5. Results will be saved in JSON format with detailed metrics"
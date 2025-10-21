# 训练 CLIP 模型
torchrun --standalone --nproc_per_node=2 run_model.py \
    --config config/clip.yaml

# 测试 CLIP 模型
python evaluate_clip.py \
    --model_path log/clip/last.ckpt \
    --config_path config/clip.yaml \
    --output_path results/batch_evaluation_results.json
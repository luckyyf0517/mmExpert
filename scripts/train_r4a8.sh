master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$1")
data_path=$1
dir_path=/root/autodl-tmp/mmExpert
cd $dir_path
output_dir=output/$filename

model_name_or_path=/root/autodl-tmp/mmExpert/huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85

PYTHONPATH=$dir_path:$PYTHONPATH \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port src/trainer/train_llm.py \
    --model_name_or_path $model_name_or_path\
    --model_max_length 2048 \
    --model_debug False\
    --output_dir $output_dir \
    --data_root $data_path\
    --split train\
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing False \
    --run_name $filename \
    --lora True\
    --lora_r 4\
    --lora_alpha 8\
    --lora_dropout 0.05\
    --lora_bias none\
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Phi3DecoderLayer' \
    --cache_dir /root/autodl-tmp/mmExpert/huggingface
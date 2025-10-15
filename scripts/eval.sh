dir_path=/root/autodl-tmp/mmExpert

model_name=/root/autodl-tmp/mmExpert/huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85
data_path=$1
adapter_name=$2

PYTHONPATH=$dir_path:$PYTHONPATH \
python src/trainer/eval_llm.py\
 --model_name $model_name\
 --adapter_name $adapter_name\
 --data_root $data_path
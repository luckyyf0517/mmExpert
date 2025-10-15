#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
# Set HuggingFace to offline mode to avoid network calls during training
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List
from peft import get_peft_model, LoraConfig

import torch
import transformers
from src.trainer.wavellm_trainer import WaveLLMTrainer
import os.path as osp

from src.wavellm import conversation as conversation_lib
from src.wavellm import *
from src.datasets.wavellm_dataset import make_object_wave_data_module

# * logger
from src.llm_utils import build_logger

# * SwanLab integration
from swanlab.integration.transformers import SwanLabCallback
SWANLAB_AVAILABLE = True

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_root: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    split: str = "train"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    model_debug: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    fix_llm: bool = True
    lora: Optional[bool] = field(default=False, metadata={"help": "Enable LoRA in stage2"})
    do_not_add_lora_model=['mm_projection_layers']
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    fix_vae: bool = True

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state(named_params, state_dict, bias):
    to_return = {}
    if bias == "none":
        to_return = {k: state_dict[k] for k in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        maybe_lora_bias = {}
        lora_bias_names = set()

        for k in named_params:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = state_dict[k]

        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError("The specified bias setting is not implemented.")
    
    return to_return

def get_peft_state_non_lora(named_params, state_dict, require_grad_only=True):
    to_return = {k: t for k, t in named_params.items() if "lora_" not in k}
    if require_grad_only:
        to_return = {k: state_dict[k] for k, t in to_return.items() if t.requires_grad}
    return to_return

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def find_all_linear_names(model,multimodal_keywords=[]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info"  # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')
    
    disable_torch_init()

    if training_args.model_debug:
        # * do not load checkpoint, load from config
        logger.info("================= under debug mode ====================")

        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=
            "flash_attention_2",
        )

        model = WaveLLMForCausalLM._from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
    else:
        
        model = WaveLLMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation = "flash_attention_2",
        )
        model.get_model().load_wave_encoder(data_args.data_root)

    model.config.use_cache = False

    if training_args.fix_llm:
        # * This will fix all the parameters
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        model.requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().mm_projection_layers.requires_grad_(True)
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    conversation_lib.default_conversation = conversation_lib.conv_templates["conv_phi3"]

    print(
        "detect using lora, the other parameters to config the model trainable weights will be disabled!"
    )

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model,training_args.do_not_add_lora_model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if training_args.tune_mm_mlp_adapter:
        model.get_model().mm_projection_layers.requires_grad_(True)
        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().mm_projection_layers.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    print('trainable lora weight:')
    
    model.initialize_tokenizer_wave_backbone_config(
            tokenizer=tokenizer,
            device=training_args.device)

    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad
    ]

    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'
                    .format(len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'
                    .format(len(params_no_grad),
                            ', '.join(params_no_grad[:10])))

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):

                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args,
                                **kwargs,
                                use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_wave_data_module(tokenizer=tokenizer,
                                                data_args=data_args)

    # Add SwanLab callback if available
    callbacks = []
    if SWANLAB_AVAILABLE:
        swanlab_callback = SwanLabCallback(
            project="mmExpert-LLM",
            experiment_name=f"train-{training_args.output_dir.split('/')[-1]}",
            config={
                "model_name": model_args.model_name_or_path,
                "data_root": data_args.data_root,
                "split": data_args.split,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "lora_r": training_args.lora_r,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": training_args.lora_dropout,
            }
        )
        callbacks.append(swanlab_callback)
        logger.info("SwanLab callback added for experiment tracking")

    trainer = WaveLLMTrainer(model=model,
                              tokenizer=tokenizer,
                              args=training_args,
                              callbacks=callbacks,
                              **data_module)

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel()} parameters")  # numel() returns the total number of elements in the parameter tensor
        print_trainable_parameters(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if training_args.lora:
        import os
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        named_params = dict(model.named_parameters())
        named_params = {k.replace('_fsdp_wrapped_module.', ''):t for k,t in named_params.items()}
        _state_dict = trainer.accelerator.get_state_dict(trainer.model)
        
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            
            state_dict = get_peft_state(
                named_params,
                _state_dict, 
                training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora(
                named_params,
                _state_dict
            )

            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

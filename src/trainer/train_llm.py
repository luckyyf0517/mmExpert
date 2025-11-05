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
import warnings
import logging

# Set HuggingFace to offline mode to avoid network calls during training
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.training_args")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Resized position embedding.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Loading pretrained weights.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Safe alternative available.*", category=UserWarning)

# Suppress timm logger INFO messages
logging.getLogger("timm").setLevel(logging.WARNING)
logging.getLogger("timm.models").setLevel(logging.WARNING)
logging.getLogger("timm.models._builder").setLevel(logging.WARNING)
logging.getLogger("timm.models._hub").setLevel(logging.WARNING)
logging.getLogger("timm.layers.pos_embed").setLevel(logging.WARNING)

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
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

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

def _convert_model_to_dtype(model, target_dtype):
    """
    Recursively convert all parameters and buffers in the model to target dtype.
    This is critical for FSDP which requires uniform dtype across all parameters.
    """
    converted_params = 0
    converted_buffers = 0
    
    for name, param in model.named_parameters():
        if param.dtype != target_dtype and param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.data = param.data.to(dtype=target_dtype)
            converted_params += 1
    
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype and buffer.dtype in (torch.float32, torch.float16, torch.bfloat16):
            buffer.data = buffer.data.to(dtype=target_dtype)
            converted_buffers += 1
    
    if converted_params > 0 or converted_buffers > 0:
        print(f"Converted {converted_params} parameters and {converted_buffers} buffers to {target_dtype}")
    
    return model


class ModelInitializer:
    """
    Handles model initialization with improved error handling and logging.
    """

    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.training_args = training_args
        self.logger = build_logger("ModelInitializer", "logs/model_init.log")

    def initialize_model(self):
        """Initialize the WaveLLM model with proper error handling."""
        try:
            if self.training_args.model_debug:
                return self._initialize_debug_model()
            else:
                return self._initialize_production_model()
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _initialize_debug_model(self):
        """Initialize model in debug mode."""
        self.logger.debug("Initializing model in debug mode")
        config = transformers.AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.training_args.cache_dir,
            attn_implementation="flash_attention_2",
        )
        model = WaveLLMForCausalLM._from_config(
            config,
            dtype=torch.bfloat16,
        )
        return model

    def _initialize_production_model(self):
        """Initialize model in production mode."""
        self.logger.info(f"Initializing model from {self.model_args.model_name_or_path}")
        model = WaveLLMForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        return model

class TokenizerManager:
    """
    Handles tokenizer initialization and configuration.
    """

    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.training_args = training_args
        self.logger = build_logger("TokenizerManager", "logs/tokenizer.log")

    def initialize_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        try:
            self.logger.debug("Initializing tokenizer")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                model_max_length=self.training_args.model_max_length,
                padding_side="right",
                use_fast=True,
            )

            # Set special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise

class LoRAManager:
    """
    Handles LoRA configuration and application with improved logic.
    """

    def __init__(self, training_args: TrainingArguments):
        self.training_args = training_args
        self.logger = build_logger("LoRAManager", "logs/lora.log")

    def apply_lora(self, model):
        """Apply LoRA to model if enabled."""
        if not self.training_args.lora:
            self.logger.debug("LoRA is disabled")
            return model

        try:
            target_modules = self._find_target_modules(model)
            lora_config = LoraConfig(
                r=self.training_args.lora_r,
                lora_alpha=self.training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.training_args.lora_dropout,
                bias=self.training_args.lora_bias,
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            self.logger.debug(f"LoRA applied with {len(target_modules)} target modules")
            return model
        except Exception as e:
            self.logger.error(f"Failed to apply LoRA: {str(e)}")
            raise

    def _find_target_modules(self, model):
        """Find linear modules for LoRA application."""
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in self.training_args.do_not_add_lora_model):
                continue
            if isinstance(module, cls):
                lora_module_names.add(name)

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

class WaveTokenManager:
    """
    Handles wave-specific token initialization with improved logic.
    """

    def __init__(self, training_args: TrainingArguments):
        self.training_args = training_args
        self.logger = build_logger("WaveTokenManager", "logs/wave_tokens.log")

        # Default wave tokens
        self.wave_patch_token = "<wave_patch>"
        self.wave_start_token = "<wave_bos>"
        self.wave_end_token = "<wave_eos>"

    def initialize_wave_tokens(self, model, tokenizer):
        """Initialize wave-specific tokens in model and tokenizer."""
        try:
            self.logger.info("Initializing wave tokens")

            # Add wave patch token
            self._add_wave_patch_token(model, tokenizer)

            # Add start/end tokens if needed
            if self.training_args.tune_mm_mlp_adapter:
                self._add_wave_start_end_tokens(model, tokenizer)

            # Configure model
            self._configure_model_tokens(model)

            self.logger.debug("Wave tokens initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize wave tokens: {str(e)}")
            raise

    def _add_wave_patch_token(self, model, tokenizer):
        """Add wave patch token."""
        tokenizer.add_tokens([self.wave_patch_token], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        model.config.wave_patch_token = tokenizer.convert_tokens_to_ids([self.wave_patch_token])[0]

    def _add_wave_start_end_tokens(self, model, tokenizer):
        """Add wave start and end tokens."""
        num_new_tokens = tokenizer.add_tokens(
            [self.wave_start_token, self.wave_end_token], special_tokens=True
        )
        model.resize_token_embeddings(len(tokenizer))

        model.config.wave_start_token = tokenizer.convert_tokens_to_ids([self.wave_start_token])[0]
        model.config.wave_end_token = tokenizer.convert_tokens_to_ids([self.wave_end_token])[0]
        model.config.mm_use_wave_start_end = True

        # Initialize new token embeddings
        if num_new_tokens > 0:
            self._initialize_new_token_embeddings(model, num_new_tokens)

    def _initialize_new_token_embeddings(self, model, num_new_tokens):
        """Initialize new token embeddings."""
        device = next(model.parameters()).device

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # Use average of existing embeddings
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # Configure training parameters
        if self.training_args.fix_llm:
            model.get_model().orig_embeds_params = [
                model.get_input_embeddings().weight.data.clone().to(device=device)
            ]
            for p in model.get_output_embeddings().parameters():
                p.requires_grad = False

    def _configure_model_tokens(self, model):
        """Configure model token settings."""
        model.config.default_wave_patch_token = self.wave_patch_token
        if hasattr(model.config, 'mm_use_wave_start_end') and model.config.mm_use_wave_start_end:
            model.config.default_wave_start_token = self.wave_start_token
            model.config.default_wave_end_token = self.wave_end_token

class RadarEncoderManager:
    """
    Handles radar encoder loading with improved error handling.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.logger = build_logger("RadarEncoderManager", "logs/radar_encoder.log")

    def load_radar_encoder(self, model):
        """Load radar encoder into the model."""
        try:
            self.logger.info(f"Loading radar encoder from {self.data_args.data_root}")
            model.get_model().load_wave_encoder(self.data_args.data_root)
            
            # Ensure radar encoder uses the same dtype as the main model
            # This is critical for FSDP which requires uniform dtype across all parameters
            model_dtype = next(model.parameters()).dtype
            
            # Convert radar encoder and all its submodules to match model dtype
            if hasattr(model.get_model(), 'wave_encoder') and model.get_model().wave_encoder is not None:
                # Recursively convert all parameters to match model dtype
                for param in model.get_model().wave_encoder.parameters():
                    param.data = param.data.to(dtype=model_dtype)
                # Also convert buffers (e.g., running_mean, running_var in BatchNorm)
                for buffer in model.get_model().wave_encoder.buffers():
                    if buffer.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        buffer.data = buffer.data.to(dtype=model_dtype)
                self.logger.debug(f"Converted radar encoder to {model_dtype}")
            
            # Also ensure projection layers use the same dtype
            if hasattr(model.get_model(), 'mm_projection_layers'):
                for param in model.get_model().mm_projection_layers.parameters():
                    param.data = param.data.to(dtype=model_dtype)
                self.logger.debug(f"Converted projection layers to {model_dtype}")
            
            self.logger.debug("Radar encoder loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load radar encoder: {str(e)}")
            raise

class ParameterManager:
    """
    Manages parameter freezing strategy.
    """

    def __init__(self, training_args: TrainingArguments):
        self.training_args = training_args
        self.logger = build_logger("ParameterManager", "logs/parameters.log")

    def apply_parameter_strategy(self, model):
        """Apply parameter freezing strategy."""
        try:
            if self.training_args.fix_llm:
                self.logger.info("Applying parameter freezing strategy")
                model.requires_grad_(False)
                model.get_model().fix_llm = True

                if self.training_args.tune_mm_mlp_adapter:
                    model.get_model().mm_projection_layers.requires_grad_(True)
                    self.logger.info("MM projection layers set to trainable")
            else:
                self.logger.warning("LLM parameters are trainable (fix_llm=False)")

        except Exception as e:
            self.logger.error(f"Failed to apply parameter strategy: {str(e)}")
            raise

    def print_trainable_parameters(self, model):
        """Print information about trainable parameters."""
        trainable_params = 0
        all_param = 0
        trainable_layers = {}

        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                layer_name = name.split('.')[0]
                if layer_name not in trainable_layers:
                    trainable_layers[layer_name] = 0
                trainable_layers[layer_name] += param.numel()

        self.logger.info(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}")

        self.logger.info("Trainable parameters by layer:")
        for layer, params in trainable_layers.items():
            self.logger.info(f"  {layer}: {params} parameters")

class TrainingPipeline:
    """
    Main training pipeline that coordinates all components.
    """

    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.logger = build_logger("TrainingPipeline", "logs/training.log")

        # Initialize managers
        self.model_initializer = ModelInitializer(model_args, training_args)
        self.tokenizer_manager = TokenizerManager(model_args, training_args)
        self.lora_manager = LoRAManager(training_args)
        self.wave_token_manager = WaveTokenManager(training_args)
        self.radar_encoder_manager = RadarEncoderManager(data_args)
        self.parameter_manager = ParameterManager(training_args)

    def run_training(self):
        """Execute the complete training pipeline."""
        try:
            self.logger.info("Starting WaveLLM training pipeline")

            # Set default conversation template to phi3 (MPT style)
            conversation_lib.default_conversation = conversation_lib.conv_templates["conv_phi3"]

            # Initialize model
            model = self.model_initializer.initialize_model()
            model.config.use_cache = False

            # Initialize tokenizer
            tokenizer = self.tokenizer_manager.initialize_tokenizer()

            # Apply LoRA
            model = self.lora_manager.apply_lora(model)

            # Initialize wave tokens
            self.wave_token_manager.initialize_wave_tokens(model, tokenizer)

            # Load radar encoder
            self.radar_encoder_manager.load_radar_encoder(model)

            # Apply parameter strategy
            self.parameter_manager.apply_parameter_strategy(model)

            # Ensure all parameters have uniform dtype for FSDP compatibility
            # This must be done after all model modifications (LoRA, tokens, encoder, etc.)
            model_dtype = next(model.parameters()).dtype
            self.logger.debug(f"Converting all model parameters to {model_dtype} for FSDP compatibility")
            _convert_model_to_dtype(model, model_dtype)

            # Print trainable parameters
            self.parameter_manager.print_trainable_parameters(model)

            # Create data module
            data_module = make_object_wave_data_module(tokenizer=tokenizer, data_args=self.data_args)

            # Create trainer
            trainer = self._create_trainer(model, tokenizer, data_module)

            # Run training
            self._run_training(trainer)

            # Save model
            self._save_model(trainer)

            self.logger.info("Training pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def _create_trainer(self, model, tokenizer, data_module):
        """Create the trainer with callbacks."""
        # Add callbacks
        callbacks = []

        # Add SwanLab callback if available
        if SWANLAB_AVAILABLE:
            try:
                swanlab_callback = SwanLabCallback(
                    project="mmExpert-LLM",
                    experiment_name=f"train-{self.training_args.output_dir.split('/')[-1]}",
                    config={
                        "model_name": self.model_args.model_name_or_path,
                        "data_root": self.data_args.data_root,
                        "split": self.data_args.split,
                        "learning_rate": self.training_args.learning_rate,
                        "batch_size": self.training_args.per_device_train_batch_size,
                        "num_epochs": self.training_args.num_train_epochs,
                        "lora_r": self.training_args.lora_r,
                        "lora_alpha": self.training_args.lora_alpha,
                        "lora_dropout": self.training_args.lora_dropout,
                    }
                )
                callbacks.append(swanlab_callback)
                self.logger.debug("SwanLab callback added for experiment tracking")
            except Exception as e:
                self.logger.warning(f"Failed to add SwanLab callback: {str(e)}")

        # Create trainer
        trainer = WaveLLMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            callbacks=callbacks,
            **data_module
        )

        return trainer

    def _run_training(self, trainer):
        """Run the training process."""
        self.logger.info("Starting training...")

        if list(pathlib.Path(self.training_args.output_dir).glob("checkpoint-*")):
            self.logger.info("Resuming from checkpoint")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

    def _save_model(self, trainer):
        """Save the trained model."""
        self.logger.info("Saving model...")

        if self.training_args.lora:
            self._save_lora_model(trainer)
        else:
            self._save_full_model(trainer)

    def _save_lora_model(self, trainer):
        """Save LoRA model weights."""
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        if self.training_args.local_rank == 0 or self.training_args.local_rank == -1:
            named_params = dict(trainer.model.named_parameters())
            named_params = {k.replace('_fsdp_wrapped_module.', ''):t for k,t in named_params.items()}
            _state_dict = trainer.accelerator.get_state_dict(trainer.model)

            state_dict = self._get_peft_state(named_params, _state_dict, self.training_args.lora_bias)
            non_lora_state_dict = self._get_peft_state_non_lora(named_params, _state_dict)

            trainer.model.config.save_pretrained(self.training_args.output_dir)
            trainer.model.save_pretrained(self.training_args.output_dir, state_dict=state_dict)

            import torch
            torch.save(non_lora_state_dict, os.path.join(self.training_args.output_dir, 'non_lora_trainables.bin'))

    def _save_full_model(self, trainer):
        """Save full model."""
        from src.trainer.train_llm import safe_save_model_for_hf_trainer
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=self.training_args.output_dir)

    def _get_peft_state(self, named_params, state_dict, bias):
        """Get PEFT state dict."""
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

    def _get_peft_state_non_lora(self, named_params, state_dict, require_grad_only=True):
        """Get non-LoRA state dict."""
        to_return = {k: t for k, t in named_params.items() if "lora_" not in k}
        if require_grad_only:
            to_return = {k: state_dict[k] for k, t in to_return.items() if t.requires_grad}
        return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
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
        trainer._save(output_dir, state_dict=cpu_state_dict)

def find_all_linear_names(model, multimodal_keywords=[]):
    """Find all linear layer names in the model."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train():
    """Main training function - maintains original interface."""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    training_args.log_level = "info"
    logger = build_logger(__name__, training_args.output_dir + '/train.log')

    disable_torch_init()

    # Create and run training pipeline
    pipeline = TrainingPipeline(model_args, data_args, training_args)
    pipeline.run_training()

if __name__ == "__main__":
    train()
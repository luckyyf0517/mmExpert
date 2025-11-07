"""Common utility functions for training and evaluation"""

import os
import json
import warnings
from termcolor import colored
from easydict import EasyDict


def is_rank_0():
    """Check if current process is rank 0"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    # Fallback to environment variable
    rank = os.environ.get('RANK', None)
    if rank is not None:
        return int(rank) == 0
    # Default to True if not in distributed mode
    return True


def log_info(message):
    """Print info message only on rank 0"""
    if is_rank_0():
        print(colored("[INFO]", "green") + f" {message}")


def override_config(cfg_obj, attr_name, arg_value, arg_name):
    """Override config attribute with command line argument if provided"""
    if arg_value is not None:
        setattr(cfg_obj, attr_name, arg_value)
        log_info(f"Using {arg_name} from command line: {arg_value}")


def load_non_lora_trainables(model, checkpoint_path, require_grad_metadata=True):
    """Load non-LoRA trainable parameters (reference code approach)

    Args:
        model: WaveLLMTrainer model instance
        checkpoint_path: Path to checkpoint directory containing non_lora_trainables.safetensors

    Returns:
        None (raises exception if loading fails)
    """
    from safetensors.torch import load_file

    # Load non-LoRA trainable parameters
    non_lora_path = os.path.join(checkpoint_path, "non_lora_trainables.safetensors")
    if not os.path.exists(non_lora_path):
        raise FileNotFoundError(f"Non-LoRA parameters file not found: {non_lora_path}")

    log_info(f"Loading non-LoRA trainable parameters from {non_lora_path}")
    non_lora_state = load_file(non_lora_path)

    # Enumerate trainable parameters (when available) to derive expected keys.
    # During evaluation (without PEFT wrapping), requires_grad might be False,
    # so fall back to trusting the checkpoint keys.
    model_trainable_non_lora = {}
    if require_grad_metadata:
        model_trainable_non_lora = {
            name: param
            for name, param in model.model.named_parameters()
            if "lora_" not in name and param.requires_grad
        }

    actual_keys = set(non_lora_state.keys())
    expected_keys = set(model_trainable_non_lora.keys()) if model_trainable_non_lora else actual_keys

    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise ValueError(f"Missing expected parameters in checkpoint: {missing_keys}")

    # Ensure all keys from checkpoint exist in model state dict
    model_state_dict = model.model.state_dict()
    invalid_keys = {key for key in actual_keys if key not in model_state_dict}
    if invalid_keys:
        raise ValueError(f"Checkpoint contains parameters not present in model: {invalid_keys}")

    # Merge the provided non-LoRA tensors into the full state dict so we can load strictly
    merged_state = model_state_dict.copy()
    for key in actual_keys:
        merged_state[key] = non_lora_state[key]

    # Load non-LoRA parameters into the model using strict=True
    # All expected parameters must be present, no extra parameters allowed
    try:
        model.model.load_state_dict(merged_state, strict=True)
        log_info(f"Non-LoRA trainable parameters loaded successfully: {len(non_lora_state)} parameters")
        for key in sorted(actual_keys):
            shape = tuple(non_lora_state[key].shape)
            log_info(f"  Loaded: {key} (shape: {shape})")

    except RuntimeError as e:
        raise RuntimeError(f"Failed to load non-LoRA parameters: {e}") from e


def save_training_artifacts(model, output_dir):
    """Save LoRA adapter and non-LoRA trainable parameters (reference code approach)

    Args:
        model: WaveLLMTrainer model instance
        output_dir: Directory to save artifacts
    """
    from safetensors.torch import save_file
    from peft import get_peft_model_state_dict

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    log_info(f"Saving training artifacts to {output_dir}")

    # Save LoRA adapter weights (PEFT standard)
    lora_state_dict = get_peft_model_state_dict(model.model)
    lora_state_dict = {k: v.contiguous() for k, v in lora_state_dict.items()}

    save_file(lora_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    log_info(f"LoRA adapter saved: {len(lora_state_dict)} parameters")

    # Save LoRA config
    adapter_config = model.model.peft_config['default'].to_dict()

    # Convert sets to lists for JSON serialization
    def make_json_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        return obj

    adapter_config = make_json_serializable(adapter_config)
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config, f, indent=2)
    log_info(f"LoRA config saved (adapter_config.json)")

    # Save non-LoRA trainable parameters (matching reference code exactly)
    # This includes mm_projection_layers and other trainable parameters
    # NOTE: mm_projection_layers is now registered as model.model.mm_projection_layers
    model_state_dict = model.model.state_dict()

    # Get all non-LoRA trainable parameters (same as reference code's get_peft_state_non_lora)
    non_lora_state_dict = {}
    for name, param in model.model.named_parameters():
        if "lora_" not in name and param.requires_grad:
            non_lora_state_dict[name] = model_state_dict[name].contiguous()
            log_info(f"  Including non-LoRA trainable param: {name} (shape: {param.shape})")

    # Save non-LoRA parameters
    save_file(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.safetensors"))
    log_info(f"Non-LoRA trainable parameters saved: {len(non_lora_state_dict)} parameters")

    # Calculate sizes
    lora_size = sum(p.numel() * p.element_size() for p in lora_state_dict.values()) / 1024 / 1024
    non_lora_size = sum(p.numel() * p.element_size() for p in non_lora_state_dict.values()) / 1024 / 1024

    log_info(f"LoRA adapter size: {lora_size:.2f} MB")
    log_info(f"Non-LoRA trainable size: {non_lora_size:.2f} MB")
    log_info(f"Training artifacts saved successfully! Total: {lora_size + non_lora_size:.2f} MB")


def load_model_from_checkpoint(checkpoint_path, config_path, data_root=None):
    """Load model from LoRA adapter checkpoint"""
    from src.llm.utils.config_loader import load_yaml_config
    from src.llm.datamodule import WaveLLMDataModule
    from src.llm.trainer import WaveLLMTrainer

    log_info(f"Loading model from LoRA adapter: {checkpoint_path}")

    # Load configuration
    cfg = load_yaml_config(config_path)

    # Override data_root if provided
    if data_root:
        cfg.data_cfg.data_root = data_root
        cfg.model_cfg.encoder_path = data_root

    # Prepare model config (same as training)
    model_cfg_dict = dict(cfg.model_cfg)
    model_cfg_dict['training'] = EasyDict(cfg.training)
    model_cfg_with_training = EasyDict(model_cfg_dict)

    # Check if it's a LoRA adapter directory (has adapter_model.safetensors and adapter_config.json)
    has_adapter = os.path.exists(os.path.join(checkpoint_path, 'adapter_model.safetensors'))
    has_config = os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json'))

    is_lora_adapter = os.path.isdir(checkpoint_path) and has_adapter and has_config

    if not is_lora_adapter:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}\n"
                        f"Expected LoRA adapter directory with adapter_model.safetensors and adapter_config.json")

    # Load from LoRA adapter
    log_info("Detected LoRA adapter directory")

    # Temporarily disable LoRA in config to create base model without PEFT wrapping
    original_use_peft = model_cfg_with_training.use_peft
    model_cfg_with_training.use_peft = False

    # Create base model (without LoRA)
    model = WaveLLMTrainer(model_cfg_with_training)

    # Restore original config
    model_cfg_with_training.use_peft = original_use_peft

    # Load LoRA adapter onto base model
    from peft import PeftModel

    log_info(f"Loading LoRA adapter from {checkpoint_path}")

    # Suppress PEFT warning about missing adapter keys (these are expected when filtering embeddings)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Found missing adapter keys.*')
        # PEFT supports safetensors directly, just use from_pretrained
        model.model = PeftModel.from_pretrained(model.model, checkpoint_path)

    # Load non-LoRA trainable parameters (reference code approach)
    # During evaluation we rehydrate into a fresh base model without PEFT wrapping,
    # so requires_grad metadata may be absent. Disable that check in this path.
    load_non_lora_trainables(model, checkpoint_path, require_grad_metadata=False)

    model.eval()
    # Move entire model to GPU
    model = model.cuda()
    # Explicitly move radar_encoder and projection layers to GPU
    if hasattr(model, 'radar_encoder'):
        model.radar_encoder = model.radar_encoder.cuda()
    if hasattr(model.model, 'mm_projection_layers'):
        model.model.mm_projection_layers = model.model.mm_projection_layers.cuda()

    # Set tokenizer padding_side to 'left' for decoder-only generation
    model.tokenizer.padding_side = 'left'

    # IMPORTANT: Initialize tokenizer and wave backbone config (like reference code)
    base_model = model.model.get_base_model() if hasattr(model.model, 'get_base_model') else model.model
    if hasattr(base_model, 'initialize_tokenizer_wave_backbone_config'):
        log_info("Initializing tokenizer and wave backbone config...")
        base_model.initialize_tokenizer_wave_backbone_config(model.tokenizer, 'cuda')
    else:
        log_info("Warning: initialize_tokenizer_wave_backbone_config method not found")

    # Verify wave_patch_token_id is set correctly
    wave_patch_token_id = getattr(model.model.config, 'wave_patch_token', None)
    if wave_patch_token_id is not None:
        log_info(f"Wave patch token ID: {wave_patch_token_id}")
    else:
        log_info("Warning: wave_patch_token_id not found in model config")

    log_info("Model loaded successfully")
    return model, cfg
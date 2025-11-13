"""Common utility functions for training and evaluation"""

import os
import json
import warnings
from easydict import EasyDict
from src.logger import log_message






def override_config(cfg_obj, attr_name, arg_value, arg_name):
    """Override config attribute with command line argument if provided"""
    if arg_value is not None:
        setattr(cfg_obj, attr_name, arg_value)
        log_message("CONFIG", f"Using {arg_name} from command line: {arg_value}", color="blue")


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

    log_message("LOAD", f"Loading non-LoRA trainable parameters from {non_lora_path}", color="cyan")
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
        log_message("SUCCESS", f"Non-LoRA trainable parameters loaded successfully: {len(non_lora_state)} parameters", color="green")
        for key in sorted(actual_keys):
            shape = tuple(non_lora_state[key].shape)
            log_message("LOAD", f"  Loaded: {key} (shape: {shape})", color="cyan")

    except RuntimeError as e:
        raise RuntimeError(f"Failed to load non-LoRA parameters: {e}") from e


def save_training_artifacts(model, output_dir, cfg=None, verbose=True):
    """Save LoRA adapter and non-LoRA trainable parameters (reference code approach)

    Args:
        model: WaveLLMTrainer model instance
        output_dir: Directory to save artifacts
        cfg: Complete configuration object to save (optional)
    """
    from safetensors.torch import save_file
    from peft import get_peft_model_state_dict

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        log_message("INFO", f"Saving training artifacts to {output_dir}", color="green")

    # Save complete training configuration if provided
    if cfg is not None:
        def config_to_dict(obj):
            """Convert EasyDict and nested objects to serializable dict"""
            if hasattr(obj, 'to_dict'):
                return config_to_dict(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return config_to_dict(vars(obj))
            elif isinstance(obj, dict):
                return {k: config_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [config_to_dict(item) for item in obj]
            else:
                return obj

        try:
            config_dict = config_to_dict(cfg)
            with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            if verbose:
                log_message("CONFIG", f"Complete training configuration saved (training_config.json)", color="blue")
        except Exception as e:
            log_message("WARNING", f"Failed to save training configuration: {e}", color="yellow")

    # Save LoRA adapter weights manually (only true LoRA parameters)
    lora_state_dict = get_peft_model_state_dict(model.model)

    # Filter to save only true LoRA parameters (exclude embed_tokens and lm_head)
    filtered_lora_state_dict = {}
    for name, param in lora_state_dict.items():
        # Only save tensors with 'lora_' in the name (true LoRA parameters)
        if 'lora_' in name:
            filtered_lora_state_dict[name] = param.contiguous()

    # Save LoRA adapter with filtered parameters
    save_file(filtered_lora_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    if verbose:
        log_message("SUCCESS", f"LoRA adapter saved: {len(filtered_lora_state_dict)} parameters", color="blue")

    # Save PEFT config manually (since we're not using save_pretrained anymore)
    adapter_config = model.model.peft_config['default'].to_dict()

    # Convert sets to lists for JSON serialization
    def make_json_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    adapter_config = make_json_serializable(adapter_config)
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config, f, indent=2)
    if verbose:
        log_message("CONFIG", f"LoRA config saved (adapter_config.json)", color="blue")

    # Save non-LoRA trainable parameters (matching reference code exactly)
    # This includes mm_projection_layers and other trainable parameters
    # NOTE: mm_projection_layers is now registered as model.model.mm_projection_layers
    model_state_dict = model.model.state_dict()

    # Get all non-LoRA trainable parameters (same as reference code's get_peft_state_non_lora)
    non_lora_state_dict = {}
    for name, param in model.model.named_parameters():
        if "lora_" not in name and param.requires_grad:
            non_lora_state_dict[name] = model_state_dict[name].contiguous()
            if verbose:
                log_message("INFO", f"  Including non-LoRA trainable param: {name} (shape: {param.shape})", color="green")

    # Save non-LoRA parameters
    save_file(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.safetensors"))
    if verbose:
        log_message("SUCCESS", f"Non-LoRA trainable parameters saved: {len(non_lora_state_dict)} parameters", color="green")

    # Calculate sizes (use filtered LoRA parameters)
    lora_size = sum(p.numel() * p.element_size() for p in filtered_lora_state_dict.values()) / 1024 / 1024
    non_lora_size = sum(p.numel() * p.element_size() for p in non_lora_state_dict.values()) / 1024 / 1024

    if verbose:
        log_message("INFO", f"LoRA adapter size: {lora_size:.2f} MB", color="green")
        log_message("INFO", f"Non-LoRA trainable size: {non_lora_size:.2f} MB", color="green")
        log_message("SUCCESS", f"Training artifacts saved successfully! Total: {lora_size + non_lora_size:.2f} MB", color="green")


def check_config_consistency(training_cfg, current_cfg, strict=False):
    """Check consistency between training and evaluation configurations
    
    Args:
        training_cfg: Configuration loaded from training_config.json
        current_cfg: Current configuration for evaluation
        strict: If True, raise error on mismatch; if False, only warn
    
    Returns:
        bool: True if configurations are consistent, False otherwise
    """
    from src.logger import log_message
    
    # Helper function to get all nested keys and values from config
    def get_all_nested_values(cfg, prefix=""):
        """Recursively get all nested keys and values from config"""
        result = {}
        
        if hasattr(cfg, '__dict__'):
            # Handle objects with __dict__ (like EasyDict)
            items = vars(cfg).items()
        elif isinstance(cfg, dict):
            # Handle dictionaries
            items = cfg.items()
        else:
            # Handle primitive values
            return {prefix: cfg}
        
        for key, value in items:
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, EasyDict)) or (hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list))):
                # Recursively process nested objects
                result.update(get_all_nested_values(value, new_prefix))
            else:
                # Store primitive values
                result[new_prefix] = value
        
        return result
    
    # Get all nested values from both configs
    training_values = get_all_nested_values(training_cfg)
    current_values = get_all_nested_values(current_cfg)
    
    # Find all keys from both configs
    all_keys = set(training_values.keys()).union(set(current_values.keys()))
    
    is_consistent = True
    mismatches = []
    
    # Check all keys
    for key_path in sorted(all_keys):
        training_value = training_values.get(key_path, "MISSING")
        current_value = current_values.get(key_path, "MISSING")
        
        if training_value != current_value:
            mismatches.append(f"  {key_path}: training={training_value}, current={current_value}")
            is_consistent = False
    
    # Report results
    if not is_consistent:
        log_message("WARNING", "Configuration mismatches detected:", color="yellow")
        for mismatch in mismatches:
            log_message("WARNING", mismatch, color="yellow")
        
        if strict:
            log_message("ERROR", "Configuration mismatches found in strict mode. Aborting.", color="red")
            raise ValueError("Configuration mismatch between training and evaluation")
        else:
            log_message("WARNING", "Continuing with evaluation despite configuration mismatches.", color="yellow")
    else:
        log_message("SUCCESS", "Configuration consistency check passed", color="green")
    
    return is_consistent


def load_model_from_checkpoint(checkpoint_path, config_path, data_root=None):
    """Load model from LoRA adapter checkpoint"""
    from src.llm.utils.config_loader import load_yaml_config
    from src.llm.datamodule import WaveLLMDataModule
    from src.llm.trainer import WaveLLMTrainer

    log_message("LOAD", f"Loading model from LoRA adapter: {checkpoint_path}", color="cyan")

    # Load configuration
    cfg = load_yaml_config(config_path)

    # Override data_root if provided
    if data_root:
        cfg.data_cfg.data_root = data_root
        cfg.model_cfg.encoder_path = data_root

    # Check for training_config.json in checkpoint directory
    training_config_path = os.path.join(checkpoint_path, "training_config.json")
    if os.path.exists(training_config_path):
        log_message("INFO", f"Loading training configuration from {training_config_path}", color="blue")
        with open(training_config_path, 'r') as f:
            training_cfg = json.load(f)
        
        # Convert to EasyDict for consistency
        training_cfg = EasyDict(training_cfg)
        
        # Check configuration consistency
        check_config_consistency(training_cfg, cfg)
    else:
        log_message("WARNING", f"Training configuration not found at {training_config_path}", color="yellow")

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
    log_message("INFO", "Detected LoRA adapter directory", color="green")

    # Temporarily disable LoRA in config to create base model without PEFT wrapping
    original_use_peft = model_cfg_with_training.use_peft
    model_cfg_with_training.use_peft = False

    # Create base model (without LoRA)
    model = WaveLLMTrainer(model_cfg_with_training)

    # Restore original config
    model_cfg_with_training.use_peft = original_use_peft

    # Load LoRA adapter onto base model
    from peft import PeftModel

    log_message("LOAD", f"Loading LoRA adapter from {checkpoint_path}", color="cyan")

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

    # IMPORTANT: Initialize tokenizer and wave backbone config (like reference code)
    base_model = model.model.get_base_model() if hasattr(model.model, 'get_base_model') else model.model
    if hasattr(base_model, 'initialize_tokenizer_wave_backbone_config'):
        log_message("INFO", "Initializing tokenizer and wave backbone config...", color="green")
        base_model.initialize_tokenizer_wave_backbone_config(model.tokenizer, 'cuda')
    else:
        log_message("WARNING", "initialize_tokenizer_wave_backbone_config method not found", color="yellow")

    # Verify wave_patch_token_id is set correctly
    wave_patch_token_id = getattr(model.model.config, 'wave_patch_token', None)
    if wave_patch_token_id is not None:
        log_message("INFO", f"Wave patch token ID: {wave_patch_token_id}", color="green")
    else:
        log_message("WARNING", "wave_patch_token_id not found in model config", color="yellow")

    log_message("SUCCESS", "Model loaded successfully", color="green")
    return model, cfg
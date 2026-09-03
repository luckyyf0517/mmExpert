"""Common utility functions for WaveLLM training."""

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

    # Enumerate trainable parameters when available to derive expected keys.
    # If gradient metadata is unavailable, trust the checkpoint keys.
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


def validate_lora_checkpoint_dir(checkpoint_path):
    """Validate a WaveLLM LoRA checkpoint directory and return key file paths."""
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")

    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    non_lora_path = os.path.join(checkpoint_path, "non_lora_trainables.safetensors")

    missing_files = [
        path for path in (adapter_path, adapter_config_path, non_lora_path)
        if not os.path.exists(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            "Checkpoint is missing required files:\n" + "\n".join(missing_files)
        )

    return {
        "adapter_path": adapter_path,
        "adapter_config_path": adapter_config_path,
        "non_lora_path": non_lora_path,
        "training_config_path": os.path.join(checkpoint_path, "training_config.json"),
    }


def load_lora_checkpoint_into_trainer(trainer, checkpoint_path, is_trainable=False):
    """Load LoRA + non-LoRA trainables into a trainer instance.

    If the current trainer model is already PEFT-wrapped, inject the adapter
    weights into the existing wrapper. Otherwise, attach the adapter from the
    checkpoint onto the base model first.
    """
    checkpoint_files = validate_lora_checkpoint_dir(checkpoint_path)

    # Check the instantiated model rather than relying on the configuration.
    base_model = trainer.model.model if hasattr(trainer.model, 'model') else trainer.model
    is_peft_wrapped = hasattr(base_model, 'peft_config')

    if is_peft_wrapped:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        log_message("LOAD", f"Loading LoRA adapter weights from {checkpoint_files['adapter_path']}", color="cyan")
        adapter_state = load_file(checkpoint_files["adapter_path"])
        set_peft_model_state_dict(trainer.model, adapter_state, adapter_name="default")
    else:
        from peft import PeftModel

        log_message("LOAD", f"Loading LoRA adapter from {checkpoint_path}", color="cyan")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Found missing adapter keys.*')
            trainer.model = PeftModel.from_pretrained(
                trainer.model,
                checkpoint_path,
                is_trainable=is_trainable,
            )

    load_non_lora_trainables(trainer, checkpoint_path, require_grad_metadata=False)
    return checkpoint_files


def initialize_trainer_from_checkpoint(trainer, checkpoint_path, strict_config_check=False):
    """Hydrate an already-initialized trainer from a saved WaveLLM checkpoint.

    This is intended for stage-to-stage continuation. The trainer should already
    be built using the *current* config, and this function injects compatible
    LoRA and non-LoRA trainable weights from an earlier checkpoint.
    """
    checkpoint_files = validate_lora_checkpoint_dir(checkpoint_path)

    training_config_path = checkpoint_files["training_config_path"]
    if os.path.exists(training_config_path):
        log_message("INFO", f"Loading training configuration from {training_config_path}", color="blue")
        with open(training_config_path, "r") as f:
            training_cfg = EasyDict(json.load(f))
        check_config_consistency(training_cfg, trainer.cfg, strict=strict_config_check)
    else:
        log_message("WARNING", f"Training configuration not found at {training_config_path}", color="yellow")

    if not getattr(trainer.cfg, "use_peft", False):
        raise ValueError("init_checkpoint currently requires use_peft=true in the current config")

    load_lora_checkpoint_into_trainer(trainer, checkpoint_path, is_trainable=True)
    log_message("SUCCESS", f"Initialized trainer from checkpoint: {checkpoint_path}", color="green")


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
                log_message("CONFIG", "Complete training configuration saved (training_config.json)", color="blue")
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
        log_message("CONFIG", "LoRA config saved (adapter_config.json)", color="blue")

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
    """Check consistency between saved and current training configurations.

    Args:
        training_cfg: Configuration loaded from training_config.json
        current_cfg: Current training configuration
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
            raise ValueError("Configuration mismatch between saved and current settings")
        else:
            log_message("WARNING", "Continuing despite configuration mismatches.", color="yellow")
    else:
        log_message("SUCCESS", "Configuration consistency check passed", color="green")

    return is_consistent

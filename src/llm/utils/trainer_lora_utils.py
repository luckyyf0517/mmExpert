import torch
from peft import LoraConfig, get_peft_model, TaskType

from src.logger import log_message


def find_all_linear_names(model, multimodal_keywords=None):
    """Find all linear layer names for LoRA target modules."""
    if multimodal_keywords is None:
        multimodal_keywords = []

    cls = torch.nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def setup_lora(trainer, is_rank_0_fn):
    """Configure LoRA adapters for the given trainer instance."""
    cfg = trainer.cfg
    model = trainer.model

    use_auto_find = cfg.peft_config.get('auto_find_target_modules', False)
    target_modules = cfg.peft_config.get('target_modules', None)
    multimodal_keywords = cfg.peft_config.get('multimodal_keywords', ['mm_projection_layers'])

    if use_auto_find:
        target_modules = find_all_linear_names(model, multimodal_keywords=multimodal_keywords)
        if is_rank_0_fn():
            module_types = {}
            for module_name in target_modules:
                if 'self_attn' in module_name:
                    if 'qkv_proj' in module_name:
                        module_type = 'self_attn.qkv_proj'
                    elif 'o_proj' in module_name:
                        module_type = 'self_attn.o_proj'
                    else:
                        module_type = 'self_attn.*'
                elif 'mlp' in module_name:
                    if 'gate_up_proj' in module_name:
                        module_type = 'mlp.gate_up_proj'
                    elif 'down_proj' in module_name:
                        module_type = 'mlp.down_proj'
                    else:
                        module_type = 'mlp.*'
                else:
                    module_type = 'other'
                module_types[module_type] = module_types.get(module_type, 0) + 1

            log_message("INFO", f"Auto-found {len(target_modules)} linear modules for LoRA", color="green")
            breakdown = ', '.join([f'{k}: {v}' for k, v in sorted(module_types.items())])
            log_message("CONFIG", f"Module breakdown: {breakdown}", color="blue")
            log_message(
                "CONFIG",
                f"Excluded: lm_head and modules containing [{', '.join(multimodal_keywords)}]",
                color="blue"
            )
    else:
        raise NotImplementedError("Manual setting target modules is not supported yet")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.peft_config.r,
        lora_alpha=cfg.peft_config.lora_alpha,
        lora_dropout=cfg.peft_config.lora_dropout,
        bias=cfg.peft_config.bias,
        target_modules=target_modules,
    )

    trainer.model = get_peft_model(model, peft_config)

    for name, param in trainer.model.named_parameters():
        if 'lora' not in name and 'mm_projection_layers' not in name:
            if 'embed_tokens' not in name and 'lm_head' not in name:
                param.requires_grad = False

    base_model = trainer.model.get_base_model()
    if hasattr(base_model, 'mm_projection_layers') and base_model.mm_projection_layers is not None:
        base_model.mm_projection_layers.requires_grad_(True)
        base_model.mm_projection_layers.weight.requires_grad = True
        if base_model.mm_projection_layers.bias is not None:
            base_model.mm_projection_layers.bias.requires_grad = True

    new_token_count = getattr(trainer, '_new_special_token_count', 0) or 0

    input_embeds = trainer.model.get_input_embeddings()
    if input_embeds is not None and hasattr(input_embeds, 'weight'):
        weight = input_embeds.weight
        if new_token_count > 0:
            weight.requires_grad = True
            start_idx = weight.shape[0] - new_token_count

            def _mask_old_token_grads(grad):
                if grad is None:
                    return None
                grad[:start_idx] = 0
                return grad

            handle = getattr(trainer, '_input_embed_grad_hook', None)
            if handle is not None:
                handle.remove()
            trainer._input_embed_grad_hook = weight.register_hook(_mask_old_token_grads)
        else:
            weight.requires_grad = False
            handle = getattr(trainer, '_input_embed_grad_hook', None)
            if handle is not None:
                handle.remove()
                trainer._input_embed_grad_hook = None

    if hasattr(base_model, 'lm_head') and hasattr(base_model.lm_head, 'weight'):
        lm_head_weight = base_model.lm_head.weight
    elif hasattr(trainer.model, 'lm_head') and hasattr(trainer.model.lm_head, 'weight'):
        lm_head_weight = trainer.model.lm_head.weight
    else:
        lm_head_weight = None

    if lm_head_weight is not None:
        handle = getattr(trainer, '_lm_head_grad_hook', None)
        if handle is not None:
            handle.remove()
            trainer._lm_head_grad_hook = None

        input_embeds_weight = input_embeds.weight if input_embeds is not None else None
        if lm_head_weight is input_embeds_weight:
            pass
        else:
            lm_head_weight.requires_grad = False


def print_trainable_parameters(trainer, is_rank_0_fn):
    """Print detailed information about trainable parameters."""
    total_params = 0
    trainable_params = 0

    breakdown = {
        'LoRA adapters': 0,
        'mm_projection_layers': 0,
        'embed_tokens (new rows)': 0,
        'Other trainable': 0,
    }

    new_token_count = getattr(trainer, '_new_special_token_count', 0) or 0
    embed_token_dim = None

    for name, param in trainer.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if not param.requires_grad:
            continue

        trainable_count = param_count

        if 'embed_tokens' in name:
            if embed_token_dim is None and param.dim() >= 2:
                embed_token_dim = param.shape[1]
            if new_token_count > 0 and embed_token_dim is not None:
                trainable_count = new_token_count * embed_token_dim
            else:
                trainable_count = 0
        elif 'lm_head' in name:
            trainable_count = 0

        if trainable_count == 0:
            continue

        trainable_params += trainable_count

        if 'lora_' in name:
            breakdown['LoRA adapters'] += trainable_count
        elif 'mm_projection_layers' in name:
            breakdown['mm_projection_layers'] += trainable_count
        elif 'embed_tokens' in name:
            breakdown['embed_tokens (new rows)'] += trainable_count
        else:
            breakdown['Other trainable'] += trainable_count

    if is_rank_0_fn():
        ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        header = f"Trainable params: {trainable_params:,} / {total_params:,} ({ratio:.2f}%)"
        log_message("INFO", header, color="green")

        log_message("INFO", "Breakdown (trainable):", color="green")
        for key, value in breakdown.items():
            if value == 0:
                continue
            percent = value / trainable_params * 100 if trainable_params > 0 else 0.0
            log_message("INFO", f"   - {key:<24}: {value:,} ({percent:.2f}%)", color="green")

        frozen_params = total_params - trainable_params
        log_message("INFO", f"Frozen params: {frozen_params:,}", color="green")


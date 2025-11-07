"""PyTorch Lightning trainer for fine-tuning Phi3 with mmwave features"""

import os
import torch
import torch.nn as nn
from termcolor import colored
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from .llm.model_factory import ModelFactory

import warnings
warnings.filterwarnings('ignore', message='.*Found .* module.*in eval mode.*')


def _is_rank_0():
    """Check if current process is rank 0"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except:
        pass
    # Fallback to environment variable
    local_rank = os.environ.get('LOCAL_RANK', None)
    rank = os.environ.get('RANK', None)
    if rank is not None:
        return int(rank) == 0
    if local_rank is not None:
        return int(local_rank) == 0
    # Default to True if not in distributed mode
    return True


class WaveLLMTrainer(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning Phi3 with mmwave features"""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Create model using ModelFactory
        self.model = ModelFactory.create_model(cfg.model_type, cfg.model)
        self.model.train()

        # Create tokenizer
        self.tokenizer = ModelFactory.create_tokenizer(cfg.model)

        # Setup wave tokens and conversation template
        self._setup_wave_tokens_and_conversation(cfg)

        # Load radar encoder (from feature directory)
        self._load_radar_encoder(cfg.encoder_path)

        # Initialize LoRA if enabled
        if cfg.use_peft:
            self._setup_lora()

        # Print trainable parameters
        self._print_trainable_parameters()

    def _setup_wave_tokens_and_conversation(self, cfg):
        """Setup wave tokens and conversation template in model and tokenizer"""
        # Set conversation template globally
        from src.llm.utils import conversation as conversation_lib
        conversation_template = cfg.get('conversation_template', 'conv_phi3')
        conversation_lib.default_conversation = conversation_lib.conv_templates[conversation_template].copy()

        # Add wave tokens to tokenizer
        DEFAULT_WAVE_PATCH_TOKEN = "<wave_patch>"
        DEFAULT_WAVE_START_TOKEN = "<wave_bos>"
        DEFAULT_WAVE_END_TOKEN = "<wave_eos>"

        mm_use_wave_start_end = cfg.get('mm_use_wave_start_end', True)

        # Add wave patch token
        num_new_tokens = self.tokenizer.add_tokens([DEFAULT_WAVE_PATCH_TOKEN], special_tokens=True)

        # Add start/end tokens if needed
        if mm_use_wave_start_end:
            start_end_tokens = [DEFAULT_WAVE_START_TOKEN, DEFAULT_WAVE_END_TOKEN]
            num_new_tokens += self.tokenizer.add_tokens(start_end_tokens, special_tokens=True)

        # Resize model embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Track newly-added token ids for selective training of embeddings
        self._new_special_token_count = num_new_tokens
        if num_new_tokens > 0:
            vocab_size = self.model.get_input_embeddings().weight.shape[0]
            self._new_special_token_start = vocab_size - num_new_tokens
            self._new_special_token_ids = list(range(self._new_special_token_start, vocab_size))
        else:
            self._new_special_token_start = None
            self._new_special_token_ids = []

        # Set token IDs in model config
        self.model.config.wave_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_WAVE_PATCH_TOKEN])[0]

        if mm_use_wave_start_end:
            self.model.config.wave_start_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_WAVE_START_TOKEN])[0]
            self.model.config.wave_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_WAVE_END_TOKEN])[0]
            self.model.config.mm_use_wave_start_end = True
        else:
            self.model.config.mm_use_wave_start_end = False

        # Set wave token length
        self.model.config.wave_token_len = cfg.get('wave_token_len', 248)

        # Initialize new token embeddings
        if num_new_tokens > 0:
            self._initialize_new_token_embeddings(num_new_tokens)

    def _initialize_new_token_embeddings(self, num_new_tokens):
        """Initialize new token embeddings with average of existing embeddings"""
        input_embeddings = self.model.get_input_embeddings().weight.data
        output_embeddings = self.model.get_output_embeddings().weight.data

        # Use average of existing embeddings for new tokens
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def _load_radar_encoder(self, encoder_path):
        """Load pre-trained radar encoder and projection layers"""
        import yaml
        from easydict import EasyDict

        # Load encoder config (now always plain dict format)
        config_path = os.path.join(encoder_path, 'radar_encoder_config.yaml')
        with open(config_path, 'r') as f:
            encoder_config = yaml.safe_load(f)

        # Get encoder config - convert plain dict to EasyDict for encoder compatibility
        radar_cfg = EasyDict(encoder_config['radar_encoder_cfg'])
        
        # Get encoder type and dimensions
        encoder_type = radar_cfg.get('encoder_type') or radar_cfg.get('type')
        # Use embed_dim from radar_cfg first, fallback to top-level config
        embed_dim = radar_cfg.get('embed_dim')
        if embed_dim is None:
            embed_dim = encoder_config.get('embed_dim', 768)
        # Ensure embed_dim is an integer
        embed_dim = int(embed_dim)

        # Create encoder instance
        from src.clip.encoders import get_encoder
        # Convert EasyDict to dict and expand with **kwargs
        radar_cfg_dict = dict(radar_cfg)
        encoder = get_encoder(encoder_type)(**radar_cfg_dict)

        # Load weights
        weights_path = os.path.join(encoder_path, 'radar_encoder.pth')
        encoder.load_state_dict(torch.load(weights_path, map_location='cpu'))

        # Store encoder
        self.radar_encoder = encoder
        self.radar_encoder.eval()  # Freeze encoder

        # Initialize wave projection layer in the model
        # Projection layer is created and owned by the model, not the trainer
        self.model.initialize_wave_projection(embed_dim)

        # Freeze encoder parameters
        for param in self.radar_encoder.parameters():
            param.requires_grad = False

    def _find_all_linear_names(self, model, multimodal_keywords=[]):
        """Find all linear layer names for LoRA target modules
        
        Adopted from FastChat/Stanford Alpaca implementation.
        Simple logic: find all Linear layers, exclude multimodal_keywords and lm_head.
        
        Args:
            model: The model to search
            multimodal_keywords: List of keywords to exclude (e.g., ['mm_projection', 'projection'])
        
        Returns:
            List of linear layer names
        """
        cls = torch.nn.Linear
        lora_module_names = set()
        
        for name, module in model.named_modules():
            # Skip multimodal-related modules
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                lora_module_names.add(name)
        
        # Remove lm_head if present (needed for 16-bit)
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        
        return list(lora_module_names)

    def _setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        # Check if we should auto-find target_modules
        use_auto_find = self.cfg.peft_config.get('auto_find_target_modules', False)
        target_modules = self.cfg.peft_config.get('target_modules', None)
        
        # Modules to exclude from LoRA (like reference code's do_not_add_lora_model)
        # mm_projection_layers should remain trainable but not wrapped with LoRA
        multimodal_keywords = self.cfg.peft_config.get('multimodal_keywords', ['mm_projection_layers'])
        
        # If auto_find_target_modules is True, always use auto-find (ignore configured target_modules)
        if use_auto_find:
            target_modules = self._find_all_linear_names(self.model, multimodal_keywords=multimodal_keywords)
            if _is_rank_0():
                # Group modules by type for cleaner output
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
                
                print(colored("[INFO]", "green") + f" Auto-found {len(target_modules)} linear modules for LoRA")
                print(f"   Module breakdown: {', '.join([f'{k}: {v}' for k, v in sorted(module_types.items())])}")
                print(f"   Excluded: lm_head and modules containing [{', '.join(multimodal_keywords)}]")
        else: 
            raise NotImplementedError("Manual setting target modules is not supported yet")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.cfg.peft_config.r,
            lora_alpha=self.cfg.peft_config.lora_alpha,
            lora_dropout=self.cfg.peft_config.lora_dropout,
            bias=self.cfg.peft_config.bias,
            target_modules=target_modules,
        )

        self.model = get_peft_model(self.model, peft_config)

        # Freeze base model except LoRA parameters, mm_projection_layers, and embeddings
        # mm_projection_layers is excluded from LoRA target_modules and remains trainable
        # Embeddings: only newly added tokens remain trainable via gradient masking
        for name, param in self.model.named_parameters():
            if 'lora' not in name and 'mm_projection_layers' not in name:
                if 'embed_tokens' not in name and 'lm_head' not in name:
                    param.requires_grad = False

        # Re-enable gradients for mm_projection_layers and embeddings explicitly (PEFT disables them by default)
        base_model = self.model.get_base_model()
        if hasattr(base_model, 'mm_projection_layers') and base_model.mm_projection_layers is not None:
            base_model.mm_projection_layers.requires_grad_(True)
            base_model.mm_projection_layers.weight.requires_grad = True
            if base_model.mm_projection_layers.bias is not None:
                base_model.mm_projection_layers.bias.requires_grad = True

        new_token_count = getattr(self, '_new_special_token_count', 0) or 0

        input_embeds = self.model.get_input_embeddings()
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

                handle = getattr(self, '_input_embed_grad_hook', None)
                if handle is not None:
                    handle.remove()
                self._input_embed_grad_hook = weight.register_hook(_mask_old_token_grads)
            else:
                weight.requires_grad = False
                handle = getattr(self, '_input_embed_grad_hook', None)
                if handle is not None:
                    handle.remove()
                    self._input_embed_grad_hook = None

        # Some HuggingFace models tie lm_head with input embeddings; handle both cases safely
        if hasattr(base_model, 'lm_head') and hasattr(base_model.lm_head, 'weight'):
            lm_head_weight = base_model.lm_head.weight
        elif hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
            lm_head_weight = self.model.lm_head.weight
        else:
            lm_head_weight = None

        if lm_head_weight is not None:
            handle = getattr(self, '_lm_head_grad_hook', None)
            if handle is not None:
                handle.remove()
                self._lm_head_grad_hook = None

            # Freeze lm_head (reference implementation keeps output embeddings fixed)
            if lm_head_weight is input_embeds.weight:
                # Shared embedding: avoid disabling gradients for input embeddings.
                # In Phi-3 custom model lm_head is untied, so this branch should not trigger.
                pass
            else:
                lm_head_weight.requires_grad = False

    def forward(self, batch):
        """
        Forward pass: mmwave features + conversation-based Q&A -> loss

        Args:
            batch: {
                'mmwave_features': Dict with:
                    'range_time': [B, H, W] radar range-time data
                    'doppler_time': [B, H, W] radar doppler-time data
                    'azimuth_time': [B, H, W] radar azimuth-time data
                'input_ids': [B, L],  # tokenized conversation input with wave tokens
                'labels': [B, L]  # tokenized labels with masked instructions
                'attention_mask': [B, L]  # attention mask
            }
        Returns:
            {'loss': tensor}
        """
        # Get input ids, labels, and attention mask
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask')

        # Process mmwave features through encoder and projection
        mmwave_embeds = self._process_mmwave_features(batch['mmwave_features'])

        # Forward pass with conversation-based input
        outputs = self.model(
            input_ids=input_ids,
            input_wave_embeds=mmwave_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return {'loss': outputs.loss}

    def _process_mmwave_features(self, mmwave_features):
        """
        Process raw radar data through encoder and projection layers

        Args:
            mmwave_features: Dict containing:
                'range_time': [B, H, W] radar range-time data
                'doppler_time': [B, H, W] radar doppler-time data
                'azimuth_time': [B, H, W] radar azimuth-time data

        Returns:
            mmwave_embeds: [B, T', H] processed mmwave embeddings
        """
        batch_size = mmwave_features['range_time'].shape[0]

        # Get model dtype to ensure type consistency
        model_dtype = next(self.model.parameters()).dtype
        
        # Get encoder dtype - check if encoder has been converted to bfloat16
        encoder_dtype = next(self.radar_encoder.parameters()).dtype if list(self.radar_encoder.parameters()) else torch.float32
        
        # Convert input to match encoder dtype to avoid dtype mismatch
        encoder_input = {
            'range_time': mmwave_features['range_time'].to(dtype=encoder_dtype),  # [B, H, W]
            'doppler_time': mmwave_features['doppler_time'].to(dtype=encoder_dtype),  # [B, H, W]
            'azimuth_time': mmwave_features['azimuth_time'].to(dtype=encoder_dtype)  # [B, H, W]
        }

        # Forward through radar encoder (batch processing)
        # Convert dict to ModalityData format
        from src.clip.core.base import ModalityData, ModalityType
        modality_data = ModalityData(data=encoder_input, modality=ModalityType.RADAR)
        
        with torch.no_grad():  # Radar encoder is frozen
            result = self.radar_encoder.encode(modality_data, return_sequence=True)
            # Use sequence_features instead of features when return_sequence=True
            encoder_output = result.sequence_features if result.sequence_features is not None else result.features  # [B, T, D]

        # The encoder should output [B, T, D] features
        mmwave_features_tensor = encoder_output  # [B, T, D]
        
        # Convert to model dtype after encoding
        mmwave_features_tensor = mmwave_features_tensor.to(dtype=model_dtype)

        # Ensure projection layer dtype matches input dtype
        if mmwave_features_tensor.dtype != next(self.model.mm_projection_layers.parameters()).dtype:
            self.model.mm_projection_layers = self.model.mm_projection_layers.to(dtype=mmwave_features_tensor.dtype)

        # Project to model hidden size
        mmwave_embeds = self.model.mm_projection_layers(mmwave_features_tensor)  # [B, T, H]

        return mmwave_embeds

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        
        # Explicitly specify batch_size to avoid inference from ambiguous collection
        batch_size = batch['input_ids'].shape[0]
        self.log('train/loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        # Explicitly specify batch_size to avoid inference from ambiguous collection
        batch_size = batch['input_ids'].shape[0]
        self.log('valid/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        # Convert learning_rate and weight_decay to float in case they're strings from YAML
        learning_rate = float(self.cfg.training.learning_rate)
        weight_decay = float(self.cfg.training.weight_decay)
        warmup_steps = int(self.cfg.training.warmup_steps)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    
    def on_test_epoch_start(self):
        """Initialize result JSON file at the start of test epoch"""
        import os
        import json
        
        # Get output file path from trainer or use default
        if hasattr(self.trainer, 'default_root_dir') and self.trainer.default_root_dir:
            checkpoint_dir = self.trainer.default_root_dir
        else:
            checkpoint_dir = getattr(self.cfg, 'log_dir', 'output')
        
        # Create evaluation directory
        eval_dir = os.path.join(checkpoint_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Get rank for distributed training
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0
        
        self.output_file = os.path.join(eval_dir, f"results_rank_{rank}.json")
        
        # Initialize empty results file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        
        if _is_rank_0():
            print(colored("[INFO]", "green") + f" Initializing result file: {self.output_file}")
        
    def test_step(self, batch, batch_idx):
        """Generate predictions and save to JSON file"""
        import json
        
        # Generate predictions
        preds = self._generate_response(batch)
        
        # Get questions and answers from batch (if available)
        questions = batch.get('questions', [])
        answers = batch.get('answers', [])
        
        # Get batch size
        batch_size = batch['input_ids'].shape[0]
        
        # Read current results
        with open(self.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Process each sample in batch
        for i in range(batch_size):
            # Use batch_idx and sample index within batch as unique identifier
            sample_idx = batch_idx * batch_size + i
            
            # Get question and answer from batch (if available)
            question = questions[i] if i < len(questions) else ''
            answer = answers[i] if i < len(answers) else ''
            
            # Add result
            results[f"sample_{sample_idx}"] = {
                "question": question,
                "answer": answer,
                "prediction": preds[i] if i < len(preds) else ""
            }
            
            # Print result (only on rank 0, print all samples)
            if _is_rank_0():
                print(colored(f"\n[Sample {sample_idx}]:", "cyan"))
                if question:
                    print(colored("Question:", "blue"), question)
                print(colored("Prediction:", "green"), preds[i] if i < len(preds) else "")
                if answer:
                    print(colored("Ground Truth:", "yellow"), answer)
                print("-" * 50)
        
        # Write updated results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return None

    def _generate_response(self, batch):
        """Generate response for a batch
        
        Args:
            batch: Batch dict with 'input_ids', 'mmwave_features', 'attention_mask'
            
        Returns:
            List of generated response strings
        """
        batch_size = batch['input_ids'].shape[0]
        responses = []
        
        # Get device from model
        device = next(self.model.parameters()).device
        
        # Generate one sample at a time to avoid batch processing issues
        for i in range(batch_size):
            # Extract single sample
            input_ids_single = batch['input_ids'][i:i+1].to(device)
            attention_mask_single = None
            if batch.get('attention_mask') is not None:
                attention_mask_single = batch['attention_mask'][i:i+1].to(device)
            
            # Extract single mmwave_features
            mmwave_features_single = {
                'range_time': batch['mmwave_features']['range_time'][i:i+1].to(device),
                'doppler_time': batch['mmwave_features']['doppler_time'][i:i+1].to(device),
                'azimuth_time': batch['mmwave_features']['azimuth_time'][i:i+1].to(device)
            }
            
            # Process mmwave features through encoder and projection
            mmwave_embeds_single = self._process_mmwave_features(mmwave_features_single)

            # Generate using input_ids and mmwave_embeds (reference settings)
            self.model.eval()
            with torch.inference_mode():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if attention_mask_single is None:
                        attention_mask_single = torch.ones(input_ids_single.shape, dtype=torch.long, device=input_ids_single.device)

                    outputs = self.model.generate(
                        input_ids_single,
                        input_wave_embeds=mmwave_embeds_single,
                        attention_mask=attention_mask_single,
                        do_sample=True,
                        temperature=1.0,
                        top_k=50,
                        num_beams=4,
                        max_new_tokens=30,
                        top_p=0.95
                    )

            # Decode response following reference code
            input_token_len = input_ids_single.shape[1]
            response = self.tokenizer.decode(outputs[0, input_token_len:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses

    def _print_trainable_parameters(self):
        """Print detailed information about trainable parameters"""
        total_params = 0
        trainable_params = 0

        breakdown = {
            'LoRA adapters': 0,
            'mm_projection_layers': 0,
            'embed_tokens (new rows)': 0,
            'Other trainable': 0,
        }

        new_token_count = getattr(self, '_new_special_token_count', 0) or 0
        embed_token_dim = None

        for name, param in self.named_parameters():
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
                trainable_count = 0  # lm_head remains frozen

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

        if _is_rank_0():
            ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
            header = f"Trainable params: {trainable_params:,} / {total_params:,} ({ratio:.2f}%)"
            print(colored("[INFO]", "green") + f" {header}")

            print(colored("[INFO]", "green") + " Breakdown (trainable):")
            for key, value in breakdown.items():
                if value == 0:
                    continue
                percent = value / trainable_params * 100 if trainable_params > 0 else 0.0
                print(colored("[INFO]", "green") + f"   - {key:<24}: {value:,} ({percent:.2f}%)")

            frozen_params = total_params - trainable_params
            print(colored("[INFO]", "green") + f" Frozen params: {frozen_params:,}")

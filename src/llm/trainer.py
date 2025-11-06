"""PyTorch Lightning trainer for fine-tuning Phi3 with mmwave features"""

import os
import torch
import torch.nn as nn
from termcolor import colored
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from .llm.model_factory import ModelFactory


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
        self.debug = getattr(cfg, 'debug', False)  # Enable debug mode to display Q&A/Prediction

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

        # Create projection layer
        # Use model's dtype to ensure compatibility
        model_dtype = next(self.model.parameters()).dtype
        self.mm_projection_layers = nn.Linear(embed_dim, self.model.config.hidden_size).to(dtype=model_dtype)

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
        
        # If auto_find_target_modules is True, always use auto-find (ignore configured target_modules)
        if use_auto_find:
            multimodal_keywords = self.cfg.peft_config.get('multimodal_keywords', ['mm_projection', 'projection'])
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
                print(f"   Excluded: lm_head and modules containing {multimodal_keywords}")
        # If target_modules is None or empty, auto-find all linear layers
        elif target_modules is None or len(target_modules) == 0:
            multimodal_keywords = self.cfg.peft_config.get('multimodal_keywords', ['mm_projection', 'projection'])
            target_modules = self._find_all_linear_names(self.model, multimodal_keywords=multimodal_keywords)
            if _is_rank_0():
                print(colored("[INFO]", "green") + f" Auto-found {len(target_modules)} linear modules for LoRA (target_modules was empty)")
        else:
            if _is_rank_0():
                print(colored("[INFO]", "green") + f" Using configured target_modules ({len(target_modules)} modules)")
        
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

        # Freeze base model except LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        if _is_rank_0():
            print(colored("[INFO]", "green") + f" Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")

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
        
        # Debug: Print batch info (only in debug mode to avoid spam)
        import os
        if os.environ.get('DEBUG_BATCH_SIZE', '0') == '1':
            print(colored("[DEBUG]", "cyan") + f" forward input_ids shape: {input_ids.shape}, batch_size: {input_ids.shape[0]}")
            print(colored("[DEBUG]", "cyan") + f" forward labels shape: {labels.shape}, batch_size: {labels.shape[0]}")
            if isinstance(batch['mmwave_features'], dict):
                for key, tensor in batch['mmwave_features'].items():
                    print(colored("[DEBUG]", "cyan") + f" forward mmwave_features['{key}'] shape: {tensor.shape}, batch_size: {tensor.shape[0]}")
            elif isinstance(batch['mmwave_features'], list):
                print(colored("[DEBUG]", "cyan") + f" forward mmwave_features list length: {len(batch['mmwave_features'])}")
                if len(batch['mmwave_features']) > 0:
                    first_feat = batch['mmwave_features'][0]
                    for key, tensor in first_feat.items():
                        print(colored("[DEBUG]", "cyan") + f" forward mmwave_features[0][{key}] shape: {tensor.shape}")

        # Process mmwave features through encoder and projection
        mmwave_embeds = self._process_mmwave_features(batch['mmwave_features'])
        
        # Debug: Print processed mmwave embeds
        if os.environ.get('DEBUG_BATCH_SIZE', '0') == '1':
            print(colored("[DEBUG]", "cyan") + f" forward mmwave_embeds shape: {mmwave_embeds.shape}, batch_size: {mmwave_embeds.shape[0]}")

        # Forward pass with conversation-based input
        outputs = self.model(
            input_ids=input_ids,
            mmwave_embeds=mmwave_embeds,
            attention_mask=attention_mask,
            labels=labels
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
        if mmwave_features_tensor.dtype != next(self.mm_projection_layers.parameters()).dtype:
            self.mm_projection_layers = self.mm_projection_layers.to(dtype=mmwave_features_tensor.dtype)

        # Project to model hidden size
        mmwave_embeds = self.mm_projection_layers(mmwave_features_tensor)  # [B, T, H]

        return mmwave_embeds

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        
        # Debug: Display Question, Answer, and Prediction for each batch
        # Only print on rank 0 to avoid duplicate output in distributed training
        if self.debug:
            # Check if we're on rank 0 (only show debug output once)
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except:
                rank = 0
            if rank == 0:
                self._debug_print_batch(batch, outputs, batch_idx)
        
        # Explicitly specify batch_size to avoid inference from ambiguous collection
        batch_size = batch['input_ids'].shape[0]
        self.log('train/loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def _debug_print_batch(self, batch, outputs, batch_idx):
        """Print Question, Answer, and Prediction for debugging"""
        import torch
        
        # Get input_ids and labels
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Extract Question and Answer from labels
        # Labels have -100 for tokens that should be ignored (question part)
        answer_texts = []
        question_texts = []
        
        for i in range(input_ids.shape[0]):
            # Get question: tokens before the first non-IGNORE_INDEX token in labels
            label_seq = labels[i].cpu()
            answer_start_idx = None
            for j, token_id in enumerate(label_seq):
                if token_id != -100:
                    answer_start_idx = j
                    break
            
            if answer_start_idx is not None:
                # Question is everything before answer
                question_ids = input_ids[i][:answer_start_idx].cpu()
                question_text = self.tokenizer.decode(question_ids, skip_special_tokens=True)
                question_texts.append(question_text)
                
                # Answer is the non-IGNORE_INDEX part of labels
                answer_mask = label_seq != -100
                answer_ids = label_seq[answer_mask]
                answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
                answer_texts.append(answer_text)
            else:
                question_texts.append("")
                answer_texts.append("")
        
        # Get prediction: generate from model output
        # Re-run forward to get logits (without labels for generation)
        predictions = []
        with torch.no_grad():
            # Temporarily set model to eval mode for inference
            was_training = self.model.training
            self.model.eval()
            
            try:
                mmwave_embeds = self._process_mmwave_features(batch['mmwave_features'])
                model_outputs = self.model(
                    input_ids=input_ids,
                    mmwave_embeds=mmwave_embeds,
                    attention_mask=batch.get('attention_mask')
                )
                logits = model_outputs.logits
                
                # Greedy decoding: take argmax
                predicted_ids = torch.argmax(logits, dim=-1).cpu()
                
                # Extract prediction from the answer part (same as labels)
                for i in range(input_ids.shape[0]):
                    label_seq = labels[i].cpu()
                    answer_start_idx = None
                    for j, token_id in enumerate(label_seq):
                        if token_id != -100:
                            answer_start_idx = j
                            break
                    
                    if answer_start_idx is not None:
                        # Get predicted tokens for the answer part (same length as answer)
                        answer_mask = label_seq != -100
                        answer_length = answer_mask.sum().item()
                        pred_answer_ids = predicted_ids[i][answer_start_idx:answer_start_idx+answer_length]
                        pred_text = self.tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
                        predictions.append(pred_text)
                    else:
                        predictions.append("")
            finally:
                # Restore training mode
                if was_training:
                    self.model.train()
        
        # Print debug information
        print("\n" + "=" * 80)
        print(colored("[DEBUG]", "cyan") + f" Batch {batch_idx} (showing first 2 samples):")
        print("=" * 80)
        for i in range(min(2, len(question_texts))):
            print(f"\n[Sample {i+1}]")
            print(f"Question: {question_texts[i][:300]}..." if len(question_texts[i]) > 300 else f"Question: {question_texts[i]}")
            print(f"Answer:   {answer_texts[i][:300]}..." if len(answer_texts[i]) > 300 else f"Answer:   {answer_texts[i]}")
            print(f"Prediction: {predictions[i][:300]}..." if len(predictions[i]) > 300 else f"Prediction: {predictions[i]}")
        print("=" * 80 + "\n")

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

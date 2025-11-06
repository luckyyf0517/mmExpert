"""PyTorch Lightning trainer for fine-tuning Phi3 with mmwave features"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from .llm.model_factory import ModelFactory


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
        self.mm_projection_layers = nn.Linear(embed_dim, self.model.config.hidden_size)

        # Freeze encoder parameters
        for param in self.radar_encoder.parameters():
            param.requires_grad = False

    def _setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.cfg.peft_config.r,
            lora_alpha=self.cfg.peft_config.lora_alpha,
            lora_dropout=self.cfg.peft_config.lora_dropout,
            bias=self.cfg.peft_config.bias,
            target_modules=self.cfg.peft_config.get('target_modules', None),
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

        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param}")

    def forward(self, batch):
        """
        Forward pass: mmwave features + conversation-based Q&A -> loss

        Args:
            batch: {
                'mmwave_features': List[Dict]  # List of radar data dicts with range/doppler/azimuth
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
            mmwave_embeds=mmwave_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return {'loss': outputs.loss}

    def _process_mmwave_features(self, mmwave_features_list):
        """
        Process raw radar data through encoder and projection layers

        Args:
            mmwave_features_list: List[Dict] where each dict contains:
                'range_time': [T, H, W] radar range-time data
                'doppler_time': [T, H, W] radar doppler-time data
                'azimuth_time': [T, H, W] radar azimuth-time data

        Returns:
            mmwave_embeds: [B, T', H] processed mmwave embeddings
        """
        batch_size = len(mmwave_features_list)

        # Get model dtype to ensure type consistency
        model_dtype = next(self.model.parameters()).dtype
        
        # Process each sample through radar encoder
        all_features = []
        for i, wave_data in enumerate(mmwave_features_list):
            # Prepare input for encoder (format expected by radar encoder)
            # Convert input to float32 to avoid dtype mismatch in frozen encoder
            encoder_input = {
                'range_time': wave_data['range_time'].unsqueeze(0).to(dtype=torch.float32),  # Add batch dim
                'doppler_time': wave_data['doppler_time'].unsqueeze(0).to(dtype=torch.float32),
                'azimuth_time': wave_data['azimuth_time'].unsqueeze(0).to(dtype=torch.float32)
            }

            # Forward through radar encoder
            # Convert dict to ModalityData format
            from src.clip.core.base import ModalityData, ModalityType
            modality_data = ModalityData(data=encoder_input, modality=ModalityType.RADAR)
            
            with torch.no_grad():  # Radar encoder is frozen
                result = self.radar_encoder.encode(modality_data, return_sequence=True)
                # Use sequence_features instead of features when return_sequence=True
                encoder_output = result.sequence_features if result.sequence_features is not None else result.features  # [1, T, D]

            # The encoder should output [1, T, D] features
            features = encoder_output.squeeze(0)  # Remove batch dim -> [T, D]
            # Convert to model dtype after encoding
            features = features.to(dtype=model_dtype)
            all_features.append(features)

        # Pad or stack features to have consistent sequence length
        max_len = max(feat.shape[0] for feat in all_features)

        # Pad features to max length (features already converted to model_dtype above)
        padded_features = []
        for feat in all_features:
            if feat.shape[0] < max_len:
                # Pad with zeros (using same dtype as feat)
                pad_size = max_len - feat.shape[0]
                padding = torch.zeros(pad_size, feat.shape[1], device=feat.device, dtype=feat.dtype)
                padded_feat = torch.cat([feat, padding], dim=0)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)

        # Stack batch: [B, T, D]
        mmwave_features_tensor = torch.stack(padded_features)

        # Project to model hidden size
        mmwave_embeds = self.mm_projection_layers(mmwave_features_tensor)  # [B, T, H]

        return mmwave_embeds

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        self.log('valid/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
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

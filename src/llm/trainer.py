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
        if mmwave_features_tensor.dtype != next(self.mm_projection_layers.parameters()).dtype:
            self.mm_projection_layers = self.mm_projection_layers.to(dtype=mmwave_features_tensor.dtype)

        # Project to model hidden size
        mmwave_embeds = self.mm_projection_layers(mmwave_features_tensor)  # [B, T, H]

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
            
            # Debug: Check if wave_patch_token exists in tokenizer and input
            wave_patch_token_id = getattr(self.model.config, 'wave_patch_token', None)
            if wave_patch_token_id is not None and _is_rank_0():
                print(f"[DEBUG] wave_patch_token_id: {wave_patch_token_id}")
                # Check if input_ids contains wave_patch_token
                if wave_patch_token_id in input_ids_single[0]:
                    wave_count = (input_ids_single[0] == wave_patch_token_id).sum().item()
                    print(f"[DEBUG] Found {wave_count} wave_patch_tokens in input")
                else:
                    print(f"[DEBUG] No wave_patch_token found in input_ids")
                    # Try to decode the input to see what's happening
                    decoded_input = self.tokenizer.decode(input_ids_single[0], skip_special_tokens=False)
                    print(f"[DEBUG] Decoded input: '{decoded_input}'")
            elif _is_rank_0():
                print(f"[DEBUG] wave_patch_token_id is None!")

            # Process mmwave features through encoder and projection
            mmwave_embeds_single = self._process_mmwave_features(mmwave_features_single)

            if _is_rank_0():
                print(f"[DEBUG] mmwave_embeds_single shape: {mmwave_embeds_single.shape}")

            # Debug: Test wave feature fusion by doing a forward pass
            with torch.no_grad():
                test_outputs = self.model(
                    input_ids=input_ids_single,
                    input_wave_embeds=mmwave_embeds_single,
                    attention_mask=attention_mask_single,
                    return_dict=True
                )
                if _is_rank_0():
                    print(f"[DEBUG] Test forward pass successful")
                    print(f"[DEBUG] Test logits shape: {test_outputs.logits.shape}")
                    max_logit = test_outputs.logits.max().item()
                    min_logit = test_outputs.logits.min().item()
                    print(f"[DEBUG] Logits range: [{min_logit:.3f}, {max_logit:.3f}]")

                    # Check first few generated tokens via greedy decoding from logits
                    greedy_ids = torch.argmax(test_outputs.logits[0, -10:, :], dim=-1)  # Last 10 positions
                    greedy_tokens = self.tokenizer.decode(greedy_ids, skip_special_tokens=True)
                    print(f"[DEBUG] Greedy prediction from last 10 logits: '{greedy_tokens}'")

                    # Deep dive into the problematic tokens
                    print(f"[DEBUG] Last 10 greedy token IDs: {greedy_ids.tolist()}")
                    for i, token_id in enumerate(greedy_ids):
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                        token_text_clean = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"[DEBUG] Token {i}: ID={token_id}, no_skip='{token_text}', clean='{token_text_clean}'")

                    # Check what the most likely tokens are for the last position
                    last_logits = test_outputs.logits[0, -1, :]
                    top_10_ids = torch.topk(last_logits, 10).indices.tolist()
                    print(f"[DEBUG] Top 10 most likely tokens at last position:")
                    for i, token_id in enumerate(top_10_ids):
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(f"[DEBUG]   {i}: ID={token_id} -> '{token_text}'")

                    # Check if we have the same token repeated
                    unique_tokens = torch.unique(greedy_ids)
                    print(f"[DEBUG] Unique tokens in greedy prediction: {unique_tokens.tolist()} (out of {len(greedy_ids)} tokens)")

            # Generate using input_ids and mmwave_embeds
            # Following the reference code's strategy exactly
            self.model.eval()
            with torch.inference_mode():
                # Use autocast like reference code
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    # Create attention mask if None like reference code
                    if attention_mask_single is None:
                        attention_mask_single = torch.ones(input_ids_single.shape, dtype=torch.long, device=input_ids_single.device)

                    if _is_rank_0():
                        print(f"[DEBUG] Starting generation...")

                    # Use reference code parameters exactly
                    outputs = self.model.generate(
                        input_ids_single,
                        input_wave_embeds=mmwave_embeds_single,
                        attention_mask=attention_mask_single,
                        do_sample=True,         # Same as reference
                        temperature=1.0,         # Same as reference
                        top_k=50,               # Same as reference
                        num_beams=4,            # Same as reference (beam_size=4)
                        max_new_tokens=30,      # Same as reference
                        top_p=0.95              # Same as reference
                    )

                    if _is_rank_0():
                        print(f"[DEBUG] Generation completed")
                        print(f"[DEBUG] Output shape: {outputs.shape}")

            # Decode response following reference code
            input_token_len = input_ids_single.shape[1]
            n_diff_input_output = (input_ids_single != outputs[0, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            response = self.tokenizer.decode(outputs[0, input_token_len:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses

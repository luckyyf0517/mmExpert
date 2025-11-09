"""PyTorch Lightning trainer for fine-tuning Phi3 with mmwave features"""

import os
import torch
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from src.logger import log_message

from .llm.model_factory import ModelFactory
from src.llm.utils.trainer_lora_utils import (
    setup_lora,
    print_trainable_parameters,
)
from src.llm.utils.trainer_debug_utils import (
    predict_text_from_logits,
    log_sample_output,
)

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
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|end|>']})
        self.model.model.tokenizer = self.tokenizer

        # Setup wave tokens and conversation template
        self._setup_wave_tokens_and_conversation(cfg)

        # Load radar encoder (from feature directory)
        self._load_radar_encoder(cfg.encoder_path)

        # Initialize LoRA if enabled
        if cfg.use_peft:
            setup_lora(self, _is_rank_0)

        # Print trainable parameters
        print_trainable_parameters(self, _is_rank_0)
        
        # Debug mode while forward pass
        self.debug_mode = False

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
            encoder_output = result.sequence_features # [B, T, D]

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

        if self.debug_mode and _is_rank_0():
            try:
                question = batch['questions'][0]    
                answer = batch['answers'][0]
                prediction = predict_text_from_logits(self.tokenizer, outputs.logits, labels)
                log_sample_output(question, prediction, answer, prefix="FORWARD")
            except Exception as exc:
                log_message("WARNING", f"Debug logging failed: {exc}", color="yellow")

        return {'loss': outputs.loss}

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
        
        log_message("INFO", f"Initializing result file: {self.output_file}", color="green")
        
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

        questions = batch.get('questions', [])

        # Process mmwave features for entire batch at once
        mmwave_embeds = self._process_mmwave_features(batch['mmwave_features'])

        # Use input_ids that are already formatted for inference (when is_inference=True)
        input_ids_batch = batch['input_ids'].to(device)
        attention_mask_batch = batch['attention_mask'].to(device)
        
        # Generate for entire batch at once
        self.model.eval()

        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids_batch,
                    input_wave_embeds=mmwave_embeds,
                    attention_mask=attention_mask_batch,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    num_beams=4,
                    max_new_tokens=30,
                    top_p=0.95,
                )
        
        input_token_len = input_ids_batch.shape[1]
        n_diff_input_output = (input_ids_batch != outputs[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            log_message("WARNING", f"{n_diff_input_output} output_ids are not the same as the input_ids", color="yellow")

        
        # Extract generated tokens (excluding input)
        generated_tokens = outputs[:, input_token_len:]
        
        # Decode generated tokens
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        responses = [response.strip() for response in responses]
        

        for i in range(batch_size):
            question = questions[i]
            answer = batch['answers'][i]
            response = responses[i]
            log_sample_output(question, response, answer, prefix="GENERATE")
        
        return responses

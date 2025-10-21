import os
import torch
import torch.nn as nn
from typing import Union
from transformers import AutoTokenizer, AutoModel

# Constants
DEFAULT_HF_CACHE_DIR = '/root/autodl-tmp/mmExpert/huggingface'


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, text_pooling: str = 'pooler', unfreeze_last_layer_num: int = 0, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.text_pooling = text_pooling
        self.unfreeze_last_layer_num = unfreeze_last_layer_num

        # Set cache directory for Hugging Face models
        cache_dir = DEFAULT_HF_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Load tokenizer and model with offline mode
        try:
            # Set environment variables for offline mode
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,  # Only use local files
                use_fast=True
            )
            self.text_encoder = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True  # Only use local files
            )
        except Exception as e:
            print(f"Error loading model {model_name} in offline mode: {e}")
            print("Trying to load with network access...")
            # Fallback to online mode if offline fails
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.text_encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            except Exception as e2:
                print(f"Error loading model {model_name} with network: {e2}")
                raise
        self.text_encoder.eval()
        for name, param in self.text_encoder.named_parameters():
            # Handle different model architectures
            if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                # BERT-style models
                num_layers = len(self.text_encoder.encoder.layer)
                unfreeze_param = False
                for i in range(self.unfreeze_last_layer_num):
                    if 'layer.%d' % (num_layers - i) in name:
                        unfreeze_param = True
                    if 'pooler' in name:
                        unfreeze_param = True
            elif hasattr(self.text_encoder, 'layers') or hasattr(self.text_encoder, 'model'):
                # Handle other architectures (like Phi-3)
                unfreeze_param = False
                # For simplicity, unfreeze all parameters for these models
                if self.unfreeze_last_layer_num > 0:
                    unfreeze_param = True
            else:
                # Default case
                unfreeze_param = False

            param.requires_grad = unfreeze_param

    def encode(self, text, device='cuda', return_sequence=False):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)

        if return_sequence:
            # Return full sequence for sequence-level processing
            return outputs.last_hidden_state  # [b, seq_len, text_embed_dim]
        else:
            # Apply pooling as before for compatibility
            if self.text_pooling == 'mean':
                out = outputs.last_hidden_state.mean(dim=1)
            elif self.text_pooling == 'pooler':
                out = outputs.pooler_output
            elif self.text_pooling == 'max':
                out = outputs.last_hidden_state.max(dim=1)[0]
            return out
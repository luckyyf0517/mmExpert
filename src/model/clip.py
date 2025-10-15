import os

# Set offline mode for Hugging Face to avoid network requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import json
import einops
import torchvision
import numpy as np
import pytorch_lightning as pl

from typing import Tuple, Union
from collections import OrderedDict
from einops.layers.torch import Rearrange   
from termcolor import colored
from easydict import EasyDict as edict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from timm.models.vision_transformer import _create_vision_transformer

import warnings
from src.model.clip_loss import create_loss

# Suppress specific warning from transformers
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")

# Constants
DEFAULT_HF_CACHE_DIR = '/root/autodl-tmp/mmExpert/huggingface'


def build_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1) # zero out the lower diagonal
    return mask


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_ratio)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_ratio, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: int = 4, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, mlp_ratio, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

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
            num_layers = len(self.text_encoder.encoder.layer)
            unfreeze_param = False
            for i in range(self.unfreeze_last_layer_num): 
                if 'layer.%d' % (num_layers - i) in name: 
                    unfreeze_param = True
                if 'pooler' in name: 
                    unfreeze_param = True
            param.requires_grad = unfreeze_param
        
    def encode(self, text, device='cuda'):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        if self.text_pooling == 'mean': 
            out = outputs.last_hidden_state.mean(dim=1)
        elif self.text_pooling == 'pooler': 
            out = outputs.pooler_output
        elif self.text_pooling == 'max': 
            out = outputs.last_hidden_state.max(dim=1)[0]
        return out
    

class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str,
                 embed_dim: int,
                 image_resolution: int, 
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        if image_resolution == [496, 128]: 
            self.window_size = None
        else: 
            self.window_size = 128
            self.stride = 16
            self.unfold = nn.Sequential(
                nn.Unfold(kernel_size=(self.window_size, self.window_size), stride=self.stride),
                Rearrange('b (c h w) n -> (b n) c h w', c=1, h=self.window_size, w=self.window_size))
        
        self.model_name = model_name
        if 'vit' in model_name: 
            self.visual = _create_vision_transformer(model_name, pretrained=pretrained, in_chans=1, img_size=image_resolution, num_classes=embed_dim, **kwargs)
        elif 'resnet' in model_name: 
            self.visual = torchvision.models.resnet18(pretrained=pretrained)
            self.visual.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.visual.fc = nn.Linear(self.visual.fc.in_features, embed_dim)
        
    def forward(self, x):
        if self.window_size is not None: 
            batch_size = x.size(0)
            x = self.unfold(x)
            x = self.visual(x)
            x = einops.rearrange(x, '(b n) c -> b n c', b=batch_size)
        else: 
            x = self.visual(x)
        return x
    
    def encode_to_sequence(self, x):
        if self.window_size is not None: 
            batch_size = x.size(0)
            x = self.unfold(x)
            x = self.visual(x)
            x = einops.rearrange(x, '(b n) c -> b n c', b=batch_size)
        else: 
            if 'vit' in self.model_name: 
                x = self.visual.forward_features(x)
                x = x[:, 1:, :] # [b, n, c]
            elif 'resnet' in self.model_name: 
                x = self.visual.conv1(x)
                x = self.visual.bn1(x)
                x = self.visual.relu(x)
                x = self.visual.maxpool(x)
                x = self.visual.layer1(x)
                x = self.visual.layer2(x)
                x = self.visual.layer3(x)
                x = einops.rearrange(x, 'b c h w -> b (h w) c')
        return x
            

class CLIP(pl.LightningModule):
    """
    CLIP model from OpenAI's implementation
    - the transformer is not used for text encoding, but for time series encoding
    """
    def __init__(self,
                 encoder_cfg: dict,
                 text_cfg: dict,
                 # transformer cfg
                 context_length: int,
                 transformer_width: int,
                 transformer_layers: int,
                 transformer_heads: int,
                 # training cfg
                 temperature: float,
                 use_siglip: bool = False,
                 learning_rate: float = 1.0e-04,
                 max_epochs: int = 50):
        super().__init__()
        self.image_encoder = ImageEncoder(**encoder_cfg)
        self.text_encoder = TextEncoder(**text_cfg)
        self.text_projection = nn.Linear(text_cfg.embed_dim, encoder_cfg.embed_dim)
        
        if self.image_encoder.window_size is not None: 
            self.context_length = context_length + 1
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=build_attention_mask(context_length + 1)
            )
            scale = transformer_width ** -0.5
            self.eot_token = nn.Parameter(scale * torch.randn(transformer_width))
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)
        
        self.use_siglip = use_siglip
        if use_siglip: 
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        else:
            self.logit_scale = 1 / temperature
        
        loss_args = edict({
            'gather_with_grad': True,
            'rank': int(os.environ.get('RANK', 0)),
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'horovod': False, 
            'local_loss': False,
            'siglip': use_siglip,
            'loss_dist_impl': 'bidir',
        })
        self.loss_fn = create_loss(loss_args)
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.initialize_parameters()
        self.configure_optimizers()
        
    # def state_dict(self, *, destination=None, prefix='', keep_vars=False):
    #     state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    #     # do not save text encoder
    #     for k in list(state_dict.keys()): 
    #         if 'text_encoder' in k: 
    #             state_dict.pop(k)
    #     return state_dict

    def initialize_parameters(self):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=0.01)
        if self.image_encoder.window_size is not None: 
            nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.image_encoder.visual.conv1.weight.dtype

    def encode_image(self, image):
        x = self.image_encoder(image) # shape: [b, n, c]
        if self.image_encoder.window_size is not None: 
            x = torch.cat([x, self.eot_token.to(x.dtype) + torch.zeros_like(x[:, :1, :])], dim=1)
            x = x + self.positional_embedding.to(x.dtype)
            x = self.transformer(x)
            x = self.ln_final(x[:, -1, :])
        return x

    def encode_text(self, text, device='cuda'):
        x = self.text_encoder.encode(text, device=device)
        x = self.text_projection(x)
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, device=image.device)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True) 
        return image_features, text_features
    
    def compute_loss(self, batch):
        image_features = batch['image_features']
        text_features = batch['text_features']
        if not self.use_siglip: 
            loss_clip = self.loss_fn(image_features, text_features, logit_scale=self.logit_scale)
        else: 
            loss_clip = self.loss_fn(image_features, text_features, logit_scale=self.logit_scale, logit_bias=self.logit_bias)
        return {'loss_clip': loss_clip}
    
    def shared_step(self, batch, batch_idx, phase='train'):
        self.batch_size = batch['motion'].size(0)
        batch['motion'] = batch['motion'].unsqueeze(1).float() # [b, c, h, w]
        motion = batch['motion']
        text = batch['caption']
        image_features, text_features = self.forward(motion, text)
        batch['image_features'] = image_features
        batch['text_features'] = text_features
        return batch
    
    def training_step(self, batch, batch_idx):
        batch = self.shared_step(batch, batch_idx, phase='train')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='train')
        
        # Display loss in progress bar with more details
        self.log('train_loss', loss_dict['loss_clip'], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        # Also log learning rate for monitoring (safely)
        try:
            if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log('lr', current_lr, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        except (AttributeError, IndexError):
            pass  # Skip if trainer or optimizers not available
        
        return loss_dict['loss_clip']
        
    def validation_step(self, batch, batch_idx):
        batch = self.shared_step(batch, batch_idx, phase='valid')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='valid')
        
        # Display loss in progress bar
        self.log('val_loss', loss_dict['loss_clip'], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        return loss_dict['loss_clip']
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW([
            {'params': self.text_encoder.parameters(), 'lr': lr / 2},
            {'params': self.text_projection.parameters(), 'lr': lr},
            {'params': self.image_encoder.parameters(), 'lr': lr},
        ], betas=(0.5, 0.9), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=0)
        return [opt], [scheduler]
    
    def log_loss(self, loss_dict, phase='train'): 
        for k, v in loss_dict.items(): 
            self.log(phase + '/' + k, v, on_step=self.training, on_epoch=not self.training, 
                     logger=True, batch_size=self.batch_size, rank_zero_only=True, sync_dist=False, add_dataloader_idx=False)
        del loss_dict

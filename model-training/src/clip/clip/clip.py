# Import the refactored CLIP model
from .clip_model import CLIPModel as CLIP
from ..encoders.radar_encoder_temporal import RadarEncoderTemporal as RadarEncoder
from ..encoders.text_encoder import TextEncoder

# Import transformer components
from .clip_transformer import Transformer, build_attention_mask, LayerNorm

# Legacy imports for backward compatibility
__all__ = [
    'CLIP',
    'RadarEncoder',
    'TextEncoder',
    'Transformer',
    'build_attention_mask',
    'LayerNorm'
]

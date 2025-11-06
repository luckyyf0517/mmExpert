from .radar_encoder_temporal import RadarEncoderTemporal as RadarEncoder
from .radar_encoder_vit import RadarEncoderViT
from .text_encoder import TextEncoder
from ..core.registry import registry

def get_encoder(encoder_type: str):
    """Get encoder class by type name from registry."""
    info = registry.get(encoder_type, category="encoders")
    if info is None:
        raise ValueError(f"Encoder '{encoder_type}' not found in registry")
    return info.component_class

__all__ = [
    'RadarEncoder',
    'RadarEncoderViT',
    'TextEncoder',
    'get_encoder',
]


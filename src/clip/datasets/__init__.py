"""Datasets for CLIP module - deprecated, use src.clip.dataset instead"""

# Backward compatibility imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..dataset import (
    Text2DopplerDatasetV2,
    load_radar_data,
    _normalize_per_frame,
    _normalize_global,
    _normalize_log,
    DEFAULT_RADAR_OPT,
)

__all__ = [
    'Text2DopplerDatasetV2',
    'load_radar_data',
    '_normalize_per_frame',
    '_normalize_global',
    '_normalize_log',
    'DEFAULT_RADAR_OPT',
]


"""Compatibility exports for the CLIP dataset module."""

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

"""Configuration loader utilities"""

import yaml
from easydict import EasyDict


def load_yaml_config(config_path):
    """Load YAML configuration file

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        EasyDict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)

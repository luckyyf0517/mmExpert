import os
import yaml
import glob
import torch
import numpy as np
from easydict import EasyDict as edict


def load_yaml(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return edict(cfg)


def load_config(path, data_config_path=None, model_config_path=None):
    """
    Load configuration with automatic data and model config resolution.

    Args:
        path: Main config file path (optional, for backward compatibility)
        data_config_path: Explicit data config file path
        model_config_path: Explicit model config file path
    """
    # If explicit configs are provided, use them
    if data_config_path is not None and model_config_path is not None:
        return {
            'data_cfg': load_yaml(data_config_path),
            'model_cfg': load_yaml(model_config_path)
        }

    # Otherwise, load from main config file (backward compatibility)
    cfg = load_yaml(path)

    # Resolve data config
    if isinstance(cfg.get('data_cfg'), str):
        data_path = cfg['data_cfg']
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(path), data_path)
        cfg['data_cfg'] = load_yaml(data_path)

    # Resolve model config
    if isinstance(cfg.get('model_cfg'), str):
        model_path = cfg['model_cfg']
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(path), model_path)
        cfg['model_cfg'] = load_yaml(model_path)

    return cfg
        
        
def load_file_list(path): 
    return sorted(glob.glob(path))

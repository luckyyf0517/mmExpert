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
        
        
def load_file_list(path): 
    return sorted(glob.glob(path))

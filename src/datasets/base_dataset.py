import torch
import os
import glob
import json
import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from termcolor import colored
from torch.utils.data._utils.collate import default_collate

# Constants for motion processing
DEFAULT_NOISE_STD = 0.001
DEFAULT_THRESHOLD = 0.1
DEFAULT_EPSILON = 1e-6
DEFAULT_LOG_SCALE = 10
DEFAULT_ENERGY_THRESHOLD = 60
DEFAULT_RANDOM_SHIFT_MAX = 0.01

# Default configuration for real dataset
DEFAULT_REAL_OPT = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'raw': True,
    'thresholding': True,
}


def _apply_downsample(motion, opt):
    """Apply downsampling if specified in options."""
    if opt.get('downsample') is not None and opt.downsample:
        downsample_start = random.randint(0, opt.downsample - 1)
        motion = motion[downsample_start::opt.downsample]
        max_motion_length = opt.max_motion_length // opt.downsample
    else: 
        max_motion_length = opt.max_motion_length
    return motion, max_motion_length

def _crop_motion(motion, opt):
    """Crop motion to unit length boundaries."""
    m_length = motion.shape[0]
    m_length = (m_length // opt.unit_length) * opt.unit_length
    idx = random.randint(0, motion.shape[0] - m_length)
    motion = motion[idx: idx + m_length]
    return motion, m_length

def _apply_separation(motion, opt):
    """Apply motion separation if specified."""
    if opt.get('separate'):
        coin = random.choice([0, 0, 0, 1, 2])
        weight_options = {
            0: np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            1: np.array([0.0, 1.0, 0.0, 1.0, 1.0]),
            2: np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        }
        weights = weight_options[coin]
        motion = np.einsum('ijk,j->ik', motion, weights)
    return motion

def _apply_log_transforms(motion, opt):
    """Apply various logarithmic transformations."""
    if opt.get('logged'):
        motion = np.power(DEFAULT_LOG_SCALE, motion)
    
    if opt.get('text2doppler'):
        motion = 20 * np.log10(motion / motion.max() + DEFAULT_EPSILON)
        motion = np.clip(motion, motion.max() - DEFAULT_ENERGY_THRESHOLD, motion.max())
    
    if opt.get('raw'):
        # DO NOT MODIFY 
        motion = np.log10(motion + DEFAULT_EPSILON)
    
    if opt.get('random_scale'):
        random_shift = random.random() * DEFAULT_RANDOM_SHIFT_MAX
        motion = np.log10(motion / motion.max() + random_shift + DEFAULT_EPSILON)
    
    if opt.get('log_norm'):
        motion = np.log10(motion / motion.max() + DEFAULT_EPSILON)
    
    return motion

def _apply_normalization(motion, opt):
    """Apply normalization and thresholding."""
    if opt.get('ignore_energy'):
        motion = (motion - motion.min(axis=1, keepdims=True)) / (motion.max(axis=1, keepdims=True) - motion.min(axis=1, keepdims=True) + DEFAULT_EPSILON)
    
    # Global normalization
    motion = (motion - motion.min()) / (motion.max() - motion.min() + DEFAULT_EPSILON)
    
    if opt.get('thresholding'):
        motion[motion < DEFAULT_THRESHOLD] = 0
    
    return motion

def _pad_motion(motion, m_length, max_motion_length):
    """Pad motion to max_motion_length and create mask."""
    mask = np.ones((max_motion_length, motion.shape[1]), dtype=np.float32)
    
    if m_length < max_motion_length:
        motion = np.concatenate([
            motion,
            np.zeros((max_motion_length - m_length, motion.shape[1]))], axis=0)
        mask[m_length:] = 0
    else: 
        motion = motion[:max_motion_length]
    
    return motion, mask

def _add_noise(motion, opt):
    """Add white noise if specified."""
    if opt.get('add_noise'):
        noise = np.random.normal(0, DEFAULT_NOISE_STD, motion.shape)
        noise[motion > 0] = 0
        motion = motion + noise
    return motion

def process_motion(motion, opt): 
    """Process motion data with various transformations and normalizations."""
    # Apply downsampling
    motion, max_motion_length = _apply_downsample(motion, opt)
    
    # Crop motion to unit length boundaries
    motion, m_length = _crop_motion(motion, opt)
    
    # Apply separation
    motion = _apply_separation(motion, opt)
    
    # Apply logarithmic transformations
    motion = _apply_log_transforms(motion, opt)
    
    # Apply normalization and thresholding
    motion = _apply_normalization(motion, opt)
    
    # Pad motion and create mask
    motion, mask = _pad_motion(motion, m_length, max_motion_length)
    
    # Add noise
    motion = _add_noise(motion, opt)
    
    return motion, mask, m_length


class Text2DopplerDatasetV2():
    """Dataset for training text motion matching model and evaluations."""
    
    # Constants
    MAX_LENGTH = 20
    ROTATE_POSTFIXES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    PERSON_SYNONYMS = ['person', 'man', 'human']
    
    def __init__(self, opt, split_file, data_scale=1.0):
        self.opt = deepcopy(opt)
        self._configure_for_split(split_file)
        self.max_length = self.MAX_LENGTH
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = opt.min_motion_len 
        self.data_list = self._load_data(split_file, data_scale)
        self.udoppler_postfix = opt.get('udoppler_postfix')

    def _configure_for_split(self, split_file):
        """Configure options based on split file type."""
        if 'MOMASK' in split_file:
            self.opt.separate = True
        elif 'REAL' in split_file:
            self.opt.separate = False
            self.opt.random_rotate = False
            self.opt.random_scale = False
            self.opt.text2doppler = False
            self.opt.raw = True
        elif 'HumanML3D' in split_file: 
            self.opt.separate = False
            self.opt.random_rotate = True
            self.opt.random_scale = False
            self.opt.text2doppler = False
            self.opt.raw = False
        else: 
            raise ValueError(f"Invalid split file: {split_file}")

    def _load_data(self, split_file, data_scale):
        """Load and scale dataset."""
        data_dict = json.load(open(split_file, 'r'))
        data_amount = int(len(data_dict) * data_scale)  
        print(f"Total number of data {data_amount} (scale factor: {data_scale}) from {split_file}")
        return list(data_dict.values())[:data_amount]

    def _get_motion_path(self, data_dict):
        """Get motion file path with appropriate postfix."""
        motion_folder = data_dict['filefolder']
        motion_folder = motion_folder.replace('udoppler', 'udoppler_' + self.udoppler_postfix)
        motion_index = data_dict['fileindex']
        
        if self.opt.get('raw'):
            return os.path.join(motion_folder, f'{motion_index}.npy')
        else:
            if self.opt.get('random_rotate'):
                rotate_postfix = random.choice(self.ROTATE_POSTFIXES)
                return os.path.join(motion_folder, f'{motion_index}{rotate_postfix}.npy')
            else: 
                path = os.path.join(motion_folder, f'{motion_index}A.npy')
                if os.path.exists(path):
                    return path
                else: 
                    return os.path.join(motion_folder, f'{motion_index}.npy')

    def _process_caption(self, data_dict):
        """Process caption text with person synonym replacement."""
        if 'captions' in data_dict:
            text_list = data_dict['captions']
            caption = random.choice(text_list)
            caption = caption.replace('person', random.choice(self.PERSON_SYNONYMS))
        else: 
            caption = ''
        return caption.lower()

    def _create_item_dict(self, data_dict, motion_path, motion):
        """Create item dictionary with all required fields."""
        item_dict = {
            'filename': motion_path,
            'motion': motion,
            'caption': self._process_caption(data_dict)
        }
        
        if 'classid' in data_dict:
            item_dict.update({
                'classid': data_dict['classid'],
                'classname': data_dict['classname'],
                'caption': f"a person {data_dict['classname']}"
            })
        else: 
            item_dict.update({
                'classid': -1,
                'classname': 'null'
            })
        
        return item_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        motion_path = self._get_motion_path(data_dict)
        motion = np.load(motion_path)
        motion, _, _ = process_motion(motion, self.opt)
        return self._create_item_dict(data_dict, motion_path, motion)


if __name__ == "__main__":
    # Test with actual dataset
    opt = edict({
        'max_motion_length': 496,
        'min_motion_len': 96,
        'unit_length': 16,
        'raw': True,
        'thresholding': True
    })
    
    # Use actual dataset file
    dataset_file = 'dataset/HumanML3D/_split/all.json'
    if os.path.exists(dataset_file):
        dataset = Text2DopplerDatasetV2(opt, dataset_file, data_scale=0.01)  # Use small subset
        if len(dataset) > 0:
            item = dataset[0]  # Get first item
            print("Dataset item content:")
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape={value.shape}")
                    if value.size < 20:  # Only print small arrays
                        print(f"  data: {value}")
                else:
                    print(f"{key}: {value}")
        else:
            print("Dataset is empty")
    else:
        print(f"Dataset file not found: {dataset_file}")
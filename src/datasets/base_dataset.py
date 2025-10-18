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

# Default configuration for radar dataset
DEFAULT_RADAR_OPT = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'raw': True,
    'thresholding': True,
    'normalize': 'per_frame',  # 'none', 'per_frame', 'global', or 'log'
}

# Default configuration for real dataset (legacy)
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


# Radar data processing functions (for DATA_FORMAT.md .npz files)
def _normalize_per_frame(data):
    """Normalize each time frame to [0, 1] for visualization."""
    normalized = data.copy()
    num_bins, T = data.shape

    for t in range(T):
        frame = normalized[:, t]
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            normalized[:, t] = (frame - frame_min) / (frame_max - frame_min)

    return normalized


def _normalize_global(data):
    """Global normalization across all frames."""
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        return (data - dmin) / (dmax - dmin)
    return data


def _normalize_log(data):
    """Log-scale normalization."""
    return np.log10(data + 1e-10)


def _apply_radar_normalization(radar_view, opt):
    """Apply normalization based on strategy for a single radar view."""
    normalize_type = opt.get('normalize', 'none')

    if normalize_type == 'per_frame':
        return _normalize_per_frame(radar_view)
    elif normalize_type == 'global':
        return _normalize_global(radar_view)
    elif normalize_type == 'log':
        return _normalize_log(radar_view)
    else:  # 'none'
        return radar_view


def _crop_radar_view(radar_view, opt):
    """Crop radar view to unit length boundaries."""
    T = radar_view.shape[1]
    T = (T // opt.unit_length) * opt.unit_length
    idx = random.randint(0, radar_view.shape[1] - T)
    radar_view = radar_view[:, idx: idx + T]
    return radar_view, T


def _pad_radar_view(radar_view, T, max_motion_length):
    """Pad radar view to max_motion_length and create mask."""
    num_bins, current_T = radar_view.shape
    mask = np.ones((max_motion_length,), dtype=np.float32)

    if T < max_motion_length:
        padding = np.zeros((num_bins, max_motion_length - T))
        radar_view = np.concatenate([radar_view, padding], axis=1)
        mask[T:] = 0
    else:
        radar_view = radar_view[:, :max_motion_length]
        T = max_motion_length

    return radar_view, mask, T


def process_radar_view(radar_view, opt):
    """Process a single radar view with transformations and normalizations."""
    # Crop to unit length boundaries
    radar_view, T = _crop_radar_view(radar_view, opt)

    # Apply normalization
    radar_view = _apply_radar_normalization(radar_view, opt)

    # Pad and create mask
    radar_view, mask, T = _pad_radar_view(radar_view, T, opt.max_motion_length)

    return radar_view, mask, T


def load_radar_data(npz_file_path, opt):
    """Load radar data from NPZ file and return three separate views."""
    try:
        data = np.load(npz_file_path)

        # Extract raw data (unnormalized as specified in DATA_FORMAT.md)
        range_time = data['range_time']      # (256, T) - keep original resolution
        doppler_time = data['doppler_time']  # (128, T)
        azimuth_time = data['azimuth_time']  # (128, T)

        # Process each view separately
        range_processed, range_mask, range_T = process_radar_view(range_time, opt)
        doppler_processed, doppler_mask, doppler_T = process_radar_view(doppler_time, opt)
        azimuth_processed, azimuth_mask, azimuth_T = process_radar_view(azimuth_time, opt)

        # Return as dictionary with three separate views
        return {
            'range_time': range_processed,    # (256, T_processed)
            'doppler_time': doppler_processed,  # (128, T_processed)
            'azimuth_time': azimuth_processed,  # (128, T_processed)
            'mask': range_mask,  # All views should have same T after processing
            'T': range_T
        }

    except Exception as e:
        print(f"Error loading radar data from {npz_file_path}: {e}")
        return None


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
            self.opt.separate = False
        elif 'REAL' in split_file:
            self.opt.separate = False
            self.opt.random_rotate = False
            self.opt.random_scale = False
            self.opt.text2doppler = False
            self.opt.raw = True
        elif 'HumanML3D' in split_file: 
            self.opt.separate = False
            self.opt.random_rotate = False
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
        """Get radar NPZ file path."""
        motion_folder = data_dict['filefolder']
        motion_folder = motion_folder.replace('udoppler', 'udoppler_' + self.udoppler_postfix)
        motion_index = data_dict['fileindex']

        # Return NPZ file path (radar format from DATA_FORMAT.md)
        return os.path.join(motion_folder, f'{motion_index}.npz')

    def _process_caption(self, data_dict):
        """Process caption text with person synonym replacement."""
        if 'captions' in data_dict:
            text_list = data_dict['captions']
            caption = random.choice(text_list)
            caption = caption.replace('person', random.choice(self.PERSON_SYNONYMS))
        else: 
            caption = ''
        return caption.lower()

    def _create_item_dict(self, data_dict, motion_path, motion_data):
        """Create item dictionary with all required fields."""
        # Handle both radar data (dict) and legacy motion data (array)
        if isinstance(motion_data, dict):
            # New radar format with three separate views
            item_dict = {
                'filename': motion_path,
                'radar_data': motion_data,  # Dict with range_time, doppler_time, azimuth_time
                'caption': self._process_caption(data_dict)
            }
        else:
            # Legacy motion format
            item_dict = {
                'filename': motion_path,
                'motion': motion_data,
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

        # Load radar data from NPZ file
        radar_data = load_radar_data(motion_path, self.opt)
        if radar_data is None:
            # Create dummy data if loading fails
            radar_data = {
                'range_time': np.zeros((256, self.opt.max_motion_length), dtype=np.float32),
                'doppler_time': np.zeros((128, self.opt.max_motion_length), dtype=np.float32),
                'azimuth_time': np.zeros((128, self.opt.max_motion_length), dtype=np.float32),
                'mask': np.ones((self.opt.max_motion_length,), dtype=np.float32),
                'T': self.opt.max_motion_length
            }
        return self._create_item_dict(data_dict, motion_path, radar_data)


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
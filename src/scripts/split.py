import os
import json
import glob
import numpy as np
import codecs as cs
from tqdm import tqdm

# Constants
DATASETS = ['HumanML3D']
SAVE_DIR = 'dataset/HumanML3D/_split'
MIN_MOTION_LEN = 96  # From DEFAULT_RADAR_OPT
MAX_MOTION_LEN = 496  # From DEFAULT_RADAR_OPT
TRAIN_VAL_SPLIT_RATIO = 10  # Every 10th file goes to validation

def collect_file_paths(datasets):
    """Collect all NPZ file paths from specified datasets."""
    all_filelist = []
    for dataset_name in datasets:
        # Look for NPZ files (radar format from DATA_FORMAT.md)
        filelist = sorted(glob.glob(f'dataset/{dataset_name}/udoppler/*.npz'))
        print(f'Found {len(filelist)} NPZ files in {dataset_name}')
        all_filelist += filelist
    return all_filelist

def get_text_filename(motion_filename):
    """Convert radar filename to corresponding text filename."""
    textname = motion_filename.replace('udoppler', 'texts').replace('.npz', '.txt')
    return textname

def validate_radar_data(radar_data):
    """Validate radar data for length and data quality."""
    # Check time dimension length for all views
    range_T = radar_data['range_time'].shape[1]
    doppler_T = radar_data['doppler_time'].shape[1]
    azimuth_T = radar_data['azimuth_time'].shape[1]

    # All views should have same time dimension
    if not (range_T == doppler_T == azimuth_T):
        return False

    # Check length constraints
    if range_T < MIN_MOTION_LEN or range_T >= MAX_MOTION_LEN:
        return False

    # Check for invalid data (NaN or Inf) in all views
    for view_name, view_data in radar_data.items():
        if view_name in ['range_time', 'doppler_time', 'azimuth_time']:
            if np.isnan(view_data).sum() > 0 or np.isinf(view_data).sum() > 0:
                return False

    return True

def process_text_file(text_filename):
    """Process text file and extract captions."""
    captions = []
    try:
        with cs.open(text_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # Remove comments (everything after #)
                line_split = line.strip().split('#')[0]
                # Remove trailing period
                if line_split and line_split[-1] == '.':
                    line_split = line_split[:-1]
                if line_split:  # Only add non-empty lines
                    captions.append(line_split.lower())
    except Exception as e:
        print(f"Warning: Could not read text file {text_filename}: {e}")
        return []
    
    return captions

def process_radar_file(filename):
    """Process a single radar NPZ file and return data dictionary entry."""
    text_filename = get_text_filename(filename)
    if not os.path.exists(text_filename):
        print(f"Warning: Text file {text_filename} not found")
        return None

    # Load and validate radar data
    try:
        radar_data = np.load(filename)
        if not validate_radar_data(radar_data):
            return None
    except Exception as e:
        print(f"Warning: Could not load radar file {filename}: {e}")
        return None

    # Process text captions
    captions = process_text_file(text_filename)
    if not captions:
        return None

    file_index = text_filename.split('/')[-1].split('.')[0]
    file_folder = os.path.dirname(filename)

    return {
        'filefolder': file_folder,
        'fileindex': file_index,
        'captions': captions
    }

def split_data(data_dict):
    """Split data into train and validation sets."""
    train_dict = {}
    val_dict = {}
    
    for key, value in data_dict.items():
        file_index = int(value['fileindex'])
        if file_index % TRAIN_VAL_SPLIT_RATIO == 0:
            val_dict[key] = value
        else:
            train_dict[key] = value
    
    return train_dict, val_dict

def save_json_formatted(data, filepath):
    """Save data to JSON file with proper formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to process radar datasets and create splits."""
    print("Starting radar dataset processing...")

    # Collect all NPZ file paths
    all_filelist = collect_file_paths(DATASETS)
    print(f'Total {len(all_filelist)} NPZ files to process')

    # Process files
    data_dict = {}
    index = 0

    for filename in tqdm(all_filelist, desc="Processing radar files"):
        result = process_radar_file(filename)
        if result is not None:
            data_dict[f'{index:06d}'] = result
            index += 1

    print(f'Successfully processed {len(data_dict)} radar files')

    # Split data
    train_dict, val_dict = split_data(data_dict)
    print(f'Train set: {len(train_dict)} files')
    print(f'Validation set: {len(val_dict)} files')

    # Save formatted JSON files
    print(f'Saving files to {SAVE_DIR}...')
    save_json_formatted(train_dict, f'{SAVE_DIR}/train.json')
    save_json_formatted(val_dict, f'{SAVE_DIR}/val.json')
    save_json_formatted(val_dict, f'{SAVE_DIR}/test.json')
    save_json_formatted(data_dict, f'{SAVE_DIR}/all.json')

    print('Radar dataset processing completed successfully!')

if __name__ == "__main__":
    main()
import os
import json
import glob
import numpy as np
import codecs as cs
from tqdm import tqdm

# Constants
DATASETS = ['humanml3d_ext']
SAVE_DIR = 'data/humanml3d_ext'
MIN_MOTION_LEN = 100  # From DEFAULT_RADAR_OPT
MAX_MOTION_LEN = 500  # From DEFAULT_RADAR_OPT

def collect_file_paths(datasets):
    """Collect uDoppler NPY file paths from specified datasets."""
    all_filelist = []
    for dataset_name in datasets:
        # Look for uDoppler NPY files (current default)
        filelist = sorted(glob.glob(f'data/{dataset_name}/udoppler/*.npy'))
        print(f'Found {len(filelist)} uDoppler NPY files in {dataset_name}')
        all_filelist += filelist
    return all_filelist

def get_text_filename(motion_filename):
    """Convert radar filename to corresponding text filename."""
    textname = motion_filename.replace('udoppler', 'texts').replace('mmwave', 'texts').replace('.npy', '.txt').replace('.npz', '.txt')
    return textname

def validate_radar_data(radar_data):
    """Validate uDoppler data for length and data quality.

    Current default simulator output is a single-view `.npy` stored as
    (T, 128). Historical `.npz` inputs are still tolerated for manual use,
    but split generation defaults to uDoppler NPY files.
    """
    if isinstance(radar_data, np.ndarray):
        if radar_data.ndim != 2:
            return False
        T = radar_data.shape[0] if radar_data.shape[1] == 128 else radar_data.shape[1]
        if T < MIN_MOTION_LEN or T >= MAX_MOTION_LEN:
            return False
        return not (np.isnan(radar_data).sum() > 0 or np.isinf(radar_data).sum() > 0)

    doppler = radar_data['udoppler'] if 'udoppler' in radar_data else radar_data['doppler_time']
    T = doppler.shape[1]
    if T < MIN_MOTION_LEN or T >= MAX_MOTION_LEN:
        return False
    return not (np.isnan(doppler).sum() > 0 or np.isinf(doppler).sum() > 0)

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
    """Process a single uDoppler NPY file and return data dictionary entry."""
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

def save_json_formatted(data, filepath):
    """Save data to JSON file with proper formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to process radar datasets and create all.json."""
    print("Starting HumanML3DExt radar dataset processing...")

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

    # Save formatted JSON file
    print(f'Saving all.json to {SAVE_DIR}...')
    save_json_formatted(data_dict, f'{SAVE_DIR}/all.json')

    print('HumanML3DExt radar dataset processing completed successfully!')

if __name__ == "__main__":
    main()

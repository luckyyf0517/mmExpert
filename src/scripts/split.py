import os
import json
import glob
import numpy as np
import codecs as cs
from tqdm import tqdm

# Constants
DATASETS = ['HumanML3D']
SAVE_DIR = 'dataset/HumanML3D/_split'
REQUIRED_POSTFIX = 'A' 
MIN_MOTION_LEN = 100
MAX_MOTION_LEN = 500
TRAIN_VAL_SPLIT_RATIO = 10  # Every 10th file goes to validation

def collect_file_paths(datasets):
    """Collect all file paths from specified datasets."""
    all_filelist = []
    for dataset_name in datasets:
        filelist = sorted(glob.glob(f'dataset/{dataset_name}/udoppler/*.npy'))
        print(f'Found {len(filelist)} files in {dataset_name}')
        all_filelist += filelist
    return all_filelist

def get_text_filename(motion_filename):
    """Convert motion filename to corresponding text filename."""
    textname = motion_filename.replace('udoppler', 'texts').replace('.npy', '.txt')
    # Remove the postfix (e.g., 'A' from filename)
    textname = textname[:-5] + textname[-4:]
    return textname

def is_valid_motion(filename):
    """Check if motion file has required postfix."""
    return REQUIRED_POSTFIX in filename

def validate_motion_data(motion):
    """Validate motion data for length and data quality."""
    motion_len = motion.shape[0]
    
    # Check length constraints
    if motion_len < MIN_MOTION_LEN or motion_len >= MAX_MOTION_LEN:
        return False
    
    # Check for invalid data (NaN or Inf)
    if np.isnan(motion).sum() > 0 or np.isinf(motion).sum() > 0:
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

def process_motion_file(filename):
    """Process a single motion file and return data dictionary entry."""
    if not is_valid_motion(filename):
        return None
    
    text_filename = get_text_filename(filename)
    if not os.path.exists(text_filename):
        print(f"Warning: Text file {text_filename} not found")
        return None
    
    # Load and validate motion data
    try:
        motion = np.load(filename)
        if not validate_motion_data(motion):
            return None
    except Exception as e:
        print(f"Warning: Could not load motion file {filename}: {e}")
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
    """Main function to process datasets and create splits."""
    print("Starting dataset processing...")
    
    # Collect all file paths
    all_filelist = collect_file_paths(DATASETS)
    print(f'Total {len(all_filelist)} files to process')
    
    # Process files
    data_dict = {}
    index = 0
    
    for filename in tqdm(all_filelist, desc="Processing files"):
        result = process_motion_file(filename)
        if result is not None:
            data_dict[f'{index:06d}'] = result
            index += 1
    
    print(f'Successfully processed {len(data_dict)} files')
    
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
    
    print('Dataset processing completed successfully!')

if __name__ == "__main__":
    main()
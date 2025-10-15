import os
import json
import numpy as np
import codecs as cs
from tqdm import tqdm

# Constants
SAVE_DIR = 'dataset/HumanML3D/_split'
REQUIRED_POSTFIX = 'A' 
MIN_MOTION_LEN = 100
MAX_MOTION_LEN = 500

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

def load_file_list(txt_file):
    """Load file list from txt file, filtering out M-prefixed files."""
    file_list = []
    try:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('M'):
                    file_list.append(line)
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return []
    
    return file_list

def check_file_exists(file_id):
    """Check if both motion and text files exist for given file ID."""
    motion_file = f'dataset/HumanML3D/udoppler/{file_id}A.npy'
    text_file = f'dataset/HumanML3D/texts/{file_id}.txt'
    
    return os.path.exists(motion_file) and os.path.exists(text_file)

def process_file_list(file_list, split_name):
    """Process a list of file IDs and return data dictionary."""
    data_dict = {}
    valid_count = 0
    missing_count = 0
    
    print(f"Processing {split_name} files...")
    
    for file_id in tqdm(file_list, desc=f"Processing {split_name}"):
        # Check if files exist
        if not check_file_exists(file_id):
            missing_count += 1
            continue
        
        motion_file = f'dataset/HumanML3D/udoppler/{file_id}A.npy'
        result = process_motion_file(motion_file)
        
        if result is not None:
            data_dict[f'{valid_count:06d}'] = result
            valid_count += 1
    
    print(f"{split_name}: {valid_count} valid files, {missing_count} missing files")
    return data_dict

def save_json_formatted(data, filepath):
    """Save data to JSON file with proper formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to process datasets and create splits."""
    print("Starting HumanML3D dataset processing...")
    
    # Load file lists
    train_val_list = load_file_list('dataset/HumanML3D/train_val.txt')
    test_list = load_file_list('dataset/HumanML3D/test.txt')
    
    print(f'Loaded {len(train_val_list)} files from train_val.txt')
    print(f'Loaded {len(test_list)} files from test.txt')
    
    # Process train files
    train_dict = process_file_list(train_val_list, "Train")
    
    # Process val/test files
    val_dict = process_file_list(test_list, "Val/Test")
    
    print(f'Train set: {len(train_dict)} files')
    print(f'Validation/Test set: {len(val_dict)} files')
    
    # Save formatted JSON files
    print(f'Saving files to {SAVE_DIR}...')
    save_json_formatted(train_dict, f'{SAVE_DIR}/train.json')
    save_json_formatted(val_dict, f'{SAVE_DIR}/val.json')
    save_json_formatted(val_dict, f'{SAVE_DIR}/test.json')
    
    # Create combined all.json
    all_dict = {**train_dict, **{f'val_{k}': v for k, v in val_dict.items()}}
    save_json_formatted(all_dict, f'{SAVE_DIR}/all.json')
    
    print('Dataset processing completed successfully!')

if __name__ == "__main__":
    main()

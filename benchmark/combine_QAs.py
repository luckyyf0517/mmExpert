import os
import json
from glob import glob

def combine_qa_files():
    """Combine all JSON files in train_QAs directory into one file."""
    
    # Get all part files
    qa_dir = "dataset/HumanML3D/_split/train_QAs"
    pattern = os.path.join(qa_dir, "part_*.json")
    part_files = sorted(glob(pattern))
    
    if not part_files:
        print(f"No part files found in {qa_dir}")
        return
    
    print(f"Found {len(part_files)} part files:")
    for file in part_files:
        print(f"  - {file}")
    
    # Combine all data
    combined_data = {}
    
    for part_file in part_files:
        print(f"Processing {part_file}...")
        with open(part_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_data.update(data)
    
    # Save combined file with same name as folder
    output_file = os.path.join(qa_dir, "train_QAs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined {len(combined_data)} items into {output_file}")
    print(f"Total items: {len(combined_data)}")

if __name__ == "__main__":
    combine_qa_files()

import os
import json
import glob


def merge_rank_files(evaluation_dir, pattern="results_rank_*.json"):
    """Merge all rank files in the evaluation directory

    Args:
        evaluation_dir: Path to evaluation directory
        pattern: Glob pattern for rank files (default: "results_rank_*.json")

    Returns:
        dict: Merged data from all rank files

    Raises:
        FileNotFoundError: If no rank files found
    """
    pattern_path = os.path.join(evaluation_dir, pattern)
    rank_files = glob.glob(pattern_path)

    if not rank_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {evaluation_dir}")

    merged_data = {}
    for rank_file in sorted(rank_files):
        print(f"Loading {rank_file}")
        data = json.load(open(rank_file))
        merged_data.update(data)

    print(f"Merged {len(rank_files)} files with total {len(merged_data)} items")
    return merged_data


def load_evaluation_data(input_path, merged_filename="results.json", pattern="results_rank_*.json"):
    """Load evaluation data from either a directory (merging rank files) or a single file

    Args:
        input_path: Path to directory containing rank files or path to single JSON file
        merged_filename: Name of the merged file to create/use when loading from directory
        pattern: Glob pattern for rank files when loading from directory

    Returns:
        dict: Loaded evaluation data
    """
    if os.path.isdir(input_path):
        # If it's a directory, merge rank files first
        merged_file = os.path.join(input_path, merged_filename)
        if not os.path.exists(merged_file):
            print(f"Step 1: Merging rank files in {input_path}...")
            all_data = merge_rank_files(input_path, pattern)
            json.dump(all_data, open(merged_file, 'w'), indent=2)
            print(f"Data merged and saved to {merged_file}")
        else:
            print(f"Using existing merged file: {merged_file}")
            all_data = json.load(open(merged_file))
    else:
        # If it's a single file, load directly
        print(f"Loading data from single file: {input_path}")
        with open(input_path, 'r') as file:
            all_data = json.load(file)

    return all_data


def extract_field_safely(container: dict, keys: list):
    """Safely extract a field from a container using multiple possible keys

    Args:
        container: Dictionary to extract from
        keys: List of possible keys to try

    Returns:
        The value if found, None otherwise
    """
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def save_evaluation_results(results: dict, output_path: str, indent: int = 2):
    """Save evaluation results to JSON file

    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the results
        indent: JSON indentation level
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=indent, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
import os
import json
import glob
from collections import OrderedDict


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
    # Look for rank files in the ranks subdirectory first
    ranks_dir = os.path.join(evaluation_dir, "ranks")
    if os.path.exists(ranks_dir):
        pattern_path = os.path.join(ranks_dir, pattern)
    else:
        pattern_path = os.path.join(evaluation_dir, pattern)

    rank_files = glob.glob(pattern_path)

    if not rank_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {evaluation_dir} or {ranks_dir}")

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
            save_evaluation_results(all_data, merged_file, preprocess_answers=True)
            print(f"Data merged and saved to {merged_file}")
        else:
            print(f"Using existing merged file: {merged_file}")
            all_data = json.load(open(merged_file))
            # Preprocess answer fields to split # delimited answers into lists
            all_data = preprocess_answer_fields(all_data)
    else:
        # If it's a single file, load directly
        print(f"Loading data from single file: {input_path}")
        with open(input_path, 'r') as file:
            all_data = json.load(file)
        # Preprocess answer fields to split # delimited answers into lists
        all_data = preprocess_answer_fields(all_data)

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


def split_answer_field(answer_value):
    """Split answer field by # delimiter to handle multiple ground truth answers

    Args:
        answer_value: The answer field value (string or list)

    Returns:
        list: List of individual answers
    """
    if isinstance(answer_value, list):
        return answer_value

    if not isinstance(answer_value, str):
        return [str(answer_value)] if answer_value is not None else []

    # Split by # and strip whitespace
    if '#' in answer_value:
        return [part.strip() for part in answer_value.split('#') if part.strip()]
    else:
        return [answer_value.strip()]


def preprocess_answer_fields(data: dict):
    """Preprocess answer fields in evaluation data to split # delimited answers into lists

    Args:
        data: Dictionary containing evaluation data

    Returns:
        dict: Data with answer fields preprocessed
    """
    processed_data = OrderedDict()

    for key, item in data.items():
        processed_item = OrderedDict()

        # Copy all fields in their original order, only modifying answer field
        for field_name, field_value in item.items():
            if field_name == 'answer':
                processed_item[field_name] = split_answer_field(field_value)
            else:
                processed_item[field_name] = field_value

        # Ensure answer field exists even if not in original item
        if 'answer' not in processed_item:
            processed_item['answer'] = []

        processed_data[key] = processed_item

    return processed_data


def save_evaluation_results(results: dict, output_path: str, indent: int = 2, preprocess_answers: bool = False):
    """Save evaluation results to JSON file

    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the results
        indent: JSON indentation level
        preprocess_answers: Whether to preprocess answer fields to split # delimited answers into lists
    """
    data_to_save = preprocess_answer_fields(results) if preprocess_answers else results

    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=indent, ensure_ascii=False, sort_keys=False)
    print(f"Results saved to: {output_path}")
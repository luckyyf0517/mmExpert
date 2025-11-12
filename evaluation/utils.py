import os
import json
import glob
import re
import copy
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

    rank_files = sorted(glob.glob(pattern_path))

    if not rank_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {evaluation_dir} or {ranks_dir}")

    merged_data = OrderedDict()
    index_offset = 0

    for rank_file in rank_files:
        print(f"Loading {rank_file}")
        data = json.load(open(rank_file))
        
        # First rank file keeps original keys, subsequent ranks reindex
        for key, value in data.items():
            if index_offset == 0:
                # First rank: keep original keys
                merged_data[key] = value
            else:
                # Subsequent ranks: reindex keys by adding offset to original index
                # Extract base name and number (e.g., "sample" and "0" from "sample_0")
                if '_' in key:
                    base_name = key.rsplit('_', 1)[0]
                    try:
                        original_num = int(key.rsplit('_', 1)[1])
                        new_key = f"{base_name}_{original_num + index_offset}"
                    except ValueError:
                        # If number extraction fails, find next available index
                        # This should be rare, so we scan existing keys
                        max_idx = index_offset - 1
                        for existing_key in merged_data.keys():
                            if existing_key.startswith(base_name + '_'):
                                try:
                                    idx = int(existing_key.rsplit('_', 1)[1])
                                    max_idx = max(max_idx, idx)
                                except ValueError:
                                    pass
                        new_key = f"{base_name}_{max_idx + 1}"
                else:
                    base_name = key
                    # Find next available index for base_name
                    max_idx = index_offset - 1
                    for existing_key in merged_data.keys():
                        if existing_key.startswith(base_name + '_'):
                            try:
                                idx = int(existing_key.rsplit('_', 1)[1])
                                max_idx = max(max_idx, idx)
                            except ValueError:
                                pass
                    new_key = f"{base_name}_{max_idx + 1}"
                
                merged_data[new_key] = value
        
        # After processing each rank file, update offset for next rank file
        # Find the maximum index in merged_data
        max_index = -1
        for key in merged_data.keys():
            if '_' in key:
                try:
                    idx = int(key.rsplit('_', 1)[1])
                    max_index = max(max_index, idx)
                except ValueError:
                    pass
        if max_index >= 0:
            index_offset = max_index + 1

    print(f"Merged {len(rank_files)} files with total {len(merged_data)} items")
    return merged_data


def load_evaluation_data(input_path, merged_filename="results.json", pattern="results_rank_*.json", use_exist=False):
    """Load evaluation data from either a directory (merging rank files) or a single file

    Args:
        input_path: Path to directory containing rank files or path to single JSON file
        merged_filename: Name of the merged file to create/use when loading from directory
        pattern: Glob pattern for rank files when loading from directory
        use_exist: If True, use existing merged file when loading from directory

    Returns:
        dict: Loaded evaluation data
    """
    all_data = None

    if os.path.isdir(input_path):
        # If it's a directory, check for existing merged file
        merged_file = os.path.join(input_path, merged_filename)

        if use_exist and os.path.exists(merged_file):
            print(f"Using existing merged file: {merged_file}")
            with open(merged_file, 'r') as file:
                all_data = json.load(file)
            # Preprocess answer fields to split # delimited answers into lists
            all_data = preprocess_answer_fields(all_data)
        else:
            print(f"Step 1: Merging rank files in {input_path}...")
            all_data = merge_rank_files(input_path, pattern)
            # Preprocess the data in memory AND save it
            all_data = preprocess_answer_fields(all_data)
            save_evaluation_results(all_data, merged_file, preprocess_answers=False)
            print(f"Data merged and saved to {merged_file}")
    else:
        # If it's a single file, load directly
        print(f"Loading data from single file: {input_path}")
        with open(input_path, 'r') as file:
            all_data = json.load(file)
        # Preprocess answer fields to split # delimited answers into lists
        all_data = preprocess_answer_fields(all_data)

    # Explicit return to ensure processed data is always returned
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


def clean_answer(data):
    """
    LEO clean strategy
    """
    data = data.lower()
    data = re.sub('[ ]+$', '', data)
    data = re.sub('^[ ]+', '', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r'[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç', 'c', data)
    data = re.sub('\u2019', '\'', data)  # Right single quotation mark to apostrophe
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', 'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b', 'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def special_token_filter(lan, clean=True, truncation=True, max_length=256):
    """
    Usage:
        clean the language, remove stop words and special tokens
    Args:
        lan: str, language to be cleaned
        clean: bool, if apply LEO clean strategy
        truncation: to avoid crash the input sentence will be truncated to max_length
        max_length: You may set this to the max length of possible gt answer
    """
    replacements = {
        "ASSISTANT:": "",
        "ASSISTANT: ": "",
        "\n": "",
        "<s>": "",
        "</s>": "",
        "<unk>": "",
        "<p>": "",
        "</p>": "",
        "<ref>": "",
        "<|endoftext|>": ""  # for GPT2
    }
    for old, new in replacements.items():
        lan = lan.replace(old, new)
    lan = lan.strip()
    lan = re.sub(r'\s{2,}', ' ', lan)
    if truncation:
        if len(lan) > max_length:
            lan = lan[:max_length]
    if clean:
        lan = clean_answer(lan)
    return lan


def refined_EM(data, gt, set_zero_as_error=True, not_refine=False):
    EM = []
    _data = copy.deepcopy(data)
    if not_refine:
        for ins in _data:
            pred = _data[ins][0]
            if pred in gt[ins]:
                EM.append(1)
            else:
                EM.append(0)
    else:
        for ins in _data:
            to_append = 0
            pred = _data[ins][0]
            if set_zero_as_error:
                if pred in [" ", ""]:
                    pred = "@@@@@@@@-= Empty Answer =-@@@@@@@@@"
            for _gt in gt[ins]:
                if pred == _gt:
                    to_append = 1
                    continue
                elif "".join(pred.split()) in "".join(_gt.split()):
                    to_append = 1
                    continue
                elif "".join(_gt.split()) in "".join(pred.split()):
                    to_append = 1
                    continue
            EM.append(to_append)
    return EM
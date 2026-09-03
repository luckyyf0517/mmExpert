"""
Generate WaveLLM question-answer supervision from motion captions.
"""

import os
import json
import httpx
from openai import OpenAI
from copy import deepcopy
import multiprocessing as mp
import argparse
import time
from pathlib import Path
import hashlib
import re


REQUIRED_QA_IDS = ("QA01", "QA02", "QA03", "QA04")
MAX_RETRIES = 8
BASE_RETRY_DELAY_SEC = 2.0
QA03_BANNED_PATTERNS = (
    r"\bgreet(?:ing)?\b",
    r"\bget attention\b",
    r"\battention\b",
    r"\bshowing respect\b",
    r"\brespect\b",
    r"\bcommunicat(?:e|ing|ion)\b",
    r"\bexercise\b",
    r"\bflexibility\b",
    r"\bbalance\b",
    r"\bjoy\b",
    r"\benjoy\b",
    r"\bhumorous\b",
    r"\bplayful\b",
    r"\bplayfulness\b",
    r"\bdefensive\b",
    r"\bclosed posture\b",
)
QUESTION_STYLE_BANKS = {
    "QA01": (
        "Ask about the action category using a wording family like 'what action is being performed'.",
        "Ask about the action category using a wording family like 'what kind of movement is shown'.",
        "Ask about the action category using a wording family like 'which activity best describes this motion'.",
        "Ask about the action category using a wording family like 'how would you describe the activity category of this motion'.",
        "Ask about the action category using a wording family like 'what motion is the person carrying out'.",
        "Ask about the action category using a wording family like 'which broad action label best fits this movement'.",
        "Ask about the action category using a wording family like 'what overall action does this sample depict'.",
        "Ask about the action category using a wording family like 'which movement category matches this motion best'.",
        "Ask about the action category using a wording family like 'what kind of action is illustrated by this motion'.",
        "Ask about the action category using a wording family like 'how should this movement be categorized at a high level'.",
    ),
    "QA02": (
        "Ask about temporal structure using a wording family like 'how does the motion unfold over time'.",
        "Ask about temporal structure using a wording family like 'what happens over the course of the motion'.",
        "Ask about temporal structure using a wording family like 'how is the movement organized in time'.",
        "Ask about temporal structure using a wording family like 'what sequence does the action follow'.",
        "Ask about temporal structure using a wording family like 'how does the action progress from start to finish'.",
        "Ask about temporal structure using a wording family like 'what temporal pattern does the motion follow'.",
        "Ask about temporal structure using a wording family like 'what order do the main motion components follow'.",
        "Ask about temporal structure using a wording family like 'how is the action structured across time'.",
        "Ask about temporal structure using a wording family like 'what is the time course of the movement'.",
        "Ask about temporal structure using a wording family like 'how do the main steps of the motion play out'.",
    ),
    "QA03": (
        "Ask about high-level intent using a wording family like 'what is the person trying to do overall'.",
        "Ask about high-level intent using a wording family like 'what overall movement goal is described'.",
        "Ask about high-level intent using a wording family like 'what broader goal does this motion suggest'.",
        "Ask about high-level intent using a wording family like 'what is the main objective of the movement'.",
        "Ask about high-level intent using a wording family like 'what is the person aiming to accomplish through the motion'.",
        "Ask about high-level intent using a wording family like 'what general movement purpose is conveyed here'.",
        "Ask about high-level intent using a wording family like 'what is the movement mainly trying to achieve'.",
        "Ask about high-level intent using a wording family like 'what overall physical goal does the motion indicate'.",
        "Ask about high-level intent using a wording family like 'what is the person attempting to accomplish physically'.",
        "Ask about high-level intent using a wording family like 'what broad movement goal best matches this action'.",
    ),
    "QA04": (
        "Ask about the specific motion detail using a wording family like 'what specific motion detail best characterizes the movement'.",
        "Ask about the specific motion detail using a wording family like 'which concrete detail is most informative'.",
        "Ask about the specific motion detail using a wording family like 'what distinctive detail stands out in the motion'.",
        "Ask about the specific motion detail using a wording family like 'which specific aspect most clearly distinguishes this movement'.",
        "Ask about the specific motion detail using a wording family like 'what concrete movement detail is most diagnostic'.",
        "Ask about the specific motion detail using a wording family like 'which detail best captures what is unique about the motion'.",
        "Ask about the specific motion detail using a wording family like 'what concrete aspect most clearly sets this motion apart'.",
        "Ask about the specific motion detail using a wording family like 'which motion detail provides the clearest distinction here'.",
        "Ask about the specific motion detail using a wording family like 'what observable detail is most characteristic of the movement'.",
        "Ask about the specific motion detail using a wording family like 'which concrete feature most clearly identifies this motion'.",
    ),
}


def clean_unicode_chars(text):
    """Clean problematic Unicode characters that can cause JSON parsing issues"""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "*",
        "\u00b0": " deg", "\u00d7": "x", "\u00f7": "/", "\u2264": "<=",
        "\u2265": ">=", "\u2260": "!=", "\u221e": "infinity",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
        "\u03b5": "epsilon", "\u03b8": "theta", "\u03bb": "lambda", "\u03bc": "mu",
        "\u03c0": "pi", "\u03c3": "sigma", "\u03c4": "tau", "\u03c6": "phi", "\u03c9": "omega",
    }
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    return text


def load_env_file(env_file):
    """Load environment variables from an explicit .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip("\"'")


def validate_questions_payload(payload):
    """Validate generated QA payload before writing it into training data."""
    if not isinstance(payload, dict):
        raise ValueError(f"QA payload must be a dict, got {type(payload).__name__}")

    missing_qa_ids = [qa_id for qa_id in REQUIRED_QA_IDS if qa_id not in payload]
    if missing_qa_ids:
        raise ValueError(f"Missing QA entries: {missing_qa_ids}")

    normalized_payload = {}
    for qa_id in REQUIRED_QA_IDS:
        qa_item = payload[qa_id]
        if not isinstance(qa_item, dict):
            raise ValueError(f"{qa_id} must be a dict, got {type(qa_item).__name__}")

        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        full_marks = qa_item.get("full marks", None)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{qa_id}.question must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{qa_id}.answer must be a non-empty string")
        if full_marks is None:
            raise ValueError(f"{qa_id}.full marks is missing")
        if not isinstance(full_marks, int):
            raise ValueError(f"{qa_id}.full marks must be an int, got {type(full_marks).__name__}")

        normalized_payload[qa_id] = {
            "question": normalize_question_text(question),
            "answer": normalize_answer_text(answer),
            "full marks": full_marks,
        }

    normalized_payload["QA03"]["answer"] = repair_qa03_answer_if_needed(
        normalized_payload["QA03"]["answer"],
        normalized_payload["QA01"]["answer"],
    )

    return normalized_payload


def normalize_question_text(question):
    question = question.strip()
    question = question[0].upper() + question[1:] if question else question
    question = question.rstrip(" .")
    if not question.endswith("?"):
        question = f"{question}?"
    return question


def normalize_answer_text(answer):
    answer = answer.strip()
    answer = answer[0].upper() + answer[1:] if answer else answer
    answer = answer.rstrip()
    if not answer.endswith("."):
        answer = answer.rstrip(".!?").rstrip() + "."
    return answer


def is_qa03_answer_too_speculative(answer):
    for pattern in QA03_BANNED_PATTERNS:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def simplify_activity_answer(activity_answer):
    text = activity_answer.strip().rstrip(".")
    prefixes = (
        "The person is ",
        "A person is ",
        "The activity is ",
        "The activity involves ",
        "The movement is ",
        "The movement involves ",
        "This movement can be categorized as ",
        "This action can be categorized as ",
        "The action is ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip().rstrip(".")


def repair_qa03_answer_if_needed(answer, qa01_answer):
    if not is_qa03_answer_too_speculative(answer):
        return answer

    activity_text = simplify_activity_answer(qa01_answer)
    if activity_text:
        repaired = f"The overall physical goal is to perform {activity_text}."
    else:
        repaired = "The overall physical goal is to perform the described movement."
    return repaired


def build_question_style_guidance(sample_key):
    """Assign deterministic wording families so surface forms diversify across samples."""
    guidance_lines = []
    for qa_id in REQUIRED_QA_IDS:
        style_bank = QUESTION_STYLE_BANKS[qa_id]
        digest = hashlib.md5(f"{sample_key}:{qa_id}".encode("utf-8")).hexdigest()
        style_index = int(digest, 16) % len(style_bank)
        guidance_lines.append(f"- {qa_id}: {style_bank[style_index]}")
    return "\n".join(guidance_lines)


def load_existing_partial_results(partial_output_dir, local_rank):
    """Load an existing partial file so reruns can resume from completed keys."""
    partial_path = os.path.join(partial_output_dir, f'part_{local_rank}.json')
    if not os.path.exists(partial_path):
        return {}, partial_path

    with open(partial_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    if not isinstance(existing_data, dict):
        raise ValueError(f"Partial output must be a dict: {partial_path}")

    return existing_data, partial_path


def generate_qa_payload(client, prompt_for_qa_generation, local_rank):
    """Generate a validated QA payload with bounded retries and backoff."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_for_qa_generation}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            response = response.choices[0].message.content
            response = response.replace('```json', '').replace('```', '')
            response = clean_unicode_chars(response)
            response_json = json.loads(response)
            return validate_questions_payload(response_json)
        except (httpx.TimeoutException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            delay = BASE_RETRY_DELAY_SEC * attempt
            print(f"[Rank {local_rank}] Attempt {attempt}/{MAX_RETRIES} failed: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            unrecoverable_markers = [
                "invalid api key",
                "incorrect api key",
                "authentication",
                "unauthorized",
                "permission",
                "model_not_found",
            ]
            if any(marker in message for marker in unrecoverable_markers):
                raise RuntimeError(f"[Rank {local_rank}] Unrecoverable API error: {exc}") from exc
            delay = BASE_RETRY_DELAY_SEC * attempt
            print(f"[Rank {local_rank}] Attempt {attempt}/{MAX_RETRIES} failed with API error: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError(f"[Rank {local_rank}] Failed to generate QA payload after {MAX_RETRIES} retries: {last_error}")


def make_QAs(source_json, output_dir, split_name, local_rank=0, world_size=1):
    """Generate QA pairs from caption data."""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the environment")

    client = OpenAI(base_url=base_url, api_key=api_key,
        http_client=httpx.Client(base_url=base_url, follow_redirects=True))

    prompt_template_path = os.path.join(os.path.dirname(__file__), 'make_QAs_prompt.txt')
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    partial_output_dir = os.path.join(output_dir, f'{split_name}_QAs')
    os.makedirs(partial_output_dir, exist_ok=True)

    data_new, partial_output_path = load_existing_partial_results(partial_output_dir, local_rank)
    with open(source_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    len_data = len(data)
    data_items = list(data.items())[local_rank * len_data // world_size: (local_rank + 1) * len_data // world_size]
    completed_keys = set(data_new.keys())
    print(f'[Rank {local_rank}] Processing {len(data_items)} items for {split_name} split ({len(completed_keys)} already done)')

    for idx, (key, item) in enumerate(data_items):
        if key in completed_keys:
            print(f'[Rank {local_rank:02d}] Skipping existing QAs ({idx+1:04d}/{len(data_items):04d}) for {split_name}: {key}')
            continue
        item_new = deepcopy(item)
        captions = '\n'.join(item['captions'])
        print(f'[Rank {local_rank:02d}] Generating QAs ({idx+1:04d}/{len(data_items):04d}) for {split_name}')
        prompt_for_qa_generation = prompt_template.format(
            captions=captions,
            question_style_guidance=build_question_style_guidance(key),
        )
        response_json = generate_qa_payload(client, prompt_for_qa_generation, local_rank)

        item_new['questions'] = response_json
        data_new[key] = item_new
        with open(partial_output_path, 'w', encoding='utf-8') as f:
            json.dump(data_new, f, indent=2, ensure_ascii=False)
    return data_new


def worker(local_rank, world_size, train_json, test_json, output_dir):
    """Worker function for parallel processing"""
    if train_json and os.path.exists(train_json):
        make_QAs(train_json, output_dir, 'train', local_rank=local_rank, world_size=world_size)
    if test_json and os.path.exists(test_json):
        make_QAs(test_json, output_dir, 'test', local_rank=local_rank, world_size=world_size)


def main():
    default_env_file = Path(__file__).resolve().parents[2] / "data-generate-pipeline" / ".env"
    parser = argparse.ArgumentParser(description='Generate QA pairs from caption data')
    parser.add_argument('--train_json', type=str, required=True,
                        help='Path to training split JSON file with captions')
    parser.add_argument('--test_json', type=str, required=True,
                        help='Path to test split JSON file with captions')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save generated QA files')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--skip_combine', action='store_true',
                        help='Skip automatic combining of partial results')
    parser.add_argument('--env_file', type=str, default=None,
                        help='Optional path to a .env file containing OpenAI credentials')
    args = parser.parse_args()

    env_file = Path(args.env_file) if args.env_file else default_env_file
    if env_file.exists():
        load_env_file(env_file)
    elif args.env_file:
        raise FileNotFoundError(f".env file not found: {env_file}")

    print(f"Starting QA generation with {args.num_workers} workers...")
    print(f"Train JSON: {args.train_json}")
    print(f"Test JSON: {args.test_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Env file: {env_file if env_file.exists() else 'environment variables'}")

    processes = []
    for local_rank in range(args.num_workers):
        p = mp.Process(target=worker, args=(local_rank, args.num_workers,
                                             args.train_json, args.test_json, args.output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed_ranks = [idx for idx, p in enumerate(processes) if p.exitcode != 0]
    if failed_ranks:
        raise RuntimeError(f"QA generation failed for worker ranks: {failed_ranks}")

    if not args.skip_combine:
        print("\nAll parallel processes completed. Merging results...")
        try:
            import subprocess
            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))
            combine_script = os.path.join(script_dir, 'combine_QAs.py')
            result = subprocess.run([sys.executable, combine_script,
                                    '--output_dir', args.output_dir],
                                  capture_output=True, text=True, check=True)
            print("Results merged successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error combining files: {e}")
            print(f"Error output: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

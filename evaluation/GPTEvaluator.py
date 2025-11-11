import os
import json
import httpx
import re
import time
import random
from typing import Any, List
from tqdm import tqdm
from openai import OpenAI
import openai


prompt = (
    "You are provided with a question, prediction, and reference answers. "
    "Your task is to evaluate the quality of the model's prediction. "

    "Evaluation guidelines:"
    "- Be generous with partial correctness - similar concepts should get credit"
    "- If prediction mentions the same body part or general action type, award partial points"
    "- Don't penalize for different but similar descriptions"
    "- Left-right differences are ignored"
    "- Uncertain language (e.g., 'probably', 'seems like') reduces hallucination points"
    "- Focus on the core action, not minor details"

    "Scoring:"
    "- Correctness: 0-2 points (0=wrong, 1=partially correct, 2=fully correct)"
    "- Hallucination: 0-2 points (0=no hallucination, 2=major hallucination)"

    "Respond in exactly this format:"
    "- Correctness: [0-2] # [brief justification]"
    "- Hallucination: [0-2] # [brief justification]"

    "Keep justifications under 20 words each."
)



def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip()
    normalized = normalized.replace("left", "right").replace("Left", "Right")
    return normalized


def resolve_field(container: dict, keys: List[str]) -> Any:
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def extract_single_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            result = extract_single_text(item)
            if result:
                return result
        return ""
    return normalize_text(value)


def extract_reference_list(value: Any) -> List[str]:
    """Extract multiple references from value - answers should already be lists"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [normalize_text(item) for item in value if item is not None]
    # Handle string values for backward compatibility
    text = normalize_text(value)
    if not text:
        return []
    # Use # as separator for multiple ground truth answers (legacy support)
    segments = text.split("#") if "#" in text else [text]
    return [segment.strip() for segment in segments if segment.strip()]


def extract_prediction_list(value: Any) -> List[str]:
    """Extract multiple predictions from value, supporting # separation for multiple predictions"""
    if value is None:
        return []
    predictions: List[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            predictions.extend(extract_prediction_list(item))
        return predictions
    text = normalize_text(value)
    if not text:
        return []
    # Use # as separator for multiple predictions
    segments = text.split("#") if "#" in text else [text]
    return [segment.strip() for segment in segments if segment.strip()]




def format_as_bullet_list(items: List[str]) -> str:
    if not items:
        return "  - None"
    return "\n".join(f"  - {item}" for item in items)


def extract_explanation(response: str) -> str:
    """Extract the explanation part from GPT response"""
    if not response:
        return ""

    # Extract explanations from the # comments in score lines
    explanations = []

    # Look for score lines with explanations
    correctness_match = re.search(r'- Correctness: \d+ # (.+)', response)
    hallucination_match = re.search(r'- Hallucination: \d+ # (.+)', response)

    if correctness_match:
        explanations.append(correctness_match.group(1).strip())
    if hallucination_match:
        explanations.append(hallucination_match.group(1).strip())

    # If no structured explanations found, extract other content
    if not explanations:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Skip score lines without explanations
            if re.match(r'- (Correctness|Hallucination): \d+$', line):
                continue
            # Skip percentage or overall score lines
            if re.match(r'Overall Score:|Partial Correctness:|.*\d+/\d+.*', line):
                continue
            # Skip empty lines and labels
            if not line or line in ['<Input>', '</Input>', '<Output>', '</Output>']:
                continue
            # Keep other content as explanation
            if line:
                explanations.append(line)

    # Join explanations
    explanation = ' '.join(explanations) if explanations else "No explanation provided"

    # Clean up and limit length
    explanation = explanation.strip('. \n\t')
    if len(explanation) > 150:
        explanation = explanation[:150] + "..."

    return explanation


def evaluate_single_qa(prediction: str, references: List[str], question: str, client, prompt: str, max_retries: int = 2, initial_delay: float = 1.0) -> tuple:
    """Evaluate a single QA pair and return correctness and hallucination scores with retry logic"""
    qa_input = (
        "<Input>\n"
        f"- question: {question}\n"
        f"- prediction: {prediction}\n"
        "- reference answers:\n"
        f"{format_as_bullet_list(references)}\n"
        "</Input>"
    )

    retry_count = 0
    delay = initial_delay

    while retry_count <= max_retries:
        try:
            response_obj = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": qa_input},
                ],
                timeout=30.0  # Add timeout to prevent hanging
            )
            full_response = response_obj.choices[0].message.content
            match_correctness = re.search(r'Correctness: (\d+) #', full_response)
            match_hallucination = re.search(r'Hallucination: (\d+) #', full_response)
            if match_correctness and match_hallucination:
                correctness = int(match_correctness.group(1))
                hallucination = int(match_hallucination.group(1))
                # Extract only the explanation part
                explanation = extract_explanation(full_response)
                return correctness, hallucination, explanation
            else:
                # If response format is incorrect, treat as a failure and retry
                raise ValueError("Response format validation failed")

        except (httpx.RequestError, openai.APITimeoutError, ValueError) as e:
            retry_count += 1
            if retry_count > max_retries:
                # Return default values on failure
                return 0, 2, "API call failed"

            # Exponential backoff with jitter
            jitter = random.uniform(0.8, 1.2)
            sleep_time = delay * jitter
            time.sleep(sleep_time)
            delay *= 2  # Exponential backoff

    # This should never be reached, but just in case
    return 0, 2, "Unknown error"


def evaluate_multiple_predictions(predictions: List[str], references: List[str], question: str, client, prompt: str) -> tuple:
    """Evaluate multiple predictions and take the best score for each metric"""
    best_correctness = 0
    best_hallucination = float('inf')  # Lower is better for hallucination
    best_response = ""

    for pred in predictions:
        if not pred.strip():  # Skip empty predictions
            continue
        correctness, hallucination, response = evaluate_single_qa(pred, references, question, client, prompt)

        # Update best scores - prioritize correctness, then minimize hallucination
        if correctness > best_correctness:
            best_correctness = correctness
            best_hallucination = hallucination
            best_response = response
        elif correctness == best_correctness and hallucination < best_hallucination:
            best_hallucination = hallucination
            best_response = response

    return best_correctness, best_hallucination, best_response


def evaluate(output_path, local_rank=0, world_size=1):
    """Main evaluation function - treats both caption and QA as QA evaluation"""

    # Initialize the client with environment variables
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
        ),
    )

    savedict = {}
    total_correctness = 0
    total_hallucination = 0
    failed_calls = 0
    total_calls = 0
    json_to_evaluate = json.load(open(output_path))
    json_to_evaluate = list(json_to_evaluate.items())[local_rank * len(json_to_evaluate) // world_size: (local_rank + 1) * len(json_to_evaluate) // world_size]

    # Create workers directory if it doesn't exist
    workers_dir = os.path.join(os.path.dirname(output_path), "workers")
    os.makedirs(workers_dir, exist_ok=True)

    # Save to workers directory with results_worker_*.json naming convention
    save_path = os.path.join(workers_dir, f'results_worker_{local_rank}.json')

    index = 0
    for key, item in tqdm(json_to_evaluate) if local_rank == 0 else json_to_evaluate:
        outdict = item
        question = normalize_text(resolve_field(outdict, ["question"]) or "")

        # Extract predictions and references
        predictions = extract_prediction_list(resolve_field(outdict, ["pred", "prediction", "predictions", "pred_caption", "prediction_caption"]))
        references = extract_reference_list(resolve_field(outdict, ["gt", "answer", "answers", "caption", "captions"]))  # Already returns list format

        # Evaluate predictions (if available)
        response = ""
        correctness = 0
        hallucination = 0

        if predictions and references:
            correctness, hallucination, response = evaluate_multiple_predictions(
                predictions, references, question, client, prompt
            )
            total_correctness += correctness
            total_hallucination += hallucination
            total_calls += 1
            if response == "API call failed":
                failed_calls += 1

        # Save results
        savedict['%06d' % index] = {
            'response': response,
            'correctness': correctness,
            'hallucination': hallucination,
            'question': question,
            'predictions': predictions,
            'references': references
        }
        # Save to workers directory
        json.dump(savedict, open(save_path, 'w'), indent=2)
        index += 1

    # Optional: Print failure statistics for this worker
    if failed_calls > 0 and local_rank == 0:
        print(f"Worker {local_rank}: {failed_calls}/{total_calls} API calls failed")

import multiprocessing as mp
import argparse
import glob
import os
from utils import load_evaluation_data, save_evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-based evaluation results")
    parser.add_argument('--evaluation_dir', type=str, required=True,
                       help='Path to evaluation directory containing results_rank_*.json files')
    parser.add_argument('--world_size', type=int, default=10,
                       help='Number of parallel processes for evaluation')
    args = parser.parse_args()

    # Step 1: Load and merge all rank files
    merged_file = os.path.join(args.evaluation_dir, "results.json")
    merged_data = load_evaluation_data(args.evaluation_dir, merged_filename="results.json")

    # Step 2: Run evaluation
    print("Step 2: Running GPT evaluation...")
    processes = []
    for local_rank in range(args.world_size):
        p = mp.Process(target=evaluate, args=(merged_file, local_rank, args.world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Step 3: Summarize results from workers directory
    print("Step 3: Summarizing results...")
    workers_dir = os.path.join(args.evaluation_dir, "workers")

    files = glob.glob(os.path.join(workers_dir, 'results_worker_*.json'))
    if not files:
        files = glob.glob(os.path.join(args.evaluation_dir, 'results_gpt_eval_*.json'))
    correctness = []
    hallucination = []
    evaluation_details = []

    for file in files:
        savedict = json.load(open(file))
        for key, item in savedict.items():
            # Only include items that have actual evaluation results
            if item.get('response'):
                correctness.append(item['correctness'])
                hallucination.append(item['hallucination'])

                # Collect detailed information for analysis
                evaluation_details.append({
                    'sample_id': key,
                    'question': item.get('question', ''),
                    'predictions': item.get('predictions', []),
                    'references': item.get('references', []),
                    'correctness': item['correctness'],
                    'hallucination': item['hallucination']
                })

    # Calculate totals and precision
    total_correctness = sum(correctness)
    total_hallucination = sum(hallucination)

    precision = total_correctness / (total_correctness + total_hallucination) if (total_correctness + total_hallucination) > 0 else 0

    # Additional statistics
    avg_correctness = sum(correctness) / len(correctness) if correctness else 0
    avg_hallucination = sum(hallucination) / len(hallucination) if hallucination else 0

    # Prepare evaluation results
    evaluation_results = []

    for detail in evaluation_details:
        # Calculate precision for this sample
        sample_precision = detail['correctness'] / (detail['correctness'] + detail['hallucination']) if (detail['correctness'] + detail['hallucination']) > 0 else 0

        evaluation_results.append({
            'question': detail['question'],
            'answer': detail['references'],  # List format
            'prediction': detail['predictions'][0] if detail['predictions'] else '',  # Best prediction
            'response': detail.get('response', ''),  # GPT model explanation only
            'corr': detail['correctness'],
            'halu': detail['hallucination'],
            'precise': round(sample_precision, 4)
        })

    # Calculate overall statistics
    overall_precision = total_correctness / (total_correctness + total_hallucination) if (total_correctness + total_hallucination) > 0 else 0

    # Prepare final summary
    summary = {
        'total_samples': len(evaluation_results),
        'stats': {
            'total_corr': total_correctness,
            'total_halu': total_hallucination,
            'avg_corr': round(avg_correctness, 2),
            'avg_halu': round(avg_hallucination, 2),
            'precise': round(overall_precision, 4)
        },
        'results': evaluation_results
    }

    # Save detailed results using save_evaluation_results
    detailed_results_file = os.path.join(args.evaluation_dir, "detailed_results_gpt.json")
    save_evaluation_results({
        'total_samples': len(evaluation_results),
        'results': evaluation_results,
        'detailed_explanations': evaluation_details  # Include full details with explanations
    }, detailed_results_file)

    # Save summary without detailed results (for quick overview)
    summary_for_saving = summary.copy()
    summary_without_results = {k: v for k, v in summary_for_saving.items() if k != 'results'}
    summary_file = os.path.join(args.evaluation_dir, "summary_gpt.json")
    save_evaluation_results(summary_without_results, summary_file)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(evaluation_results)}")

    print(f"\nEvaluation Results:")
    print(f"  - Total Corr: {summary['stats']['total_corr']}")
    print(f"  - Total Halu: {summary['stats']['total_halu']}")
    print(f"  - Avg Corr: {summary['stats']['avg_corr']}")
    print(f"  - Avg Halu: {summary['stats']['avg_halu']}")
    print(f"  - Precision: {summary['stats']['precise']}")

    print(f"\nEvaluation completed successfully!")
    print(f"Detailed results saved to: {detailed_results_file}")
    print(f"Summary saved to: {summary_file}")
    print("="*60)
        
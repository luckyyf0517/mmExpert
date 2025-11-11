import os
import json
import httpx
import re
from typing import Any, List
from tqdm import tqdm
from openai import OpenAI


prompt = (
    "You are provided with 2 sets of QA about the same action. "
    "Your task is to evaluate the quality of the QA of the model. "
    "Identify the key aspects (e.g. categories, temporal orders) of the action. "
    "The captions will also be provided, to help you judge the correctness. "
    "Assign correctness points for each distinct correct attribute. Each correct attribute contributes 1 point. If two actions are the same, the score is 2. "
    "And calculate the percentage of these aspects that are correctly mentioned or partially matched in the model-generated caption." 
    "Partial correctness should be graded on a scale of 0 to 1 depending on accuracy, with similar concepts considered for partial scores."
    "Each aspect contributes equally to the final score, with no extra penalties for repeated inaccuracies of the same attribute. "
    "Also, assign hallucination points for incorrect details in the model output, with 1 point per incorrect attribute. "
    "Repetitive inaccuracies based on one attribute should incur only a single hallucination point. "
    "However, if a detail is described uncertainly or speculatively (e.g., using phrases like 'probably for', 'possibly', 'resemble' or 'indicating'), assign fewer or no hallucination points."
    "Provide your score and a short justification (less than 25 words) in the format of example."
    "If there are redundant actions, please analyze whether it is reasonable. If it's reasonable or normal, it's not an error."
    "Note: there is no need to judge left-right. For example, the action is about left, but the caption says right is OK."
    "You should response in the following format:"
    "Example1: "
    "<Input>"
    "- question: Describe the human motion based on the provided wave signal representing joint movements."
    "- pred: [a person does a golf swing]"
    "- gt: [a person is golf putting a ball; a person holds their hands together below their waist and swings their arm; a person grasps something and then does a hitting motion]"
    "<Output>"
    "- Correctness: 1 # Correct identification of the action category"
    "- Hallucination: 0 # No incorrect details"
    # "Example2: "
    # "<Input>"
    # "- question: Describe the human motion based on the provided wave signal representing joint movements."
    # "- pred: [a person picks something up with their right hand, then bends down and picks something up with their left hand]"
    # "- gt: [the person is bowing over, a person bows twice, step to the side and back, person is making bowing gestures]"
    # "<Output>"
    # "- Correctness: 2 # Correct identification of the action category and temporal order (twice)"
    # "- Hallucination: 0 # No incorrect details"
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
    """Extract multiple references from value, supporting # separation for multiple answers"""
    if value is None:
        return []
    references: List[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            references.extend(extract_reference_list(item))
        return references
    text = normalize_text(value)
    if not text:
        return []
    # Use # as separator for multiple ground truth answers
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


def extract_caption_options(value: Any) -> List[str]:
    """Extract caption options - now treated as another QA input"""
    if value is None:
        return []
    captions: List[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            captions.extend(extract_caption_options(item))
        return captions
    text = normalize_text(value)
    if not text:
        return []
    # Handle various separators but also support # for multiple captions
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
    elif "#" in text:
        parts = [part.strip() for part in text.split("#")]
    else:
        parts = [text]
    return [part for part in parts if part]


def format_as_bullet_list(items: List[str]) -> str:
    if not items:
        return "  - None"
    return "\n".join(f"  - {item}" for item in items)


def extract_explanation(response: str) -> str:
    """Extract only the explanation part from GPT response"""
    # Look for justification/explanation after the scores
    explanation_match = re.search(r'# (.+)', response, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
        # Remove any remaining score lines
        explanation = re.sub(r'- (Correctness|Hallucination): \d+ #.*', '', explanation).strip()
        return explanation
    return ""


def evaluate_single_qa(prediction: str, references: List[str], question: str, client, prompt: str) -> tuple:
    """Evaluate a single QA pair and return correctness and hallucination scores"""
    qa_input = (
        "<Input>\n"
        f"- question: {question}\n"
        f"- prediction: {prediction}\n"
        "- reference answers:\n"
        f"{format_as_bullet_list(references)}\n"
        "</Input>"
    )

    while True:
        response_obj = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": qa_input},
            ],
        )
        full_response = response_obj.choices[0].message.content
        match_correctness = re.search(r'Correctness: (\d+) #', full_response)
        match_hallucination = re.search(r'Hallucination: (\d+) #', full_response)
        if match_correctness and match_hallucination:
            correctness = int(match_correctness.group(1))
            hallucination = int(match_hallucination.group(1))
            break

    # Extract only the explanation part
    explanation = extract_explanation(full_response)
    return correctness, hallucination, explanation


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
    total_correctness_cap = 0
    total_hallucination_cap = 0
    json_to_evaluate = json.load(open(output_path))
    json_to_evaluate = list(json_to_evaluate.items())[local_rank * len(json_to_evaluate) // world_size: (local_rank + 1) * len(json_to_evaluate) // world_size]
    save_path = os.path.join(os.path.dirname(output_path), f'results_gpt_eval_{local_rank}.json')

    index = 0
    for key, item in tqdm(json_to_evaluate) if local_rank == 0 else json_to_evaluate:
        outdict = item
        question = normalize_text(resolve_field(outdict, ["question"]) or "")

        # Extract caption-related data (treated as QA input)
        caption_predictions = extract_prediction_list(resolve_field(outdict, ["pred_caption", "prediction_caption"]))
        caption_references = extract_caption_options(resolve_field(outdict, ["caption", "captions"]))

        # Extract QA-related data
        qa_predictions = extract_prediction_list(resolve_field(outdict, ["pred", "prediction", "predictions"]))
        qa_references = extract_reference_list(resolve_field(outdict, ["gt", "answer", "answers"]))  # Already returns list format

        # Evaluate caption predictions (if available)
        response_cap = ""
        correctness_cap = 0
        hallucination_cap = 0

        if caption_predictions and caption_references:
            correctness_cap, hallucination_cap, response_cap = evaluate_multiple_predictions(
                caption_predictions, caption_references, question, client, prompt
            )
            total_correctness_cap += correctness_cap
            total_hallucination_cap += hallucination_cap

        # Evaluate QA predictions (if available)
        response = ""
        correctness = 0
        hallucination = 0

        if qa_predictions and qa_references:
            correctness, hallucination, response = evaluate_multiple_predictions(
                qa_predictions, qa_references, question, client, prompt
            )
            total_correctness += correctness
            total_hallucination += hallucination

        # Save results
        savedict['%06d' % index] = {
            'response': response,
            'response_cap': response_cap,
            'correctness': correctness,
            'hallucination': hallucination,
            'correctness_cap': correctness_cap,
            'hallucination_cap': hallucination_cap,
            'question': question,
            'qa_predictions': qa_predictions,
            'qa_references': qa_references,
            'caption_predictions': caption_predictions,
            'caption_references': caption_references
        }
        json.dump(savedict, open(save_path, 'w'), indent=2)
        index += 1

import multiprocessing as mp
import argparse
import glob
import os
from utils import merge_rank_files, load_evaluation_data, save_evaluation_results


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

    # Step 3: Summarize results
    print("Step 3: Summarizing results...")
    files = glob.glob(os.path.join(args.evaluation_dir, 'results_gpt_eval_*.json'))
    correctness = []
    hallucination = []
    correctness_cap = []
    hallucination_cap = []
    evaluation_details = []

    for file in files:
        savedict = json.load(open(file))
        for key, item in savedict.items():
            # Only include items that have actual evaluation results (skip empty ones)
            if item.get('response') or item.get('response_cap'):
                correctness.append(item['correctness'])
                hallucination.append(item['hallucination'])
                correctness_cap.append(item['correctness_cap'])
                hallucination_cap.append(item['hallucination_cap'])

                # Collect detailed information for analysis
                evaluation_details.append({
                    'sample_id': key,
                    'question': item.get('question', ''),
                    'qa_predictions': item.get('qa_predictions', []),
                    'qa_references': item.get('qa_references', []),
                    'caption_predictions': item.get('caption_predictions', []),
                    'caption_references': item.get('caption_references', []),
                    'scores': {
                        'qa_correctness': item['correctness'],
                        'qa_hallucination': item['hallucination'],
                        'caption_correctness': item['correctness_cap'],
                        'caption_hallucination': item['hallucination_cap']
                    }
                })

    # Calculate totals and precision
    total_correctness = sum(correctness)
    total_hallucination = sum(hallucination)
    total_correctness_cap = sum(correctness_cap)
    total_hallucination_cap = sum(hallucination_cap)

    precision = total_correctness / (total_correctness + total_hallucination) if (total_correctness + total_hallucination) > 0 else 0
    precision_cap = total_correctness_cap / (total_correctness_cap + total_hallucination_cap) if (total_correctness_cap + total_hallucination_cap) > 0 else 0

    # Additional statistics
    avg_correctness = sum(correctness) / len(correctness) if correctness else 0
    avg_hallucination = sum(hallucination) / len(hallucination) if hallucination else 0
    avg_correctness_cap = sum(correctness_cap) / len(correctness_cap) if correctness_cap else 0
    avg_hallucination_cap = sum(hallucination_cap) / len(hallucination_cap) if hallucination_cap else 0

    # Prepare evaluation results with new field structure
    evaluation_results = []

    for detail in evaluation_details:
        # Calculate precision for this sample
        sample_precision = detail['scores']['qa_correctness'] / (detail['scores']['qa_correctness'] + detail['scores']['qa_hallucination']) if (detail['scores']['qa_correctness'] + detail['scores']['qa_hallucination']) > 0 else 0

        evaluation_results.append({
            'question': detail['question'],
            'answer': detail['qa_references'],  # List format
            'prediction': detail['qa_predictions'][0] if detail['qa_predictions'] else '',  # Best prediction
            'response': detail.get('response', ''),  # GPT model explanation only
            'corr': detail['scores']['qa_correctness'],
            'halu': detail['scores']['qa_hallucination'],
            'precise': round(sample_precision, 4)
        })

    # Calculate overall statistics
    overall_precision = total_correctness / (total_correctness + total_hallucination) if (total_correctness + total_hallucination) > 0 else 0
    caption_precision = total_correctness_cap / (total_correctness_cap + total_hallucination_cap) if (total_correctness_cap + total_hallucination_cap) > 0 else 0

    # Prepare final summary with new structure
    summary = {
        'total_samples': len(evaluation_results),
        'qa_stats': {
            'total_corr': total_correctness,
            'total_halu': total_hallucination,
            'avg_corr': round(avg_correctness, 2),
            'avg_halu': round(avg_hallucination, 2),
            'precise': round(overall_precision, 4)
        },
        'caption_stats': {
            'total_corr': total_correctness_cap,
            'total_halu': total_hallucination_cap,
            'avg_corr': round(avg_correctness_cap, 2),
            'avg_halu': round(avg_hallucination_cap, 2),
            'precise': round(caption_precision, 4)
        },
        'overall_stats': {
            'total_corr': total_correctness + total_correctness_cap,
            'total_halu': total_hallucination + total_hallucination_cap,
            'precise': round((total_correctness + total_correctness_cap) / ((total_correctness + total_hallucination) + (total_correctness_cap + total_hallucination_cap)), 4) if ((total_correctness + total_hallucination) + (total_correctness_cap + total_hallucination_cap)) > 0 else 0
        },
        'results': evaluation_results
    }

    # Save summary to file
    summary_file = os.path.join(args.evaluation_dir, "evaluation_summary.json")
    json.dump(summary, open(summary_file, 'w'), indent=2)

    # Print results with new field structure
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(evaluation_results)}")

    print(f"\nQA Evaluation:")
    print(f"  - Total Corr: {summary['qa_stats']['total_corr']}")
    print(f"  - Total Halu: {summary['qa_stats']['total_halu']}")
    print(f"  - Avg Corr: {summary['qa_stats']['avg_corr']}")
    print(f"  - Avg Halu: {summary['qa_stats']['avg_halu']}")
    print(f"  - Precision: {summary['qa_stats']['precise']}")

    print(f"\nCaption Evaluation:")
    print(f"  - Total Corr: {summary['caption_stats']['total_corr']}")
    print(f"  - Total Halu: {summary['caption_stats']['total_halu']}")
    print(f"  - Avg Corr: {summary['caption_stats']['avg_corr']}")
    print(f"  - Avg Halu: {summary['caption_stats']['avg_halu']}")
    print(f"  - Precision: {summary['caption_stats']['precise']}")

    print(f"\nOverall Performance:")
    print(f"  - Total Corr: {summary['overall_stats']['total_corr']}")
    print(f"  - Total Halu: {summary['overall_stats']['total_halu']}")
    print(f"  - Precision: {summary['overall_stats']['precise']}")

    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {summary_file}")
    print("="*60)
        
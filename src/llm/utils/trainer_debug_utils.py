from typing import Tuple

import torch

from src.logger import log_message


def decode_question_answer(tokenizer, batch, index: int = 0) -> Tuple[str, str]:
    """Decode question and answer strings from a batch."""
    question = ""
    answer = ""

    questions = batch.get('questions')
    if questions:
        if isinstance(questions, (list, tuple)):
            if index < len(questions):
                question = questions[index]
        else:
            question = str(questions)

    answers = batch.get('answers')
    if answers:
        if isinstance(answers, (list, tuple)):
            if index < len(answers):
                answer = answers[index]
        else:
            answer = str(answers)

    if not question and 'input_ids' in batch:
        decoded_input = tokenizer.decode(
            batch['input_ids'][index].tolist(),
            skip_special_tokens=True
        )
        if '<|end|>' in decoded_input:
            parts = decoded_input.split('<|end|>')
            question = parts[-2].strip() if len(parts) > 1 else decoded_input.strip()
        else:
            question = decoded_input.strip()

    if not answer and 'labels' in batch and batch['labels'] is not None:
        labels = batch['labels'][index]
        valid_mask = labels != -100
        if valid_mask.any():
            valid_tokens = labels[valid_mask]
            answer = tokenizer.decode(
                valid_tokens.tolist(),
                skip_special_tokens=True
            ).strip()

    return question, answer


def predict_text_from_logits(tokenizer, logits, labels, index: int = 0) -> str:
    """Convert logits to decoded assistant text."""
    if logits is None:
        return ""

    token_ids = logits.argmax(dim=-1)[index]
    if labels is not None:
        label_row = labels[index]
        mask = label_row != -100
        if mask.any():
            token_ids = token_ids[mask]
        else:
            # Fallback: take a window from the end (likely assistant response)
            window = min(64, token_ids.size(-1))
            token_ids = token_ids[-window:]

    decoded = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
    return decoded


def log_sample_output(question: str, prediction: str, answer: str, prefix: str = ""):
    """Standardized logging of question/prediction/answer triples."""
    log_message(f"{prefix} SAMPLE" if prefix else "SAMPLE", "", color="cyan")
    log_message("QUESTION", question, color="blue")
    log_message("PREDICTION", prediction, color="green")
    log_message("GROUND_TRUTH", answer, color="yellow")
    print("-" * 50)


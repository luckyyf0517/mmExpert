from typing import Tuple

import torch

from src.logger import log_message


def predict_text_from_logits(tokenizer, logits, labels, index: int = 0) -> str:
    """Convert logits to decoded assistant text aligned with shifted labels."""
    if logits is None:
        return ""

    token_predictions = logits.argmax(dim=-1)[index]

    if labels is not None:
        label_row = labels[index]
        if label_row.numel() > 1:
            shift_logits = logits[index, :-1, :]
            shift_labels = label_row[1:]
            mask = shift_labels != -100
            if mask.any():
                token_predictions = shift_logits.argmax(dim=-1)[mask]
            else:
                token_predictions = shift_logits.argmax(dim=-1)
        else:
            token_predictions = token_predictions[:-1]
    else:
        token_predictions = token_predictions[:-1]

    if token_predictions.numel() == 0:
        return ""

    decoded = tokenizer.decode(token_predictions.tolist(), skip_special_tokens=False)
    return decoded


def log_sample_output(question: str, prediction: str, answer: str, prefix: str = ""):
    """Standardized logging of question/prediction/answer triples."""
    log_message(f"{prefix} SAMPLE" if prefix else "SAMPLE", "", color="cyan")
    log_message("QUESTION", question, color="blue")
    
    # Format GROUND_TRUTH: split by '#' and display on separate lines if multiple
    if '#' in answer:
        ground_truth_lines = [line.strip() for line in answer.split('#') if line.strip()]
        # Print remaining lines with indentation to align with GROUND_TRUTH content
        # [GROUND_TRUTH] tag is approximately 16 characters, so use 16 spaces for alignment
        for line in ground_truth_lines:
            print(f" -  {line}")
    else:
        log_message("GROUND_TRUTH", answer, color="green")
    
    log_message("PREDICTION", prediction, color="yellow")
    print("-" * 50)


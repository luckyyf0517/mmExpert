from typing import Tuple

import torch

from src.logger import log_message


def decode_question_answer(tokenizer, batch, index: int = 0) -> Tuple[str, str]:
    """
    Decode question and answer strings from a batch.
    Format: 
    <|syetem|> ... <|end|>
    <|user|> ... <|end|>
    <|assistant|> ... <|end|>
    """
    
    question = ""
    answer = ""

    decoded_input = tokenizer.decode(
        batch['input_ids'][index].tolist(),
        skip_special_tokens=False
    )

    parts = decoded_input.split('<|end|>')
    if len(parts) >= 3:
        question_raw = parts[1].strip()
        answer_raw = parts[2].strip()
    else:
        question_raw = decoded_input
        answer_raw = ""

    def _normalize_whitespace(text: str) -> str:
        return " ".join(text.split())

    wave_start = "<wave_bos>"
    wave_end = "<wave_eos>"
    if wave_start in question_raw and wave_end in question_raw:
        prefix, rest = question_raw.split(wave_start, 1)
        _, suffix = rest.split(wave_end, 1)
        question_raw = (prefix + suffix).strip()

    question_raw = _normalize_whitespace(question_raw)
    answer_raw = _normalize_whitespace(answer_raw)

    if question_raw.startswith("<|user|>"):
        question_raw = question_raw[len("<|user|>"):].strip()

    question = f"{question_raw} <|end|>" if question_raw else ""
    answer = f"{answer_raw} <|end|>" if answer_raw else ""

    return question, answer


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
    log_message("GROUND_TRUTH", answer, color="green")
    log_message("PREDICTION", prediction, color="yellow")
    print("-" * 50)


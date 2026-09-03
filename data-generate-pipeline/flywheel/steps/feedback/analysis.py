"""Classifier-feedback artifact loading and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClassifierFeedbackAnalysis:
    """Normalized classifier-feedback report used by Step 4."""

    overall_accuracy: float = 0.0
    total_evaluated: int = 0
    total_misclassified: int = 0
    common_issues: list[str] = field(default_factory=list)
    new_constraints: list[str] = field(default_factory=list)
    modified_constraints: list[str] = field(default_factory=list)
    style_adjustments: dict[str, Any] = field(default_factory=dict)
    per_action_summary: dict[str, Any] = field(default_factory=dict)
    source_analysis_file: str = ""
    source_results_file: str = ""


def load_classifier_feedback(
    analysis_file: str | Path,
    test_results_file: str | Path | None = None,
) -> ClassifierFeedbackAnalysis:
    """Load and normalize classifier-feedback artifacts from JSON files."""
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Classifier feedback analysis file not found: {analysis_path}"
        )

    analysis_data = _load_json(analysis_path)
    test_results_data: dict[str, Any] = {}
    if test_results_file is not None:
        test_path = Path(test_results_file)
        if not test_path.exists():
            raise FileNotFoundError(
                f"Classifier test-results file not found: {test_path}"
            )
        test_results_data = _load_json(test_path)
    else:
        test_path = None

    report = ClassifierFeedbackAnalysis()
    report.source_analysis_file = str(analysis_path)
    report.source_results_file = str(test_path) if test_path else ""

    report.overall_accuracy = _normalize_accuracy(
        _find_first_value(
            analysis_data,
            test_results_data,
            keys=("overall_accuracy", "accuracy", "test_accuracy", "acc"),
        )
    )
    report.total_evaluated = int(
        _find_first_value(
            analysis_data,
            test_results_data,
            keys=("total_evaluated", "total_samples", "num_samples", "sample_count"),
            default=0,
        )
        or 0
    )
    report.total_misclassified = int(
        _find_first_value(
            analysis_data,
            test_results_data,
            keys=("total_misclassified", "misclassified_count", "num_errors", "error_count"),
            default=0,
        )
        or 0
    )
    if report.total_misclassified == 0 and report.total_evaluated and report.overall_accuracy > 0:
        report.total_misclassified = round(
            report.total_evaluated * (1.0 - report.overall_accuracy)
        )

    report.common_issues = _coerce_str_list(
        _find_first_value(
            analysis_data,
            keys=("common_issues", "error_patterns", "top_failure_modes", "issues"),
            default=[],
        )
    )
    if not report.common_issues:
        report.common_issues = _coerce_str_list(
            _find_first_value(
                analysis_data,
                keys=("summary", "analysis", "diagnosis"),
                default="",
            )
        )

    report.new_constraints = _coerce_str_list(
        _find_first_value(
            analysis_data,
            keys=(
                "new_constraints",
                "improvement_rules",
                "prompt_rules",
                "recommended_constraints",
                "refinement_rules",
            ),
            default=[],
        )
    )
    report.modified_constraints = _coerce_str_list(
        _find_first_value(
            analysis_data,
            keys=("modified_constraints",),
            default=[],
        )
    )
    style_adjustments = _find_first_value(
        analysis_data,
        keys=("style_adjustments", "prompt_template_updates", "generation_hints"),
        default={},
    )
    report.style_adjustments = style_adjustments if isinstance(style_adjustments, dict) else {}

    per_action = _find_first_value(
        analysis_data,
        test_results_data,
        keys=("per_action_summary", "per_class_summary", "per_action"),
        default={},
    )
    report.per_action_summary = per_action if isinstance(per_action, dict) else {}

    return report


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def _find_first_value(
    *objects: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    for obj in objects:
        value = _find_in_object(obj, keys)
        if value is not None:
            return value
    return default


def _find_in_object(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = _find_in_object(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_in_object(item, keys)
            if found is not None:
                return found
    return None


def _normalize_accuracy(value: Any) -> float:
    try:
        accuracy = float(value)
    except (TypeError, ValueError):
        return 0.0
    if accuracy > 1.0:
        accuracy /= 100.0
    return max(0.0, min(accuracy, 1.0))


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = (
                    item.get("issue")
                    or item.get("rule")
                    or item.get("text")
                    or json.dumps(item, ensure_ascii=False)
                )
            else:
                text = str(item)
            text = text.strip()
            if text:
                items.append(text)
        return items
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    text = str(value).strip()
    return [text] if text else []

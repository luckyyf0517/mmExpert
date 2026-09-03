"""Config update and classifier-feedback report persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ...config import FlywheelConfig
from ...logging_utils import get_console, get_logger
from .analysis import ClassifierFeedbackAnalysis

logger = get_logger()


def update_config(
    config: FlywheelConfig,
    analysis: ClassifierFeedbackAnalysis,
) -> FlywheelConfig:
    """Write classifier-feedback results back to config for the next round."""
    data = dict(config.data)

    current_constraints = list(data.get("constraints", []))
    for constraint in analysis.new_constraints:
        if constraint and constraint not in current_constraints:
            current_constraints.append(constraint)
    data["constraints"] = current_constraints

    history_entry = {
        "round": config.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classifier_accuracy": round(analysis.overall_accuracy, 4),
        "total_evaluated": analysis.total_evaluated,
        "total_misclassified": analysis.total_misclassified,
        "common_issues": analysis.common_issues[:10],
        "new_constraints_added": len(analysis.new_constraints),
        "style_adjustments": analysis.style_adjustments,
        "per_action": analysis.per_action_summary,
        "source_analysis_file": analysis.source_analysis_file,
        "source_results_file": analysis.source_results_file,
    }
    data.setdefault("feedback_history", []).append(history_entry)

    config.save(data)
    next_version, next_cfg = config.create_next_round_config(data)

    console = get_console()
    console.print(f"  Config updated. Next round: [bold]{next_version}[/]")
    console.print(f"  New constraints: [bold]{len(analysis.new_constraints)}[/]")
    return next_cfg


def save_feedback_report(
    report: ClassifierFeedbackAnalysis,
    output_dir: Path,
) -> None:
    """Save normalized classifier feedback report to disk."""
    report_path = output_dir / "classifier_feedback.json"
    report_data = {
        "overall_accuracy": report.overall_accuracy,
        "total_evaluated": report.total_evaluated,
        "total_misclassified": report.total_misclassified,
        "per_action_summary": report.per_action_summary,
        "common_issues": report.common_issues,
        "new_constraints": report.new_constraints,
        "modified_constraints": report.modified_constraints,
        "style_adjustments": report.style_adjustments,
        "source_analysis_file": report.source_analysis_file,
        "source_results_file": report.source_results_file,
    }
    report_path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Feedback report saved to %s", report_path)

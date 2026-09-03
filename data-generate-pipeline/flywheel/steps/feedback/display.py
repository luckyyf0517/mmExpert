"""Rich-based display for classifier-feedback step summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...logging_utils import print_table, print_tree, summary_panel
from .analysis import ClassifierFeedbackAnalysis


@dataclass
class ClassifierFeedbackResult:
    """Result of the classifier-feedback step."""

    overall_accuracy: float = 0.0
    total_evaluated: int = 0
    total_misclassified: int = 0
    constraints_added: int = 0
    analysis: ClassifierFeedbackAnalysis = field(default_factory=ClassifierFeedbackAnalysis)
    config_updated: bool = False
    elapsed: float = 0.0
    analysis_file: str = ""
    test_results_file: str = ""


def print_summary(
    result: ClassifierFeedbackResult,
    config: Any,
) -> None:
    """Print the final classifier-feedback summary."""
    analysis = result.analysis

    metric_rows = [
        ("Accuracy", f"{result.overall_accuracy * 100:.2f}%"),
        ("Evaluated samples", str(result.total_evaluated)),
        ("Misclassified samples", str(result.total_misclassified)),
        ("New constraints", str(result.constraints_added)),
    ]
    print_table(
        "Classifier Feedback Metrics",
        ["Metric", "Value"],
        metric_rows,
        styles=["cyan", "green"],
    )

    if analysis.common_issues:
        issue_rows = [
            (str(i + 1), issue)
            for i, issue in enumerate(analysis.common_issues[:10])
        ]
        print_table(
            "Common Issues (ranked)",
            ["#", "Issue"],
            issue_rows,
            styles=["dim", "red"],
        )

    improvements: dict[str, Any] = {}
    if analysis.new_constraints:
        improvements["New Constraints"] = analysis.new_constraints[:5]
    if analysis.modified_constraints:
        improvements["Modified Constraints"] = analysis.modified_constraints[:5]
    if analysis.style_adjustments:
        improvements["Style Adjustments"] = analysis.style_adjustments
    if improvements:
        print_tree("Suggested Improvements", tree_dict=improvements)

    next_round = config.version
    summary_lines = [
        f"Classifier acc.:   {result.overall_accuracy * 100:.2f}%",
        f"Evaluated samples: [bold]{result.total_evaluated}[/]",
        f"Misclassified:     {result.total_misclassified}",
        f"New constraints:   {result.constraints_added}",
        f"Config updated:    {'[green]yes[/]' if result.config_updated else '[red]no[/]'}",
        f"Next round:        [bold cyan]{next_round}[/]",
        f"Elapsed:           {result.elapsed:.1f}s",
    ]
    if result.analysis_file:
        summary_lines.append(f"Analysis JSON:     {result.analysis_file}")
    if result.test_results_file:
        summary_lines.append(f"Results JSON:      {result.test_results_file}")
    if analysis.common_issues:
        summary_lines.append(f"Top issue:         [red]{analysis.common_issues[0]}[/]")

    summary_panel("\n".join(summary_lines), title="Step 4 Complete")

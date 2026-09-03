"""Rich display helpers for prompt generation step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...logging_utils import get_console, print_table

if TYPE_CHECKING:
    from ...config import FlywheelConfig
    from .generator import PromptGenResult


def print_config_summary(
    config: FlywheelConfig,
) -> None:
    """Print configuration summary."""
    console = get_console()
    console.print(
        f"Planner: [bold]{config.planner_model}[/]  |  "
        f"Worker: [bold]{config.worker_model}[/]  |  "
        f"Target: [bold]{config.total_count}[/]  |  "
        f"Batch Size: [bold]{config.batch_size}[/]"
    )

    rows = [
        ("Task Scope", config.tasks),
        ("Constraints", str(len(config.constraints))),
    ]
    print_table(
        "Task Configuration",
        ["Field", "Value"],
        rows,
        styles=["cyan", None],
    )


def print_results_table(
    result: PromptGenResult,
    elapsed: float,
) -> None:
    """Print results table and summary."""
    console = get_console()
    rows = [
        ("Planner strategies", result.planned_strategies),
        ("Planner subtasks", result.planned_subtasks),
        ("Worker batches", result.total_batches),
        ("Existing prompts", result.existing_prompts),
        ("New prompts", result.total_prompts),
        ("Final prompts", result.final_prompt_count),
        ("Duplicates removed", result.duplicates_removed),
        ("Repair batches", result.repair_batches),
    ]
    print_table(
        "Generation Results",
        ["Metric", "Value"],
        rows,
        styles=["cyan", "green"],
    )

    console.print(
        f"Total prompts: [bold]{result.final_prompt_count}[/]  |  "
        f"Tokens: {result.total_input_tokens:,}in/{result.total_output_tokens:,}out  |  "
        f"Cost: ${result.total_cost:.4f}  |  Time: {elapsed:.1f}s"
    )

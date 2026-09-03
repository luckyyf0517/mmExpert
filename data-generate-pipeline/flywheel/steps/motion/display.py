"""Rich output display for motion generation step."""

from __future__ import annotations

import subprocess
import sys

from ...logging_utils import (
    get_console,
    print_table,
)
from ...config import FlywheelConfig
from .generator import MotionGenResult


def print_gpu_config(num_gpus: int, batch_size: int) -> None:
    """Print GPU configuration."""
    console = get_console()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.cuda.device_count()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        available_gpus = int(lines[0]) if lines else 0
        gpu_name = lines[1] if len(lines) > 1 else "N/A"
    except Exception:
        available_gpus = 0
        gpu_name = "N/A"

    gpu_info = f"{gpu_name}" if gpu_name != "N/A" else ""
    console.print(f"GPUs: [bold]{available_gpus}[/] available, [bold]{num_gpus}[/] used"
                  + (f" ([dim]{gpu_info}[/])" if gpu_info else "")
                  + f"  |  Batch: [bold]{batch_size}[/]")

    if available_gpus == 0:
        console.print("[yellow]Warning:[] No GPUs detected.")


def print_results_table(
    result: MotionGenResult, config: FlywheelConfig
) -> None:
    """Print per-action results and summary."""
    stats = result.stats
    console = get_console()

    # Per-action table
    rows: list[tuple[str, ...]] = []
    for action in config.actions:
        action_stats = stats.per_action.get(action.id, {})
        rows.append(
            (
                action.id,
                action.name,
                str(action_stats.get("prompts", 0)),
                str(action_stats.get("joints", 0)),
                str(action_stats.get("bvh", 0)),
                str(action_stats.get("video", 0)),
            )
        )

    if rows:
        print_table(
            "Motion Generation Results",
            ["Action ID", "Action", "Prompts", "Joints", "BVH", "Videos"],
            rows,
            styles=["cyan", None, "green", "bold", "bold", "bold"],
        )

    # Summary
    error_info = f"  [red]Errors: {len(stats.errors)}[/]\n" if stats.errors else ""
    console.print(
        f"Total motions: [bold]{stats.joints_count}[/]  |  "
        f"BVH: {stats.bvh_count}  |  Videos: {stats.video_count}  |  "
        f"Time: {result.gen_elapsed:.1f}s\n{error_info}"
    )

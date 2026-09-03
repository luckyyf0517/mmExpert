"""Command: status -- show current flywheel status dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flywheel.config import FlywheelConfig, load_env
from flywheel.logging_utils import get_console, print_data_flow, print_table
from flywheel.path_manager import PathManager
from flywheel.state import load_state
from flywheel.steps.motion.backends import resolve_motion_backend

from .run_cmd import resolve_config


def cmd_status(args: object) -> None:
    """Show current flywheel status dashboard."""
    console = get_console()
    args._print_banner()
    load_env()

    output_dir: Path = args.output_dir  # type: ignore[attr-defined]
    saved = load_state(output_dir)

    # Header info
    try:
        config = resolve_config(args)
        console.print(
            f"Output: [bold]{output_dir}[/]  |  "
            f"Config: [dim]{config.path}[/]  |  "
            f"Total Count: {config.total_count}  |  Model: {config.model}"
        )
    except FileNotFoundError:
        console.print(f"Output: [bold]{output_dir}[/]  [yellow]No config found.[/]")

    console.print()

    if not saved:
        console.print("[yellow]No flywheel state found. Run 'init' first.[/]")
        return

    completed_steps = saved.get("completed_steps", [])
    rounds_data: dict[str, Any] = saved.get("rounds", {})

    # Data flow diagram
    print_data_flow(console, tuple(completed_steps))

    # Round status table
    if rounds_data:
        rows = []
        for round_key in sorted(rounds_data.keys()):
            rd = rounds_data[round_key]
            steps_done = [s for s in rd if rd[s].get("status") == "completed"]
            steps_failed = [s for s in rd if rd[s].get("status") == "failed"]
            status = (
                "completed" if len(steps_done) == 4 else (
                    "failed" if steps_failed else "in progress"
                )
            )
            rows.append((
                round_key,
                str(len(steps_done)),
                "/ 4",
                status,
                rd.get("step1", {}).get("timestamp", "-"),
            ))
        print_table(
            "Round History",
            ["Round", "Steps", "", "Status", "Started"],
            rows,
            styles=["cyan", "green", "dim", "bold", "dim"],
        )

    # Cumulative statistics
    cumulative = _collect_cumulative_stats(output_dir, rounds_data)
    if cumulative:
        rows = [(k, str(v)) for k, v in cumulative.items()]
        print_table("Cumulative Statistics", ["Metric", "Value"], rows, styles=["cyan", "green"])

    # Quality trend
    quality_trend = _collect_quality_trend(rounds_data)
    if quality_trend:
        console.print("\n[bold]Quality Trend[/]")
        for round_key, accuracy in quality_trend:
            bar_len = int(accuracy * 20)
            bar = "[green]" + "█" * bar_len + "[/][dim]" + "░" * (20 - bar_len) + "[/]"
            console.print(f"  {round_key}: {bar}  {accuracy * 100:.2f}%")

    # Per-round directory trees
    for round_key in sorted(rounds_data.keys()):
        paths = PathManager(output_dir, round_key)
        if paths.version_dir.exists():
            console.print(f"\n[bold]{round_key}/[/]")
            console.print(paths.tree_summary())

    console.print()


def _collect_cumulative_stats(
    output_dir: Path,
    rounds_data: dict[str, Any],
) -> dict[str, int]:
    total_prompts = 0
    total_motions = 0
    total_videos = 0

    for round_key in rounds_data:
        paths = PathManager(output_dir, round_key)
        if paths.version_dir.exists():
            for prompt_file in paths.prompts_dir.rglob("*.txt"):
                content = prompt_file.read_text(encoding="utf-8").strip()
                if content:
                    total_prompts += len([line for line in content.splitlines() if line.strip()])
            active_joints_dir = _active_joints_dir(paths)
            total_motions += sum(1 for _ in active_joints_dir.glob("*.npy"))
            total_videos += sum(1 for _ in paths.video_dir.glob("*.mp4"))

    return {
        "Total prompts": total_prompts,
        "Total joints files": total_motions,
        "Total videos": total_videos,
        "Rounds tracked": len(rounds_data),
    }


def _active_joints_dir(paths: PathManager) -> Path:
    """Return the manifest/config-selected joints dir for a round."""
    if paths.config_path.exists():
        config = FlywheelConfig(paths.config_path)
        config.load()
    else:
        config = _EmptyConfig()
    try:
        return resolve_motion_backend(config, paths, prefer_manifest=True).joints_dir
    except ValueError:
        return paths.joints_dir


class _EmptyConfig:
    data: dict[str, Any] = {}


def _collect_quality_trend(
    rounds_data: dict[str, Any],
) -> list[tuple[str, float]]:
    trend = []
    for round_key in sorted(rounds_data.keys()):
        rd = rounds_data[round_key]
        step4 = rd.get("step4", {})
        result = step4.get("result", {})
        accuracy = result.get("overall_accuracy", 0.0)
        if accuracy > 0:
            trend.append((round_key, accuracy))
    return trend

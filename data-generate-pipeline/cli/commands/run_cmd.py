"""Command: run -- execute the full flywheel loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flywheel.config import FlywheelConfig, config_to_state, load_env
from flywheel.logging_utils import (
    error_panel,
    get_console,
    print_table,
    summary_panel,
)
from flywheel.path_manager import PathManager
from flywheel.state import load_state
from flywheel.types import RoundSummary, StepStatus
from utils.gpu import detect_gpus


# ---------------------------------------------------------------------------
# Config resolution (shared with other commands)
# ---------------------------------------------------------------------------


def resolve_config(args: object) -> FlywheelConfig:
    """Resolve or create the flywheel configuration."""
    output_dir: Path = args.output_dir  # type: ignore[attr-defined]
    config_arg = getattr(args, "config", None)
    round_arg = getattr(args, "round", None)

    if config_arg is not None:
        config_path = Path(config_arg)
    elif round_arg is not None:
        config_path = output_dir / f"round_{round_arg}" / "info.json"
    else:
        config_path = _find_latest_config(output_dir)

    if config_path is not None and config_path.exists():
        config = FlywheelConfig(config_path)
        config.load()
        return config

    raise FileNotFoundError(
        "No config.json found. Run 'init' first or specify --config."
    )


def _find_latest_config(output_dir: Path) -> Path | None:
    """Find the most recent round_*/info.json."""
    if not output_dir.exists():
        return None

    round_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else -1,
    )
    for rd in reversed(round_dirs):
        cfg = rd / "info.json"
        if cfg.exists():
            return cfg
    return None


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------


def _print_round_header(round_num: int, version: str, max_rounds: int) -> None:
    console = get_console()
    console.print()
    console.rule(f"[bold]Round {round_num}/{max_rounds}")
    console.print()


def _print_round_summary(round_num: int, summary: RoundSummary, console: Any) -> None:
    console.rule(f"[bold]Round {round_num} Summary")

    step_names = {
        "step1": "Prompt Generation",
        "step2": "Motion Generation",
        "step3": "mmWave Simulation",
        "step4": "Classifier Feedback",
    }
    rows = []
    for step_key, status in summary.step_statuses.items():
        name = step_names.get(step_key, step_key)
        style = "green" if status == StepStatus.COMPLETED else (
            "red" if status == StepStatus.FAILED else "yellow"
        )
        rows.append((name, f"[{style}]{status.value}[//]"))

    print_table(f"Round {round_num} Steps", ["Step", "Status"], rows, styles=["cyan", None])

    stats_rows = [
        ("Prompts generated", str(summary.total_prompts)),
        ("Motions generated", str(summary.total_motions)),
        ("mmWave files generated", str(summary.total_simulations)),
        (
            "Classifier accuracy",
            f"{summary.classifier_accuracy * 100:.2f}%"
            if summary.classifier_accuracy > 0
            else "-",
        ),
        ("Constraints added", str(summary.feedback_constraints)),
    ]
    print_table("Round Statistics", ["Metric", "Count"], stats_rows, styles=["cyan", "green"])


def _print_final_summary(summaries: list[RoundSummary], console: Any) -> None:
    if not summaries:
        return

    console.rule("[bold cyan]Final Summary")

    total_prompts = sum(s.total_prompts for s in summaries)
    total_motions = sum(s.total_motions for s in summaries)
    total_simulations = sum(s.total_simulations for s in summaries)
    accuracies = [s.classifier_accuracy for s in summaries if s.classifier_accuracy > 0]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    rows = []
    for s in summaries:
        steps_ok = sum(1 for st in s.step_statuses.values() if st == StepStatus.COMPLETED)
        steps_fail = sum(1 for st in s.step_statuses.values() if st == StepStatus.FAILED)
        status_str = f"{steps_ok}/4 completed"
        if steps_fail:
            status_str += f" ({steps_fail} failed)"
        rows.append((str(s.round_number), str(s.total_prompts), str(s.total_motions), status_str))

    print_table("All Rounds", ["Round", "Prompts", "Motions", "Status"], rows,
                styles=["cyan", "green", "green", "bold"])

    summary_text = (
        f"Rounds completed:       {len(summaries)}\n"
        f"Total prompts:          {total_prompts}\n"
        f"Total motions:          {total_motions}\n"
        f"Total mmWave files:     {total_simulations}"
    )
    if accuracies:
        summary_text += f"\nAvg classifier acc.:    {avg_accuracy * 100:.2f}%"

    summary_panel(summary_text, title="Flywheel Pipeline Complete")


# ---------------------------------------------------------------------------
# Command implementation
# ---------------------------------------------------------------------------


def cmd_run(args: object) -> None:
    """Run the full flywheel loop (one or more rounds)."""
    from flywheel.engine import FlywheelEngine

    console = get_console()
    args._print_banner()
    load_env()
    args._install_signal_handlers()

    dry_run = getattr(args, "dry_run", False)
    output_dir: Path = args.output_dir  # type: ignore[attr-defined]

    config = resolve_config(args)
    state = config_to_state(config)

    # Determine starting round
    saved_state = load_state(output_dir)
    round_arg = getattr(args, "round", None)
    if round_arg is not None:
        start_round = round_arg
    elif saved_state.get("current_round") is not None:
        start_round = saved_state["current_round"]
        completed = saved_state.get("completed_steps", [])
        if completed:
            console.print(
                f"Resuming from round {start_round}, "
                f"completed steps: {', '.join(completed)}"
            )
    else:
        start_round = int(config.version.split("_")[-1])

    max_rounds = getattr(args, "max_rounds", 4)
    gpu_count_val = getattr(args, "gpu_count", None)
    gpu_count = gpu_count_val if gpu_count_val else detect_gpus()
    state.max_rounds = max_rounds

    # Pipeline header
    console.print(
        f"Config: [bold]{config.path}[/]  |  "
        f"Output: [bold]{output_dir}[/]  |  "
        f"Rounds: {start_round}-{max_rounds - 1}  |  "
        f"GPUs: {gpu_count}"
    )
    if dry_run:
        console.print("  [bold yellow]DRY RUN MODE[/]")
    console.print()

    # Create engine
    engine = FlywheelEngine(output_dir, gpu_count=gpu_count)

    # Per-round loop (with display)
    all_summaries: list[RoundSummary] = []
    batch_size = getattr(args, "batch_size", 8)
    no_feedback = getattr(args, "no_feedback", False)
    analysis_file = getattr(args, "analysis_file", None)
    test_results_file = getattr(args, "test_results_file", None)
    fps = getattr(args, "fps", 30)
    interpolation_points = getattr(args, "interpolation_points", 50)
    output_mode = getattr(args, "output_mode", "udoppler")
    simulation_config = getattr(args, "simulation_config", None)
    motion_backend = getattr(args, "motion_backend", None)
    motion_joints_subdir = getattr(args, "motion_joints_subdir", None)

    for round_num in range(start_round, max_rounds):
        if args._is_interrupted():
            console.print("[yellow]Interrupted. State saved.[/]")
            break

        version = f"round_{round_num}"

        # Load per-round config
        round_config_path = output_dir / version / "info.json"
        if round_config_path.exists():
            config = FlywheelConfig(round_config_path)
            config.load()

        paths = PathManager(output_dir, version)
        paths.create_dirs()

        _print_round_header(round_num, version, max_rounds)

        # Run this round via engine
        try:
            round_summary = engine.run_single_round(
                round_num, config,
                dry_run=dry_run,
                batch_size=batch_size,
                no_feedback=no_feedback,
                analysis_file=str(analysis_file) if analysis_file else None,
                test_results_file=str(test_results_file) if test_results_file else None,
                fps=fps,
                interpolation_points=interpolation_points,
                output_mode=output_mode,
                simulation_config=str(simulation_config) if simulation_config else None,
                motion_backend=motion_backend,
                motion_joints_subdir=motion_joints_subdir,
                num_workers=getattr(args, "num_workers", 8),
                interrupted=args._is_interrupted,
            )
            all_summaries.append(round_summary)
            _print_round_summary(round_num, round_summary, console)
        except Exception as e:
            error_panel(str(e), title=f"Round {round_num} Failed")
            break

    _print_final_summary(all_summaries, console)

"""Command: step1/step2/step3/step4 -- run a single flywheel step."""

from __future__ import annotations

import sys
from pathlib import Path

from flywheel.config import load_env
from flywheel.engine import FlywheelEngine
from flywheel.logging_utils import error_panel, get_console
from flywheel.state import load_state, save_round_state

from .run_cmd import resolve_config

_STEP_NAMES = {
    "step1": (1, "Prompt Generation"),
    "step2": (2, "Motion Generation"),
    "step3": (3, "mmWave Simulation"),
    "step4": (4, "Classifier Feedback"),
}


def cmd_step(args: object) -> None:
    """Run a single step of the flywheel."""
    console = get_console()
    args._print_banner()
    load_env()

    command: str = args.command  # type: ignore[attr-defined]
    step_num, step_name = _STEP_NAMES[command]

    output_dir: Path = args.output_dir  # type: ignore[attr-defined]

    # Determine which round to run the step in
    round_num = getattr(args, "round", None)
    if round_num is None:
        saved = load_state(output_dir)
        round_num = saved.get("current_round", 0)

    config = resolve_config(args)

    console.print(
        f"Running [bold]Step {step_num}: {step_name}[/]  |  "
        f"Round: {round_num}  |  Config: [dim]{config.path}[/]"
    )

    gpu_count = getattr(args, "gpu_count", None)
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
    no_approval = getattr(args, "no_approval", False)

    engine = FlywheelEngine(output_dir, gpu_count=gpu_count)
    num_workers = getattr(args, "num_workers", 8)

    try:
        result = engine.run_single_step(
            step_num, round_num, config,
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
            num_workers=num_workers,
            require_approval=not no_approval,
        )

        save_round_state(
            output_dir, round_num, f"step{step_num}", [f"step{step_num}"], result
        )
        console.print(f"\n[green]Step {step_num} completed successfully.[/]")

    except Exception as e:
        error_panel(str(e), title=f"Step {step_num} Failed")
        sys.exit(1)

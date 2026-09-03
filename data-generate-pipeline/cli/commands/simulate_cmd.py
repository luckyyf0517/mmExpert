"""Command: simulate -- run mmWave simulation on completed motion data."""

from __future__ import annotations

import sys
from pathlib import Path

from flywheel.config import load_env
from flywheel.logging_utils import error_panel, get_console
from flywheel.path_manager import PathManager
from flywheel.state import load_state
from flywheel.steps.motion.backends import resolve_motion_backend
from simulator.runner import mmWaveSimulator

from .run_cmd import resolve_config


def cmd_simulate(args: object) -> None:
    """Run mmWave simulation on completed motion data.

    This command is also exposed as a standalone utility for rerunning the
    simulation stage outside the full flywheel loop.
    """
    console = get_console()
    args._print_banner()
    load_env()

    output_dir: Path = args.output_dir  # type: ignore[attr-defined]

    round_num = getattr(args, "round", None)
    if round_num is None:
        saved = load_state(output_dir)
        round_num = saved.get("current_round", 0)

    version = f"round_{round_num}"
    config = resolve_config(args)
    paths = PathManager(output_dir, version)

    fps: int = getattr(args, "fps", 30)
    interpolation_points: int = getattr(args, "interpolation_points", 50)
    output_mode: str = getattr(args, "output_mode", "udoppler")
    simulation_config = getattr(args, "simulation_config", None)
    motion_backend = getattr(args, "motion_backend", None)
    motion_joints_subdir = getattr(args, "motion_joints_subdir", None)
    selection = resolve_motion_backend(
        config,
        paths,
        backend_override=motion_backend,
        joints_subdir_override=motion_joints_subdir,
        prefer_manifest=True,
    )

    console.print(
        f"Running [bold]mmWave Simulation[/]  |  Round: {round_num}  |  "
        f"FPS: {fps}  |  Interp: {interpolation_points}  |  Mode: {output_mode}  |  "
        f"Motion backend: {selection.backend} ({selection.joints_subdir})"
    )

    try:
        sim = mmWaveSimulator()
        result = sim.run(
            version, config, paths,
            fps=fps,
            interpolation_points=interpolation_points,
            output_mode=output_mode,
            simulation_config=str(simulation_config) if simulation_config else None,
            joints_dir=selection.joints_dir,
            motion_backend=selection.backend,
            joints_subdir=selection.joints_subdir,
        )
        console.print(
            f"\n[green]mmWave simulation complete: "
            f"{result.total_simulations} files generated in {result.elapsed:.1f}s[/]"
        )

    except Exception as e:
        error_panel(str(e), title="Simulation Failed")
        sys.exit(1)

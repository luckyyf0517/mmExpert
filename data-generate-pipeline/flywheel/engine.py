"""Flywheel engine -- pure orchestration logic, no CLI I/O.

``FlywheelEngine.run_single_round()`` executes the 4-step flywheel loop for a
single round and returns a ``RoundSummary`` object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import FlywheelConfig
from .logging_utils import get_console
from .path_manager import PathManager
from .state import load_state, save_round_state
from .types import RoundSummary, StepStatus

# Step modules
from .steps.feedback import Step4ClassifierFeedback
from .steps.motion import Step2MotionGen
from .steps.prompt import Step1PromptGen
from .steps.motion.backends import resolve_motion_backend

from simulator.runner import mmWaveSimulator
from utils.gpu import detect_gpus


# ---------------------------------------------------------------------------
# Step runner type
# ---------------------------------------------------------------------------

StepRunner = Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------------------------


def _run_step1(
    version: str,
    config: FlywheelConfig,
    paths: PathManager,
    gpu_count: int,
    *,
    num_workers: int = 8,
    require_approval: bool = True,
    **kw: Any,
) -> dict[str, Any]:
    step = Step1PromptGen()
    result = step.run(version, config, paths, num_workers=num_workers, require_approval=require_approval)
    return {"total_prompts": result.total_prompts, "elapsed": result.elapsed}


def _run_step2(
    version: str,
    config: FlywheelConfig,
    paths: PathManager,
    gpu_count: int,
    batch_size: int = 8,
    motion_backend: str | None = None,
    motion_joints_subdir: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    step = Step2MotionGen()
    result = step.run(
        version,
        config,
        paths,
        num_gpus=gpu_count,
        batch_size=batch_size,
        motion_backend=motion_backend,
        motion_joints_subdir=motion_joints_subdir,
    )
    return {
        "total_motions": result.total_motions,
        "elapsed": result.gen_elapsed,
        "motion_backend": result.backend,
        "motion_joints_subdir": result.joints_subdir,
    }


def _run_step3(
    version: str,
    config: FlywheelConfig,
    paths: PathManager,
    gpu_count: int,
    *,
    fps: int = 20,
    interpolation_points: int = 50,
    output_mode: str = "udoppler",
    simulation_config: str | None = None,
    motion_backend: str | None = None,
    motion_joints_subdir: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    selection = resolve_motion_backend(
        config,
        paths,
        backend_override=motion_backend,
        joints_subdir_override=motion_joints_subdir,
        prefer_manifest=True,
    )
    step = mmWaveSimulator()
    result = step.run(
        version,
        config,
        paths,
        fps=fps,
        interpolation_points=interpolation_points,
        output_mode=output_mode,
        gpu_count=gpu_count,
        simulation_config=simulation_config,
        joints_dir=selection.joints_dir,
        motion_backend=selection.backend,
        joints_subdir=selection.joints_subdir,
    )
    return {"total_simulations": result.total_simulations, "elapsed": result.elapsed}


def _run_step4(
    version: str,
    config: FlywheelConfig,
    paths: PathManager,
    *,
    no_feedback: bool = False,
    analysis_file: str | None = None,
    test_results_file: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    if no_feedback:
        return {
            "overall_accuracy": 0.0,
            "total_evaluated": 0,
            "constraints_added": 0,
            "config_updated": False,
        }

    fb = Step4ClassifierFeedback()
    result = fb.run(
        version,
        config,
        paths,
        analysis_file=analysis_file,
        test_results_file=test_results_file,
    )
    return {
        "overall_accuracy": result.overall_accuracy,
        "total_evaluated": result.total_evaluated,
        "constraints_added": result.constraints_added,
        "config_updated": result.config_updated,
    }


# Step dispatch table
_STEP_RUNNERS: dict[int, tuple[str, StepRunner]] = {
    1: ("Prompt Generation", _run_step1),
    2: ("Motion Generation", _run_step2),
    3: ("mmWave Simulation", _run_step3),
    4: ("Classifier Feedback", _run_step4),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FlywheelEngine:
    """Orchestrate the flywheel loop across rounds.

    This class is intentionally free of CLI concerns (argparse, banners,
    signal handling).  The CLI layer in ``cli/`` wraps this engine.
    """

    def __init__(self, output_dir: Path, gpu_count: int | None = None) -> None:
        self.output_dir = output_dir
        self.gpu_count = gpu_count or detect_gpus()

    # -- public entry point --------------------------------------------------

    def run_single_round(
        self,
        round_num: int,
        config: FlywheelConfig,
        *,
        dry_run: bool = False,
        batch_size: int = 8,
        no_feedback: bool = False,
        analysis_file: str | None = None,
        test_results_file: str | None = None,
        fps: int = 20,
        interpolation_points: int = 50,
        output_mode: str | None = None,
        simulation_config: str | None = None,
        motion_backend: str | None = None,
        motion_joints_subdir: str | None = None,
        num_workers: int = 8,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> RoundSummary:
        """Execute a single flywheel round and return its summary."""
        console = get_console()

        if interrupted():
            console.print("[yellow]Interrupted. State saved.[/]")
            return RoundSummary(round_number=round_num, step_statuses={})

        version = f"round_{round_num}"

        # Load per-round config
        round_config_path = self.output_dir / version / "info.json"
        if round_config_path.exists():
            config = FlywheelConfig(round_config_path)
            config.load()

        paths = PathManager(self.output_dir, version)
        paths.create_dirs()

        # Resolve output_mode from config (default: udoppler)
        resolved_output_mode = output_mode or config.data.get("output_mode", "udoppler")

        # Determine already-completed steps (for resume)
        flywheel_state = load_state(self.output_dir)
        if flywheel_state.get("current_round") == round_num:
            completed_steps = flywheel_state.get("completed_steps", [])
        else:
            completed_steps = []

        summary = RoundSummary(round_number=round_num, step_statuses={})
        round_results: dict[str, Any] = {}

        for step_num in range(1, 5):
            if interrupted():
                break

            step_name, step_fn = _STEP_RUNNERS[step_num]
            step_key = f"step{step_num}"

            if step_key in completed_steps:
                summary.step_statuses[step_key] = StepStatus.COMPLETED
                continue

            try:
                if dry_run:
                    result: dict[str, Any] = {}
                else:
                    result = step_fn(
                        version,
                        config,
                        paths,
                        gpu_count=self.gpu_count,
                        batch_size=batch_size,
                        no_feedback=no_feedback,
                        analysis_file=analysis_file,
                        test_results_file=test_results_file,
                        fps=fps,
                        interpolation_points=interpolation_points,
                        output_mode=resolved_output_mode,
                        simulation_config=simulation_config,
                        num_workers=num_workers,
                        motion_backend=motion_backend,
                        motion_joints_subdir=motion_joints_subdir,
                    )

                round_results[step_key] = result
                summary.step_statuses[step_key] = StepStatus.COMPLETED

                if step_num == 1:
                    summary.total_prompts = result.get("total_prompts", 0)
                elif step_num == 2:
                    summary.total_motions = result.get("total_motions", 0)
                elif step_num == 3:
                    summary.total_simulations = result.get("total_simulations", 0)
                elif step_num == 4:
                    summary.classifier_accuracy = result.get("overall_accuracy", 0.0)
                    summary.feedback_constraints = result.get("constraints_added", 0)

                completed_steps.append(step_key)
                save_round_state(
                    self.output_dir, round_num, step_key, completed_steps, round_results,
                )

            except Exception:
                summary.step_statuses[step_key] = StepStatus.FAILED
                try:
                    save_round_state(
                        self.output_dir, round_num, step_key, completed_steps, round_results,
                        failed=True,
                    )
                except Exception:
                    pass  # Don't mask the original error
                raise

        return summary

    # -- helpers -------------------------------------------------------------

    def run_single_step(
        self,
        step_num: int,
        round_num: int,
        config: FlywheelConfig,
        *,
        batch_size: int = 8,
        no_feedback: bool = False,
        analysis_file: str | None = None,
        test_results_file: str | None = None,
        fps: int = 20,
        interpolation_points: int = 50,
        output_mode: str | None = None,
        simulation_config: str | None = None,
        motion_backend: str | None = None,
        motion_joints_subdir: str | None = None,
        num_workers: int = 8,
        require_approval: bool = True,
    ) -> dict[str, Any]:
        """Run a single step and return its result dict."""
        version = f"round_{round_num}"
        paths = PathManager(self.output_dir, version)
        paths.create_dirs()

        resolved_output_mode = output_mode or config.data.get("output_mode", "udoppler")
        _, step_fn = _STEP_RUNNERS[step_num]
        return step_fn(
            version,
            config,
            paths,
            gpu_count=self.gpu_count,
            batch_size=batch_size,
            no_feedback=no_feedback,
            analysis_file=analysis_file,
            test_results_file=test_results_file,
            fps=fps,
            interpolation_points=interpolation_points,
            output_mode=resolved_output_mode,
            simulation_config=simulation_config,
            num_workers=num_workers,
            require_approval=require_approval,
            motion_backend=motion_backend,
            motion_joints_subdir=motion_joints_subdir,
        )

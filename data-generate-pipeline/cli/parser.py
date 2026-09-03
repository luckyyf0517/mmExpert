"""Argument parser definitions for the flywheel CLI.

All argparse configuration lives here so that command modules stay focused
on behaviour rather than option parsing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve pipeline root for default paths
# ---------------------------------------------------------------------------
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="run_flywheel",
        description="Flywheel data-generation pipeline orchestrator",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.json (default: auto-detect from output-dir)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path(_PIPELINE_ROOT) / "datasets" / "flywheel_demo",
        help="Output directory (default: datasets/flywheel_demo/)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- init ----------------------------------------------------------------
    p_init = sub.add_parser("init", help="Initialize a new flywheel project")
    p_init.add_argument(
        "--round",
        type=int,
        required=True,
        help="Round number to initialize (e.g. 0 or 1)",
    )
    p_init.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for prompt generation (default: gpt-4o-mini)",
    )
    p_init.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Natural-language description of the motion task scope",
    )
    p_init.add_argument(
        "--total-count",
        type=int,
        required=True,
        help="Total number of motion items to generate for this task scope",
    )

    # -- run -----------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run the full flywheel loop")
    _add_common_options(p_run)
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # -- step1 ---------------------------------------------------------------
    p_step1 = sub.add_parser("step1", help="Run Step 1: Prompt Generation")
    _add_common_options(p_step1)
    p_step1.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip user approval checkpoint for planner strategies",
    )

    # -- step2 ---------------------------------------------------------------
    p_step2 = sub.add_parser("step2", help="Run Step 2: Motion Generation")
    _add_common_options(p_step2)

    # -- step3 ---------------------------------------------------------------
    p_step3 = sub.add_parser("step3", help="Run Step 3: mmWave Simulation")
    _add_common_options(p_step3)

    # -- step4 ---------------------------------------------------------------
    p_step4 = sub.add_parser("step4", help="Run Step 4: Classifier Feedback")
    _add_common_options(p_step4)

    # -- simulate ------------------------------------------------------------
    p_sim = sub.add_parser("simulate", help="Run mmWave simulation on completed motion data")
    p_sim.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round number to simulate (default: latest)",
    )
    p_sim.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Simulation input motion frame rate (default: 30)",
    )
    p_sim.add_argument(
        "--interpolation-points",
        type=int,
        default=50,
        help="Interpolation points between joints (default: 50)",
    )
    p_sim.add_argument(
        "--output-mode",
        type=str,
        choices=["udoppler", "full"],
        default="udoppler",
        help="Simulation output mode (default: udoppler)",
    )
    p_sim.add_argument(
        "--simulation-config",
        type=Path,
        default=None,
        help="Path to a dedicated simulation YAML config",
    )
    _add_motion_options(p_sim)

    # -- status --------------------------------------------------------------
    sub.add_parser("status", help="Show current flywheel status dashboard")

    # -- resume --------------------------------------------------------------
    p_resume = sub.add_parser("resume", help="Resume interrupted flywheel")
    _add_common_options(p_resume)

    return parser


def _add_common_options(p: argparse.ArgumentParser) -> None:
    """Add shared options to a subcommand parser."""
    p.add_argument(
        "--round",
        type=int,
        default=None,
        help="Start from round N (default: auto-detect)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Maximum number of rounds (default: 4)",
    )
    p.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-GPU batch size for Step 2 (default: 8)",
    )
    p.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip classifier feedback step (leave config unchanged)",
    )
    p.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="Path to classifier feedback analysis JSON",
    )
    p.add_argument(
        "--test-results-file",
        type=Path,
        default=None,
        help="Path to classifier test-results JSON",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Simulation input motion frame rate for Step 3 (default: 30)",
    )
    p.add_argument(
        "--interpolation-points",
        type=int,
        default=50,
        help="Interpolation points between joints for Step 3 (default: 50)",
    )
    p.add_argument(
        "--output-mode",
        type=str,
        choices=["udoppler", "full"],
        default="udoppler",
        help="Simulation output mode for Step 3 (default: udoppler)",
    )
    p.add_argument(
        "--simulation-config",
        type=Path,
        default=None,
        help="Path to a dedicated simulation YAML config for Step 3",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for prompt generation (default: 8)",
    )
    _add_motion_options(p)


def _add_motion_options(p: argparse.ArgumentParser) -> None:
    """Add motion-backend selection options."""
    p.add_argument(
        "--motion-backend",
        type=str,
        default=None,
        help="Motion generator backend for Step 2/3 (default: config or hunyuan_motion)",
    )
    p.add_argument(
        "--motion-joints-subdir",
        type=str,
        default=None,
        help="Step 2 joints subdirectory to use for motion simulation",
    )

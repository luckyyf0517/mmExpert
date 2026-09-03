"""Command: resume -- resume an interrupted flywheel from last checkpoint."""

from __future__ import annotations

from pathlib import Path

from flywheel.logging_utils import get_console
from flywheel.state import load_state

from .run_cmd import cmd_run


def cmd_resume(args: object) -> None:
    """Resume interrupted flywheel from last checkpoint."""
    console = get_console()
    output_dir: Path = args.output_dir  # type: ignore[attr-defined]
    saved = load_state(output_dir)

    if not saved:
        console.print("[yellow]No saved state found. Nothing to resume.[/]")
        console.print("Run [bold]python run_flywheel.py run[/] to start a new flywheel.")
        return

    current_round = saved.get("current_round", 0)
    completed_steps = saved.get("completed_steps", [])

    console.print(f"Resuming from round {current_round}")
    console.print(f"Completed steps: {', '.join(completed_steps) or 'none'}")
    console.print()

    # Delegate to run command with the saved round
    args.round = current_round  # type: ignore[attr-defined]
    cmd_run(args)

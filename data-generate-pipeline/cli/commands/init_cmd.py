"""Command: init -- initialise a new flywheel project."""

from __future__ import annotations

from pathlib import Path

from flywheel.config import FlywheelConfig, load_env
from flywheel.logging_utils import get_console
from flywheel.path_manager import PathManager
from flywheel.state import init_state, save_state


def cmd_init(args: object) -> None:
    """Initialize a new flywheel project."""
    console = get_console()
    args._print_banner()
    load_env()

    output_dir: Path = args.output_dir  # type: ignore[attr-defined]
    output_dir.mkdir(parents=True, exist_ok=True)

    round_arg = getattr(args, "round", None)
    if round_arg is None or round_arg < 0:
        raise ValueError("`--round` must be a non-negative integer.")

    version = f"round_{round_arg}"
    paths = PathManager(output_dir, version)
    paths.create_dirs()

    model_arg = getattr(args, "model", "gpt-4o-mini")
    tasks_arg = getattr(args, "tasks", "").strip()
    total_count_arg = getattr(args, "total_count", 0)

    if not tasks_arg:
        raise ValueError("`--tasks` must be a non-empty natural-language description.")
    if total_count_arg <= 0:
        raise ValueError("`--total-count` must be a positive integer.")

    FlywheelConfig.from_template(
        paths.config_path,
        version=version,
        model=model_arg,
        tasks=tasks_arg,
        total_count=total_count_arg,
    )

    # Save initial state
    state = init_state(round_arg)
    state["completed_steps"] = []
    save_state(output_dir, state)

    console.print(
        f"Output: [bold]{output_dir}[/]  |  "
        f"Config: [dim]{paths.config_path}[/]  |  "
        f"Total Count: [bold]{total_count_arg}[/]"
    )
    console.print(paths.tree_summary())
    console.print(
        "[green]Init complete.[/] Run [bold]python run_flywheel.py run[/] to start."
    )

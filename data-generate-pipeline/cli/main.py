"""Top-level CLI entry point: banner, signal handling, command dispatch."""

from __future__ import annotations

from importlib import import_module
import signal
import sys
from pathlib import Path
from typing import Any

from .parser import build_parser

# ---------------------------------------------------------------------------
# Ensure the pipeline root is importable
# ---------------------------------------------------------------------------
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_interrupted = False


def _is_interrupted() -> bool:
    return _interrupted


def _signal_handler(sig: int, frame: Any) -> None:
    from flywheel.logging_utils import get_console

    global _interrupted
    if _interrupted:
        get_console().print("\n[bold red]Force exit.[/]")
        sys.exit(1)
    _interrupted = True
    get_console().print("\n[yellow]Interrupt received. Saving state and exiting...[/]")


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    from flywheel.logging_utils import get_console

    console = get_console()
    console.print("[bold cyan]Flywheel Pipeline v0.1.0[/]")
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DISPATCH = {
    "init": ("init_cmd", "cmd_init"),
    "run": ("run_cmd", "cmd_run"),
    "step1": ("step_cmd", "cmd_step"),
    "step2": ("step_cmd", "cmd_step"),
    "step3": ("step_cmd", "cmd_step"),
    "step4": ("step_cmd", "cmd_step"),
    "simulate": ("simulate_cmd", "cmd_simulate"),
    "status": ("status_cmd", "cmd_status"),
    "resume": ("resume_cmd", "cmd_resume"),
}


def main() -> None:
    """Parse args and dispatch to the appropriate command."""
    parser = build_parser()
    args = parser.parse_args()

    command = args.command

    if command is None:
        parser.print_help()
        sys.exit(0)

    target = _DISPATCH.get(command)
    if target is None:
        parser.print_help()
        sys.exit(1)

    module_name, function_name = target
    module = import_module(f".commands.{module_name}", package=__package__)
    fn = getattr(module, function_name)

    # Attach helpers to args namespace so commands can use them
    args._is_interrupted = _is_interrupted  # type: ignore[attr-defined]
    args._install_signal_handlers = _install_signal_handlers  # type: ignore[attr-defined]
    args._print_banner = _print_banner  # type: ignore[attr-defined]
    args._pipeline_root = _PIPELINE_ROOT  # type: ignore[attr-defined]

    fn(args)

"""Rich-based beautiful logging for the flywheel pipeline.

Provides:
- ``get_console()``  – colourised console with timestamps
- ``get_logger()``   – dual-output logger (console + rotating file)
- Step / progress helpers (panels, progress bars, tables, trees, spinners)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.tree import Tree

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_console: Console | None = None
_logger: logging.Logger | None = None



def get_console() -> Console:
    """Return a shared ``rich.Console``."""
    global _console
    if _console is None:
        _console = Console(
            stderr=True,
            log_time=True,
            log_time_format="[%X]",
        )
    return _console


def get_logger(name: str = "flywheel", log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger that writes to both the Rich console and an optional file.

    Parameters
    ----------
    name:
        Logger name (default ``"flywheel"``).
    log_file:
        If given, also write plain-text logs to this file.
    """
    global _logger
    if _logger is not None and log_file is None:
        return _logger

    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)

    # Rich console handler (INFO+)
    console_handler = RichHandler(
        console=get_console(),
        level=logging.INFO,
        show_time=True,
        show_path=False,
        markup=True,
    )
    lg.addHandler(console_handler)

    # File handler (DEBUG+) – optional
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        lg.addHandler(fh)

    lg.propagate = False
    _logger = lg
    return lg


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def step_panel(title: str, subtitle: str = "", step_num: int | None = None) -> None:
    """Print a prominent step-title line."""
    label = title
    if step_num is not None:
        label = f"Step {step_num}: {title}"
    parts = [f"[bold cyan]{label}[/]"]
    if subtitle:
        parts.append(f"[dim]({subtitle})[/]")
    get_console().print(" ".join(parts))


def summary_panel(content: str, title: str = "Summary") -> None:
    """Print a summary panel."""
    get_console().print(Panel(content, title=title, style="bold green"))


def error_panel(content: str, title: str = "Error") -> None:
    """Print an error panel."""
    get_console().print(Panel(content, title=title, style="bold red"))


def print_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    styles: Sequence[str] | None = None,
) -> None:
    """Print a Rich table."""
    table = Table(title=title, show_lines=True)
    for i, col in enumerate(columns):
        style = styles[i] if styles and i < len(styles) else None
        table.add_column(col, style=style)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    get_console().print(table)


def print_tree(label: str, tree_dict: dict[str, Any] | None = None, text: str | None = None) -> None:
    """Print a Rich tree.

    Use *tree_dict* for a nested dict structure, or *text* for a plain-text
    tree (e.g. from ``PathManager.tree_summary()``).
    """
    if tree_dict is not None:
        tree = Tree(label)
        _dict_to_tree(tree, tree_dict)
        get_console().print(tree)
    elif text is not None:
        get_console().print(text)
    else:
        get_console().print(Tree(label))


def _dict_to_tree(node: Tree, d: dict[str, Any]) -> None:
    for key, val in d.items():
        if isinstance(val, dict):
            branch = node.add(f"[bold]{key}[/]")
            _dict_to_tree(branch, val)
        elif isinstance(val, list):
            branch = node.add(f"[bold]{key}[/] ({len(val)} items)")
            for item in val[:5]:
                branch.add(str(item))
            if len(val) > 5:
                branch.add("...")
        else:
            node.add(f"{key}: {val}")


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------

def print_data_flow(console: Console, completed_steps: tuple[str, ...] = ()) -> None:
    """Print a data-flow diagram showing pipeline step status."""
    steps = [
        ("step1", "Prompt Gen", "LLM -> text"),
        ("step2", "Motion Gen", "text -> motion artifacts"),
        ("step3", "mmWave Sim", "motion -> radar"),
        ("step4", "Feedback", "classifier diagnostics -> config"),
    ]
    lines = ["[bold]Data Flow:[/]"]
    for key, label, desc in steps:
        done = key in completed_steps
        check = "[green]✓[/]" if done else "[dim]○[/]"
        lines.append(f"  {check}  {label:<14s} {desc}")
    lines.append("       └──> next round")
    console.print("\n".join(lines))
    console.print()


def make_progress() -> Progress:
    """Return a pre-configured ``rich.progress.Progress`` instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    )

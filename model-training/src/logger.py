"""Utility helpers for consistently tagged console logging."""

from __future__ import annotations

import os
from termcolor import colored


def format_tag(tag: str) -> str:
    if not tag:
        raise ValueError("Tag must be a non-empty string")
    tag = tag.strip().upper()
    return f"[{tag}]"


def log_message(
    tag: str,
    message: str,
    *,
    color: str = "green",
    attrs: tuple[str, ...] | None = None,
    prefix: str = "",
    suffix: str = "",
    end: str | None = "\n",
    rank_0_only: bool = True,
) -> None:
    """Print colored message to console, optionally only on rank 0.

    Args:
        tag: Message tag (will be formatted as [TAG])
        message: Message content
        color: Color for the tag only
        attrs: Additional text attributes for the tag
        prefix: Text to prepend
        suffix: Text to append
        end: String to append at the end (default: newline)
        rank_0_only: If True, only print on rank 0 (default: True)
    """
    # Check rank restriction
    if rank_0_only:
        # First try torch.distributed
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        # Fallback to environment variable
        rank = os.environ.get('RANK', None)
        if rank is not None and int(rank) != 0:
            return

    label = colored(format_tag(tag), color, attrs=list(attrs) if attrs else None)
    separator = " " if message else ""
    output = f"{prefix}{label}{separator}{message}{suffix}"
    print(output, end=end)


def log_block(tag: str, lines: list[str], *, color: str = "green") -> None:
    if not lines:
        log_message(tag, "", color=color)
        return
    log_message(tag, lines[0], color=color)
    for line in lines[1:]:
        log_message(tag, line, color=color)

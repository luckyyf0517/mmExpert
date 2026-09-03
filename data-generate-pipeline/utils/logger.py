"""Utility helpers for consistently tagged console logging."""

from __future__ import annotations

from termcolor import colored


def format_tag(tag: str) -> str:
    if not tag:
        raise ValueError("Tag must be a non-empty string")
    tag = tag.strip().upper()
    return f"[{tag}]"


def build_log_message(
    tag: str,
    message: str,
    *,
    color: str = "green",
    attrs: tuple[str, ...] | None = None,
    prefix: str = "",
    suffix: str = "",
) -> str:
    label = colored(format_tag(tag), color, attrs=list(attrs) if attrs else None)
    separator = " " if message else ""
    return f"{prefix}{label}{separator}{message}{suffix}"


def log_message(
    tag: str,
    message: str,
    *,
    color: str = "green",
    attrs: tuple[str, ...] | None = None,
    prefix: str = "",
    suffix: str = "",
    end: str | None = "\n",
) -> None:
    output = build_log_message(
        tag,
        message,
        color=color,
        attrs=attrs,
        prefix=prefix,
        suffix=suffix,
    )
    print(output, end=end, flush=True)


def log_block(tag: str, lines: list[str], *, color: str = "green") -> None:
    if not lines:
        log_message(tag, "", color=color)
        return
    log_message(tag, lines[0], color=color)
    for line in lines[1:]:
        log_message(tag, line, color=color)

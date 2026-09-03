"""Expand ${VAR} placeholders in config values using os.environ (no defaults)."""

from __future__ import annotations

import os
import re

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")


def expand_env_vars(obj):
    if isinstance(obj, str):

        def repl(match: re.Match) -> str:
            return os.environ[match.group(1)]

        return _PLACEHOLDER.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    return obj

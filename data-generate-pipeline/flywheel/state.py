"""Flywheel state persistence (load / save / init).

The state file ``.flywheel_state.json`` tracks round progress, completed
steps, and per-step results so that interrupted runs can be resumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_STATE_FILENAME = ".flywheel_state.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def state_path(output_dir: Path) -> Path:
    """Return the path to the state file for *output_dir*."""
    return output_dir / _STATE_FILENAME


def load_state(output_dir: Path) -> dict[str, Any]:
    """Load flywheel state from ``.flywheel_state.json``.

    Returns an empty dict when the file does not exist.
    """
    sp = state_path(output_dir)
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    """Persist *state* to ``.flywheel_state.json``."""
    sp = state_path(output_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def init_state(round_num: int) -> dict[str, Any]:
    """Return a fresh state dict for a new flywheel run."""
    return {
        "current_round": round_num,
        "completed_steps": [],
        "rounds": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_round_state(
    output_dir: Path,
    round_num: int,
    current_step: str,
    completed_steps: list[str],
    results: dict[str, Any],
    failed: bool = False,
) -> None:
    """Save flywheel checkpoint state after each step."""
    state = load_state(output_dir)
    if not state:
        state = init_state(round_num)

    state["current_round"] = round_num
    state["completed_steps"] = list(completed_steps)

    round_key = f"round_{round_num}"
    if round_key not in state["rounds"]:
        state["rounds"][round_key] = {}

    state["rounds"][round_key][current_step] = {
        "status": "failed" if failed else "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result": {
            k: v
            for k, v in results.get(current_step, {}).items()
            if isinstance(v, (int, float, str, bool))
        },
    }

    save_state(output_dir, state)

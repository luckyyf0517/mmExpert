"""Configuration management for the flywheel pipeline.

Handles loading/saving config.json, environment variables (.env),
and version tracking across flywheel rounds.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .types import ActionCategory, FlywheelState


# ---------------------------------------------------------------------------
# Environment variable handling
# ---------------------------------------------------------------------------

def load_env(env_file: str | Path | None = None) -> None:
    """Load environment variables from a .env file (minimal parser).

    Does not overwrite variables already set in the environment.
    """
    if env_file is None:
        # Walk up from this file to find .env in the pipeline root
        env_file = Path(__file__).resolve().parent.parent / ".env"
    env_path = Path(env_file)
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ[key] = value


def get_env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable."""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """Read an environment variable, raising if missing."""
    value = os.environ.get(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file or set it before running."
        )
    return value


# ---------------------------------------------------------------------------
# config.json read / write
# ---------------------------------------------------------------------------

class FlywheelConfig:
    """Read/write access to the flywheel config.json.

    The config file lives at ``<pipeline_root>/<version>/info.json``
    and stores prompt templates, task requirements, generation rules, and
    feedback history.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.path = Path(config_path)
        self._data: dict[str, Any] = {}

    # -- I/O -----------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load config from disk. Returns the raw dict."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with open(self.path, encoding="utf-8") as f:
            self._data = json.load(f)
        return self._data

    def save(self, data: dict[str, Any] | None = None) -> None:
        """Write config to disk (pretty-printed JSON)."""
        if data is not None:
            self._data = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # -- Convenience accessors -----------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        if not self._data:
            self.load()
        return self._data

    @property
    def version(self) -> str:
        return self.data.get("version", "round_0")

    @property
    def model(self) -> str:
        return self.data.get("model", "gpt-4o-mini")

    @property
    def actions(self) -> list[ActionCategory]:
        raw = self.data.get("actions", [])
        return [ActionCategory.from_dict(a) for a in raw]

    @property
    def tasks(self) -> str:
        return self.data.get("tasks", "")

    @property
    def total_count(self) -> int:
        return int(self.data.get("total_count", 0))

    @property
    def planner_model(self) -> str:
        return self.data.get("planner_model", "gpt-4o")

    @property
    def worker_model(self) -> str:
        return self.data.get("worker_model", "gpt-4o-mini")

    @property
    def batch_size(self) -> int:
        return int(self.data.get("batch_size", 10))

    @property
    def prompt_template(self) -> list[str]:
        return self.data.get("prompt_template", [])

    @property
    def constraints(self) -> list[str]:
        return self.data.get("constraints", [])

    @property
    def feedback_history(self) -> list[dict[str, Any]]:
        return self.data.get("feedback_history", [])

    def add_feedback(self, entry: dict[str, Any]) -> None:
        """Append a feedback entry and persist."""
        self.data.setdefault("feedback_history", []).append(entry)
        self.save()

    def advance_version(self) -> str:
        """Bump the version string (round_N -> round_N+1) and persist."""
        current = self.version
        num = int(current.split("_")[-1]) + 1
        new_version = f"round_{num}"
        self.data["version"] = new_version
        self.save()
        return new_version

    def create_next_round_config(
        self,
        data: dict[str, Any] | None = None,
    ) -> tuple[str, FlywheelConfig]:
        """Create and persist the next round's ``info.json``.

        The current round config remains unchanged on disk. A cloned config is
        written to ``../round_{N+1}/info.json`` and returned.
        """
        current_data = deepcopy(data if data is not None else self.data)
        current = current_data.get("version", self.version)
        num = int(current.split("_")[-1]) + 1
        new_version = f"round_{num}"
        current_data["version"] = new_version

        next_path = self.path.parent.parent / new_version / "info.json"
        next_cfg = FlywheelConfig(next_path)
        next_cfg.save(current_data)
        return new_version, next_cfg

    # -- Factory -------------------------------------------------------------

    @classmethod
    def from_template(
        cls,
        dest_path: str | Path,
        *,
        version: str = "round_0",
        model: str = "gpt-4o-mini",
        tasks: str,
        total_count: int,
        planner_model: str = "gpt-4o",
        worker_model: str = "gpt-4o-mini",
        batch_size: int = 10,
        prompt_template: list[str] | None = None,
    ) -> FlywheelConfig:
        """Create a new config.json from defaults / overrides."""
        cfg = cls(dest_path)
        cfg._data = {
            "version": version,
            "model": model,
            "tasks": tasks,
            "total_count": total_count,
            "planner_model": planner_model,
            "worker_model": worker_model,
            "batch_size": batch_size,
            "prompt_template": prompt_template or _default_prompt_template(),
            "constraints": [],
            "feedback_history": [],
            "motion_generation": {
                "backend": "hunyuan_motion",
                "joints_subdir": "joints_hunyuan_motion",
                "backend_config": {
                    "disable_rewrite": True,
                    "disable_duration_est": True,
                    "duration_policy": "uniform_random",
                    "duration_min_frames": 180,
                    "duration_max_frames": 300,
                    "seed": 42,
                },
            },
            "output_mode": "udoppler",
            "price_per_1000_input_tokens": 0.15,
            "price_per_1000_output_tokens": 0.60,
        }
        cfg.save()
        return cfg


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

def _default_prompt_template() -> list[str]:
    return [
        "Generate diverse, concise descriptions of human motion for the task scope: {tasks}.",
        "Each description should be a single sentence focusing on body parts, actions, speed, direction, and repetition.",
        "Avoid emotional words, scenic descriptions, and literary filler. Write like a motion-capture annotation.",
        "Vary sentence structure. Do not repeat the same opening pattern.",
        "Do not number the descriptions.",
    ]


# ---------------------------------------------------------------------------
# FlywheelState helpers
# ---------------------------------------------------------------------------

def config_to_state(config: FlywheelConfig) -> FlywheelState:
    """Build a FlywheelState from a loaded config."""
    data = config.data
    round_num = int(data.get("version", "round_0").split("_")[-1])
    return FlywheelState(
        current_round=round_num,
        version=data.get("version", "round_0"),
        actions=config.actions,
        model=config.model,
    )

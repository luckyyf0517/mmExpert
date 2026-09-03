"""Type definitions for the flywheel pipeline.

All core data structures used across pipeline steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    """Status of a single pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActionCategory:
    """A single action category in the dataset."""
    id: str          # e.g. "A00"
    name: str        # e.g. "walk"
    count: int = 500 # target sample count

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "count": self.count}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionCategory:
        return cls(id=d["id"], name=d["name"], count=d.get("count", 500))


@dataclass
class PromptEntry:
    """A single generated motion description prompt."""
    action_id: str       # e.g. "A00"
    source_file: str = ""  # originating .txt file name
    index: int = 0       # line index within source file
    prompt_count: int = 1  # number of prompts contained in source_file


@dataclass
class MotionResult:
    """Result from motion generation for a single prompt."""
    prompt: str
    action_id: str
    joints_path: Path | None = None   # .npy joints file
    bvh_path: Path | None = None      # .bvh animation file
    video_path: Path | None = None    # .mp4 video file
    text_path: Path | None = None     # .txt description file
    success: bool = True
    error: str = ""


@dataclass
class RoundSummary:
    """Summary statistics for a completed flywheel round."""
    round_number: int
    total_prompts: int = 0
    total_motions: int = 0
    total_simulations: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_revised: int = 0
    classifier_accuracy: float = 0.0
    feedback_constraints: int = 0
    step_statuses: dict[str, StepStatus] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        total = self.total_accepted + self.total_rejected + self.total_revised
        return self.total_accepted / total if total > 0 else 0.0


@dataclass
class FlywheelState:
    """Complete state of the flywheel pipeline across rounds."""
    current_round: int = 0
    max_rounds: int = 3
    version: str = "round_0"
    actions: list[ActionCategory] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    round_summaries: list[RoundSummary] = field(default_factory=list)

    def next_round_version(self) -> str:
        return f"round_{self.current_round + 1}"

    def current_round_version(self) -> str:
        return f"round_{self.current_round}"

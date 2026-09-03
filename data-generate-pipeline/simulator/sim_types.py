"""Data types for the mmWave simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationStats:
    """Verification statistics for mmWave simulation outputs."""

    total_input: int = 0
    total_output_npy: int = 0
    total_output_npz: int = 0
    avg_file_size_kb: float = 0.0
    per_action: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of the simulation step."""

    total_simulations: int = 0
    stats: SimulationStats = field(default_factory=SimulationStats)
    elapsed: float = 0.0
    skipped: bool = False

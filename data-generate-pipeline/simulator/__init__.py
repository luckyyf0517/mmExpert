"""mmWave simulator package.

Contains both the physics engine (FMCW radar simulation) and the high-level
orchestration layer for batch processing after flywheel iterations complete.
"""

from simulator.runner import mmWaveSimulator

__all__ = ["mmWaveSimulator"]

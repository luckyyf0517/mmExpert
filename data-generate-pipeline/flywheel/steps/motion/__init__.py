"""Step 2: Motion Generation sub-package.

Converts text prompts into 3D human motion sequences using HY-Motion 1.0.
"""

from .generator import MotionGenResult, MotionGenStats, Step2MotionGen

__all__ = [
    "MotionGenResult",
    "MotionGenStats",
    "Step2MotionGen",
]

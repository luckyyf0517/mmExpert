#!/usr/bin/env python3
"""Flywheel Pipeline -- thin wrapper that delegates to the CLI package.

Usage::

    python run_flywheel.py --output-dir outputs/daily init --round 0 \
        --tasks "Generate diverse daily activities" --total-count 100
    python run_flywheel.py --output-dir outputs/daily run --max-rounds 1
    python run_flywheel.py step2 --round 1
    python run_flywheel.py simulate --round 2
    python run_flywheel.py status
    python run_flywheel.py resume
"""

import sys
from pathlib import Path

# Ensure pipeline root is importable
_PIPELINE_ROOT = str(Path(__file__).resolve().parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from cli import main

if __name__ == "__main__":
    main()

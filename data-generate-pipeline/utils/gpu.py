"""GPU detection utilities."""

from __future__ import annotations


def detect_gpus() -> int:
    """Return the number of available GPUs.

    Tries ``torch.cuda.device_count()`` first, then falls back to
    ``nvidia-smi``.  Returns 1 as a minimum.
    """
    try:
        import torch
        return torch.cuda.device_count() or 1
    except ImportError:
        pass

    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        return max(1, len(out.strip().splitlines()))
    except Exception:
        return 1

"""Support running the pipeline directory with Python's ``-m`` option."""

import sys
from pathlib import Path

_PIPELINE_ROOT = str(Path(__file__).resolve().parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from cli import main

if __name__ == "__main__":
    main()

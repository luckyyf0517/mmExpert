"""Unified path manager for the flywheel pipeline.

Given a pipeline root directory and a version string (e.g. "round_0"),
generates and manages all data paths used across pipeline steps.

Directory layout for a single version::

    <root>/<version>/
        info.json
        step1/
            prompts/
        step2/
            joints/
            bvh/
            video/
            texts/
        step3/
            udoppler/
            mmwave/
            texts/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


class PathManager:
    """Centralised path generation for one flywheel version."""

    def __init__(self, root: str | Path, version: str) -> None:
        self.root = Path(root).resolve()
        self.version = version

    # -- Top-level -----------------------------------------------------------

    @property
    def version_dir(self) -> Path:
        return self.root / self.version

    @property
    def real_data_dir(self) -> Path:
        return self.root / "_real_data"

    @property
    def evaluate_dir(self) -> Path:
        return self.root / "evaluate"

    @property
    def preprocessed_data_dir(self) -> Path:
        return self.evaluate_dir / "preprocessed-data"

    @property
    def classifier_outputs_dir(self) -> Path:
        return self.evaluate_dir / "cnn_outputs"

    @property
    def feedback_outputs_dir(self) -> Path:
        return self.evaluate_dir / "feedback_outputs"

    # -- Config --------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.version_dir / "info.json"

    # -- Step 1: Prompt Generation -------------------------------------------

    @property
    def step1_dir(self) -> Path:
        return self.version_dir / "step1"

    @property
    def prompts_dir(self) -> Path:
        return self.step1_dir / "prompts"

    # -- Step 2: Motion Generation -------------------------------------------

    @property
    def step2_dir(self) -> Path:
        return self.version_dir / "step2"

    @property
    def joints_dir(self) -> Path:
        return self.step2_dir / "joints"

    @property
    def motion_backend_manifest_path(self) -> Path:
        return self.step2_dir / "motion_backend.json"

    def motion_joints_dir(self, joints_subdir: str) -> Path:
        """Return the active Step 2 joints directory for a backend subdir."""
        return self.step2_dir / joints_subdir

    @property
    def bvh_dir(self) -> Path:
        return self.step2_dir / "bvh"

    @property
    def video_dir(self) -> Path:
        return self.step2_dir / "video"

    @property
    def texts_dir(self) -> Path:
        return self.step2_dir / "texts"

    # -- Step 3: Simulation --------------------------------------------------

    @property
    def step3_dir(self) -> Path:
        return self.version_dir / "step3"

    @property
    def udoppler_dir(self) -> Path:
        return self.step3_dir / "udoppler"

    @property
    def mmwave_dir(self) -> Path:
        return self.step3_dir / "mmwave"

    @property
    def step3_texts_dir(self) -> Path:
        return self.step3_dir / "texts"

    # -- Step 4: Classifier Feedback -----------------------------------------

    @property
    def step4_dir(self) -> Path:
        return self.version_dir / "step4"

    @property
    def feedback_dir(self) -> Path:
        return self.step4_dir / "feedback"

    # -- Other ---------------------------------------------------------------

    # -- Action-specific paths -----------------------------------------------

    def prompt_action_dir(self, action_id: str) -> Path:
        """Deprecated action-based path; retained only as a placeholder."""
        return self.prompts_dir / action_id

    # -- Directory operations ------------------------------------------------

    def all_dirs(self) -> list[Path]:
        """Return all managed directories in a flat list."""
        return [
            self.step1_dir,
            self.prompts_dir,
            self.step2_dir,
            self.joints_dir,
            self.bvh_dir,
            self.video_dir,
            self.texts_dir,
            self.step3_dir,
            self.udoppler_dir,
            self.mmwave_dir,
            self.step3_texts_dir,
        ]

    def create_dirs(self) -> None:
        """Create all managed directories (idempotent)."""
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    def validate_dirs(self) -> dict[str, bool]:
        """Check which managed directories exist on disk."""
        return {d.name: d.exists() for d in self.all_dirs()}

    def iter_prompt_files(self) -> Iterator[Path]:
        """Yield all .txt files under step1/prompts/."""
        if not self.prompts_dir.exists():
            return
        yield from sorted(self.prompts_dir.glob("*.txt"))

    def iter_joints_files(self) -> Iterator[Path]:
        """Yield all .npy files in the step2/joints directory."""
        if not self.joints_dir.exists():
            return
        yield from sorted(self.joints_dir.glob("*.npy"))

    def count_files(self, subdir: str, pattern: str = "*") -> int:
        """Count files matching *pattern* in a named sub-directory."""
        d = self.version_dir / subdir
        if not d.exists():
            return 0
        return sum(1 for _ in d.glob(pattern) if _.is_file())

    # -- Tree representation -------------------------------------------------

    def tree_summary(self) -> str:
        """Return a compact text tree of the version directory (for logging)."""
        lines = [f"{self.version}/"]
        self._build_tree(self.version_dir, lines, prefix="")
        return "\n".join(lines)

    def _build_tree(self, path: Path, lines: list[str], prefix: str) -> None:
        if not path.exists():
            return
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            count = ""
            if entry.is_dir():
                n = sum(1 for _ in entry.iterdir() if _.is_file())
                if n:
                    count = f"  ({n} files)"
            lines.append(f"{prefix}{connector}{entry.name}{count}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._build_tree(entry, lines, prefix + extension)
                if len(lines) > 50:  # truncate large trees
                    lines.append(f"{prefix}{extension}...")
                    return

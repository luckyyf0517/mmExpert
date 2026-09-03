"""Step 2 core: motion generation orchestration.

Converts text prompts into 3D human motion sequences via HY-Motion 1.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ...config import FlywheelConfig
from ...logging_utils import (
    get_console,
    step_panel,
)
from ...path_manager import PathManager
from ...types import PromptEntry
from .backends import (
    MotionBackendContext,
    get_backend,
    resolve_motion_backend,
    write_motion_backend_manifest,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MotionGenStats:
    """Verification statistics for generated motion data."""

    joints_count: int = 0
    bvh_count: int = 0
    texts_count: int = 0
    video_count: int = 0
    per_action: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class MotionGenResult:
    """Result of the motion generation step."""

    total_motions: int = 0
    stats: MotionGenStats = field(default_factory=MotionGenStats)
    gen_elapsed: float = 0.0
    backend: str = "hunyuan_motion"
    joints_subdir: str = "joints_hunyuan_motion"
    manifest_path: Path | None = None


# ---------------------------------------------------------------------------
# Step implementation
# ---------------------------------------------------------------------------


class Step2MotionGen:
    """Generate 3D human motion sequences from text prompts."""

    def __init__(self) -> None:
        self._num_gpus: int = 4
        self._batch_size: int = 8

    # -- Public API ----------------------------------------------------------

    def run(
        self,
        version: str,
        config: FlywheelConfig,
        paths: PathManager,
        *,
        num_gpus: int = 4,
        batch_size: int = 8,
        motion_backend: str | None = None,
        motion_joints_subdir: str | None = None,
    ) -> MotionGenResult:
        """Execute Step 2 -- motion generation.

        Parameters
        ----------
        version:
            Version string (e.g. ``"round_0"``).
        config:
            Loaded flywheel configuration.
        paths:
            PathManager for the current version.
        num_gpus:
            Number of GPUs for parallel generation.
        batch_size:
            Per-GPU batch size.

        Returns
        -------
        MotionGenResult with generation stats and timing.
        """
        self._num_gpus = num_gpus
        self._batch_size = batch_size
        selection = resolve_motion_backend(
            config,
            paths,
            backend_override=motion_backend,
            joints_subdir_override=motion_joints_subdir,
        )
        backend = get_backend(selection.backend)

        step_panel(
            "Motion Generation",
            subtitle=(
                f"Version: {version}  |  Backend: {selection.backend}  |  "
                f"Joints: {selection.joints_subdir}  |  GPUs: {num_gpus}  |  Batch: {batch_size}"
            ),
            step_num=2,
        )

        # Validate prerequisites
        prompts_dir = paths.prompts_dir
        if not prompts_dir.exists():
            raise FileNotFoundError(
                f"Step 1 prompts directory not found: {prompts_dir}. Run Step 1 first."
            )

        # Ensure output directories exist
        for d in [selection.joints_dir, paths.bvh_dir, paths.video_dir, paths.texts_dir]:
            d.mkdir(parents=True, exist_ok=True)

        result = MotionGenResult(
            backend=selection.backend,
            joints_subdir=selection.joints_subdir,
        )

        # Prepare input
        entries = self._prepare_input_files(prompts_dir, paths.texts_dir)
        if not entries:
            raise FileNotFoundError("No Step 1 prompt .txt files found. Run Step 1 first.")

        console = get_console()
        console.print(f"Found [bold]{len(entries)}[/] prompts to convert into motions")
        console.print(
            f"Motion backend: [bold]{selection.backend}[/]  |  "
            f"Active joints dir: [bold]{selection.joints_dir}[/]"
        )

        # Display GPU info
        from .display import print_gpu_config

        print_gpu_config(num_gpus, batch_size)

        # Run the selected GPU motion backend.
        gen_start = time.time()
        backend_result = backend.run(
            MotionBackendContext(
                version=version,
                config=config,
                paths=paths,
                entries=entries,
                num_gpus=num_gpus,
                batch_size=batch_size,
                joints_subdir=selection.joints_subdir,
                backend_config=selection.backend_config,
            )
        )
        result.gen_elapsed = time.time() - gen_start

        # Verify outputs
        result.stats = self._verify_outputs(paths, selection.joints_dir)
        result.stats.errors.extend(backend_result.errors)
        result.stats.errors.extend(
            self._validate_required_outputs(
                paths,
                selection.joints_dir,
                entries,
                expected_ids_override=self._expected_ids_from_backend_config(
                    selection.backend_config
                ),
            )
        )
        if result.stats.errors:
            preview = "\n".join(f"- {err}" for err in result.stats.errors[:10])
            remaining = len(result.stats.errors) - 10
            if remaining > 0:
                preview += f"\n- ... and {remaining} more"
            raise RuntimeError(
                "Motion generation did not produce a complete required output set "
                f"for backend {selection.backend!r} ({selection.joints_subdir!r}).\n"
                f"{preview}"
            )
        result.total_motions = result.stats.joints_count
        result.manifest_path = write_motion_backend_manifest(
            paths,
            backend=selection.backend,
            joints_subdir=selection.joints_subdir,
            counts={
                "joints": result.stats.joints_count,
                "bvh": result.stats.bvh_count,
                "texts": result.stats.texts_count,
                "video": result.stats.video_count,
            },
            runtime_args={
                "version": version,
                "num_gpus": num_gpus,
                "batch_size": batch_size,
                "motion_backend_override": motion_backend,
                "motion_joints_subdir_override": motion_joints_subdir,
                "backend_config": selection.backend_config,
            },
        )

        # Print results
        from .display import print_results_table

        print_results_table(result, config)

        return result

    # -- Input preparation ---------------------------------------------------

    def _prepare_input_files(self, prompts_dir: Path, texts_dir: Path) -> list[PromptEntry]:
        """Use step1 prompt files directly, with each file assigned a global start offset."""
        entries: list[PromptEntry] = []
        start_index = 0
        for txt_file in sorted(prompts_dir.glob("*.txt")):
            prompt_count = sum(
                1
                for line in txt_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if prompt_count == 0:
                continue
            entries.append(
                PromptEntry(
                    action_id=f"{start_index:06d}",
                    source_file=str(txt_file),
                    index=start_index,
                    prompt_count=prompt_count,
                )
            )
            start_index += prompt_count
        return entries

    # -- Output verification -------------------------------------------------

    def _verify_outputs(self, paths: PathManager, joints_dir: Path | None = None) -> MotionGenStats:
        """Verify generated outputs by counting files and checking integrity."""
        import numpy as np

        stats = MotionGenStats()
        active_joints_dir = joints_dir or paths.joints_dir

        # Global counts
        stats.joints_count = (
            sum(1 for _ in active_joints_dir.glob("*.npy"))
            if active_joints_dir.exists()
            else 0
        )
        stats.bvh_count = paths.count_files("step2/bvh", "*.bvh")
        stats.texts_count = paths.count_files("step2/texts", "*.txt")
        stats.video_count = paths.count_files("step2/video", "*.mp4")

        # Per-action counts
        prompts_dir = paths.prompts_dir
        if prompts_dir.exists():
            prompt_count = sum(
                1
                for txt in prompts_dir.glob("*.txt")
                for line in txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            stats.per_action["TASK"] = {
                "texts": prompt_count,
                "joints": stats.joints_count,
                "bvh": stats.bvh_count,
                "step2_texts": stats.texts_count,
                "video": stats.video_count,
            }

        # Spot-check a few joints files for shape integrity
        if active_joints_dir.exists():
            sample_files = list(active_joints_dir.glob("*.npy"))[:3]
            for npy_file in sample_files:
                try:
                    arr = np.load(str(npy_file))
                    if arr.ndim < 2 or arr.shape[0] < 10:
                        stats.errors.append(
                            f"{npy_file.name}: unexpected shape {arr.shape}"
                        )
                except Exception as exc:
                    stats.errors.append(f"{npy_file.name}: load error: {exc}")

        return stats

    def _validate_required_outputs(
        self,
        paths: PathManager,
        joints_dir: Path,
        entries: list[PromptEntry],
        expected_ids_override: list[str] | None = None,
    ) -> list[str]:
        """Check required prompt-aligned outputs for the active backend."""
        expected_ids = expected_ids_override
        if expected_ids is None:
            expected_ids = []
            for entry in entries:
                expected_ids.extend(
                    f"{idx:06d}" for idx in range(entry.index, entry.index + entry.prompt_count)
                )

        errors: list[str] = []
        missing_joints = [
            motion_id for motion_id in expected_ids
            if not (joints_dir / f"{motion_id}.npy").exists()
        ]
        missing_texts = [
            motion_id for motion_id in expected_ids
            if not (paths.texts_dir / f"{motion_id}.txt").exists()
        ]
        if missing_joints:
            errors.append(
                "missing required joints files: "
                + ", ".join(missing_joints[:10])
                + (" ..." if len(missing_joints) > 10 else "")
            )
        if missing_texts:
            errors.append(
                "missing required text files: "
                + ", ".join(missing_texts[:10])
                + (" ..." if len(missing_texts) > 10 else "")
            )
        return errors

    @staticmethod
    def _expected_ids_from_backend_config(cfg: dict | None) -> list[str] | None:
        """Load explicit output ids from an ordered duration manifest when present."""
        if not cfg:
            return None
        raw_path = cfg.get("duration_manifest_path")
        if not raw_path:
            return None
        import json

        path = Path(str(raw_path))
        if not path.exists():
            return None

        items: list[dict] = []
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        items.append(item)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                for key in ("items", "prompts", "extensions"):
                    value = data.get(key)
                    if isinstance(value, list):
                        items = [item for item in value if isinstance(item, dict)]
                        break

        ids = [str(item.get("id", "")).strip() for item in items]
        ids = [motion_id for motion_id in ids if motion_id]
        return ids or None

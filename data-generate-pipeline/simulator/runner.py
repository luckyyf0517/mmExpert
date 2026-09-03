"""mmWave Radar Simulation runner.

Converts 3D joint sequences into simulated mmWave radar returns using an
FMCW radar model (77 GHz, TI IWR1843 8-antenna configuration).

This utility can be used either as Step 3 of the iterative flywheel loop
or as a standalone rerun tool via::

    python run_flywheel.py simulate --round N

Invokes ``simulator/run_simulation.py`` as a subprocess, feeding it a
temporary YAML config derived from the template
``simulator/configs/simulation/simulation_joints.yaml``.

Outputs: mmwave spectrograms (.npy, .npz).
"""

from __future__ import annotations

import re
import os
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import shutil

import numpy as np

from flywheel.config import FlywheelConfig
from flywheel.logging_utils import (
    error_panel,
    get_console,
    get_logger,
    make_progress,
    print_table,
    step_panel,
    summary_panel,
)
from flywheel.path_manager import PathManager

from simulator.sim_types import SimulationResult, SimulationStats

logger = get_logger()
PROGRESS_RELAY_PREFIX = "__MMEXPERT_STEP3_PROGRESS__"
SIMULATION_BACKEND_MANIFEST = "simulation_backend.json"


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *_args, total: int | None = None, **_kwargs) -> int:
        return 0

    def update(self, *_args, **_kwargs) -> None:
        return None


def _emit_progress_event(event: dict[str, object]) -> None:
    print(f"{PROGRESS_RELAY_PREFIX} {json.dumps(event, ensure_ascii=False)}", flush=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _replace_yaml_value(text: str, key: str, value: str) -> str:
    """Replace a top-level scalar value in a simple YAML config string."""
    return re.sub(
        rf"^{key}\s*:.*$",
        f"{key}: {value}",
        text,
        flags=re.MULTILINE,
    )


# ---------------------------------------------------------------------------
# Step implementation
# ---------------------------------------------------------------------------


class mmWaveSimulator:
    """Simulate mmWave radar returns from 3D joint sequences."""

    # Radar configuration constants
    RADAR_FREQ_GHZ = 77
    NUM_ANTENNAS = 8
    DEFAULT_FPS = 30
    DEFAULT_INTERP_POINTS = 50

    def __init__(self) -> None:
        self._pipeline_root: Path = Path(__file__).resolve().parent.parent

    # -- Public API ----------------------------------------------------------

    def run(
        self,
        version: str,
        config: FlywheelConfig,
        paths: PathManager,
        *,
        fps: int = 30,
        interpolation_points: int = 50,
        skip_exist: bool = True,
        output_mode: str = "udoppler",
        gpu_count: int = 1,
        simulation_config: str | None = None,
        joints_dir: str | Path | None = None,
        motion_backend: str | None = None,
        joints_subdir: str | None = None,
    ) -> SimulationResult:
        """Execute mmWave radar simulation.

        Parameters
        ----------
        version:
            Version string (e.g. ``"round_0"``).
        config:
            Loaded flywheel configuration.
        paths:
            PathManager for the current version.
        fps:
            Simulation frame rate (default 20).
        interpolation_points:
            Number of interpolation points between joints (default 50).
        skip_exist:
            Skip files that already have mmwave outputs.
        output_mode:
            "full" for multi-view .npz (range+doppler+azimuth),
            "udoppler" for doppler-only .npy.

        Returns
        -------
        SimulationResult with simulation stats and timing.
        """
        self._output_mode = output_mode
        step_panel(
            "mmWave Radar Simulation",
            subtitle=(
                f"Version: {version}  |  "
                f"Freq: {self.RADAR_FREQ_GHZ} GHz  |  "
                f"Antennas: {self.NUM_ANTENNAS}  |  "
                f"FPS: {fps}  |  "
                f"Mode: {output_mode}"
            ),
        )

        # Validate prerequisites
        active_joints_dir = Path(joints_dir) if joints_dir is not None else paths.joints_dir
        backend_label = motion_backend or "hunyuan_motion"
        subdir_label = joints_subdir or active_joints_dir.name
        if not active_joints_dir.exists():
            raise FileNotFoundError(
                "Joints directory not found for motion backend "
                f"{backend_label!r} (subdir {subdir_label!r}): {active_joints_dir}. "
                "Run motion generation first."
            )

        # Discover joints files
        joints_files = self._discover_joints_files(active_joints_dir)
        if not joints_files:
            raise FileNotFoundError(
                "No joints .npy files found for motion backend "
                f"{backend_label!r} (subdir {subdir_label!r}) in {active_joints_dir}. "
                "Run motion generation first."
            )

        console = get_console()
        console.print(
            f"Found [bold]{len(joints_files)}[/] joint sequences to simulate"
        )

        # Ensure output directories exist
        output_dir = paths.udoppler_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        paths.step3_texts_dir.mkdir(parents=True, exist_ok=True)

        effective_skip_exist = skip_exist
        if skip_exist:
            should_force, reason = self._should_disable_skip_for_backend_change(
                paths=paths,
                output_dir=output_dir,
                backend=backend_label,
                joints_subdir=subdir_label,
                joints_dir=active_joints_dir,
            )
            if should_force:
                effective_skip_exist = False
                console.print(
                    "[yellow]Existing Step 3 outputs were produced by a different "
                    f"motion backend; rerunning instead of skipping ({reason}).[/]"
                )

        result = SimulationResult()

        # Run simulation
        start = time.time()
        self._run_simulation(
            joints_files=joints_files,
            output_dir=output_dir,
            fps=fps,
            interpolation_points=interpolation_points,
            skip_exist=effective_skip_exist,
            gpu_count=gpu_count,
            simulation_config=simulation_config,
        )
        self._sync_training_texts(paths)
        result.elapsed = time.time() - start

        # Verify outputs
        result.stats = self._verify_outputs(output_dir, active_joints_dir, config)
        result.total_simulations = (
            result.stats.total_output_npy + result.stats.total_output_npz
        )
        self._write_simulation_backend_manifest(
            paths=paths,
            backend=backend_label,
            joints_subdir=subdir_label,
            joints_dir=active_joints_dir,
            output_dir=output_dir,
            output_mode=output_mode,
            fps=fps,
            interpolation_points=interpolation_points,
            simulation_config=simulation_config,
            skip_exist=effective_skip_exist,
            counts={
                "input_joints": result.stats.total_input,
                "output_npy": result.stats.total_output_npy,
                "output_npz": result.stats.total_output_npz,
            },
        )

        # Print results
        self._print_results_table(result, config)
        console.print(self._generate_report(result.stats))

        return result

    def _sync_training_texts(self, paths: PathManager) -> None:
        """Copy matching step2 texts into step3/texts for training pairs."""
        source_dir = paths.texts_dir
        target_dir = paths.step3_texts_dir
        if not source_dir.exists():
            return
        for text_file in sorted(source_dir.glob("*.txt")):
            target_path = target_dir / text_file.name
            shutil.copy2(text_file, target_path)

    def _should_disable_skip_for_backend_change(
        self,
        *,
        paths: PathManager,
        output_dir: Path,
        backend: str,
        joints_subdir: str,
        joints_dir: Path,
    ) -> tuple[bool, str | None]:
        """Return whether existing outputs should be overwritten for backend safety."""
        if not self._has_simulation_outputs(output_dir):
            return False, None

        manifest = self._read_simulation_backend_manifest(paths)
        current = {
            "backend": backend,
            "joints_subdir": joints_subdir,
            "joints_dir": str(joints_dir.resolve()),
        }
        if not manifest:
            return True, "no previous Step 3 backend manifest"

        previous = {
            "backend": manifest.get("backend"),
            "joints_subdir": manifest.get("joints_subdir"),
            "joints_dir": manifest.get("joints_dir"),
        }
        if previous != current:
            return True, f"previous={previous}, current={current}"
        return False, None

    @staticmethod
    def _has_simulation_outputs(output_dir: Path) -> bool:
        if not output_dir.exists():
            return False
        return any(output_dir.glob("*.npy")) or any(output_dir.glob("*.npz"))

    @staticmethod
    def _read_simulation_backend_manifest(paths: PathManager) -> dict:
        manifest_path = paths.step3_dir / SIMULATION_BACKEND_MANIFEST
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_simulation_backend_manifest(
        *,
        paths: PathManager,
        backend: str,
        joints_subdir: str,
        joints_dir: Path,
        output_dir: Path,
        output_mode: str,
        fps: int,
        interpolation_points: int,
        simulation_config: str | None,
        skip_exist: bool,
        counts: dict[str, int],
    ) -> None:
        from datetime import datetime, timezone

        manifest_path = paths.step3_dir / SIMULATION_BACKEND_MANIFEST
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": backend,
            "joints_subdir": joints_subdir,
            "joints_dir": str(joints_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "output_mode": output_mode,
            "fps": fps,
            "interpolation_points": interpolation_points,
            "simulation_config": simulation_config,
            "skip_exist": skip_exist,
            "counts": counts,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # -- File discovery ------------------------------------------------------

    def _discover_joints_files(self, joints_dir: Path) -> list[Path]:
        """Collect all .npy joint files from the step2 output directory.

        Returns a sorted list of paths for deterministic processing order.
        """
        if not joints_dir.exists():
            return []
        return sorted(joints_dir.glob("*.npy"))

    # -- Simulation ----------------------------------------------------------

    def _run_simulation(
        self,
        joints_files: list[Path],
        output_dir: Path,
        fps: int,
        interpolation_points: int,
        skip_exist: bool,
        gpu_count: int,
        simulation_config: str | None,
    ) -> None:
        """Run the mmWave radar simulation via ``simulator/run_simulation.py``.

        Constructs a temporary YAML config from the template and invokes
        the simulator as a subprocess.  Progress is tracked per-file by
        monitoring stdout from the subprocess.
        """
        console = get_console()
        simulator_dir = self._pipeline_root / "simulator"
        config_template = (
            Path(simulation_config)
            if simulation_config
            else simulator_dir / "configs" / "simulation" / "simulation_joints.yaml"
        )

        if not simulator_dir.exists():
            raise FileNotFoundError(
                f"Simulator directory not found: {simulator_dir}"
            )
        if not config_template.exists():
            raise FileNotFoundError(
                f"Simulation config template not found: {config_template}"
            )

        tmp_file = None
        try:
            config_text = config_template.read_text(encoding="utf-8")
            joints_dir = joints_files[0].parent
            data_pattern = f"{joints_dir}/*.npy"
            config_text = _replace_yaml_value(config_text, "data_pattern", data_pattern)
            config_text = _replace_yaml_value(config_text, "output_dir", str(output_dir))
            if not simulation_config:
                config_text = _replace_yaml_value(
                    config_text, "output_mode", getattr(self, "_output_mode", "udoppler")
                )
                config_text = _replace_yaml_value(config_text, "fps", str(fps))
                config_text = _replace_yaml_value(
                    config_text, "interp_n", str(interpolation_points)
                )
            config_text = _replace_yaml_value(
                config_text,
                "skip_exist",
                "true" if skip_exist else "false",
            )

            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            )
            tmp_file.write(config_text)
            tmp_file.close()
            config_path = tmp_file.name

            base_cmd = [
                sys.executable,
                str(simulator_dir / "run_simulation.py"),
                "--config",
                config_path,
                "--yes",
                "--no-rich-progress",
            ]

            worker_count = max(1, min(gpu_count, len(joints_files)))
            slices = self._make_slices(len(joints_files), worker_count)
            relay_progress = os.environ.get("MMEXPERT_TOPLEVEL_PROGRESS") == "1"

            logger.info("Simulation cmd: %s", " ".join(base_cmd))
            console.print(
                f"\n[bold]Running simulation[/] "
                f"({len(joints_files)} files, method=points, "
                f"fps={fps}, interp={interpolation_points}, workers={worker_count})"
            )

            if relay_progress:
                _emit_progress_event(
                    {
                        "event": "start",
                        "total": len(joints_files),
                        "workers": [
                            {"gpu": worker_idx, "total": end - start}
                            for worker_idx, start, end in slices
                        ],
                    }
                )
            progress = _NullProgress() if relay_progress else make_progress()
            lock = None
            with progress:
                from threading import Lock
                from concurrent.futures import ThreadPoolExecutor, as_completed

                lock = Lock()
                overall_task = progress.add_task(
                    "[bold green]Overall[/]", total=len(joints_files)
                )
                worker_tasks = {}
                for worker_idx, start, end in slices:
                    task = progress.add_task(
                        f"[cyan]GPU {worker_idx}[/]",
                        total=end - start,
                    )
                    worker_tasks[worker_idx] = task

                errors: list[str] = []
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(
                            self._run_simulation_slice,
                            worker_idx,
                            start,
                            end,
                            base_cmd,
                            simulator_dir,
                            progress,
                            overall_task,
                            worker_tasks[worker_idx],
                            lock,
                        )
                        for worker_idx, start, end in slices
                    ]
                    for future in as_completed(futures):
                        error = future.result()
                        if error:
                            errors.append(error)

                if errors:
                    message = "\n\n".join(errors)
                    logger.error("Simulation failed:\n%s", message)
                    if relay_progress:
                        _emit_progress_event({"event": "error", "message": message})
                    error_panel(message, title="Simulation Error")
                    raise RuntimeError("Simulation failed. See full simulator output above.")
                else:
                    progress.update(overall_task, completed=len(joints_files))
                    for worker_idx, start, end in slices:
                        progress.update(
                            worker_tasks[worker_idx],
                            completed=end - start,
                        )
                    if relay_progress:
                        _emit_progress_event({"event": "done", "total": len(joints_files)})
                    logger.info("Simulation completed successfully")
        finally:
            if tmp_file is not None:
                Path(tmp_file.name).unlink(missing_ok=True)

    @staticmethod
    def _make_slices(total: int, worker_count: int) -> list[tuple[int, int, int]]:
        """Split total files into contiguous [start, end) ranges."""
        chunk = (total + worker_count - 1) // worker_count
        slices = []
        for worker_idx in range(worker_count):
            start = worker_idx * chunk
            end = min(total, start + chunk)
            if start < end:
                slices.append((worker_idx, start, end))
        return slices

    def _run_simulation_slice(
        self,
        worker_idx: int,
        start: int,
        end: int,
        base_cmd: list[str],
        simulator_dir: Path,
        progress: object,
        overall_task: int,
        worker_task: int,
        lock: object,
    ) -> str | None:
        """Run one run_simulation.py slice on one visible GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_idx)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = base_cmd + ["--start", str(start), "--end", str(end)]

        proc = subprocess.Popen(
            cmd,
            cwd=str(simulator_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        output_lines: list[str] = []
        last_completed = 0

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            output_lines.append(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            if "[PROGRESS]" in line:
                completed = self._parse_progress_completed(line)
                if completed is None:
                    continue
                completed = min(completed, end - start)
                delta = max(0, completed - last_completed)
                last_completed = completed
                if delta:
                    with lock:
                        progress.update(worker_task, completed=completed)  # type: ignore[attr-defined]
                        progress.update(overall_task, advance=delta)  # type: ignore[attr-defined]
                if os.environ.get("MMEXPERT_TOPLEVEL_PROGRESS") == "1":
                    _emit_progress_event(
                        {
                            "event": "update",
                            "gpu": worker_idx,
                            "completed": completed,
                            "total": end - start,
                        }
                    )
            elif "ERROR" in line or "Traceback" in line:
                logger.warning("Simulator GPU %d: %s", worker_idx, line)

        proc.wait()
        if proc.returncode != 0:
            combined_output = "".join(output_lines).strip()
            return (
                f"GPU {worker_idx} slice {start}:{end} failed with "
                f"exit code {proc.returncode}\n{combined_output}"
            )
        return None

    @staticmethod
    def _parse_progress_completed(line: str) -> int | None:
        """Parse '[PROGRESS] current/total ...' lines from run_simulation.py."""
        try:
            after = line.split("[PROGRESS]", 1)[1].strip()
            current, _ = after.split("/", 1)
            return int(current.strip())
        except (ValueError, IndexError):
            return None

    # -- Output verification -------------------------------------------------

    def _verify_outputs(
        self,
        output_dir: Path,
        joints_dir: Path,
        config: FlywheelConfig,
    ) -> SimulationStats:
        """Verify simulation outputs.

        Counts output files, checks data integrity on a sample, and
        aggregates per-action statistics.
        """
        stats = SimulationStats()

        # Global input count
        stats.total_input = sum(1 for _ in joints_dir.glob("*.npy")) if joints_dir.exists() else 0

        # Global output counts
        if output_dir.exists():
            npy_files = list(output_dir.glob("*.npy"))
            npz_files = list(output_dir.glob("*.npz"))
            stats.total_output_npy = len(npy_files)
            stats.total_output_npz = len(npz_files)

            # Average file size
            all_output = npy_files + npz_files
            if all_output:
                total_bytes = sum(f.stat().st_size for f in all_output)
                stats.avg_file_size_kb = total_bytes / len(all_output) / 1024

        stats.per_action["TASK"] = {
            "input_files": stats.total_input,
            "output_files": stats.total_output_npy + stats.total_output_npz,
            "avg_size_kb": round(stats.avg_file_size_kb, 1),
        }

        # Spot-check a few output files for data integrity
        if output_dir.exists():
            sample_files = sorted(output_dir.glob("*.npz"))[:3]
            for npz_file in sample_files:
                try:
                    data = np.load(str(npz_file))
                    if not hasattr(data, "files") or len(data.files) == 0:
                        stats.errors.append(
                            f"{npz_file.name}: empty npz archive"
                        )
                except Exception as exc:
                    stats.errors.append(f"{npz_file.name}: load error: {exc}")

            sample_npy = sorted(output_dir.glob("*.npy"))[:3]
            for npy_file in sample_npy:
                try:
                    arr = np.load(str(npy_file))
                    if arr.size == 0:
                        stats.errors.append(
                            f"{npy_file.name}: empty array"
                        )
                except Exception as exc:
                    stats.errors.append(f"{npy_file.name}: load error: {exc}")

        return stats

    # -- Reporting -----------------------------------------------------------

    def _generate_report(self, stats: SimulationStats) -> str:
        """Generate a human-readable summary string for the summary panel."""
        lines = [
            f"Input joints:    [bold]{stats.total_input}[/]",
            f"Output .npy:     {stats.total_output_npy}",
            f"Output .npz:     {stats.total_output_npz}",
            f"Avg file size:   {stats.avg_file_size_kb:.1f} KB",
        ]
        if stats.errors:
            lines.append(f"[red]Errors: {len(stats.errors)}[/]")
        return "\n".join(lines)

    def _print_results_table(
        self, result: SimulationResult, config: FlywheelConfig
    ) -> None:
        """Print per-action results table and summary panel."""
        stats = result.stats

        rows: list[tuple[str, ...]] = []
        for action in config.actions:
            action_stats = stats.per_action.get(action.id, {})
            rows.append(
                (
                    action.id,
                    action_stats.get("name", action.name),
                    str(action_stats.get("input_files", 0)),
                    str(action_stats.get("output_files", 0)),
                    f"{action_stats.get('avg_size_kb', 0):.1f} KB",
                )
            )

        if rows:
            print_table(
                "Simulation Results by Action",
                [
                    "Action ID",
                    "Action",
                    "Input Files",
                    "Output Files",
                    "Avg Size",
                ],
                rows,
                styles=["cyan", None, "green", "bold", "dim"],
            )

        error_info = ""
        if stats.errors:
            error_info = f"\n[red]Data errors: {len(stats.errors)}[/]"

        summary_panel(
            f"Input joints:    [bold]{stats.total_input}[/]\n"
            f"Output .npy:     {stats.total_output_npy}\n"
            f"Output .npz:     {stats.total_output_npz}\n"
            f"Avg file size:   {stats.avg_file_size_kb:.1f} KB\n"
            f"Elapsed:         {result.elapsed:.1f}s"
            f"{error_info}",
            title="mmWave Simulation Complete",
        )

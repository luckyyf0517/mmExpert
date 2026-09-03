"""Hunyuan Motion text-to-motion backend."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from ....logging_utils import get_logger, make_progress
from ....types import PromptEntry
from .base import MotionBackendContext, MotionBackendRunResult
from .hunyuan_motion_runner import PROGRESS_PREFIX

logger = get_logger()
DEFAULT_DURATION_MIN_FRAMES = 180
DEFAULT_DURATION_MAX_FRAMES = 300
DEFAULT_DURATION_SEED = 100039


class HunyuanMotionBackend:
    """Backend wrapper for HY-Motion-1.0 generation."""

    name = "hunyuan_motion"
    default_joints_subdir = "joints_hunyuan_motion"

    def run(self, context: MotionBackendContext) -> MotionBackendRunResult:
        cfg = dict(context.backend_config or {})
        tasks = self._collect_tasks(list(context.entries), cfg)
        if not tasks:
            raise FileNotFoundError("No non-empty prompts found for Hunyuan Motion")

        pipeline_root = Path(__file__).resolve().parents[4]
        hymotion_dir = pipeline_root / "backends" / "hymotion"
        if not hymotion_dir.exists():
            raise FileNotFoundError(f"Hunyuan Motion directory not found: {hymotion_dir}")

        model_path = self._resolve_model_path(cfg, hymotion_dir)
        artifacts_dir = Path(
            cfg.get("artifacts_dir") or context.paths.step2_dir / self.name
        )
        input_manifest = artifacts_dir / "input_manifest.json"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        input_manifest.write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        context.paths.texts_dir.mkdir(parents=True, exist_ok=True)
        context.paths.motion_joints_dir(context.joints_subdir).mkdir(
            parents=True,
            exist_ok=True,
        )

        errors = self._run_subprocess(
            context=context,
            cfg=cfg,
            pipeline_root=pipeline_root,
            hymotion_dir=hymotion_dir,
            model_path=model_path,
            input_manifest=input_manifest,
            artifacts_dir=artifacts_dir,
            total=len(tasks),
        )

        return MotionBackendRunResult(
            backend=self.name,
            joints_subdir=context.joints_subdir,
            errors=errors,
        )

    def _run_subprocess(
        self,
        *,
        context: MotionBackendContext,
        cfg: dict[str, Any],
        pipeline_root: Path,
        hymotion_dir: Path,
        model_path: Path,
        input_manifest: Path,
        artifacts_dir: Path,
        total: int,
    ) -> list[str]:
        cmd = self._build_command(
            context=context,
            cfg=cfg,
            runner_path=Path(__file__).with_name("hunyuan_motion_runner.py"),
            model_path=model_path,
            input_manifest=input_manifest,
            artifacts_dir=artifacts_dir,
        )
        env = self._build_env(cfg, pipeline_root, hymotion_dir)

        progress = make_progress()
        errors: list[str] = []
        output_lines: list[str] = []

        with progress:
            overall_task_id = progress.add_task("[bold green]Overall[/]", total=total)
            proc = subprocess.Popen(
                cmd,
                cwd=str(hymotion_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            completed = 0
            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                stripped = line.strip()
                if stripped.startswith(PROGRESS_PREFIX):
                    event = self._parse_progress(stripped)
                    if event.get("event") == "advance":
                        completed = min(total, completed + 1)
                        progress.update(overall_task_id, completed=completed)
                    elif event.get("event") == "done":
                        progress.update(overall_task_id, completed=total)
                elif stripped:
                    logger.debug("[hunyuan_motion] %s", stripped)

            return_code = proc.wait()
            if return_code != 0:
                snippet = "".join(output_lines[-80:]) or f"Process exited with code {return_code}"
                errors.append(snippet)
                raise RuntimeError(
                    "Hunyuan Motion backend failed with exit code "
                    f"{return_code}. Last subprocess output:\n{snippet}"
                )

        return errors

    @classmethod
    def _collect_tasks(
        cls,
        entries: list[PromptEntry],
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        duration_policy = _duration_policy(cfg)
        duration_by_id = _load_duration_manifest(cfg.get("duration_manifest_path"))
        manifest_items = _load_ordered_duration_manifest_items(
            cfg.get("duration_manifest_path")
        )
        duration_rng = random.Random(
            int(cfg.get("duration_seed", DEFAULT_DURATION_SEED))
        )
        row_index = 0
        for entry in entries:
            source = Path(entry.source_file)
            lines = source.read_text(encoding="utf-8").splitlines()
            next_index = entry.index
            for line_number, line in enumerate(lines, start=1):
                prompt = line.strip()
                if not prompt:
                    continue
                manifest_item = (
                    manifest_items[row_index]
                    if manifest_items is not None and row_index < len(manifest_items)
                    else None
                )
                motion_id = (
                    str(manifest_item.get("id", "")).strip()
                    if manifest_item is not None
                    else ""
                )
                if not motion_id:
                    motion_id = f"{next_index:06d}"
                if manifest_item is not None and manifest_item.get("duration_frames") is not None:
                    duration_frames = int(manifest_item["duration_frames"])
                    duration_source = "manifest_ordered"
                elif duration_by_id and motion_id in duration_by_id:
                    duration_frames = duration_by_id[motion_id]
                    duration_source = "manifest"
                else:
                    duration_frames, duration_source = cls._duration_for_task(
                        cfg,
                        duration_policy,
                        duration_rng,
                    )
                tasks.append(
                    {
                        "id": motion_id,
                        "prompt": prompt,
                        "duration_frames": duration_frames,
                        "duration_source": duration_source,
                        "source_file": str(source),
                        "source_line": line_number,
                    }
                )
                next_index += 1
                row_index += 1
        return tasks

    @staticmethod
    def _duration_for_task(
        cfg: dict[str, Any],
        policy: str,
        rng: random.Random,
    ) -> tuple[int, str]:
        if policy == "fixed":
            frames = int(cfg.get("duration_frames", 100))
            if frames <= 0:
                raise ValueError(f"duration_frames must be positive, got {frames}")
            return frames, "fixed"

        if policy in {"uniform_random", "random"}:
            min_frames = int(cfg.get("duration_min_frames", DEFAULT_DURATION_MIN_FRAMES))
            max_frames = int(cfg.get("duration_max_frames", DEFAULT_DURATION_MAX_FRAMES))
            if min_frames <= 0 or max_frames <= min_frames:
                raise ValueError(
                    "duration_min_frames/duration_max_frames must define a positive "
                    f"exclusive range, got {min_frames}..{max_frames}"
                )
            return rng.randrange(min_frames, max_frames), "uniform_random"

        raise ValueError(
            "Unsupported Hunyuan Motion duration_policy "
            f"{policy!r}; expected 'uniform_random' or 'fixed'"
        )

    @staticmethod
    def _resolve_model_path(cfg: dict[str, Any], hymotion_dir: Path) -> Path:
        raw = (
            cfg.get("model_path")
            or os.environ.get("HY_MOTION_MODEL_PATH")
            or hymotion_dir / "ckpts" / "tencent" / "HY-Motion-1.0"
        )
        path = Path(raw)
        if not path.is_absolute():
            pipeline_root = hymotion_dir.parents[1]
            repo_root = hymotion_dir.parents[2]
            candidates = [
                hymotion_dir / path,
                pipeline_root / path,
                repo_root / path,
                Path.cwd() / path,
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        return path.resolve()

    def _build_command(
        self,
        *,
        context: MotionBackendContext,
        cfg: dict[str, Any],
        runner_path: Path,
        model_path: Path,
        input_manifest: Path,
        artifacts_dir: Path,
    ) -> list[str]:
        cmd = [
            sys.executable,
            str(runner_path),
            "--model_path",
            str(model_path),
            "--input_manifest",
            str(input_manifest),
            "--artifacts_dir",
            str(artifacts_dir),
            "--joints_dir",
            str(context.paths.motion_joints_dir(context.joints_subdir)),
            "--texts_dir",
            str(context.paths.texts_dir),
            "--device_ids",
            self._device_ids(cfg, context.num_gpus),
            "--cfg_scale",
            str(float(cfg.get("cfg_scale", 5.0))),
            "--duration_frames",
            str(int(cfg.get("duration_frames", 100))),
            "--num_seeds",
            str(int(cfg.get("num_seeds", 1))),
            "--seed",
            str(int(cfg.get("seed", 42))),
            "--max_workers",
            str(int(cfg.get("max_workers", max(1, context.num_gpus)))),
        ]

        if cfg.get("validation_steps") is not None:
            cmd.extend(["--validation_steps", str(int(cfg["validation_steps"]))])
        if _as_bool(cfg.get("disable_rewrite", True)):
            cmd.append("--disable_rewrite")
        if _as_bool(cfg.get("disable_duration_est", True)):
            cmd.append("--disable_duration_est")

        prompt_model_path = cfg.get("prompt_engineering_model_path") or os.environ.get(
            "HY_MOTION_PROMPT_ENGINEERING_MODEL_PATH"
        )
        prompt_host = cfg.get("prompt_engineering_host") or os.environ.get(
            "HY_MOTION_PROMPT_ENGINEERING_HOST"
        )
        if prompt_model_path:
            cmd.extend(["--prompt_engineering_model_path", str(prompt_model_path)])
        if prompt_host:
            cmd.extend(["--prompt_engineering_host", str(prompt_host)])
        return cmd

    @staticmethod
    def _device_ids(cfg: dict[str, Any], num_gpus: int) -> str:
        raw = cfg.get("device_ids")
        if raw is None:
            return ",".join(str(i) for i in range(max(0, num_gpus)))
        if isinstance(raw, (list, tuple)):
            return ",".join(str(int(item)) for item in raw)
        return str(raw)

    @staticmethod
    def _build_env(
        cfg: dict[str, Any],
        pipeline_root: Path,
        hymotion_dir: Path,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        python_path_parts = [str(pipeline_root), str(hymotion_dir)]
        existing_python_path = env.get("PYTHONPATH")
        if existing_python_path:
            python_path_parts.append(existing_python_path)
        env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

        if cfg.get("use_hf_models") is not None:
            env["USE_HF_MODELS"] = "1" if _as_bool(cfg["use_hf_models"]) else "0"
        return env

    @staticmethod
    def _parse_progress(line: str) -> dict[str, Any]:
        payload = line[len(PROGRESS_PREFIX):].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _duration_policy(cfg: dict[str, Any]) -> str:
    return (
        str(cfg.get("duration_policy", "uniform_random"))
        .strip()
        .lower()
        .replace("-", "_")
    )


def _load_duration_manifest(raw_path: object) -> dict[str, int] | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"Hunyuan Motion duration manifest not found: {path}")

    def add_item(target: dict[str, int], item: object) -> None:
        if not isinstance(item, dict):
            return
        motion_id = str(item.get("id", "")).strip()
        duration = item.get("duration_frames", item.get("motion_length"))
        if motion_id and duration is not None:
            target[motion_id] = int(duration)

    durations: dict[str, int] = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                add_item(durations, json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                add_item(durations, item)
        elif isinstance(data, dict):
            for key in ("items", "prompts", "extensions"):
                if isinstance(data.get(key), list):
                    for item in data[key]:
                        add_item(durations, item)
                    break
            else:
                for motion_id, duration in data.items():
                    if isinstance(duration, int):
                        durations[str(motion_id)] = duration
                    elif isinstance(duration, dict):
                        add_item(durations, {"id": motion_id, **duration})
    return durations


def _load_ordered_duration_manifest_items(raw_path: object) -> list[dict[str, Any]] | None:
    """Return ordered manifest rows when duration_manifest_path is a JSON/JSONL list.

    Older configs only used the manifest as an id -> duration lookup and relied on
    sequential ids such as 000000. Table3 caption-overlap needs explicit ids such
    as 000000_000 so external and local outputs share one naming contract.
    """
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        return None

    items: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                items.append(item)
        return items or None

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)] or None
    if isinstance(data, dict):
        for key in ("items", "prompts", "extensions"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)] or None
    return None

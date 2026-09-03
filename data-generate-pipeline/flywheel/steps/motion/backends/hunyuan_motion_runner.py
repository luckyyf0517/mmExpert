"""Subprocess runner for the Hunyuan Motion backend.

This module lives in the mmExpert wrapper layer rather than the HY-Motion
submodule. It imports HY-Motion at runtime, generates one motion per prompt,
retains the pose/translation artifacts used by the mesh simulator, and exports
body joints for the optional point-scatterer path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROGRESS_PREFIX = "__MMEXPERT_HYMOTION_PROGRESS__"
HUNYUAN_BODY_JOINT_COUNT = 22


def main() -> int:
    args = _parse_args()

    manifest_path = Path(args.input_manifest)
    model_path = Path(args.model_path).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    joints_dir = Path(args.joints_dir).resolve()
    texts_dir = Path(args.texts_dir).resolve()

    _validate_model_path(model_path)
    _validate_hymotion_lfs_assets()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError(f"Input manifest must be a list: {manifest_path}")

    if args.num_seeds != 1:
        raise ValueError(
            "Hunyuan Motion backend currently supports exactly one seed per prompt "
            f"to keep one prompt -> one joints/text pair; got num_seeds={args.num_seeds}"
        )

    _write_schema(artifacts_dir)

    from hymotion.utils.t2m_runtime import T2MRuntime

    cfg_path = _prepare_runtime_config(model_path, artifacts_dir)
    ckpt_path = model_path / "latest.ckpt"
    device_ids = _parse_device_ids(args.device_ids)

    print(">>> Initializing Hunyuan Motion runtime...", flush=True)
    runtime = T2MRuntime(
        config_path=str(cfg_path),
        ckpt_name=str(ckpt_path),
        device_ids=device_ids,
        disable_prompt_engineering=args.disable_duration_est and args.disable_rewrite,
        prompt_engineering_host=args.prompt_engineering_host,
        prompt_engineering_model_path=args.prompt_engineering_model_path,
    )
    if args.validation_steps is not None:
        for pipeline in runtime.pipelines:
            pipeline.validation_steps = args.validation_steps

    rng = random.Random(args.seed)
    work_items = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        prompt = str(task.get("prompt", "")).strip()
        motion_id = str(task.get("id", "")).strip()
        if not prompt or not motion_id:
            continue
        work_items.append(
            {
                **task,
                "prompt": prompt,
                "id": motion_id,
                "seed": rng.randint(0, 999999),
            }
        )

    results: list[dict[str, Any]] = []
    failed = 0
    max_workers = args.max_workers or max(1, len(device_ids) if device_ids else 1)
    max_workers = max(1, min(max_workers, len(work_items) or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_one,
                runtime,
                item,
                args,
                joints_dir,
                texts_dir,
                artifacts_dir,
            ): item
            for item in work_items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                _emit_progress({"event": "advance", "id": item["id"], "status": "success"})
            except Exception as exc:
                failed += 1
                result = {
                    "id": item.get("id"),
                    "prompt": item.get("prompt"),
                    "status": "failed",
                    "error": str(exc),
                }
                results.append(result)
                print(f">>> Hunyuan task failed for {item.get('id')}: {exc}", flush=True)
                _emit_progress({"event": "advance", "id": item.get("id"), "status": "failed"})

    summary = {
        "backend": "hunyuan_motion",
        "joint_schema": "hunyuan_motion_body22_keypoints3d",
        "total": len(work_items),
        "success": len(work_items) - failed,
        "failed": failed,
        "results": sorted(results, key=lambda item: str(item.get("id", ""))),
    }
    (artifacts_dir / "generation_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _emit_progress({"event": "done", "total": len(work_items), "failed": failed})
    return 1 if failed else 0


def _generate_one(
    runtime: Any,
    item: dict[str, Any],
    args: argparse.Namespace,
    joints_dir: Path,
    texts_dir: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    prompt = item["prompt"]
    motion_id = item["id"]
    duration_frames = int(item.get("duration_frames") or args.duration_frames)
    if duration_frames <= 0:
        raise ValueError(f"duration_frames must be positive, got {duration_frames}")
    duration = duration_frames / 30.0
    duration_source = str(item.get("duration_source") or "fixed")
    rewritten_text = prompt

    if not (args.disable_duration_est and args.disable_rewrite):
        predicted_duration, predicted_text = runtime.rewrite_text_and_infer_time(text=prompt)
        if not args.disable_duration_est:
            duration = predicted_duration
            duration_frames = int(round(duration * 30))
            duration_source = "hunyuan_prompt_engineering"
        if not args.disable_rewrite:
            rewritten_text = predicted_text

    html_content, _fbx_files, model_output = runtime.generate_motion(
        text=rewritten_text,
        seeds_csv=str(item["seed"]),
        duration=duration,
        cfg_scale=args.cfg_scale,
        output_format="dict",
        original_text=prompt,
        output_dir=str(artifacts_dir),
        output_filename=motion_id,
    )

    keypoints = model_output.get("keypoints3d")
    if keypoints is None:
        raise ValueError("HY-Motion model output did not contain keypoints3d")
    transl = model_output.get("transl")
    if transl is None:
        raise ValueError("HY-Motion model output did not contain transl")

    keypoints_np = _to_numpy(keypoints)
    if keypoints_np.ndim != 4 or keypoints_np.shape[0] < 1:
        raise ValueError(f"Unexpected keypoints3d shape: {keypoints_np.shape}")
    transl_np = _to_numpy(transl)
    if transl_np.ndim != 3 or transl_np.shape[0] < 1:
        raise ValueError(f"Unexpected transl shape: {transl_np.shape}")
    if transl_np.shape[1] != keypoints_np.shape[1]:
        raise ValueError(
            "HY-Motion transl frame count does not match keypoints3d: "
            f"transl={transl_np.shape}, keypoints3d={keypoints_np.shape}"
        )

    joints = _select_body_joints(keypoints_np[0] + transl_np[0, :, None, :]).astype(
        np.float32,
        copy=False,
    )
    np.save(joints_dir / f"{motion_id}.npy", joints)
    (texts_dir / f"{motion_id}.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
    (artifacts_dir / f"{motion_id}.html").write_text(str(html_content), encoding="utf-8")
    wooden_npz_files = sorted(artifacts_dir.glob(f"{motion_id}_*.npz"))
    wooden_npz_path = str(wooden_npz_files[0]) if wooden_npz_files else ""

    return {
        "id": motion_id,
        "status": "success",
        "prompt": prompt,
        "rewritten_text": rewritten_text,
        "duration": duration,
        "duration_frames": duration_frames,
        "duration_source": duration_source,
        "seed": item["seed"],
        "joints_path": str(joints_dir / f"{motion_id}.npy"),
        "wooden_npz_path": wooden_npz_path,
        "wooden_npz_schema": "hunyuan_wooden_smplx_npz" if wooden_npz_path else "",
        "shape": list(joints.shape),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mmExpert HY-Motion backend runner")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--artifacts_dir", required=True)
    parser.add_argument("--joints_dir", required=True)
    parser.add_argument("--texts_dir", required=True)
    parser.add_argument("--device_ids", default="")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--duration_frames", type=int, default=100)
    parser.add_argument("--disable_rewrite", action="store_true")
    parser.add_argument("--disable_duration_est", action="store_true")
    parser.add_argument("--prompt_engineering_model_path", default=None)
    parser.add_argument("--prompt_engineering_host", default=None)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=None)
    return parser.parse_args()


def _validate_model_path(model_path: Path) -> None:
    cfg_path = model_path / "config.yml"
    ckpt_path = model_path / "latest.ckpt"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Hunyuan Motion config not found: {cfg_path}. "
            "Set motion_generation.backend_config.model_path or HY_MOTION_MODEL_PATH."
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Hunyuan Motion checkpoint not found: {ckpt_path}. "
            "Download/copy the HY-Motion weights before running Step 2."
        )


def _prepare_runtime_config(model_path: Path, artifacts_dir: Path) -> Path:
    cfg_path = model_path / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected Hunyuan Motion config format: {cfg_path}")

    repo_root = Path.cwd()
    stats_dir = repo_root / "stats"
    if not (stats_dir / "Mean.npy").exists() or not (stats_dir / "Std.npy").exists():
        raise FileNotFoundError(
            f"Hunyuan Motion stats not found: {stats_dir}. "
            "Run `git -C data-generate-pipeline/backends/hymotion lfs pull`."
        )

    train_args = cfg.setdefault("train_pipeline_args", {})
    if not isinstance(train_args, dict):
        raise ValueError("Hunyuan Motion config train_pipeline_args must be a mapping")
    test_cfg = train_args.setdefault("test_cfg", {})
    if not isinstance(test_cfg, dict):
        raise ValueError("Hunyuan Motion config train_pipeline_args.test_cfg must be a mapping")

    configured_stats = Path(str(test_cfg.get("mean_std_dir", "")))
    configured_stats_path = (
        configured_stats
        if configured_stats.is_absolute()
        else repo_root / configured_stats
    )
    if not configured_stats_path.is_dir():
        test_cfg["mean_std_dir"] = str(stats_dir.resolve())

    runtime_cfg = artifacts_dir / "runtime_config.yml"
    runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return runtime_cfg


def _validate_hymotion_lfs_assets() -> None:
    required = [
        Path("scripts/gradio/static/assets/dump_wooden/kintree.bin"),
        Path("scripts/gradio/static/assets/dump_wooden/j_template.bin"),
        Path("scripts/gradio/static/assets/dump_wooden/v_template.bin"),
        Path("scripts/gradio/static/assets/dump_wooden/skinWeights.bin"),
        Path("scripts/gradio/static/assets/dump_wooden/skinIndice.bin"),
    ]
    for path in required:
        if not path.exists() or _looks_like_lfs_pointer(path):
            raise FileNotFoundError(
                f"HY-Motion asset is missing or still a Git LFS pointer: {path}. "
                "Run `git -C data-generate-pipeline/backends/hymotion lfs pull` "
                "before Hunyuan Motion generation."
            )


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(
            b"version https://git-lfs.github.com/spec"
        )
    except OSError:
        return False


def _parse_device_ids(value: str) -> list[int] | None:
    text = value.strip()
    if not text:
        return None
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    return ids or None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_body_joints(joints: np.ndarray) -> np.ndarray:
    if joints.ndim != 3:
        raise ValueError(f"Unexpected joints shape: {joints.shape}")
    if joints.shape[1] < HUNYUAN_BODY_JOINT_COUNT:
        raise ValueError(
            "HY-Motion keypoints3d does not contain enough body joints: "
            f"expected at least {HUNYUAN_BODY_JOINT_COUNT}, got {joints.shape[1]}"
        )
    return joints[:, :HUNYUAN_BODY_JOINT_COUNT, :]


def _write_schema(artifacts_dir: Path) -> None:
    schema = {
        "name": "hunyuan_motion_body22_keypoints3d",
        "description": (
            "HY-Motion keypoints3d body joints saved without retargeting. "
            "Finger joints are omitted from Step 2 .npy outputs."
        ),
        "array_shape": f"[frames, {HUNYUAN_BODY_JOINT_COUNT}, 3]",
        "fps": 30,
        "unit": "meter",
        "source": (
            "model_output['keypoints3d'][:, :22] plus model_output['transl'] "
            "from HY-Motion-1.0 wooden skeleton"
        ),
    }
    names_path = Path("scripts/gradio/static/assets/dump_wooden/joint_names.json")
    if names_path.exists():
        try:
            names = json.loads(names_path.read_text(encoding="utf-8"))
            schema["joint_names"] = names[:HUNYUAN_BODY_JOINT_COUNT]
        except json.JSONDecodeError:
            pass
    (artifacts_dir / "joints_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _emit_progress(event: dict[str, Any]) -> None:
    print(f"{PROGRESS_PREFIX} {json.dumps(event, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

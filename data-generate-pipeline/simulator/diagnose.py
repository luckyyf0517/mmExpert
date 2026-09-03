"""Statistical anomaly detection on generated joints and mmWave data.

Outputs a structured diagnosis.json describing all detected problems,
ready for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from flywheel.logging_utils import get_console, get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    "A00": "walking",
    "A01": "bending",
    "A02": "running",
    "A03": "jumping",
    "A04": "sitting",
    "A05": "waving",
}


def parse_sample_id(filename: str) -> dict | None:
    """Parse 'A01_0000_0042.npy' -> {class_id, batch, index, stem}."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    return {
        "class_id": parts[0],
        "batch": parts[1],
        "index": int(parts[2]),
        "stem": stem,
    }


def scan_version_dir(version_dir: str) -> dict:
    """Scan a version directory and group samples by class.

    Returns {class_id: [{"sample_id", "joints_path", "mmwave_path", "text_path"}, ...]}
    """
    vdir = Path(version_dir)
    joints_dir = vdir / "joints"
    mmwave_dir = vdir / "mmwave"
    text_dir = vdir / "texts"

    samples_by_class = defaultdict(list)

    for npy_file in sorted(joints_dir.glob("*.npy")):
        info = parse_sample_id(npy_file.name)
        if info is None:
            continue
        sample_id = info["stem"]
        mmwave_path = mmwave_dir / f"{sample_id}.npz"
        text_path = text_dir / f"{sample_id}.txt"
        samples_by_class[info["class_id"]].append({
            "sample_id": sample_id,
            "class_id": info["class_id"],
            "batch": info["batch"],
            "index": info["index"],
            "joints_path": str(npy_file),
            "mmwave_path": str(mmwave_path),
            "text_path": str(text_path),
            "mmwave_exists": mmwave_path.exists(),
            "text_exists": text_path.exists(),
        })

    return dict(samples_by_class)


# ---------------------------------------------------------------------------
# Statistical checks
# ---------------------------------------------------------------------------

def compute_velocity(joints: np.ndarray) -> np.ndarray:
    """Frame-to-frame displacement, shape (T-1, 22, 3)."""
    return np.diff(joints, axis=0)


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    return np.diff(velocity, axis=0)


def check_joint_velocity_outlier(
    joints: np.ndarray,
    class_mean: float,
    class_std: float,
    z_thresh: float,
    sample_id: str,
    class_id: str,
) -> list:
    """Flag frames where any joint velocity exceeds z_thresh std from class mean."""
    problems = []
    vel = compute_velocity(joints)  # (T-1, 22, 3)
    speed = np.linalg.norm(vel, axis=2)  # (T-1, 22)

    # Per-frame max speed across joints
    max_speed = speed.max(axis=1)  # (T-1,)

    if class_std < 1e-8:
        return problems

    z_scores = (max_speed - class_mean) / class_std
    outlier_frames = np.where(z_scores > z_thresh)[0]

    for frame_idx in outlier_frames:
        joint_speeds = speed[frame_idx]
        top_joints = np.where(joint_speeds > class_mean + z_thresh * class_std)[0]
        problems.append({
            "problem_id": None,  # assigned later
            "sample_id": sample_id,
            "class": class_id,
            "action_name": CLASS_NAMES.get(class_id, class_id),
            "category": "joint_velocity_outlier",
            "severity": "high" if z_scores[frame_idx] > 5.0 else "medium",
            "description": (
                f"Joint velocity at frame {frame_idx} exceeds {z_thresh:.1f} std "
                f"from class mean (z={z_scores[frame_idx]:.1f}). "
                f"Joints {top_joints.tolist()} show high displacement "
                f"({max_speed[frame_idx]:.3f}m, class mean={class_mean:.3f}m)."
            ),
            "detail": {
                "frame_idx": int(frame_idx),
                "joint_indices": top_joints.tolist(),
                "velocity": float(max_speed[frame_idx]),
                "class_mean_velocity": float(class_mean),
                "z_score": float(z_scores[frame_idx]),
            },
        })
    return problems


def check_joint_discontinuity(
    joints: np.ndarray,
    threshold: float,
    sample_id: str,
    class_id: str,
) -> list:
    """Detect sudden position jumps exceeding threshold meters between consecutive frames."""
    problems = []
    vel = compute_velocity(joints)
    displacement = np.linalg.norm(vel, axis=2)  # (T-1, 22)
    outlier_frames_joints = np.argwhere(displacement > threshold)

    for frame_idx, joint_idx in outlier_frames_joints:
        disp = displacement[frame_idx, joint_idx]
        problems.append({
            "problem_id": None,
            "sample_id": sample_id,
            "class": class_id,
            "action_name": CLASS_NAMES.get(class_id, class_id),
            "category": "joint_discontinuity",
            "severity": "high" if disp > threshold * 2 else "medium",
            "description": (
                f"Joint {joint_idx} displaced {disp:.3f}m between frames "
                f"{frame_idx} and {frame_idx + 1}, exceeding threshold {threshold:.2f}m."
            ),
            "detail": {
                "frame_idx": int(frame_idx),
                "joint_index": int(joint_idx),
                "displacement": float(disp),
                "threshold": float(threshold),
            },
        })
    return problems


def check_mmwave_nan(
    mmwave_path: str,
    sample_id: str,
    class_id: str,
) -> list:
    """Check for NaN values in mmWave modalities."""
    problems = []
    data = np.load(mmwave_path)
    for key in ["range_time", "doppler_time", "azimuth_time"]:
        if key not in data:
            continue
        arr = data[key]
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            problems.append({
                "problem_id": None,
                "sample_id": sample_id,
                "class": class_id,
                "action_name": CLASS_NAMES.get(class_id, class_id),
                "category": "mmwave_nan",
                "severity": "high",
                "description": (
                    f"NaN values found in {key}: {nan_count}/{arr.size} "
                    f"({nan_count / arr.size:.2%})."
                ),
                "detail": {
                    "modality": key,
                    "nan_count": int(nan_count),
                    "total_elements": int(arr.size),
                },
            })
    return problems


def check_mmwave_padding(
    mmwave_path: str,
    padding_thresh: float,
    sample_id: str,
    class_id: str,
) -> list:
    """Check for excessive padding (-1 columns) in mmWave data."""
    problems = []
    data = np.load(mmwave_path)
    for key in ["range_time", "doppler_time", "azimuth_time"]:
        if key not in data:
            continue
        arr = data[key]
        # Count columns where all values are -1
        if arr.ndim == 2:
            padded_cols = np.all(arr == -1, axis=0).sum()
            total_cols = arr.shape[1]
        else:
            continue

        if total_cols == 0:
            continue
        ratio = padded_cols / total_cols
        if ratio > padding_thresh:
            problems.append({
                "problem_id": None,
                "sample_id": sample_id,
                "class": class_id,
                "action_name": CLASS_NAMES.get(class_id, class_id),
                "category": "mmwave_padding_ratio_high",
                "severity": "high" if ratio > 0.5 else "medium",
                "description": (
                    f"{key} has {ratio:.1%} padding columns ({padded_cols}/{total_cols}), "
                    f"exceeding threshold {padding_thresh:.0%}."
                ),
                "detail": {
                    "modality": key,
                    "padded_columns": int(padded_cols),
                    "total_columns": int(total_cols),
                    "padding_ratio": float(ratio),
                },
            })
    return problems


def check_mmwave_low_energy(
    mmwave_path: str,
    class_energy_10p: float,
    sample_id: str,
    class_id: str,
) -> list:
    """Flag samples whose range-time energy is below the class 10th percentile."""
    problems = []
    data = np.load(mmwave_path)
    if "range_time" not in data:
        return problems
    arr = data["range_time"]
    # Mean of absolute values of non-padding entries
    valid = arr[arr != -1]
    if valid.size == 0:
        energy = 0.0
    else:
        energy = float(np.abs(valid).mean())

    if class_energy_10p > 0 and energy < class_energy_10p:
        problems.append({
            "problem_id": None,
            "sample_id": sample_id,
            "class": class_id,
            "action_name": CLASS_NAMES.get(class_id, class_id),
            "category": "mmwave_low_energy",
            "severity": "low",
            "description": (
                f"Range-time energy ({energy:.2f}) below class 10th percentile "
                f"({class_energy_10p:.2f})."
            ),
            "detail": {
                "modality": "range_time",
                "energy": float(energy),
                "class_10th_percentile": float(class_energy_10p),
            },
        })
    return problems


# ---------------------------------------------------------------------------
# Class-level statistics computation
# ---------------------------------------------------------------------------

def compute_class_statistics(samples: list) -> dict:
    """Compute per-class baseline statistics for z-score detection."""
    velocities = []
    energies = []

    for s in samples:
        try:
            joints = np.load(s["joints_path"])
        except Exception:
            continue
        vel = compute_velocity(joints)
        speed = np.linalg.norm(vel, axis=2)
        velocities.append(speed)

        if s["mmwave_exists"]:
            try:
                data = np.load(s["mmwave_path"])
                if "range_time" in data:
                    valid = data["range_time"][data["range_time"] != -1]
                    if valid.size > 0:
                        energies.append(float(np.abs(valid).mean()))
            except Exception:
                pass

    all_vel = np.concatenate(velocities) if velocities else np.array([0.0])
    stats = {
        "sample_count": len(samples),
        "joints_mean_velocity": float(all_vel.mean()),
        "joints_std_velocity": float(all_vel.std()) if len(velocities) > 0 else 0.0,
        "mmwave_mean_range_energy": float(np.mean(energies)) if energies else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_diagnosis(
    version_dir: str,
    output_path: str,
    velocity_z_thresh: float = 3.0,
    discontinuity_thresh: float = 0.3,
    padding_thresh: float = 0.3,
) -> dict:
    """Run statistical anomaly detection and write diagnosis.json."""
    console = get_console()
    console.print(f"[bold cyan]Scanning {version_dir} ...[/]")
    samples_by_class = scan_version_dir(version_dir)

    if not samples_by_class:
        console.print("[bold red]No samples found![/]")
        sys.exit(1)

    total_samples = sum(len(v) for v in samples_by_class.values())
    console.print(
        f"[green]Found {total_samples} samples "
        f"across {len(samples_by_class)} classes[/]"
    )

    # Phase 1: Compute per-class statistics
    console.print("[cyan]Computing per-class statistics ...[/]")
    class_stats = {}
    for class_id, samples in sorted(samples_by_class.items()):
        class_stats[class_id] = compute_class_statistics(samples)
        cs = class_stats[class_id]
        logger.info(
            "%s (%s): n=%d, vel=%.4f+/-%.4f",
            class_id, CLASS_NAMES.get(class_id, "?"),
            cs["sample_count"],
            cs["joints_mean_velocity"], cs["joints_std_velocity"],
        )

    # Compute 10th percentile of range-energy per class
    class_energy_10p = {}
    for class_id, samples in sorted(samples_by_class.items()):
        energies = []
        for s in samples:
            if not s["mmwave_exists"]:
                continue
            try:
                data = np.load(s["mmwave_path"])
                if "range_time" in data:
                    valid = data["range_time"][data["range_time"] != -1]
                    if valid.size > 0:
                        energies.append(float(np.abs(valid).mean()))
            except Exception:
                pass
        class_energy_10p[class_id] = float(np.percentile(energies, 10)) if len(energies) >= 5 else 0.0

    # Phase 2: Run checks on each sample
    console.print("[cyan]Running anomaly detection ...[/]")
    all_problems = []
    total_samples = 0

    for class_id, samples in sorted(samples_by_class.items()):
        cs = class_stats[class_id]
        for s in samples:
            total_samples += 1
            sid = s["sample_id"]

            # Load joints
            try:
                joints = np.load(s["joints_path"])
            except Exception as e:
                all_problems.append({
                    "problem_id": None,
                    "sample_id": sid,
                    "class": class_id,
                    "action_name": CLASS_NAMES.get(class_id, class_id),
                    "category": "joints_load_error",
                    "severity": "high",
                    "description": f"Failed to load joints: {e}",
                    "detail": {"error": str(e)},
                })
                continue

            # Joint velocity outlier
            all_problems.extend(check_joint_velocity_outlier(
                joints, cs["joints_mean_velocity"], cs["joints_std_velocity"],
                velocity_z_thresh, sid, class_id,
            ))

            # Joint discontinuity
            all_problems.extend(check_joint_discontinuity(
                joints, discontinuity_thresh, sid, class_id,
            ))

            # mmWave checks
            if s["mmwave_exists"]:
                all_problems.extend(check_mmwave_nan(s["mmwave_path"], sid, class_id))
                all_problems.extend(check_mmwave_padding(s["mmwave_path"], padding_thresh, sid, class_id))
                all_problems.extend(check_mmwave_low_energy(
                    s["mmwave_path"], class_energy_10p.get(class_id, 0.0), sid, class_id,
                ))

    # Assign sequential problem IDs
    for i, p in enumerate(all_problems):
        p["problem_id"] = f"P{i + 1:04d}"

    # Build summary
    problems_by_category = defaultdict(int)
    problems_by_class = defaultdict(lambda: {"count": 0, "total": 0})
    for class_id in samples_by_class:
        problems_by_class[class_id]["total"] = len(samples_by_class[class_id])
    for p in all_problems:
        problems_by_category[p["category"]] += 1
        problems_by_class[p["class"]]["count"] += 1

    summary = {
        "total_samples": total_samples,
        "total_problems": len(all_problems),
        "problem_rate": len(all_problems) / total_samples if total_samples > 0 else 0.0,
        "problems_by_category": dict(problems_by_category),
        "problems_by_class": {k: dict(v) for k, v in sorted(problems_by_class.items())},
    }

    # Add rate to problems_by_class
    for cls_id in summary["problems_by_class"]:
        info = summary["problems_by_class"][cls_id]
        info["rate"] = info["count"] / info["total"] if info["total"] > 0 else 0.0

    diagnosis = {
        "version": Path(version_dir).name,
        "timestamp": datetime.now().isoformat(),
        "pipeline_stage": "feedback_diagnosis",
        "summary": summary,
        "class_statistics": {
            cls: {**stats, "action_name": CLASS_NAMES.get(cls, cls)}
            for cls, stats in class_stats.items()
        },
        "problems": all_problems,
    }

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green]Found {len(all_problems)} problems in {total_samples} samples "
        f"({summary['problem_rate']:.2%})[/]"
    )
    console.print(f"[green]Diagnosis saved to {output_path}[/]")

    return diagnosis


def main():
    parser = argparse.ArgumentParser(description="Statistical anomaly detection on generated data")
    parser.add_argument("--version-dir", required=True, help="Path to dataset version directory")
    parser.add_argument("--output", required=True, help="Output path for diagnosis.json")
    parser.add_argument("--velocity-z-threshold", type=float, default=3.0)
    parser.add_argument("--discontinuity-threshold", type=float, default=0.3)
    parser.add_argument("--padding-ratio-threshold", type=float, default=0.3)
    args = parser.parse_args()

    console = get_console()
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Data Quality Diagnosis[/]")
    console.print("=" * 80)
    console.print(f"[yellow]Version dir: {args.version_dir}[/]")
    console.print(f"[yellow]Output: {args.output}[/]")
    console.print(f"[yellow]Velocity z-threshold: {args.velocity_z_threshold}[/]")
    console.print(f"[yellow]Discontinuity threshold: {args.discontinuity_threshold}m[/]")
    console.print(f"[yellow]Padding ratio threshold: {args.padding_ratio_threshold}[/]")
    console.print("=" * 80 + "\n")

    run_diagnosis(
        version_dir=args.version_dir,
        output_path=args.output,
        velocity_z_thresh=args.velocity_z_threshold,
        discontinuity_thresh=args.discontinuity_threshold,
        padding_thresh=args.padding_ratio_threshold,
    )


if __name__ == "__main__":
    main()

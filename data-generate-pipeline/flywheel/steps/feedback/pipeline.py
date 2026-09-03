"""End-to-end classifier-feedback pipeline for doppler-only flywheel data."""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...config import FlywheelConfig
from ...logging_utils import get_console, get_logger
from ...path_manager import PathManager

logger = get_logger()

TARGET_SHAPE = (128, 128)
MODEL_FILENAME = "best_model_finetuned.pth"
TEST_RESULTS_FILENAME = "test_results.json"
ANALYSIS_FILENAME = "misclassification_analysis.json"
PREPROCESSED_REAL_DIRNAME = "real"
PREPROCESSED_SYNTH_DIRNAME = "synthetic"


@dataclass
class DopplerSample:
    file_path: Path
    label_id: str
    label_name: str
    class_index: int
    sample_id: str
    prompt_text: str = ""


class DopplerDataset(Dataset):
    """Torch dataset over normalized doppler maps."""

    def __init__(self, samples: list[DopplerSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        arr = _load_doppler_file(sample.file_path)
        tensor = _prepare_tensor(arr)
        return tensor, sample.class_index


class DopplerClassifier(nn.Module):
    """Lightweight 2D CNN for doppler classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def run_classifier_feedback_pipeline(
    version: str,
    config: FlywheelConfig,
    paths: PathManager,
    *,
    epochs: int = 12,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> tuple[Path, Path]:
    """Train a real-data classifier, evaluate synthetic doppler, and emit artifacts."""
    console = get_console()
    console.print("[bold]Preparing classifier feedback pipeline[/]")

    paths.classifier_outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.feedback_outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    label_maps = _build_label_maps(config)
    real_samples = _discover_real_samples(paths, config, label_maps)
    synthetic_samples = _discover_synthetic_samples(paths, config, label_maps)

    _validate_label_coverage(config, real_samples, synthetic_samples)

    train_samples, val_samples = _split_real_samples(real_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DopplerClassifier(num_classes=len(config.actions)).to(device)

    best_state, train_summary = _train_model(
        model,
        train_samples,
        val_samples,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    model.load_state_dict(best_state)

    model_path = paths.classifier_outputs_dir / MODEL_FILENAME
    torch.save(
        {
            "state_dict": best_state,
            "actions": [a.to_dict() for a in config.actions],
            "version": version,
            "train_summary": train_summary,
            "target_shape": TARGET_SHAPE,
            "view": "doppler_only",
        },
        model_path,
    )

    results = _evaluate_synthetic(model, synthetic_samples, config, device=device)
    results["model_path"] = str(model_path)
    results["version"] = version
    results["train_summary"] = train_summary
    results["real_data_dir"] = str(paths.real_data_dir)
    results["synthetic_mmwave_dir"] = str(paths.mmwave_dir)
    results["view"] = "doppler_only"

    analysis = _build_feedback_analysis(results, config)

    test_results_path = paths.feedback_outputs_dir / TEST_RESULTS_FILENAME
    analysis_path = paths.feedback_outputs_dir / ANALYSIS_FILENAME
    _write_json(test_results_path, results)
    _write_json(analysis_path, analysis)

    console.print(
        f"[green]Classifier feedback artifacts written:[/] "
        f"{test_results_path.name}, {analysis_path.name}"
    )
    return analysis_path, test_results_path


def _build_label_maps(config: FlywheelConfig) -> dict[str, tuple[str, int]]:
    mapping: dict[str, tuple[str, int]] = {}
    for idx, action in enumerate(config.actions):
        for alias in {_normalize_key(action.id), _normalize_key(action.name)}:
            mapping[alias] = (action.id, idx)
    return mapping


def _discover_real_samples(
    paths: PathManager,
    config: FlywheelConfig,
    label_maps: dict[str, tuple[str, int]],
) -> list[DopplerSample]:
    real_root = _materialize_real_data(paths.real_data_dir)
    files = [
        p for p in real_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".npy", ".npz"}
    ]
    if not files:
        raise FileNotFoundError(
            f"No real doppler files found under {real_root}. "
            "Expected per-class .npy/.npz files in _real_data."
        )

    samples: list[DopplerSample] = []
    for file_path in sorted(files):
        label = _infer_label(file_path, label_maps)
        if label is None:
            continue
        label_id, class_index = label
        label_name = config.actions[class_index].name
        cached = _cache_preprocessed(
            file_path,
            paths.preprocessed_data_dir / PREPROCESSED_REAL_DIRNAME / label_id,
        )
        samples.append(
            DopplerSample(
                file_path=cached,
                label_id=label_id,
                label_name=label_name,
                class_index=class_index,
                sample_id=file_path.stem,
            )
        )
    return samples


def _discover_synthetic_samples(
    paths: PathManager,
    config: FlywheelConfig,
    label_maps: dict[str, tuple[str, int]],
) -> list[DopplerSample]:
    files = [
        p for p in paths.mmwave_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".npy", ".npz"}
    ]
    if not files:
        raise FileNotFoundError(
            f"No synthetic mmWave files found under {paths.mmwave_dir}."
        )

    samples: list[DopplerSample] = []
    for file_path in sorted(files):
        label = _infer_label(file_path, label_maps)
        if label is None:
            continue
        label_id, class_index = label
        label_name = config.actions[class_index].name
        cached = _cache_preprocessed(
            file_path,
            paths.preprocessed_data_dir / PREPROCESSED_SYNTH_DIRNAME / paths.version / label_id,
        )
        stem = file_path.stem
        prompt_path = paths.texts_dir / f"{stem}.txt"
        prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
        samples.append(
            DopplerSample(
                file_path=cached,
                label_id=label_id,
                label_name=label_name,
                class_index=class_index,
                sample_id=stem,
                prompt_text=prompt_text,
            )
        )
    return samples


def _validate_label_coverage(
    config: FlywheelConfig,
    real_samples: list[DopplerSample],
    synthetic_samples: list[DopplerSample],
) -> None:
    required = {a.id for a in config.actions}
    real_present = {s.label_id for s in real_samples}
    synth_present = {s.label_id for s in synthetic_samples}
    missing_real = sorted(required - real_present)
    missing_synth = sorted(required - synth_present)
    if missing_real:
        raise ValueError(
            "Real-data classes missing for flywheel feedback: "
            + ", ".join(missing_real)
        )
    if missing_synth:
        raise ValueError(
            "Synthetic classes missing in current round mmWave outputs: "
            + ", ".join(missing_synth)
        )


def _split_real_samples(
    samples: list[DopplerSample],
    val_ratio: float = 0.2,
) -> tuple[list[DopplerSample], list[DopplerSample]]:
    grouped: dict[int, list[DopplerSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.class_index].append(sample)

    train: list[DopplerSample] = []
    val: list[DopplerSample] = []
    rng = np.random.default_rng(seed=42)
    for class_samples in grouped.values():
        shuffled = list(class_samples)
        rng.shuffle(shuffled)
        if len(shuffled) <= 2:
            train.extend(shuffled)
            continue
        val_count = max(1, int(math.floor(len(shuffled) * val_ratio)))
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    return train, val


def _train_model(
    model: DopplerClassifier,
    train_samples: list[DopplerSample],
    val_samples: list[DopplerSample],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    train_loader = DataLoader(DopplerDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(DopplerDataset(val_samples), batch_size=batch_size, shuffle=False)
        if val_samples else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        train_acc = correct / total if total else 0.0
        train_loss = running_loss / total if total else 0.0
        val_acc = _evaluate_loader(model, val_loader, device=device) if val_loader else train_acc
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(val_acc, 6),
            }
        )
        if val_acc >= best_metric:
            best_metric = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_accuracy": round(best_metric, 6),
        "history": history,
    }


def _evaluate_loader(
    model: DopplerClassifier,
    loader: DataLoader | None,
    *,
    device: torch.device,
) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / total if total else 0.0


def _evaluate_synthetic(
    model: DopplerClassifier,
    samples: list[DopplerSample],
    config: FlywheelConfig,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    dataset = DopplerDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, _labels in loader:
            logits = model(inputs.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    confusion = np.zeros((len(config.actions), len(config.actions)), dtype=int)
    misclassified: list[dict[str, Any]] = []
    per_action_counts: Counter[str] = Counter()
    per_action_correct: Counter[str] = Counter()

    for sample, pred_idx in zip(samples, predictions):
        true_idx = sample.class_index
        pred_action = config.actions[pred_idx]
        true_action = config.actions[true_idx]
        confusion[true_idx, pred_idx] += 1
        per_action_counts[true_action.id] += 1
        if pred_idx == true_idx:
            per_action_correct[true_action.id] += 1
        else:
            misclassified.append(
                {
                    "sample_id": sample.sample_id,
                    "file_path": str(sample.file_path),
                    "true_label": true_action.id,
                    "true_action": true_action.name,
                    "pred_label": pred_action.id,
                    "pred_action": pred_action.name,
                    "prompt_text": sample.prompt_text,
                }
            )

    total = len(samples)
    total_misclassified = len(misclassified)
    per_action_summary: dict[str, Any] = {}
    for action in config.actions:
        count = per_action_counts[action.id]
        correct = per_action_correct[action.id]
        accuracy = correct / count if count else 0.0
        action_idx = _action_index(config, action.id)
        pred_breakdown = {
            config.actions[pred_idx].id: int(confusion[action_idx, pred_idx])
            for pred_idx in range(len(config.actions))
            if int(confusion[action_idx, pred_idx]) > 0
        }
        per_action_summary[action.id] = {
            "name": action.name,
            "count": count,
            "correct": correct,
            "accuracy": round(accuracy, 6),
            "predicted_as": pred_breakdown,
        }

    return {
        "overall_accuracy": round((total - total_misclassified) / total, 6) if total else 0.0,
        "total_evaluated": total,
        "total_misclassified": total_misclassified,
        "confusion_matrix": confusion.tolist(),
        "actions": [a.to_dict() for a in config.actions],
        "per_action_summary": per_action_summary,
        "misclassifications": misclassified,
    }


def _build_feedback_analysis(
    results: dict[str, Any],
    config: FlywheelConfig,
) -> dict[str, Any]:
    per_action = results.get("per_action_summary", {})
    misclassified = results.get("misclassifications", [])
    confusion = np.asarray(results.get("confusion_matrix", []), dtype=int)

    low_acc_actions = sorted(
        per_action.items(),
        key=lambda item: item[1].get("accuracy", 0.0),
    )

    common_issues: list[str] = []
    new_constraints: list[str] = []

    for action_id, summary in low_acc_actions[:3]:
        action_name = summary.get("name", action_id)
        accuracy = summary.get("accuracy", 0.0)
        predicted_as = summary.get("predicted_as", {})
        if predicted_as:
            top_confused = max(
                ((pred, count) for pred, count in predicted_as.items() if pred != action_id),
                key=lambda x: x[1],
                default=None,
            )
        else:
            top_confused = None

        if top_confused is None:
            common_issues.append(
                f"{action_name} has limited synthetic separability in doppler space "
                f"(accuracy {accuracy * 100:.1f}%)."
            )
            new_constraints.append(
                f"For action '{action_name}', add clearer Doppler-temporal cues "
                "covering speed, duration, and phase transitions."
            )
            continue

        pred_id, count = top_confused
        pred_name = next((a.name for a in config.actions if a.id == pred_id), pred_id)
        common_issues.append(
            f"{action_name} is frequently confused with {pred_name} "
            f"({count} synthetic samples; per-class accuracy {accuracy * 100:.1f}%)."
        )
        new_constraints.append(
            f"For action '{action_name}', strengthen prompt constraints that separate it "
            f"from '{pred_name}' in doppler-only observations by emphasizing motion speed, "
            "temporal cadence, body-direction change, and transition sharpness."
        )

    if misclassified:
        prompt_examples = [
            item["prompt_text"] for item in misclassified if item.get("prompt_text")
        ][:3]
    else:
        prompt_examples = []

    if prompt_examples:
        common_issues.append(
            "Some prompts may underspecify doppler-discriminative details, making "
            "different actions produce similar temporal velocity patterns."
        )
        new_constraints.append(
            "Require each prompt to specify observable motion tempo, repetition pattern, "
            "and onset/offset transitions that are visible in doppler-time signatures."
        )

    confusion_pairs = []
    if confusion.size:
        for i, true_action in enumerate(config.actions):
            for j, pred_action in enumerate(config.actions):
                if i == j or confusion[i, j] <= 0:
                    continue
                confusion_pairs.append(
                    {
                        "true_label": true_action.id,
                        "true_action": true_action.name,
                        "pred_label": pred_action.id,
                        "pred_action": pred_action.name,
                        "count": int(confusion[i, j]),
                    }
                )
        confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    return {
        "overall_accuracy": results.get("overall_accuracy", 0.0),
        "total_evaluated": results.get("total_evaluated", 0),
        "total_misclassified": results.get("total_misclassified", 0),
        "per_action_summary": per_action,
        "common_issues": common_issues[:10],
        "new_constraints": _unique_preserve_order(new_constraints)[:10],
        "modified_constraints": [],
        "style_adjustments": {},
        "top_confusions": confusion_pairs[:10],
        "example_misclassifications": misclassified[:20],
        "analysis_mode": "heuristic_doppler_only",
    }


def _materialize_real_data(real_data_dir: Path) -> Path:
    real_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = real_data_dir / "extracted"
    files = list(real_data_dir.rglob("*.npy")) + list(real_data_dir.rglob("*.npz"))
    if files:
        return real_data_dir

    zip_files = sorted(real_data_dir.glob("*.zip"))
    if not zip_files:
        return real_data_dir

    extracted_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                target = extracted_dir / zip_path.stem
                if not target.exists():
                    zf.extractall(target)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"Real-data archive is not a valid zip file: {zip_path}"
            ) from exc
    return extracted_dir


def _infer_label(
    file_path: Path,
    label_maps: dict[str, tuple[str, int]],
) -> tuple[str, int] | None:
    candidates = [file_path.stem]
    candidates.extend(parent.name for parent in file_path.parents)
    for candidate in candidates:
        key = _normalize_key(candidate)
        if key in label_maps:
            return label_maps[key]
        match = re.match(r"^(A\d{2})", candidate, flags=re.IGNORECASE)
        if match:
            key = _normalize_key(match.group(1))
            if key in label_maps:
                return label_maps[key]
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _cache_preprocessed(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{src.stem}.npy"
    arr = _load_doppler_file(src)
    np.save(dest_path, arr.astype(np.float32))
    return dest_path


def _load_doppler_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "doppler_time" in data.files:
                arr = data["doppler_time"]
            elif len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise ValueError(
                    f"Could not infer doppler array from {path}; expected 'doppler_time'."
                )
    else:
        arr = np.load(path)

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D doppler array in {path}, got shape {arr.shape}")

    if arr.shape[0] == 128:
        doppler = arr
    elif arr.shape[1] == 128:
        doppler = arr.T
    else:
        raise ValueError(
            f"Expected one dimension to be 128 in {path}, got shape {arr.shape}"
        )
    doppler = np.nan_to_num(doppler, nan=0.0, posinf=0.0, neginf=0.0)
    return doppler


def _prepare_tensor(arr: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mean = tensor.mean()
    std = tensor.std()
    if std > 1e-6:
        tensor = (tensor - mean) / std
    else:
        tensor = tensor - mean
    return tensor


def _action_index(config: FlywheelConfig, action_id: str) -> int:
    for idx, action in enumerate(config.actions):
        if action.id == action_id:
            return idx
    raise KeyError(action_id)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

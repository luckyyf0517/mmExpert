"""Shared motion-generator backend interfaces and resolution helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

DEFAULT_MOTION_BACKEND = "hunyuan_motion"
MOTION_BACKEND_MANIFEST = "motion_backend.json"
MOTION_GENERATION_CONFIG_KEY = "motion_generation"
MOTION_BACKEND_ALIASES = {
    "hymotion": "hunyuan_motion",
    "hy_motion": "hunyuan_motion",
    "hunyuan": "hunyuan_motion",
    "hunyuan_motion": "hunyuan_motion",
}


@dataclass(frozen=True)
class MotionBackendSelection:
    """Resolved motion backend and active joints directory."""

    backend: str
    joints_subdir: str
    joints_dir: Path
    backend_config: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class MotionBackendContext:
    """Runtime context passed to a motion backend."""

    version: str
    config: Any
    paths: Any
    entries: Sequence[Any]
    num_gpus: int
    batch_size: int
    joints_subdir: str
    backend_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionBackendRunResult:
    """Backend execution result before output verification."""

    backend: str
    joints_subdir: str
    errors: list[str] = field(default_factory=list)


class MotionGeneratorBackend(Protocol):
    """Protocol implemented by text-to-motion generator backends."""

    name: str
    default_joints_subdir: str

    def run(self, context: MotionBackendContext) -> MotionBackendRunResult:
        """Generate motions for the supplied prompt entries."""
        ...


def default_joints_subdir_for_backend(backend: str) -> str:
    """Return the default Step 2 joints subdirectory for a backend name."""
    backend = canonical_motion_backend_name(backend, default=DEFAULT_MOTION_BACKEND)
    if backend == "hunyuan_motion":
        return "joints_hunyuan_motion"
    return f"joints_{backend}"


def canonical_motion_backend_name(
    value: Any,
    *,
    default: str = DEFAULT_MOTION_BACKEND,
) -> str:
    """Normalize aliases to the canonical backend registry name."""
    name = _normalize_name(value, default=default)
    return MOTION_BACKEND_ALIASES.get(name, name)


def resolve_motion_backend(
    config: Any,
    paths: Any,
    *,
    backend_override: str | None = None,
    joints_subdir_override: str | None = None,
    prefer_manifest: bool = False,
) -> MotionBackendSelection:
    """Resolve backend and joints subdir from CLI, config, manifest, defaults."""
    cfg = _motion_config(config)
    manifest_path = paths.step2_dir / MOTION_BACKEND_MANIFEST
    manifest = _read_manifest(manifest_path)

    backend_sources = (
        (backend_override, manifest.get("backend"), cfg.get("backend"), DEFAULT_MOTION_BACKEND)
        if prefer_manifest
        else (backend_override, cfg.get("backend"), manifest.get("backend"), DEFAULT_MOTION_BACKEND)
    )
    backend = _first_non_empty(*backend_sources)
    backend = canonical_motion_backend_name(backend, default=DEFAULT_MOTION_BACKEND)

    joints_sources = (
        (
            joints_subdir_override,
            manifest.get("joints_subdir"),
            cfg.get("joints_subdir"),
            default_joints_subdir_for_backend(backend),
        )
        if prefer_manifest
        else (
            joints_subdir_override,
            cfg.get("joints_subdir"),
            manifest.get("joints_subdir"),
            default_joints_subdir_for_backend(backend),
        )
    )
    joints_subdir = _first_non_empty(*joints_sources)
    joints_subdir = _validate_joints_subdir(str(joints_subdir))

    backend_config = cfg.get("backend_config", {})
    if not isinstance(backend_config, dict):
        backend_config = {}

    return MotionBackendSelection(
        backend=backend,
        joints_subdir=joints_subdir,
        joints_dir=paths.step2_dir / joints_subdir,
        backend_config=dict(backend_config),
        manifest_path=manifest_path,
    )


def write_motion_backend_manifest(
    paths: Any,
    *,
    backend: str,
    joints_subdir: str,
    counts: dict[str, int],
    runtime_args: dict[str, Any],
) -> Path:
    """Write Step 2 backend metadata for later simulation runs."""
    manifest_path = paths.step2_dir / MOTION_BACKEND_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "joints_subdir": joints_subdir,
        "counts": counts,
        "runtime_args": runtime_args,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _motion_config(config: Any) -> dict[str, Any]:
    data = getattr(config, "data", {}) or {}
    raw = data.get(MOTION_GENERATION_CONFIG_KEY, {})
    return raw if isinstance(raw, dict) else {}


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_name(value: Any, *, default: str) -> str:
    text = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def _validate_joints_subdir(value: str) -> str:
    text = value.strip()
    path = Path(text)
    if not text or path.is_absolute() or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(
            "motion joints subdir must be a single relative directory name, "
            f"got {value!r}"
        )
    return text

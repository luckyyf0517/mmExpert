"""Motion-generator backend registry."""

from __future__ import annotations

from .base import (
    DEFAULT_MOTION_BACKEND,
    MOTION_BACKEND_MANIFEST,
    MotionBackendContext,
    MotionBackendRunResult,
    MotionBackendSelection,
    MotionGeneratorBackend,
    canonical_motion_backend_name,
    default_joints_subdir_for_backend,
    resolve_motion_backend,
    write_motion_backend_manifest,
)
from .hunyuan_motion import HunyuanMotionBackend

_BACKENDS: dict[str, type[MotionGeneratorBackend]] = {
    HunyuanMotionBackend.name: HunyuanMotionBackend,
}


def available_backends() -> list[str]:
    """Return registered motion backend names."""
    return sorted(_BACKENDS)


def get_backend(name: str) -> MotionGeneratorBackend:
    """Instantiate a registered motion backend."""
    key = canonical_motion_backend_name(name)
    backend_cls = _BACKENDS.get(key)
    if backend_cls is None:
        choices = ", ".join(available_backends()) or "<none>"
        raise ValueError(
            f"Unknown motion backend {name!r}. Available backends: {choices}"
        )
    return backend_cls()


__all__ = [
    "DEFAULT_MOTION_BACKEND",
    "MOTION_BACKEND_MANIFEST",
    "MotionBackendContext",
    "MotionBackendRunResult",
    "MotionBackendSelection",
    "MotionGeneratorBackend",
    "HunyuanMotionBackend",
    "available_backends",
    "canonical_motion_backend_name",
    "default_joints_subdir_for_backend",
    "get_backend",
    "resolve_motion_backend",
    "write_motion_backend_manifest",
]

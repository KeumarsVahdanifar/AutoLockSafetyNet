"""Face backend selection."""

from __future__ import annotations

import logging

from ..config import Config
from .base import (
    FaceBackend,
    FaceDet,
    detect_with_rotations,
    iou,
    normalize,
)

log = logging.getLogger(__name__)

__all__ = [
    "FaceBackend",
    "FaceDet",
    "build_backend",
    "detect_with_rotations",
    "iou",
    "normalize",
]


def build_backend(cfg: Config) -> FaceBackend:
    """Instantiate the configured backend, falling back to OpenCV."""
    from .opencv_backend import OpenCVFaceBackend

    choice = (cfg.backend or "auto").lower()

    if choice in ("auto", "insightface"):
        from . import insightface_backend as insight

        if insight.available():
            try:
                backend = insight.InsightFaceBackend(
                    det_threshold=cfg.det_threshold,
                    detect_width=cfg.detect_width,
                    min_face_px=cfg.min_face_px,
                )
                log.info("Face backend: InsightFace (SCRFD + ArcFace)")
                return backend
            except Exception as exc:
                if choice == "insightface":
                    raise
                log.warning("InsightFace unavailable (%s); using OpenCV backend", exc)
        elif choice == "insightface":
            raise RuntimeError(
                "backend='insightface' requested but insightface/onnxruntime are not installed"
            )

    backend = OpenCVFaceBackend(
        det_threshold=cfg.det_threshold,
        detect_width=cfg.detect_width,
        min_face_px=cfg.min_face_px,
    )
    log.info("Face backend: OpenCV (YuNet + SFace)")
    return backend

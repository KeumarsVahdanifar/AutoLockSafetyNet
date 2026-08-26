"""Model registry: download + cache the ONNX/TFLite weights the app needs."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR

log = logging.getLogger(__name__)

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    sha1: str | None
    purpose: str
    required: bool

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


REGISTRY: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        key="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha1=None,
        purpose="face detection (YuNet)",
        required=True,
    ),
    "sface": ModelSpec(
        key="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha1=None,
        purpose="face recognition (SFace)",
        required=True,
    ),
    "pose": ModelSpec(
        key="pose",
        filename="pose_landmarker_lite.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        ),
        sha1=None,
        purpose="body fallback for head-down / turned-away (MediaPipe Pose)",
        required=False,
    ),
}


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_present(spec: ModelSpec) -> bool:
    return spec.path.exists() and spec.path.stat().st_size > 0


def download(spec: ModelSpec, force: bool = False) -> Path:
    """Fetch one model into models/ (no-op when it is already cached)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if is_present(spec) and not force:
        return spec.path

    log.info("Downloading %s -> %s", spec.purpose, spec.filename)
    with tempfile.NamedTemporaryFile(delete=False, dir=MODELS_DIR, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "presence-lock"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        if spec.sha1 and _sha1(tmp_path) != spec.sha1:
            raise RuntimeError(f"checksum mismatch for {spec.filename}")
        tmp_path.replace(spec.path)
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Saved %s (%.1f MB)", spec.filename, spec.path.stat().st_size / 1e6)
    return spec.path


def ensure(*keys: str, force: bool = False) -> dict[str, Path]:
    """Ensure the named models exist locally, downloading what is missing."""
    return {key: download(REGISTRY[key], force=force) for key in keys}


def ensure_all(force: bool = False, include_optional: bool = True) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in REGISTRY.values():
        if not spec.required and not include_optional:
            continue
        try:
            paths[spec.key] = download(spec, force=force)
        except Exception as exc:  # optional models must never break setup
            if spec.required:
                raise
            log.warning("Optional model %s unavailable: %s", spec.key, exc)
    return paths

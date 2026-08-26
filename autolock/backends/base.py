"""Backend-neutral face types and the rotated-frame retry helper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceDet:
    """One detected face, expressed in the coordinates of the original frame."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: np.ndarray | None = None  # (5, 2) in original-frame coordinates
    angle: float = 0.0  # rotation applied to the frame this was found in
    image: np.ndarray | None = field(default=None, repr=False)  # image `raw` refers to
    raw: Any = field(default=None, repr=False)  # backend payload used for embedding
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


class FaceBackend(ABC):
    """Detect faces and turn them into L2-normalised embeddings."""

    name: str = "base"
    embedding_dim: int = 0
    default_threshold: float = 0.4  # cosine similarity for "same person"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        """Detect faces in a BGR frame."""

    @abstractmethod
    def embed(self, det: FaceDet) -> np.ndarray | None:
        """Return the L2-normalised embedding for a detection from `detect`."""

    def close(self) -> None:  # pragma: no cover - trivial
        return None


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def rotate_frame(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about the centre onto an expanded canvas.

    Returns the rotated image and the inverse affine matrix that maps points
    from the rotated image back into the original frame.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - cx
    matrix[1, 2] += new_h / 2.0 - cy
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return rotated, cv2.invertAffineTransform(matrix)


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    return (homogeneous @ matrix.T).astype(np.float32)


def map_detection_back(det: FaceDet, matrix: np.ndarray) -> FaceDet:
    """Rewrite a detection's geometry from a rotated frame into the original."""
    x, y, w, h = det.bbox
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
    )
    mapped = apply_affine(corners, matrix)
    x1, y1 = mapped.min(axis=0)
    x2, y2 = mapped.max(axis=0)
    det.bbox = (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))
    if det.landmarks is not None:
        det.landmarks = apply_affine(det.landmarks, matrix)
    return det


def detect_with_rotations(
    backend: FaceBackend,
    frame: np.ndarray,
    angles: Iterable[float],
) -> list[FaceDet]:
    """Retry detection on rotated copies of the frame.

    An upright detector loses a head tilted onto a hand or slumped toward the
    desk; rotating the frame puts that head back near-upright. Detections keep
    a reference to the rotated image so embeddings are computed from the
    better-aligned crop, while their boxes are mapped back for display.
    """
    for angle in angles:
        rotated, inverse = rotate_frame(frame, float(angle))
        found = backend.detect(rotated)
        if not found:
            continue
        out: list[FaceDet] = []
        for det in found:
            det.angle = float(angle)
            out.append(map_detection_back(det, inverse))
        return out
    return []

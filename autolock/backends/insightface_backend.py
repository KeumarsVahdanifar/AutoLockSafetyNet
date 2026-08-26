"""Optional InsightFace backend (SCRFD detection + ArcFace embeddings).

More accurate than YuNet/SFace at steep yaw and pitch, so it is preferred when
importable. It needs `insightface` + `onnxruntime`, which require a C++
toolchain on Windows and have no wheels for Python 3.14 yet — hence optional.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import FaceBackend, FaceDet, normalize

log = logging.getLogger(__name__)


def available() -> bool:
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


class InsightFaceBackend(FaceBackend):
    name = "insightface"
    embedding_dim = 512
    default_threshold = 0.38  # cosine on normalised ArcFace embeddings

    def __init__(
        self,
        det_threshold: float = 0.5,
        detect_width: int = 640,
        min_face_px: int = 48,
        model_pack: str = "buffalo_l",
    ) -> None:
        from insightface.app import FaceAnalysis  # imported lazily

        self.min_face_px = int(min_face_px)
        size = max(320, (int(detect_width) // 32) * 32)
        self._app = FaceAnalysis(name=model_pack, allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=0, det_size=(size, size), det_thresh=float(det_threshold))
        log.debug("InsightFace backend ready (%s, det_size=%d)", model_pack, size)

    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        if frame is None or frame.size == 0:
            return []
        detections: list[FaceDet] = []
        for face in self._app.get(frame):
            x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self.min_face_px:
                continue
            landmarks = None
            if getattr(face, "kps", None) is not None:
                landmarks = np.asarray(face.kps, dtype=np.float32).reshape(5, 2)
            detections.append(
                FaceDet(
                    bbox=(x1, y1, w, h),
                    score=float(face.det_score),
                    landmarks=landmarks,
                    image=frame,
                    raw=face,
                )
            )
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def embed(self, det: FaceDet) -> np.ndarray | None:
        face = det.raw
        vector = getattr(face, "normed_embedding", None)
        if vector is None:
            vector = getattr(face, "embedding", None)
        if vector is None:
            return None
        det.embedding = normalize(np.asarray(vector, dtype=np.float32))
        return det.embedding

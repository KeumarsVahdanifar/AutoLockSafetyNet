"""YuNet (detection) + SFace (recognition), both executed by OpenCV's DNN module.

Chosen as the default because it needs no inference runtime beyond
`opencv-contrib-python`, runs comfortably on a CPU, and YuNet keeps detecting
faces well past the profile angles a Haar cascade gives up on.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .. import models
from .base import FaceBackend, FaceDet, normalize

log = logging.getLogger(__name__)


class OpenCVFaceBackend(FaceBackend):
    name = "opencv"
    embedding_dim = 128
    default_threshold = 0.363  # OpenCV's published SFace cosine threshold

    def __init__(
        self,
        det_threshold: float = 0.55,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        detect_width: int = 640,
        min_face_px: int = 48,
    ) -> None:
        paths = models.ensure("yunet", "sface")
        self.detect_width = int(detect_width)
        self.min_face_px = int(min_face_px)
        self._input_size: tuple[int, int] = (0, 0)

        self._detector = cv2.FaceDetectorYN.create(
            str(paths["yunet"]),
            "",
            (320, 320),
            float(det_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(paths["sface"]), "")
        log.debug("OpenCV backend ready (YuNet + SFace)")

    # ------------------------------------------------------------------
    def set_det_threshold(self, value: float) -> None:
        self._detector.setScoreThreshold(float(value))

    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        scale = 1.0
        working = frame
        if self.detect_width and width > self.detect_width:
            scale = self.detect_width / float(width)
            working = cv2.resize(
                frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )

        size = (working.shape[1], working.shape[0])
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size

        _, raw = self._detector.detect(working)
        if raw is None:
            return []

        inv = 1.0 / scale
        detections: list[FaceDet] = []
        for row in raw:
            row = np.asarray(row, dtype=np.float32)
            full = row.copy()
            full[:14] *= inv  # box + 5 landmarks back into original-frame pixels

            x, y, w, h = (int(round(v)) for v in full[:4])
            if min(w, h) < self.min_face_px:
                continue
            detections.append(
                FaceDet(
                    bbox=(x, y, w, h),
                    score=float(full[14]),
                    landmarks=full[4:14].reshape(5, 2).copy(),
                    image=frame,  # align + embed from the full-resolution frame
                    raw=full,
                )
            )

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def embed(self, det: FaceDet) -> np.ndarray | None:
        if det.image is None or det.raw is None:
            return None
        try:
            aligned = self._recognizer.alignCrop(det.image, np.asarray(det.raw, dtype=np.float32))
            feature = self._recognizer.feature(aligned)
        except cv2.error as exc:  # face partly outside the frame
            log.debug("alignCrop/feature failed: %s", exc)
            return None
        det.embedding = normalize(feature)
        return det.embedding

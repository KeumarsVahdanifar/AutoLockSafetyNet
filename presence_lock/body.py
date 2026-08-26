"""Body/pose fallback — the answer to "my head is down or turned away".

Every face detector needs a face. Look down at a keyboard, turn to a colleague,
or rest your head on your hand and there is no face in the frame at all, yet
you are plainly still at the desk. MediaPipe's pose model still finds your
shoulders and head there, so it supplies presence evidence when the face
pipeline has nothing to say.

This signal is deliberately *identity-blind*: it can only extend a presence
that face recognition already established (see `presence.py`), never start one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from . import models

log = logging.getLogger(__name__)

# MediaPipe pose landmark indices
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


@dataclass
class BodySignal:
    present: bool = False
    head_down: bool = False
    head_visible: bool = False
    bbox: tuple[int, int, int, int] | None = None
    score: float = 0.0

    @property
    def reason(self) -> str:
        if not self.present:
            return ""
        if self.head_down:
            return "head down"
        if not self.head_visible:
            return "body only"
        return "body"


class BodyDetector:
    """Thin wrapper over MediaPipe PoseLandmarker in VIDEO mode."""

    def __init__(self, min_visibility: float = 0.55, min_confidence: float = 0.5) -> None:
        self.min_visibility = float(min_visibility)
        self.available = False
        self._landmarker = None
        self._last_ts_ms = -1

        try:
            model_path = models.download(models.REGISTRY["pose"])
        except Exception as exc:
            log.warning("Body fallback disabled — pose model unavailable: %s", exc)
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=float(min_confidence),
                min_pose_presence_confidence=float(min_confidence),
                min_tracking_confidence=float(min_confidence),
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.available = True
            log.info("Body fallback: MediaPipe Pose (head-down / turned-away coverage)")
        except Exception as exc:
            log.warning("Body fallback disabled — MediaPipe unusable: %s", exc)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> BodySignal:
        if not self.available or self._landmarker is None or frame is None:
            return BodySignal()

        height, width = frame.shape[:2]
        # detect_for_video demands strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = timestamp_ms

        try:
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            log.debug("Pose inference failed: %s", exc)
            return BodySignal()

        if not result.pose_landmarks:
            return BodySignal()

        landmarks = result.pose_landmarks[0]

        def visibility(index: int) -> float:
            if index >= len(landmarks):
                return 0.0
            landmark = landmarks[index]
            return float(min(landmark.visibility, landmark.presence))

        shoulder_vis = min(visibility(LEFT_SHOULDER), visibility(RIGHT_SHOULDER))
        head_vis = max(visibility(NOSE), visibility(LEFT_EAR), visibility(RIGHT_EAR))
        score = max(shoulder_vis, head_vis)
        if score < self.min_visibility:
            return BodySignal()

        head_visible = head_vis >= self.min_visibility
        head_down = False
        if head_visible and shoulder_vis >= self.min_visibility:
            shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2.0
            shoulder_span = abs(landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x) or 0.2
            # Chin toward the chest: the nose drops toward the shoulder line.
            head_down = landmarks[NOSE].y > shoulder_y - 0.35 * shoulder_span
        elif shoulder_vis >= self.min_visibility:
            head_down = True  # shoulders but no head at all

        # Upper-body box, clipped to the frame, for the motion ROI and the HUD.
        points = [
            (landmarks[i].x * width, landmarks[i].y * height)
            for i in (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)
            if i < len(landmarks) and visibility(i) >= self.min_visibility * 0.6
        ]
        bbox = None
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1 = max(0, int(min(xs)) - 20)
            y1 = max(0, int(min(ys)) - 60)
            x2 = min(width, int(max(xs)) + 20)
            y2 = min(height, int(max(ys)) + 20)
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        return BodySignal(
            present=True,
            head_down=head_down,
            head_visible=head_visible,
            bbox=bbox,
            score=score,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
            self.available = False

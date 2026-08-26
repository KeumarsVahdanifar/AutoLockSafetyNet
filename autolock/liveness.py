"""Anti-spoofing: is this a real face, or a picture of one?

Recognition alone answers "is this Kian?" — it cannot tell a person from a
photograph of that person held up to the camera. The threat here is narrow but
real: someone props your photo in front of the webcam so the machine never
locks while you are away. (Spoofing can never *unlock* anything; this app only
ever decides when to lock.)

The detector is Intel Open Model Zoo's `anti-spoof-mn3` — a MobileNetV3
classifier, Apache-2.0, ~12 MB — run through OpenCV's DNN module, so it adds no
new runtime dependency. It looks for the texture and context cues that give
away print and screen replays.

Two things worth knowing about the model:

* its output layer already applies softmax, so the two values it returns are
  probabilities and must NOT be softmaxed again — doing so squeezes every
  result toward [0.27, 0.73] and destroys the signal;
* it is trained on face crops. Feeding it anything else produces confident
  nonsense, so it is only ever run on a detector's bounding box, padded out to
  include the context it expects.

**Calibrate before trusting it.** `autolock liveness --test` prints your live
score continuously; hold up a photo on a phone and watch what happens. Pick a
threshold between the two clusters. This ships disabled for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from . import models
from .backends.base import FaceDet

log = logging.getLogger(__name__)

# Preprocessing constants from the model's Open Model Zoo definition.
INPUT_SIZE = 128
MEAN = np.array((151.2405, 119.5950, 107.8395), dtype=np.float32)
SCALE = np.array((63.0105, 56.4570, 55.0035), dtype=np.float32)

# Output index that carries "this is a real face". Established empirically
# against live faces; `liveness --test` is how you confirm it on your own
# camera before turning the check on.
REAL_INDEX = 1


@dataclass
class LivenessResult:
    checked: bool = False
    live: bool = True  # fail open: an unchecked face is not a rejected face
    score: float = float("nan")
    reason: str = ""

    def __bool__(self) -> bool:
        return self.live


class LivenessDetector:
    """Scores a detected face on how likely it is to be physically present."""

    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, threshold: float = 0.55, crop_padding: float = 0.25) -> None:
        self.threshold = float(threshold)
        self.crop_padding = float(crop_padding)
        self.available = False
        self.status = "not initialised"
        self._net: cv2.dnn.Net | None = None
        self._failures = 0

        # Nothing in here may propagate. A liveness check is an enhancement to
        # the lock decision, never a prerequisite for it: if the model will not
        # download, will not parse, or the OpenCV build cannot run it, the
        # monitor carries on doing its actual job without the extra check.
        try:
            path = models.download(models.REGISTRY["antispoof"])
            self._net = cv2.dnn.readNet(str(path))
            self._self_test()
            self.available = True
            self.status = "ok"
            log.info("Liveness check: anti-spoof-mn3 (threshold %.2f)", self.threshold)
        except Exception as exc:
            self.status = f"unavailable: {exc}"
            log.warning(
                "Liveness (anti-spoofing) check is NOT active: %s. "
                "Everything else runs normally; faces just are not tested for spoofing.",
                exc,
            )

    def _self_test(self) -> None:
        """Prove the model actually runs before declaring it available."""
        probe = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        blob = ((probe - MEAN) / SCALE).transpose(2, 0, 1)[None, ...]
        assert self._net is not None
        self._net.setInput(blob)
        output = self._net.forward().ravel()
        if output.size <= REAL_INDEX:
            raise RuntimeError(
                f"model returned {output.size} value(s); expected at least {REAL_INDEX + 1}"
            )

    def _note_failure(self, exc: Exception) -> None:
        self._failures += 1
        log.debug("Liveness inference failed (%d): %s", self._failures, exc)
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
            self.available = False
            self.status = f"disabled after repeated failures: {exc}"
            log.warning(
                "Liveness check disabled after %d consecutive failures (%s). "
                "The monitor continues without it.",
                self._failures,
                exc,
            )

    # ------------------------------------------------------------------
    def _crop(self, frame: np.ndarray, det: FaceDet) -> np.ndarray | None:
        x, y, w, h = det.bbox
        pad = int(self.crop_padding * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return frame[y1:y2, x1:x2]

    def score(self, frame: np.ndarray, det: FaceDet) -> float:
        """Probability that the face is physically present. NaN if unavailable."""
        if not self.available or self._net is None:
            return float("nan")

        crop = self._crop(frame, det)
        if crop is None:
            return float("nan")

        try:
            resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            blob = ((rgb - MEAN) / SCALE).transpose(2, 0, 1)[None, ...]
            self._net.setInput(blob)
            # Already a probability distribution — do not softmax this again.
            output = self._net.forward().ravel()
        except Exception as exc:  # any failure here degrades, never propagates
            self._note_failure(exc)
            return float("nan")

        if output.size <= REAL_INDEX:
            self._note_failure(RuntimeError(f"unexpected output size {output.size}"))
            return float("nan")

        self._failures = 0
        return float(output[REAL_INDEX])

    def check(self, frame: np.ndarray, det: FaceDet) -> LivenessResult:
        value = self.score(frame, det)
        if np.isnan(value):
            # Fail open: a check that could not run is not evidence of a spoof,
            # and treating it as one would lock the screen for a cropped face.
            return LivenessResult(checked=False, live=True, score=value, reason="not checked")
        live = value >= self.threshold
        return LivenessResult(
            checked=True,
            live=live,
            score=value,
            reason="" if live else f"spoof suspected ({value:.2f} < {self.threshold:.2f})",
        )

    def close(self) -> None:
        self._net = None
        self.available = False

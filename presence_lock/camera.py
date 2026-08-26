"""Camera wrapper with reconnect-with-backoff and optional release/reopen."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

_APIS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


class Camera:
    def __init__(
        self,
        index: int = 0,
        api: str = "dshow",
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 5,
    ) -> None:
        self.index = int(index)
        self.api = _APIS.get(str(api).lower(), cv2.CAP_ANY)
        self.width = int(width)
        self.height = int(height)
        self.warmup_frames = int(warmup_frames)
        self._cap: cv2.VideoCapture | None = None
        self._backoff = 1.0
        self._next_retry = 0.0

    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        self.release()
        cap = cv2.VideoCapture(self.index, self.api)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prefer the freshest frame
        for _ in range(self.warmup_frames):
            cap.read()

        self._cap = cap
        self._backoff = 1.0
        log.info(
            "Camera %d open at %dx%d",
            self.index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Return a frame, transparently reconnecting with backoff."""
        if not self.is_open:
            now = time.monotonic()
            if now < self._next_retry:
                return None
            if not self.open():
                self._next_retry = now + self._backoff
                log.warning("Camera %d unavailable; retrying in %.0fs", self.index, self._backoff)
                self._backoff = min(self._backoff * 2, 15.0)
                return None

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            log.warning("Camera read failed; reconnecting")
            self.release()
            self._next_retry = time.monotonic() + self._backoff
            self._backoff = min(self._backoff * 2, 15.0)
            return None
        return frame

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

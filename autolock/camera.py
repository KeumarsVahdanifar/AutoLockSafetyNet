"""Camera wrapper with reconnect-with-backoff and optional release/reopen."""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

_APIS = {
    "auto": cv2.CAP_ANY,
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,  # Windows
    "msmf": cv2.CAP_MSMF,  # Windows
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
    "v4l2": cv2.CAP_V4L2,  # Linux
    "gstreamer": cv2.CAP_GSTREAMER,
}


def default_api() -> str:
    """The capture backend that opens fastest and most reliably per platform."""
    if sys.platform == "win32":
        return "dshow"
    if sys.platform == "darwin":
        return "avfoundation"
    if sys.platform.startswith("linux"):
        return "v4l2"
    return "auto"


class Camera:
    def __init__(
        self,
        index: int = 0,
        api: str = "auto",
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 5,
    ) -> None:
        self.index = int(index)
        requested = str(api).lower()
        if requested in ("", "auto"):
            requested = default_api()
        self.api_name = requested
        self.api = _APIS.get(requested, cv2.CAP_ANY)
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
            # A platform default can be wrong for an unusual camera (virtual
            # devices, some USB stacks), so fall back to letting OpenCV choose.
            if self.api != cv2.CAP_ANY:
                log.debug("%s backend failed for camera %d; retrying with auto",
                          self.api_name, self.index)
                cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
                if not cap.isOpened():
                    cap.release()
                    return False
            else:
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

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

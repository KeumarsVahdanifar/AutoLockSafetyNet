"""The monitor loop."""

from __future__ import annotations

import logging
import signal
import time
from types import FrameType

import cv2
import numpy as np

from .backends import build_backend
from .body import BodyDetector
from .camera import Camera
from .config import Config
from .identity import IdentityGallery
from .lock import is_session_locked, lock_workstation
from .presence import PresenceEngine
from .ui import draw_overlay

log = logging.getLogger(__name__)


class PresenceLockApp:
    def __init__(self, cfg: Config, arm: bool = True) -> None:
        self.cfg = cfg
        self.arm = arm  # False => `test` mode: everything runs, nothing locks
        self.backend = build_backend(cfg)
        self.gallery = IdentityGallery.load(
            name=cfg.identity,
            backend_name=self.backend.name,
            threshold=cfg.match_threshold,
            margin=cfg.match_margin,
        )
        self.body = (
            BodyDetector(min_visibility=cfg.body_min_visibility)
            if cfg.use_body_fallback
            else None
        )
        self.engine = PresenceEngine(cfg, self.backend, self.gallery, self.body)
        self.camera = Camera(
            cfg.camera_index,
            cfg.camera_api,
            cfg.frame_width,
            cfg.frame_height,
            cfg.camera_warmup_frames,
        )
        self._running = False
        self._paused = False
        self._fps = 0.0
        self._lock_probe_at = 0.0
        self._lock_probe_result = False

    # ------------------------------------------------------------------
    def stop(self, *_args: object) -> None:
        self._running = False

    def run(self) -> int:
        cfg = self.cfg
        self._running = True
        previous_sigint = signal.signal(signal.SIGINT, self._on_sigint)

        log.info(
            "Watching for %s | threshold %.3f | lock after %.0fs absent%s",
            ", ".join(self.gallery.names),
            self.gallery.threshold,
            cfg.absence_timeout_s,
            "  [DRY RUN]" if cfg.dry_run else ("  [TEST — will not lock]" if not self.arm else ""),
        )
        if cfg.preview:
            cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

        frame_budget = 1.0 / max(1.0, cfg.target_fps)
        session_was_locked = False

        try:
            while self._running:
                started = time.monotonic()

                # ---- secure desktop: idle instead of burning CPU ----
                if self._session_locked():
                    if not session_was_locked:
                        log.info("Session locked — monitor idle")
                        session_was_locked = True
                        if cfg.release_camera_when_locked:
                            self.camera.release()
                    time.sleep(1.0)
                    continue

                if session_was_locked:
                    log.info("Session unlocked — monitor active")
                    session_was_locked = False
                    self.engine.reset(present=True)

                if self._paused:
                    self._show_message("PAUSED — press P to resume")
                    if self._handle_keys() is False:
                        break
                    time.sleep(0.15)
                    continue

                frame = self.camera.read()
                if frame is None:
                    if self._handle_keys() is False:
                        break
                    time.sleep(0.2)
                    continue

                result = self.engine.process(frame)

                if result.should_lock:
                    self._do_lock(result.lock_reason)

                if cfg.preview:
                    view = draw_overlay(
                        frame,
                        result,
                        timeout_s=cfg.absence_timeout_s,
                        backend_name=self.backend.name,
                        identities=self.gallery.names,
                        fps=self._fps,
                        dry_run=cfg.dry_run or not self.arm,
                        mirror=cfg.mirror_preview,
                    )
                    cv2.imshow(cfg.window_name, view)

                if self._handle_keys() is False:
                    break

                elapsed = time.monotonic() - started
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(elapsed, 1e-3))
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.close()

        return 0

    # ------------------------------------------------------------------
    def _session_locked(self, interval: float = 0.5) -> bool:
        """Cached lock-screen probe — no need to ask the OS on every frame."""
        now = time.monotonic()
        if now >= self._lock_probe_at:
            self._lock_probe_result = is_session_locked()
            self._lock_probe_at = now + interval
        return self._lock_probe_result

    def _on_sigint(self, _sig: int, _frame: FrameType | None) -> None:
        log.info("Interrupted — shutting down")
        self._running = False

    def _do_lock(self, reason: str) -> None:
        self._lock_probe_at = 0.0  # the session state is about to change
        if not self.arm:
            log.info("Would lock: %s (test mode)", reason)
            self.engine.note_locked()
            return
        if self.cfg.dry_run:
            log.warning("DRY RUN — would lock now: %s", reason)
            self.engine.note_locked()
            return

        log.warning("Locking workstation: %s", reason)
        if lock_workstation():
            self.engine.note_locked()
        else:
            log.error("Lock failed; will retry after the cooldown")
            self.engine.note_locked()

    def _handle_keys(self) -> bool | None:
        """Return False to quit the loop."""
        if not self.cfg.preview:
            return None
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        if key in (ord("p"), ord("P")):
            self._paused = not self._paused
            log.info("Monitor %s", "paused" if self._paused else "resumed")
        elif key in (ord("l"), ord("L")):
            self._do_lock("manual (L)")
        return None

    def _show_message(self, text: str) -> None:
        if not self.cfg.preview:
            return
        canvas = np.full((160, 520, 3), 30, dtype=np.uint8)
        cv2.putText(
            canvas, text, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow(self.cfg.window_name, canvas)

    def close(self) -> None:
        self.camera.release()
        if self.body is not None:
            self.body.close()
        self.backend.close()
        if self.cfg.preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

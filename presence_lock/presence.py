"""The presence state machine.

Evidence is layered, strongest first, and each weaker layer is only allowed to
*extend* a presence that face recognition already established:

    face     you were recognised in this frame                 (resets everything)
    tracked  a face sits in your tracked box but scores too
             low to recognise — steep yaw, backlight, motion blur
    body     no face at all, but MediaPipe still sees your head
             and shoulders: head down over the keyboard, turned away
    motion   nothing but movement in the region you last occupied

Each weaker layer has its own expiry measured from the last real recognition,
so an empty chair can never be propped up indefinitely by a moving curtain.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .backends import FaceBackend, FaceDet, detect_with_rotations, iou
from .body import BodyDetector, BodySignal
from .config import Config
from .identity import IdentityGallery, MatchResult

log = logging.getLogger(__name__)

EV_FACE = "face"
EV_TRACKED = "tracked"
EV_BODY = "body"
EV_MOTION = "motion"
EV_NONE = "none"

_EVIDENCE_LABEL = {
    EV_FACE: "recognised",
    EV_TRACKED: "tracked face",
    EV_BODY: "body",
    EV_MOTION: "motion",
    EV_NONE: "nothing",
}


@dataclass
class ScoredFace:
    det: FaceDet
    match: MatchResult | None = None
    tracked: bool = False

    @property
    def similarity(self) -> float:
        return self.match.similarity if self.match else float("nan")

    @property
    def label(self) -> str:
        if self.match and self.match.is_match:
            return self.match.name
        if self.tracked:
            return "tracked"
        if self.match and self.match.is_stranger:
            return "unknown"
        return "unscored"


@dataclass
class FrameResult:
    faces: list[ScoredFace] = field(default_factory=list)
    owner: ScoredFace | None = None
    strangers: list[ScoredFace] = field(default_factory=list)
    body: BodySignal = field(default_factory=BodySignal)
    motion: bool = False
    stranger_for: float = 0.0  # seconds an unrecognised face has been in frame
    evidence: str = EV_NONE
    absent_for: float = 0.0
    seconds_to_lock: float = math.inf
    should_lock: bool = False
    lock_reason: str = ""
    rotated: bool = False

    @property
    def evidence_label(self) -> str:
        if self.evidence == EV_BODY and self.body.reason:
            return self.body.reason
        return _EVIDENCE_LABEL.get(self.evidence, self.evidence)


class MotionSensor:
    """Mean absolute frame difference inside a region of interest."""

    def __init__(self, threshold: float = 3.0, work_width: int = 160) -> None:
        self.threshold = float(threshold)
        self.work_width = int(work_width)
        self._previous: np.ndarray | None = None

    def update(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> bool:
        height, width = frame.shape[:2]
        scale = self.work_width / float(width)
        small = cv2.resize(
            frame, (self.work_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return False

        delta = cv2.absdiff(previous, gray)
        if roi is not None:
            x, y, w, h = roi
            x1 = max(0, int(x * scale))
            y1 = max(0, int(y * scale))
            x2 = min(gray.shape[1], int((x + w) * scale))
            y2 = min(gray.shape[0], int((y + h) * scale))
            if x2 > x1 and y2 > y1:
                delta = delta[y1:y2, x1:x2]
        return float(delta.mean()) >= self.threshold

    def reset(self) -> None:
        self._previous = None


class PresenceEngine:
    """Turns frames into presence evidence and lock decisions."""

    def __init__(
        self,
        cfg: Config,
        backend: FaceBackend,
        gallery: IdentityGallery,
        body: BodyDetector | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.gallery = gallery
        self.body = body
        self.motion = MotionSensor(cfg.motion_threshold) if cfg.use_motion_fallback else None

        now = time.monotonic()
        self._frame_index = 0
        self._last_present = now
        # Zero, not `now`: the weak layers may only extend a presence that a
        # real recognition established, so on a cold start nobody gets a free
        # body/tracking window before the owner has been seen once.
        self._last_strong = 0.0
        self._match_streak = 0
        self._stranger_since: float | None = None
        self._track_box: tuple[int, int, int, int] | None = None
        self._track_updated = 0.0
        self._locked = False
        self._lock_until = 0.0
        self._last_evidence = EV_FACE

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def locked(self) -> bool:
        return self._locked

    def reset(self, present: bool = True) -> None:
        """Re-arm the monitor, e.g. after the session is unlocked again.

        Unlocking proves someone authenticated, so the absence countdown starts
        fresh — but not the recognition timestamp: the owner still has to be
        seen before body or motion evidence counts for anything.
        """
        now = time.monotonic()
        self._last_present = now if present else 0.0
        self._last_strong = 0.0
        self._match_streak = 0
        self._stranger_since = None
        self._track_box = None
        self._locked = False
        if self.motion:
            self.motion.reset()

    def note_locked(self) -> None:
        """Record that the workstation was locked."""
        now = time.monotonic()
        self._locked = True
        self._lock_until = now + self.cfg.lock_cooldown_s
        # Weak evidence must not resurrect a presence after a lock: only a real
        # recognition may re-arm the monitor.
        self._last_strong = 0.0
        self._last_present = 0.0
        self._track_box = None
        self._stranger_since = None

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> FrameResult:
        cfg = self.cfg
        now = time.monotonic()
        self._frame_index += 1
        result = FrameResult()

        detections = self.backend.detect(frame)
        if (
            not detections
            and cfg.rotation_retry
            and cfg.rotation_angles
            and self._frame_index % max(1, cfg.rotation_retry_every) == 0
        ):
            detections = detect_with_rotations(self.backend, frame, cfg.rotation_angles)
            result.rotated = bool(detections)

        scored = self._score(detections, now)
        result.faces = scored

        owner = next((f for f in scored if f.match and f.match.is_match), None)
        result.owner = owner
        # Sitting where you sat does not make someone you: a face scored below
        # the stranger cut-off is listed here wherever it appears in frame.
        result.strangers = [
            f for f in scored if f.match and f.match.is_stranger and f is not owner
        ]

        # --- strongest layer: a confirmed recognition -------------------
        if owner is not None:
            self._match_streak += 1
            if self._match_streak >= max(1, cfg.confirm_frames):
                self._last_strong = now
                self._last_present = now
                result.evidence = EV_FACE
            self._track_box = owner.det.bbox
            self._track_updated = now
        else:
            self._match_streak = 0

        # --- a stranger invalidates every weak layer --------------------
        # Somebody else's face at the desk must not be propped up by the body
        # or motion they themselves produce, so the graces expire immediately.
        # This holds even with `lock_on_unknown` off, where it simply means the
        # normal absence countdown runs instead of being extended.
        result.stranger_for = self._update_stranger_timer(result.strangers, owner, now)
        if result.stranger_for >= cfg.unknown_confirm_s:
            self._last_strong = 0.0
            self._track_box = None

        # --- weaker layers ---------------------------------------------
        if result.evidence == EV_NONE:
            tracked = next((f for f in scored if f.tracked), None)
            if tracked is not None and self._within(now, cfg.track_grace_s):
                self._last_present = now
                self._track_box = tracked.det.bbox
                self._track_updated = now
                result.evidence = EV_TRACKED

        if result.evidence == EV_NONE and self.body is not None and self.body.available:
            result.body = self.body.detect(frame)
            if result.body.present and self._within(now, cfg.body_grace_s):
                self._last_present = now
                if result.body.bbox:
                    self._track_box = result.body.bbox
                    self._track_updated = now
                result.evidence = EV_BODY

        if self.motion is not None:
            roi = self._expanded_track_box(frame.shape) if self._track_box else None
            result.motion = self.motion.update(frame, roi)
            if (
                result.evidence == EV_NONE
                and result.motion
                and self._within(now, cfg.motion_grace_s)
            ):
                self._last_present = now
                result.evidence = EV_MOTION

        if result.evidence != self._last_evidence:
            log.debug("presence evidence: %s -> %s", self._last_evidence, result.evidence)
            self._last_evidence = result.evidence

        # --- lock decision ---------------------------------------------
        result.absent_for = max(0.0, now - self._last_present)
        result.seconds_to_lock = max(0.0, cfg.absence_timeout_s - result.absent_for)

        if owner is not None and self._match_streak >= max(1, cfg.confirm_frames):
            self._locked = False  # owner is back: re-arm

        reason = self._lock_reason(result, owner, now)
        if reason and not self._locked and now >= self._lock_until:
            result.should_lock = True
            result.lock_reason = reason

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _within(self, now: float, grace: float) -> bool:
        """True while the last real recognition is recent enough for a weak layer."""
        return self._last_strong > 0.0 and (now - self._last_strong) <= grace

    def _expanded_track_box(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        x, y, w, h = self._track_box  # type: ignore[misc]
        height, width = shape[:2]
        pad_x, pad_y = int(w * 0.6), int(h * 0.8)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _score(self, detections: list[FaceDet], now: float) -> list[ScoredFace]:
        """Attach identity scores, embedding only as often as configured."""
        cfg = self.cfg
        scored: list[ScoredFace] = []
        track_fresh = self._track_box is not None and (now - self._track_updated) <= max(
            cfg.track_grace_s, 5.0
        )
        # The largest face (index 0) is always recognised, so noticing that you
        # came back is never delayed. Extra faces — the stranger check — are
        # scored every Nth frame, or on every frame while nobody is confirmed.
        recognise = (
            self._frame_index % max(1, cfg.recognize_every) == 0 or self._match_streak == 0
        )

        for index, det in enumerate(detections[:3]):
            face = ScoredFace(det=det)
            if index == 0 or recognise:
                embedding = self.backend.embed(det)
                if embedding is not None:
                    face.match = self.gallery.match(embedding)
            # The tracked layer exists for faces that are *probably* yours but
            # score too low to confirm. A face below the stranger cut-off is
            # not ambiguous, so it never inherits your box.
            in_box = track_fresh and iou(det.bbox, self._track_box) >= 0.3  # type: ignore[arg-type]
            face.tracked = in_box and not (face.match is not None and face.match.is_stranger)
            scored.append(face)
        return scored

    def _update_stranger_timer(
        self, strangers: list[ScoredFace], owner: ScoredFace | None, now: float
    ) -> float:
        """Seconds an unrecognised face has been in frame with you absent.

        Seeing you resets it, so a colleague reading over your shoulder while
        you sit there never registers as an intruder.
        """
        if not strangers or owner is not None:
            self._stranger_since = None
            return 0.0
        if self._stranger_since is None:
            self._stranger_since = now
        return now - self._stranger_since

    def _lock_reason(self, result: FrameResult, owner: ScoredFace | None, now: float) -> str:
        cfg = self.cfg

        if result.absent_for >= cfg.absence_timeout_s:
            return f"absent for {result.absent_for:.0f}s (last evidence: {result.evidence_label})"

        if cfg.lock_on_unknown and result.stranger_for >= cfg.unknown_confirm_s:
            best = max((f.similarity for f in result.strangers), default=float("nan"))
            return (
                f"unrecognised face for {result.stranger_for:.0f}s "
                f"(best score {best:.2f} < {self.gallery.threshold:.2f})"
            )

        return ""

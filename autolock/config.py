"""Configuration model and on-disk paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
IDENTITY_DIR = DATA_DIR / "identity"
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class Config:
    """Every tunable knob. Loaded from config.json, overridable from the CLI."""

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    camera_index: int = 0
    # auto picks the right backend per platform: DirectShow on Windows,
    # AVFoundation on macOS, V4L2 on Linux. Override only for odd hardware.
    camera_api: str = "auto"  # auto | dshow | msmf | avfoundation | v4l2 | gstreamer | any
    frame_width: int = 640
    frame_height: int = 480
    target_fps: float = 12.0  # processing rate; lower burns less CPU
    camera_warmup_frames: int = 5

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    identity: str = ""  # "" -> use the only enrolled identity, or all of them
    match_threshold: float = 0.0  # 0 -> use the backend's recommended value
    match_margin: float = 0.08  # below (threshold - margin) a face counts as a stranger
    confirm_frames: int = 2  # consecutive matches needed to (re)confirm the owner

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    backend: str = "auto"  # auto | opencv | insightface
    detect_width: int = 640  # frames are downscaled to this before detection
    det_threshold: float = 0.55  # YuNet/SCRFD score floor
    min_face_px: int = 48
    # The largest face is recognised on every frame; any *additional* faces in
    # shot (the stranger check) only every Nth, since they are rarely urgent.
    recognize_every: int = 2

    # Rotated-frame retry: catches a head tilted onto a hand or resting on a desk,
    # where an upright detector sees nothing at all.
    rotation_retry: bool = True
    rotation_angles: tuple[int, ...] = (-35, 35, -60, 60)
    rotation_retry_every: int = 3  # only every Nth frame with no face (it is not free)

    # ------------------------------------------------------------------
    # Presence fallbacks (the "head down / looking away" layer)
    # ------------------------------------------------------------------
    # Off by default: with the *_hold_s ceilings at 0 these signals cannot
    # affect the lock decision, so running them would only cost CPU. Turn one
    # on together with its hold to give yourself head-down time again.
    use_body_fallback: bool = False  # MediaPipe pose: shoulders/head still visible
    body_min_visibility: float = 0.55
    use_motion_fallback: bool = False  # last resort: movement inside the owner's region
    motion_threshold: float = 3.0  # mean abs frame delta inside the ROI

    # The countdown starts the moment recognition stops and runs continuously.
    # These are the *maximum* seconds it may reach while that evidence is in
    # shot — a ceiling on how long you may go unrecognised, never a reset.
    #
    # 0 disables the layer's influence entirely, which is the default: only
    # your recognised face keeps the session open. Raise one (e.g.
    # body_hold_s = 60) together with its use_*_fallback flag to buy back time
    # for working head-down.
    track_hold_s: float = 0.0  # unrecognised face sitting in the owner's tracked box
    body_hold_s: float = 0.0  # body visible, face not — head down, turned away
    motion_hold_s: float = 0.0  # movement only
    # A layer counts as active for this long after its last sighting, so one
    # dropped pose frame does not slam the deadline shut mid-countdown.
    evidence_hold_s: float = 1.5

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------
    absence_timeout_s: float = 3.0
    lock_cooldown_s: float = 10.0  # ignore further lock requests for this long
    lock_on_unknown: bool = True  # stranger at the desk with the owner gone -> lock now
    unknown_confirm_s: float = 2.0  # a stranger must persist this long to count
    dry_run: bool = False  # log the lock instead of performing it
    release_camera_when_locked: bool = True  # drop the handle so the webcam LED goes out

    # ------------------------------------------------------------------
    # Safety — guards against locking you out of your own machine
    # ------------------------------------------------------------------
    # Never lock until you have been recognised at least once this session.
    # At login the camera is often still warming up and you may not be seated:
    # the worst case of this guard is "it never locks", never "it locks forever".
    require_initial_recognition: bool = True
    startup_grace_s: float = 10.0  # no locking at all for this long after start
    max_locks_per_window: int = 3  # circuit breaker: 0 disables it
    lock_window_s: float = 60.0
    breaker_pause_s: float = 300.0  # how long the breaker stays tripped
    pause_file: str = "PAUSE"  # create this file in the project dir to stop locking

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    preview: bool = True
    mirror_preview: bool = True  # selfie view; the processed frame is never mirrored
    window_name: str = "AutoLock Safety Net"
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key.startswith("_") or key not in known:
                continue
            if known[key].type == "tuple[int, ...]" and isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rotation_angles"] = list(self.rotation_angles)
        return data

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_CONFIG_PATH
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def normalise(self) -> Config:
        """Switch on the detectors that the configured holds actually need.

        A hold without its detector would silently do nothing, so asking for
        `--body-hold 60` turns the pose model on. The reverse is deliberately
        not done: a detector with a zero hold still draws its evidence on the
        preview, which is useful in `test`.
        """
        if self.body_hold_s > 0:
            self.use_body_fallback = True
        if self.motion_hold_s > 0:
            self.use_motion_fallback = True
        return self

    def apply_overrides(self, overrides: dict[str, Any]) -> Config:
        """Apply non-None CLI overrides in place and return self."""
        known = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if value is None or key not in known:
                continue
            setattr(self, key, value)
        return self


def ensure_dirs() -> None:
    for directory in (MODELS_DIR, DATA_DIR, IDENTITY_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

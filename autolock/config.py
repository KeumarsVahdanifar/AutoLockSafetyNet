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


def coerce_value(cfg: Config, key: str, raw: str) -> Any:
    """Parse a string into the type the config field already holds.

    Shared by `config --set` and the GUI settings panel so both accept and
    reject exactly the same things.
    """
    current = getattr(cfg, key)  # raises AttributeError for unknown keys
    text = raw.strip()

    if isinstance(current, bool):  # before int: bool is a subclass of int
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} expects a boolean, got {raw!r}")
    if isinstance(current, int):
        try:
            return int(float(text)) if "." in text else int(text)
        except ValueError:
            raise ValueError(f"{key} expects a whole number, got {raw!r}") from None
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{key} expects a number, got {raw!r}") from None
    if isinstance(current, tuple):
        try:
            return tuple(int(part) for part in text.split(",") if part.strip())
        except ValueError:
            raise ValueError(f"{key} expects comma-separated whole numbers, got {raw!r}") from None
    return raw


# ----------------------------------------------------------------------
# Field registry — one entry per Config field.
#
# The GUI settings page is generated from this, so a new option appears there
# automatically. `tests/test_config_spec.py` fails if a field is added to
# Config without an entry here, which is what keeps the two in step.
# ----------------------------------------------------------------------
CAMERA = "Camera"
IDENTITY = "Identity"
DETECTION = "Detection"
FALLBACKS = "Working head-down"
LOCKING = "Locking"
SAFETY = "Safety"
INTERFACE = "Interface"

GROUP_ORDER = (CAMERA, IDENTITY, DETECTION, FALLBACKS, LOCKING, SAFETY, INTERFACE)


@dataclass(frozen=True)
class FieldSpec:
    group: str
    label: str
    help: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    # advanced=True keeps a setting out of the GUI's default view. The bar is
    # "would someone using this ever want to change it?" — model plumbing,
    # tuning constants and internals sit behind the toggle so the page stays
    # about decisions rather than parameters. Search still reaches them.
    advanced: bool = False


FIELD_SPECS: dict[str, FieldSpec] = {
    # --- camera ---
    "camera_index": FieldSpec(
        CAMERA, "Camera", "Which webcam to use. 0 is your default device.", 0, 8, 1
    ),
    "target_fps": FieldSpec(
        CAMERA, "Processing rate",
        "Frames analysed per second. The main dial for CPU use.", 1, 30, 1,
    ),
    "camera_api": FieldSpec(
        CAMERA, "Capture backend",
        "auto picks per platform: DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux.",
        choices=("auto", "dshow", "msmf", "avfoundation", "v4l2", "gstreamer", "any"),
        advanced=True,
    ),
    "frame_width": FieldSpec(
        CAMERA, "Frame width", "Requested capture width in pixels.", 160, 1920, 16, advanced=True
    ),
    "frame_height": FieldSpec(
        CAMERA, "Frame height", "Requested capture height in pixels.", 120, 1080, 16, advanced=True
    ),
    "camera_warmup_frames": FieldSpec(
        CAMERA, "Warm-up frames", "Frames discarded while exposure settles.",
        0, 30, 1, advanced=True,
    ),
    # --- identity ---
    "identity": FieldSpec(
        IDENTITY, "Watch for", "Leave blank to accept every enrolled identity."
    ),
    "match_threshold": FieldSpec(
        IDENTITY, "How sure it must be",
        "Similarity needed to count as you. 0 uses the backend default. Raise it if it "
        "recognises other people; lower it if it keeps missing you.", 0.0, 0.8, 0.01,
    ),
    "match_margin": FieldSpec(
        IDENTITY, "Stranger margin",
        "How far below the threshold a face must score before it counts as somebody else "
        "rather than an unclear reading of you.", 0.0, 0.4, 0.01, advanced=True,
    ),
    "confirm_frames": FieldSpec(
        IDENTITY, "Confirm frames",
        "Consecutive matching frames before you count as recognised.", 1, 10, 1, advanced=True,
    ),
    # --- detection ---
    "rotation_retry": FieldSpec(
        DETECTION, "Catch a tilted head",
        "When no face is found, retry on rotated copies of the frame. Recovers a head "
        "resting on a hand.",
    ),
    "backend": FieldSpec(
        DETECTION, "Face engine",
        "auto prefers InsightFace when installed, otherwise OpenCV (YuNet + SFace).",
        choices=("auto", "opencv", "insightface"), advanced=True,
    ),
    "det_threshold": FieldSpec(
        DETECTION, "Detector confidence", "Score floor for accepting a detected face.",
        0.1, 0.99, 0.01, advanced=True,
    ),
    "min_face_px": FieldSpec(
        DETECTION, "Smallest face (px)", "Ignore faces smaller than this.",
        16, 200, 4, advanced=True,
    ),
    "detect_width": FieldSpec(
        DETECTION, "Detection width",
        "Frames are scaled to this before detection. Lower is faster, worse at distance.",
        160, 1280, 16, advanced=True,
    ),
    "recognize_every": FieldSpec(
        DETECTION, "Recognise every Nth frame",
        "Your face is always scored; extra faces only this often.", 1, 10, 1, advanced=True,
    ),
    "rotation_angles": FieldSpec(
        DETECTION, "Rotation angles", "Comma-separated degrees to retry at.", advanced=True
    ),
    "rotation_retry_every": FieldSpec(
        DETECTION, "Retry every Nth empty frame",
        "Rotation retries cost time; do them only this often.", 1, 10, 1, advanced=True,
    ),
    # --- fallbacks ---
    "body_hold_s": FieldSpec(
        FALLBACKS, "Head-down allowance",
        "Seconds you may go unrecognised while your body is still visible. 0 means only "
        "your recognised face counts. This is a ceiling, never a reset: the countdown "
        "keeps falling.", 0, 300, 5,
    ),
    "track_hold_s": FieldSpec(
        FALLBACKS, "Unclear-face allowance",
        "Seconds you may go unrecognised while a face sits where yours was but scores too "
        "low to confirm.", 0, 120, 1,
    ),
    "motion_hold_s": FieldSpec(
        FALLBACKS, "Movement allowance",
        "Seconds you may go unrecognised while there is movement where you were. The "
        "weakest signal of the three.", 0, 120, 1,
    ),
    "evidence_hold_s": FieldSpec(
        FALLBACKS, "Signal persistence",
        "How long a signal counts as active after its last sighting, so one dropped frame "
        "does not slam the deadline shut.", 0.1, 10, 0.1, advanced=True,
    ),
    "use_body_fallback": FieldSpec(
        FALLBACKS, "Run the pose model",
        "Switched on automatically when the head-down allowance is above 0.", advanced=True,
    ),
    "body_min_visibility": FieldSpec(
        FALLBACKS, "Pose confidence", "How clearly the pose model must see you.",
        0.1, 0.99, 0.01, advanced=True,
    ),
    "use_motion_fallback": FieldSpec(
        FALLBACKS, "Run motion detection",
        "Switched on automatically when the movement allowance is above 0.", advanced=True,
    ),
    "motion_threshold": FieldSpec(
        FALLBACKS, "Motion sensitivity",
        "Mean frame difference counted as movement. Lower is more sensitive.",
        0.5, 30, 0.5, advanced=True,
    ),
    # --- locking ---
    "absence_timeout_s": FieldSpec(
        LOCKING, "Lock after",
        "Seconds from your last recognition to the screen locking. The countdown starts "
        "the moment you are not recognised.", 1, 120, 1,
    ),
    "lock_on_unknown": FieldSpec(
        LOCKING, "Lock on a stranger",
        "Lock when an unrecognised face is at the desk and you are not. Seeing you resets "
        "it, so a colleague beside you is not an intruder.",
    ),
    "unknown_confirm_s": FieldSpec(
        LOCKING, "Stranger delay",
        "How long an unrecognised face must persist before it counts.", 0.5, 30, 0.5,
    ),
    "dry_run": FieldSpec(
        LOCKING, "Dry run", "Log the decision instead of actually locking. Use while tuning.",
    ),
    "release_camera_when_locked": FieldSpec(
        LOCKING, "Free the camera when locked",
        "Drop the camera while the session is locked, so the webcam light goes out.",
    ),
    "lock_cooldown_s": FieldSpec(
        LOCKING, "Lock cooldown",
        "Ignore further lock requests for this long after locking.", 1, 120, 1, advanced=True,
    ),
    # --- safety ---
    "require_initial_recognition": FieldSpec(
        SAFETY, "Stay disarmed until it sees you",
        "Never lock until you have been recognised once. The main protection against being "
        "locked out at login.",
    ),
    "startup_grace_s": FieldSpec(
        SAFETY, "Grace period",
        "No locking for this long after starting or unlocking. Being recognised ends it "
        "immediately.", 0, 120, 1,
    ),
    "max_locks_per_window": FieldSpec(
        SAFETY, "Stop after this many locks",
        "If locks start firing repeatedly something is wrong, so locking stops. 0 turns "
        "this protection off.", 0, 20, 1,
    ),
    "lock_window_s": FieldSpec(
        SAFETY, "Counted over", "The period those locks are counted over.",
        10, 600, 10, advanced=True,
    ),
    "breaker_pause_s": FieldSpec(
        SAFETY, "Pause for", "How long locking stays off once that trips.",
        30, 1800, 30, advanced=True,
    ),
    "pause_file": FieldSpec(
        SAFETY, "Pause file name",
        "Create a file with this name in the project folder to stop locking, from anywhere.",
        advanced=True,
    ),
    # --- interface ---
    "mirror_preview": FieldSpec(
        INTERFACE, "Mirror the preview",
        "Selfie view. The frame that is analysed is never mirrored.",
    ),
    "preview": FieldSpec(
        INTERFACE, "Preview window when run from a terminal",
        "The control panel always draws its own preview; this is for `autolock run`.",
    ),
    "log_level": FieldSpec(
        INTERFACE, "Log detail", "How much goes to logs/autolock.log.",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"), advanced=True,
    ),
    "window_name": FieldSpec(
        INTERFACE, "Terminal preview title", "Title of the `autolock run` preview window.",
        advanced=True,
    ),
}


def grouped_fields(include_advanced: bool = True) -> dict[str, list[str]]:
    """Field names by group, in display order."""
    groups: dict[str, list[str]] = {name: [] for name in GROUP_ORDER}
    for key, spec in FIELD_SPECS.items():
        if spec.advanced and not include_advanced:
            continue
        groups.setdefault(spec.group, []).append(key)
    return {name: keys for name, keys in groups.items() if keys}

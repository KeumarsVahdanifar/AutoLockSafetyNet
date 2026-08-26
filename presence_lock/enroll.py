"""Enrolment — teaching the monitor your face, and only your face.

The wizard walks through a short list of head poses and collects several
embeddings for each. Enrolling the awkward poses (looking down at the keyboard,
turned to one side, head resting on a hand) is what lets recognition keep up
later, when you are working rather than posing for the camera.

Samples are quality-gated (detector score, face size, sharpness, exposure) and
de-duplicated, so holding still does not fill the template with copies of one
frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .backends import FaceBackend, FaceDet, build_backend, detect_with_rotations
from .camera import Camera
from .config import Config
from .identity import Identity, list_identities
from .ui import AMBER, FONT, GREEN, GREY, RED, WHITE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseStep:
    key: str
    instruction: str
    hint: str


POSE_STEPS: tuple[PoseStep, ...] = (
    PoseStep("front", "Look straight at the camera", "neutral expression"),
    PoseStep("left", "Turn your head to the LEFT", "about 30-45 degrees"),
    PoseStep("right", "Turn your head to the RIGHT", "about 30-45 degrees"),
    PoseStep("up", "Tilt your chin UP", "as if reading the top of the screen"),
    PoseStep("down", "Look DOWN at your keyboard", "the pose that used to lock your PC"),
    PoseStep("tilt", "Rest your head on one hand", "tilt it toward a shoulder"),
    PoseStep("natural", "Work normally", "type, read, glance around, lean back"),
)

MIN_SHARPNESS = 35.0  # variance of Laplacian
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 225.0


@dataclass
class QualityCheck:
    ok: bool
    reason: str = ""


def check_quality(frame: np.ndarray, det: FaceDet, min_score: float, min_px: int) -> QualityCheck:
    x, y, w, h = det.bbox
    if min(w, h) < min_px:
        return QualityCheck(False, "too far away")
    if det.score < min_score:
        return QualityCheck(False, "low detector score")

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return QualityCheck(False, "face outside frame")

    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        return QualityCheck(False, "too blurry — hold still")

    brightness = float(crop.mean())
    if brightness < MIN_BRIGHTNESS:
        return QualityCheck(False, "too dark")
    if brightness > MAX_BRIGHTNESS:
        return QualityCheck(False, "overexposed")

    return QualityCheck(True)


def _largest_face(backend: FaceBackend, frame: np.ndarray, cfg: Config) -> FaceDet | None:
    detections = backend.detect(frame)
    if not detections and cfg.rotation_retry and cfg.rotation_angles:
        detections = detect_with_rotations(backend, frame, cfg.rotation_angles)
    return detections[0] if detections else None


def _draw_progress(
    frame: np.ndarray,
    step: PoseStep,
    step_index: int,
    total_steps: int,
    captured: int,
    per_pose: int,
    status: str,
    status_color: tuple[int, int, int],
    total_captured: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 92), (25, 25, 25), -1)
    cv2.rectangle(overlay, (0, height - 34), (width, height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        f"[{step_index + 1}/{total_steps}] {step.instruction}",
        (12, 30),
        FONT,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, step.hint, (12, 52), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (12, 78), FONT, 0.5, status_color, 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w = width - 172, 24, 160
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (70, 70, 70), -1)
    filled = int(bar_w * min(1.0, captured / max(1, per_pose)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 12), GREEN, -1)
    cv2.putText(
        frame,
        f"{captured}/{per_pose}  (total {total_captured})",
        (bar_x, bar_y + 32),
        FONT,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "SPACE skip pose  |  ESC cancel",
        (12, height - 12),
        FONT,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def enroll_interactive(
    cfg: Config,
    name: str,
    samples_per_pose: int = 8,
    pose_timeout_s: float = 25.0,
    min_interval_s: float = 0.25,
    append: bool = False,
    poses: tuple[PoseStep, ...] = POSE_STEPS,
) -> Identity:
    """Run the guided capture wizard and write the template to disk."""
    backend = build_backend(cfg)

    identity: Identity | None = None
    if append and name in list_identities():
        identity = Identity.load_by_name(name)
        if identity.backend != backend.name:
            raise RuntimeError(
                f"'{name}' was enrolled with backend '{identity.backend}' but the active "
                f"backend is '{backend.name}'. Re-enrol without --append."
            )
        log.info("Appending to existing template for '%s' (%d samples)", name, len(identity))

    window = f"Enrol: {name}"
    aborted = False
    total_captured = 0

    with Camera(
        cfg.camera_index, cfg.camera_api, cfg.frame_width, cfg.frame_height, cfg.camera_warmup_frames
    ) as camera:
        if not camera.is_open:
            raise RuntimeError(f"cannot open camera {cfg.camera_index}")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        for step_index, step in enumerate(poses):
            captured = 0
            last_capture = 0.0
            deadline = time.monotonic() + pose_timeout_s
            settle_until = time.monotonic() + 1.6  # time to actually get into the pose

            while captured < samples_per_pose and time.monotonic() < deadline:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                status, color = "hold the pose...", AMBER
                det = _largest_face(backend, frame, cfg)

                if det is None:
                    status, color = "no face visible — adjust slightly", RED
                else:
                    quality = check_quality(frame, det, cfg.det_threshold, cfg.min_face_px)
                    if not quality.ok:
                        status, color = quality.reason, RED
                    elif now < settle_until:
                        status, color = "get into position...", AMBER
                    elif now - last_capture >= min_interval_s:
                        embedding = backend.embed(det)
                        if embedding is None:
                            status, color = "could not align face", RED
                        else:
                            if identity is None:
                                identity = Identity(
                                    name=name,
                                    embeddings=embedding[None, :],
                                    poses=[step.key],
                                    backend=backend.name,
                                    threshold=backend.default_threshold,
                                    meta={"embedding_dim": int(embedding.size)},
                                )
                                added = True
                            else:
                                added = identity.add(embedding, step.key)
                            if added:
                                captured += 1
                                total_captured += 1
                                last_capture = now
                                status, color = "captured", GREEN
                            else:
                                status, color = "too similar — move a little", AMBER
                    else:
                        status, color = "captured", GREEN

                    x, y, w, h = det.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                view = _draw_progress(
                    frame,
                    step,
                    step_index,
                    len(poses),
                    captured,
                    samples_per_pose,
                    status,
                    color,
                    total_captured,
                )
                if cfg.mirror_preview:
                    view = cv2.flip(view, 1)
                cv2.imshow(window, view)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    aborted = True
                    break
                if key == 32:  # SPACE
                    break

            if aborted:
                break
            if captured == 0:
                log.warning("Pose '%s' captured nothing — skipped", step.key)

        cv2.destroyWindow(window)
        cv2.waitKey(1)

    if aborted and (identity is None or len(identity) < 5):
        raise KeyboardInterrupt("enrolment cancelled")
    if identity is None or len(identity) < 5:
        raise RuntimeError(
            "enrolment collected too few samples — check lighting and camera framing"
        )

    identity.meta.update(
        {
            "poses_covered": sorted(set(identity.poses)),
            "backend": backend.name,
            "updated": time.time(),
        }
    )
    path = identity.save()
    _report(identity, path)
    backend.close()
    return identity


def enroll_from_images(cfg: Config, name: str, folder: Path, append: bool = False) -> Identity:
    """Build a template from a folder of photos instead of the webcam."""
    backend = build_backend(cfg)
    files = sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"no images found in {folder}")

    identity: Identity | None = None
    if append and name in list_identities():
        identity = Identity.load_by_name(name)

    used, skipped = 0, 0
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        det = _largest_face(backend, image, cfg)
        if det is None:
            log.warning("No face in %s", path.name)
            skipped += 1
            continue
        embedding = backend.embed(det)
        if embedding is None:
            skipped += 1
            continue
        if identity is None:
            identity = Identity(
                name=name,
                embeddings=embedding[None, :],
                poses=["image"],
                backend=backend.name,
                threshold=backend.default_threshold,
                meta={"embedding_dim": int(embedding.size)},
            )
            used += 1
        elif identity.add(embedding, "image"):
            used += 1
        else:
            skipped += 1

    if identity is None or len(identity) < 3:
        raise RuntimeError("not enough usable faces in the folder")

    identity.meta.update({"source": str(folder), "updated": time.time()})
    path = identity.save()
    log.info("Used %d image(s), skipped %d", used, skipped)
    _report(identity, path)
    backend.close()
    return identity


def _report(identity: Identity, path: Path) -> None:
    stats = identity.intra_stats()
    counts: dict[str, int] = {}
    for pose in identity.poses:
        counts[pose] = counts.get(pose, 0) + 1

    log.info("Saved '%s' -> %s", identity.name, path)
    log.info(
        "  %d samples across %d pose(s): %s",
        len(identity),
        len(counts),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )
    log.info(
        "  match threshold %.3f | template self-similarity min=%.3f p05=%.3f mean=%.3f",
        identity.threshold,
        stats["min"],
        stats["p05"],
        stats["mean"],
    )
    if stats["p05"] < identity.threshold:
        log.warning(
            "  Some enrolled poses score below the threshold against each other. "
            "That is expected with steep angles — the run loop falls back to tracking, "
            "body and motion evidence. Run `python plock.py test` to watch live scores."
        )

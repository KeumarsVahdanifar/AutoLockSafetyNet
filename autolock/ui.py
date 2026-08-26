"""Preview overlay: boxes, presence evidence, countdown."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .presence import EV_FACE, EV_NONE, FrameResult, ScoredFace

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (80, 220, 100)
AMBER = (60, 190, 250)
RED = (70, 70, 240)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
DARK = (28, 28, 28)


def _face_color(face: ScoredFace) -> tuple[int, int, int]:
    if face.match and face.match.is_match:
        return GREEN
    if face.tracked:
        return AMBER
    if face.match and face.match.is_stranger:
        return RED
    return GREY


def _label(face: ScoredFace) -> str:
    if face.match is None:
        return "face"
    if face.match.is_match:
        return f"{face.match.name} {face.match.similarity:.2f}"
    return f"{face.label} {face.match.similarity:.2f}"


def _banner(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not lines:
        return
    pad, line_h = 8, 22
    height = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (10, pad + line_h * index + 15), FONT, 0.5, color, 1, cv2.LINE_AA
        )


def mirror_box(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    """Reflect an (x, y, w, h) box across the vertical centre line."""
    x, y, w, h = box
    return (width - x - w, y, w, h)


def mirror_points(points: np.ndarray, width: int) -> np.ndarray:
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[:, 0] = width - 1 - flipped[:, 0]
    return flipped


def draw_overlay(
    frame: np.ndarray,
    result: FrameResult,
    *,
    timeout_s: float,
    backend_name: str,
    identities: list[str],
    fps: float = 0.0,
    dry_run: bool = False,
    mirror: bool = False,
    safety: str = "",
) -> np.ndarray:
    # Mirror the image *before* drawing, then reflect only the geometry.
    # Flipping afterwards would reverse every label as well.
    if mirror:
        frame = cv2.flip(frame, 1)
    width = frame.shape[1]

    for face in result.faces:
        box = face.det.bbox
        x, y, w, h = mirror_box(box, width) if mirror else box
        color = _face_color(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = _label(face)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
        cv2.putText(frame, text, (x + 4, y - 5), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        if face.det.landmarks is not None:
            points = mirror_points(face.det.landmarks, width) if mirror else face.det.landmarks
            for px, py in points.astype(int):
                cv2.circle(frame, (int(px), int(py)), 2, color, -1)

    if result.body.present and result.body.bbox and result.evidence != EV_FACE:
        box = result.body.bbox
        x, y, w, h = mirror_box(box, width) if mirror else box
        cv2.rectangle(frame, (x, y), (x + w, y + h), AMBER, 1)
        cv2.putText(frame, result.body.reason, (x, y - 6), FONT, 0.45, AMBER, 1, cv2.LINE_AA)

    if result.evidence == EV_NONE:
        state_color = RED
    elif result.evidence == EV_FACE:
        state_color = GREEN
    else:
        state_color = AMBER

    # Never clamp to `timeout_s`: the deadline is anchored to the last
    # recognition, so this figure has to be free to show the longer fuse a
    # weak layer buys — and to be seen ticking down the whole way.
    countdown = "--" if math.isinf(result.seconds_to_lock) else f"{result.seconds_to_lock:.1f}s"
    lines = [
        (
            f"presence: {result.evidence_label:<13} lock in: {countdown}"
            + (
                f"   unrecognised {result.unrecognised_for:.0f}s"
                if result.evidence != EV_FACE and result.unrecognised_for > 0
                else ""
            ),
            state_color,
        ),
        (
            f"{backend_name} | {', '.join(identities) or 'no identity'} | "
            f"{fps:.1f} fps{' | rotated' if result.rotated else ''}"
            f"{' | DRY-RUN' if dry_run else ''}",
            WHITE,
        ),
    ]
    if result.stranger_for > 0:
        lines.append(
            (
                f"unrecognised face for {result.stranger_for:.1f}s",
                RED,
            )
        )
    if result.withheld_reason:
        lines.append((f"lock withheld: {result.withheld_reason}", AMBER))
    elif safety and ("DISARMED" in safety or "PAUSED" in safety or "BREAKER" in safety):
        lines.append((safety, AMBER))
    _banner(frame, lines)

    if result.seconds_to_lock <= min(3.0, timeout_s) and result.evidence != EV_FACE:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 4)

    hint = "ESC quit  |  P pause  |  L lock now"
    cv2.putText(
        frame, hint, (10, frame.shape[0] - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )
    return frame

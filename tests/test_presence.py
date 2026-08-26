"""Unit tests for the presence state machine, identity store and geometry."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presence_lock import identity as identity_mod  # noqa: E402
from presence_lock import presence as presence_mod  # noqa: E402
from presence_lock.backends.base import FaceDet, iou, map_detection_back, rotate_frame  # noqa: E402
from presence_lock.body import BodySignal  # noqa: E402
from presence_lock.config import Config  # noqa: E402
from presence_lock.identity import Identity, IdentityGallery  # noqa: E402
from presence_lock.presence import EV_BODY, EV_FACE, EV_NONE, PresenceEngine  # noqa: E402

DIM = 8
OWNER = np.eye(DIM, dtype=np.float32)[0]
# Orthogonal to OWNER *and* to BLURRY_OWNER below, so it stays a stranger
# against every sample in the template.
STRANGER = np.eye(DIM, dtype=np.float32)[2]
# Similarity 0.45 to OWNER: below the 0.5 match threshold, above the stranger
# cut-off — the score an awkwardly-posed owner actually produces.
BLURRY_OWNER = np.asarray([0.45, 0.893] + [0.0] * (DIM - 2), dtype=np.float32)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubBackend:
    """Backend whose detections and embeddings the test drives directly."""

    name = "stub"
    embedding_dim = DIM
    default_threshold = 0.5

    def __init__(self) -> None:
        self.queue: list[tuple[tuple[int, int, int, int], np.ndarray | None]] = []

    def set(self, *faces: tuple[tuple[int, int, int, int], np.ndarray | None]) -> None:
        self.queue = list(faces)

    def detect(self, frame):
        return [
            FaceDet(bbox=box, score=0.9, image=frame, raw=embedding)
            for box, embedding in self.queue
        ]

    def embed(self, det):
        det.embedding = det.raw
        return det.raw

    def close(self):
        pass


class StubBody:
    def __init__(self, present: bool = False) -> None:
        self.available = True
        self.present = present

    def detect(self, _frame) -> BodySignal:
        return BodySignal(present=self.present, head_down=True, bbox=(10, 10, 40, 40))

    def close(self):
        pass


def make_gallery(threshold: float = 0.5, margin: float = 0.08) -> IdentityGallery:
    person = Identity(
        name="owner",
        embeddings=OWNER[None, :],
        poses=["front"],
        backend="stub",
        threshold=threshold,
    )
    return IdentityGallery([person], threshold=threshold, margin=margin)


def make_config(**overrides) -> Config:
    cfg = Config()
    cfg.confirm_frames = 1
    cfg.recognize_every = 1
    cfg.use_body_fallback = False
    cfg.use_motion_fallback = False
    cfg.rotation_retry = False
    cfg.absence_timeout_s = 5.0
    cfg.lock_cooldown_s = 10.0
    # Shipped defaults are all 0 (only recognition counts); tests that exercise
    # a layer opt into a hold explicitly.
    cfg.track_hold_s = 0.0
    cfg.body_hold_s = 0.0
    cfg.motion_hold_s = 0.0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


class PresenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = mock.patch.object(presence_mod.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = StubBackend()

    def build(self, cfg: Config, body: StubBody | None = None) -> PresenceEngine:
        return PresenceEngine(cfg, self.backend, make_gallery(), body)

    # ------------------------------------------------------------------
    def test_recognised_owner_keeps_session_open(self):
        engine = self.build(make_config())
        self.backend.set(((10, 10, 80, 80), OWNER))
        for _ in range(3):
            self.clock.advance(4.0)
            result = engine.process(FRAME)
        self.assertEqual(result.evidence, EV_FACE)
        self.assertFalse(result.should_lock)
        self.assertEqual(result.owner.match.name, "owner")

    def test_absence_locks_after_timeout(self):
        engine = self.build(make_config())
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set()  # nobody in frame
        self.clock.advance(4.0)
        self.assertFalse(engine.process(FRAME).should_lock)

        self.clock.advance(2.0)
        result = engine.process(FRAME)
        self.assertTrue(result.should_lock)
        self.assertEqual(result.evidence, EV_NONE)
        self.assertIn("not recognised", result.lock_reason)

    def test_countdown_runs_while_a_face_is_visible_but_unrecognised(self):
        """The headline rule: not recognised means the clock is running."""
        cfg = make_config(absence_timeout_s=5.0)
        engine = self.build(cfg)
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        # A face is right there, in the tracked box, every frame — and the
        # countdown still falls, because it is not a recognised face.
        self.backend.set(((12, 12, 80, 80), BLURRY_OWNER))
        remaining = []
        for _ in range(4):
            self.clock.advance(1.0)
            result = engine.process(FRAME)
            remaining.append(result.seconds_to_lock)

        self.assertEqual(remaining, sorted(remaining, reverse=True))
        self.assertEqual(remaining, [4.0, 3.0, 2.0, 1.0])
        self.assertGreater(result.unrecognised_for, 0.0)

        self.clock.advance(1.5)
        self.assertTrue(engine.process(FRAME).should_lock)

    def test_recognition_is_the_only_thing_that_resets_the_clock(self):
        cfg = make_config(absence_timeout_s=5.0)
        engine = self.build(cfg)
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set(((12, 12, 80, 80), BLURRY_OWNER))
        self.clock.advance(3.0)
        self.assertAlmostEqual(engine.process(FRAME).seconds_to_lock, 2.0)

        self.backend.set(((10, 10, 80, 80), OWNER))  # you look back at the camera
        self.clock.advance(1.0)
        result = engine.process(FRAME)
        self.assertAlmostEqual(result.seconds_to_lock, 5.0)
        self.assertEqual(result.unrecognised_for, 0.0)

    def test_a_hold_caps_the_countdown_without_resetting_it(self):
        """Opt-in behaviour: a hold buys a longer fuse, not a fresh one."""
        cfg = make_config(absence_timeout_s=5.0, track_hold_s=10.0)
        engine = self.build(cfg)
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set(((12, 12, 80, 80), BLURRY_OWNER))
        self.clock.advance(4.0)
        first = engine.process(FRAME)
        self.assertEqual(first.evidence, presence_mod.EV_TRACKED)
        self.assertAlmostEqual(first.seconds_to_lock, 6.0)  # 10 - 4, still falling

        self.clock.advance(4.0)
        self.assertAlmostEqual(engine.process(FRAME).seconds_to_lock, 2.0)

        self.clock.advance(2.5)  # past the 10 s ceiling
        self.assertTrue(engine.process(FRAME).should_lock)

    def test_a_body_alone_does_not_stop_the_lock_by_default(self):
        """Shipped behaviour: head down, no recognised face, clock runs out."""
        cfg = make_config(use_body_fallback=True)  # body_hold_s stays 0
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set()  # head down: no face at all
        self.clock.advance(6.0)
        result = engine.process(FRAME)
        self.assertEqual(result.evidence, EV_BODY)  # still reported...
        self.assertTrue(result.should_lock)  # ...but it holds nothing

    def test_body_hold_buys_head_down_time_when_enabled(self):
        cfg = make_config(use_body_fallback=True, body_hold_s=120.0)
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set()
        for _ in range(5):
            self.clock.advance(10.0)
            result = engine.process(FRAME)
            self.assertFalse(result.should_lock, "body evidence should hold the session open")
        self.assertEqual(result.evidence, EV_BODY)
        self.assertTrue(result.body.head_down)
        self.assertAlmostEqual(result.seconds_to_lock, 70.0)  # 120 - 50, falling

    def test_body_hold_expires_so_an_empty_chair_still_locks(self):
        cfg = make_config(use_body_fallback=True, body_hold_s=30.0)
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set()
        self.clock.advance(31.0)
        self.assertTrue(engine.process(FRAME).should_lock)

    def test_body_alone_cannot_arm_presence_after_a_lock(self):
        cfg = make_config(use_body_fallback=True, body_hold_s=30.0)
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set()
        self.clock.advance(31.0)  # past the body hold
        self.assertTrue(engine.process(FRAME).should_lock)
        engine.note_locked()

        # A body in front of a locked machine must not count as presence.
        self.clock.advance(10.0)
        result = engine.process(FRAME)
        self.assertEqual(result.seconds_to_lock, 0.0)
        self.assertTrue(engine.locked)
        self.assertFalse(result.should_lock)

    def test_body_cannot_arm_presence_on_a_cold_start(self):
        """Before the owner has ever been recognised, a body means nothing."""
        cfg = make_config(use_body_fallback=True, body_hold_s=120.0)
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set()
        self.clock.advance(6.0)
        result = engine.process(FRAME)
        self.assertEqual(result.seconds_to_lock, 0.0)
        self.assertTrue(result.should_lock)

    def test_owner_return_rearms_after_lock(self):
        engine = self.build(make_config())
        self.backend.set()
        self.clock.advance(10.0)
        engine.note_locked()

        self.clock.advance(20.0)
        self.backend.set(((10, 10, 80, 80), OWNER))
        result = engine.process(FRAME)
        self.assertFalse(engine.locked)
        self.assertEqual(result.evidence, EV_FACE)

    def test_stranger_locks_after_the_confirmation_window(self):
        cfg = make_config(lock_on_unknown=True, unknown_confirm_s=2.0)
        engine = self.build(cfg)
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set(((150, 10, 80, 80), STRANGER))
        self.clock.advance(1.0)
        self.assertFalse(engine.process(FRAME).should_lock)
        self.clock.advance(2.5)
        result = engine.process(FRAME)
        self.assertTrue(result.should_lock)
        self.assertIn("unrecognised", result.lock_reason)

    def test_stranger_in_the_owners_tracked_box_still_locks(self):
        """Someone sitting down where you sat does not inherit your session."""
        cfg = make_config(lock_on_unknown=True, unknown_confirm_s=2.0, track_hold_s=60.0)
        engine = self.build(cfg)
        box = (10, 10, 80, 80)
        self.backend.set((box, OWNER))
        engine.process(FRAME)

        self.backend.set((box, STRANGER))  # same position, different face
        self.clock.advance(1.0)
        first = engine.process(FRAME)
        self.assertFalse(first.faces[0].tracked, "a stranger must not count as tracked")
        self.assertEqual(len(first.strangers), 1)
        self.assertNotEqual(first.evidence, presence_mod.EV_TRACKED)

        self.clock.advance(2.5)
        result = engine.process(FRAME)
        self.assertTrue(result.should_lock)

    def test_stranger_cancels_the_body_fallback(self):
        """A stranger's own body must not hold your session open."""
        cfg = make_config(
            lock_on_unknown=False,  # even with the intruder lock disabled
            use_body_fallback=True,
            body_hold_s=600.0,
            unknown_confirm_s=2.0,
        )
        engine = self.build(cfg, StubBody(present=True))
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        # The stranger has to persist past unknown_confirm_s before the graces
        # are cut, which debounces a single misread frame of the real owner.
        self.backend.set(((150, 10, 80, 80), STRANGER))
        self.clock.advance(1.0)
        engine.process(FRAME)
        self.clock.advance(2.5)
        engine.process(FRAME)  # stranger confirmed -> weak layers expire

        self.backend.set()  # they look down; only a body is visible
        self.clock.advance(4.0)
        result = engine.process(FRAME)
        self.assertEqual(result.seconds_to_lock, 0.0)
        self.assertTrue(result.should_lock)

    def test_stranger_beside_the_owner_is_not_an_intruder(self):
        """Someone reading over your shoulder while you sit there is fine."""
        cfg = make_config(lock_on_unknown=True, unknown_confirm_s=2.0)
        engine = self.build(cfg)
        for _ in range(4):
            self.backend.set(((10, 10, 80, 80), OWNER), ((150, 10, 80, 80), STRANGER))
            self.clock.advance(2.0)
            result = engine.process(FRAME)
            self.assertFalse(result.should_lock)
        self.assertEqual(result.stranger_for, 0.0)
        self.assertEqual(result.evidence, EV_FACE)

    def test_low_scoring_owner_is_not_treated_as_a_stranger(self):
        """The margin keeps an awkward pose out of the intruder path."""
        cfg = make_config(lock_on_unknown=True, match_margin=0.2)
        engine = PresenceEngine(cfg, self.backend, make_gallery(margin=0.2), None)
        self.backend.set(((10, 10, 80, 80), OWNER))
        engine.process(FRAME)

        self.backend.set(((12, 12, 80, 80), BLURRY_OWNER))
        self.clock.advance(3.0)
        result = engine.process(FRAME)
        self.assertEqual(result.strangers, [])
        self.assertNotIn("unrecognised", result.lock_reason)

    def test_confirm_frames_debounces_a_single_lucky_match(self):
        cfg = make_config(confirm_frames=3)
        engine = self.build(cfg)
        self.backend.set(((10, 10, 80, 80), OWNER))
        # Until the streak is met the match is not trusted, so the clock keeps
        # running even though the face is right there.
        self.assertNotEqual(engine.process(FRAME).evidence, EV_FACE)
        self.clock.advance(1.0)
        second = engine.process(FRAME)
        self.assertNotEqual(second.evidence, EV_FACE)
        self.assertLess(second.seconds_to_lock, 5.0)

        self.clock.advance(1.0)
        third = engine.process(FRAME)
        self.assertEqual(third.evidence, EV_FACE)
        self.assertAlmostEqual(third.seconds_to_lock, 5.0)


class ShippedDefaultsTests(unittest.TestCase):
    """The defaults a user gets with no config.json."""

    def test_only_recognition_holds_the_session_open(self):
        cfg = Config()
        self.assertEqual(cfg.absence_timeout_s, 5.0)
        self.assertEqual(cfg.track_hold_s, 0.0)
        self.assertEqual(cfg.body_hold_s, 0.0)
        self.assertEqual(cfg.motion_hold_s, 0.0)
        self.assertFalse(cfg.use_body_fallback)
        self.assertFalse(cfg.use_motion_fallback)
        self.assertTrue(cfg.lock_on_unknown)

    def test_a_hold_switches_on_the_detector_it_needs(self):
        cfg = Config()
        cfg.body_hold_s = 60.0
        cfg.motion_hold_s = 10.0
        cfg.normalise()
        self.assertTrue(cfg.use_body_fallback)
        self.assertTrue(cfg.use_motion_fallback)

    def test_a_detector_without_a_hold_is_left_alone(self):
        cfg = Config()
        cfg.use_body_fallback = True  # preview-only, as `test` uses it
        cfg.normalise()
        self.assertTrue(cfg.use_body_fallback)
        self.assertEqual(cfg.body_hold_s, 0.0)


class IdentityTests(unittest.TestCase):
    def test_save_load_roundtrip_and_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(identity_mod, "IDENTITY_DIR", Path(tmp)):
                person = Identity("kian", OWNER[None, :], ["front"], "stub", 0.5)
                person.add(BLURRY_OWNER, "down")
                path = person.save()
                self.assertTrue(path.exists())

                loaded = Identity.load_by_name("kian")
                self.assertEqual(len(loaded), 2)
                self.assertEqual(loaded.poses, ["front", "down"])
                self.assertEqual(loaded.backend, "stub")

                gallery = IdentityGallery([loaded], threshold=0.5)
                self.assertTrue(gallery.match(OWNER).is_match)
                self.assertTrue(gallery.match(STRANGER).is_stranger)

    def test_duplicate_samples_are_rejected(self):
        person = Identity("kian", OWNER[None, :], ["front"], "stub", 0.5)
        self.assertFalse(person.add(OWNER.copy(), "front"))
        self.assertTrue(person.add(STRANGER, "left"))

    def test_max_similarity_beats_centroid_for_multipose(self):
        person = Identity("kian", OWNER[None, :], ["front"], "stub", 0.5)
        person.add(STRANGER, "down")  # deliberately distant pose sample
        best, _ = person.similarity(STRANGER)
        self.assertAlmostEqual(best, 1.0, places=5)
        self.assertLess(float(person.centroid @ STRANGER), best)


class GeometryTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)

    def test_rotated_detection_maps_back_to_original_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        rotated, inverse = rotate_frame(frame, 35.0)
        self.assertGreater(rotated.shape[0], frame.shape[0])

        centre_box = (rotated.shape[1] // 2 - 20, rotated.shape[0] // 2 - 20, 40, 40)
        det = FaceDet(bbox=centre_box, score=0.9)
        mapped = map_detection_back(det, inverse)
        x, y, w, h = mapped.bbox
        self.assertAlmostEqual(x + w / 2, frame.shape[1] / 2, delta=2)
        self.assertAlmostEqual(y + h / 2, frame.shape[0] / 2, delta=2)


class MotionTests(unittest.TestCase):
    def test_motion_sensor_detects_change_in_roi(self):
        sensor = presence_mod.MotionSensor(threshold=3.0)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertFalse(sensor.update(blank, None))
        self.assertFalse(sensor.update(blank, None))

        moved = blank.copy()
        moved[60:180, 80:240] = 255
        self.assertTrue(sensor.update(moved, None))

    def test_motion_outside_roi_is_ignored(self):
        sensor = presence_mod.MotionSensor(threshold=3.0)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        sensor.update(blank, (0, 0, 40, 40))

        moved = blank.copy()
        moved[150:240, 250:320] = 255  # movement far from the ROI
        self.assertFalse(sensor.update(moved, (0, 0, 40, 40)))


class OverlayTests(unittest.TestCase):
    def test_overlay_draws_without_error(self):
        from presence_lock.presence import FrameResult, ScoredFace
        from presence_lock.ui import draw_overlay

        gallery = make_gallery()
        face = ScoredFace(det=FaceDet(bbox=(10, 10, 60, 60), score=0.9))
        face.match = gallery.match(OWNER)
        result = FrameResult(faces=[face], owner=face, evidence=EV_FACE, seconds_to_lock=4.0)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = draw_overlay(
            frame, result, timeout_s=8.0, backend_name="stub", identities=["owner"], fps=12.0
        )
        self.assertEqual(out.shape, frame.shape)

    def test_mirrored_preview_reflects_boxes_but_not_text(self):
        """The image is flipped before drawing, so labels stay readable."""
        from presence_lock.presence import FrameResult, ScoredFace
        from presence_lock.ui import draw_overlay

        # A distinctive stripe on the left of the source image.
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :40] = (255, 0, 0)

        face = ScoredFace(det=FaceDet(bbox=(10, 100, 60, 60), score=0.9))
        result = FrameResult(faces=[face], evidence=EV_FACE, seconds_to_lock=4.0)

        out = draw_overlay(
            frame.copy(),
            result,
            timeout_s=8.0,
            backend_name="stub",
            identities=["owner"],
            mirror=True,
        )
        # The stripe moved to the right edge...
        self.assertTrue((out[200, 300] == (255, 0, 0)).all())
        # ...and the face box followed it: 320 - 10 - 60 = 250.
        self.assertFalse((out[100:160, 240:260] == 0).all())
        # Nothing is left behind at the box's original position.
        self.assertTrue((out[100:160, 10:70] == 0).all())

        # The status banner is drawn after the flip, so on a uniform background
        # it renders pixel-identical either way — proof the text is not reversed.
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        kwargs = dict(
            timeout_s=8.0, backend_name="stub", identities=["owner"], fps=12.0
        )
        mirrored_banner = draw_overlay(blank.copy(), FrameResult(), mirror=True, **kwargs)
        plain_banner = draw_overlay(blank.copy(), FrameResult(), mirror=False, **kwargs)
        self.assertTrue((mirrored_banner[:52] == plain_banner[:52]).all())
        self.assertGreater(int(mirrored_banner[:52].sum()), 0)  # something was drawn

    def test_mirror_box_is_an_involution(self):
        from presence_lock.ui import mirror_box

        self.assertEqual(mirror_box((10, 20, 60, 60), 320), (250, 20, 60, 60))
        self.assertEqual(mirror_box(mirror_box((10, 20, 60, 60), 320), 320), (10, 20, 60, 60))


if __name__ == "__main__":
    unittest.main(verbosity=2)

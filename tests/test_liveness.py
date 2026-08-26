"""Anti-spoofing tests, with an emphasis on failing open.

The liveness check is an enhancement to the lock decision, never a
prerequisite for it. Every one of its failure modes must leave the monitor
running and merely stop testing faces for spoofing.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolock import liveness as liveness_mod  # noqa: E402
from autolock.backends.base import FaceDet  # noqa: E402
from autolock.liveness import REAL_INDEX, LivenessDetector, LivenessResult  # noqa: E402


def setUpModule() -> None:
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


FRAME = np.full((480, 640, 3), 120, dtype=np.uint8)
DET = FaceDet(bbox=(200, 150, 160, 160), score=0.9)


def detector_with_net(net) -> LivenessDetector:
    """A detector wired to a stub network, skipping the model download."""
    with mock.patch.object(liveness_mod.models, "download", return_value=Path("stub.onnx")), \
         mock.patch.object(liveness_mod.cv2.dnn, "readNet", return_value=net):
        return LivenessDetector(threshold=0.55)


class StubNet:
    """Stands in for cv2.dnn.Net.

    `fail_after` lets a net pass the constructor's self-test and only then
    start failing, which is what an intermittent runtime fault looks like.
    """

    def __init__(
        self,
        output=(0.1, 0.9),
        raises: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.output = np.asarray(output, dtype=np.float32)
        self.raises = raises
        self.fail_after = fail_after
        self.calls = 0

    def setInput(self, _blob) -> None:
        pass

    def forward(self):
        self.calls += 1
        if self.raises is not None and (
            self.fail_after is None or self.calls > self.fail_after
        ):
            raise self.raises
        return self.output.reshape(1, -1)


class ScoringTests(unittest.TestCase):
    def test_a_live_face_scores_high(self):
        detector = detector_with_net(StubNet(output=(0.02, 0.98)))
        self.assertTrue(detector.available)

        result = detector.check(FRAME, DET)
        self.assertTrue(result.checked)
        self.assertTrue(result.live)
        self.assertAlmostEqual(result.score, 0.98, places=5)

    def test_a_spoof_scores_low_and_is_rejected(self):
        detector = detector_with_net(StubNet(output=(0.93, 0.07)))
        result = detector.check(FRAME, DET)
        self.assertTrue(result.checked)
        self.assertFalse(result.live)
        self.assertIn("spoof suspected", result.reason)

    def test_the_model_output_is_not_softmaxed_again(self):
        """The network already emits probabilities; re-softmaxing destroys them.

        A second softmax squeezes [0.02, 0.98] to about [0.28, 0.72], which
        would drag every genuine face toward the threshold.
        """
        detector = detector_with_net(StubNet(output=(0.02, 0.98)))
        self.assertAlmostEqual(detector.score(FRAME, DET), 0.98, places=5)
        self.assertGreater(detector.score(FRAME, DET), 0.9)

    def test_real_index_is_the_second_output(self):
        detector = detector_with_net(StubNet(output=(0.3, 0.7)))
        self.assertEqual(REAL_INDEX, 1)
        self.assertAlmostEqual(detector.score(FRAME, DET), 0.7, places=5)


class FailOpenTests(unittest.TestCase):
    """None of these may raise, and none may report a spoof."""

    def test_a_missing_model_leaves_the_detector_unavailable(self):
        with mock.patch.object(
            liveness_mod.models, "download", side_effect=OSError("no network")
        ):
            detector = LivenessDetector()
        self.assertFalse(detector.available)
        self.assertIn("no network", detector.status)

    def test_an_unreadable_model_does_not_raise(self):
        with mock.patch.object(liveness_mod.models, "download", return_value=Path("x.onnx")), \
             mock.patch.object(
                 liveness_mod.cv2.dnn, "readNet", side_effect=RuntimeError("bad onnx")
             ):
            detector = LivenessDetector()
        self.assertFalse(detector.available)

    def test_a_model_with_the_wrong_output_shape_is_rejected_at_startup(self):
        detector = detector_with_net(StubNet(output=(0.5,)))  # single value
        self.assertFalse(detector.available, "self-test should catch this before use")

    def test_an_unavailable_detector_reports_live_not_spoof(self):
        with mock.patch.object(
            liveness_mod.models, "download", side_effect=OSError("nope")
        ):
            detector = LivenessDetector()
        result = detector.check(FRAME, DET)
        self.assertFalse(result.checked)
        self.assertTrue(result.live, "an unchecked face must never count as a spoof")

    def test_a_model_that_fails_its_self_test_never_goes_live(self):
        detector = detector_with_net(StubNet(raises=RuntimeError("broken from the start")))
        self.assertFalse(detector.available, "a broken model is caught before first use")

    def test_inference_errors_fail_open(self):
        detector = detector_with_net(
            StubNet(raises=RuntimeError("inference blew up"), fail_after=1)
        )
        result = detector.check(FRAME, DET)
        self.assertFalse(result.checked)
        self.assertTrue(result.live)

    def test_repeated_failures_disable_the_check_instead_of_spamming(self):
        net = StubNet(raises=RuntimeError("broken"), fail_after=1)  # survives the self-test
        detector = detector_with_net(net)
        self.assertTrue(detector.available)

        for _ in range(LivenessDetector.MAX_CONSECUTIVE_FAILURES + 2):
            self.assertTrue(detector.check(FRAME, DET).live)

        self.assertFalse(detector.available)
        self.assertIn("disabled after repeated failures", detector.status)

        calls_so_far = net.calls
        detector.check(FRAME, DET)  # no further inference once disabled
        self.assertEqual(net.calls, calls_so_far)

    def test_a_face_box_outside_the_frame_fails_open(self):
        detector = detector_with_net(StubNet(output=(0.02, 0.98)))
        offscreen = FaceDet(bbox=(5000, 5000, 100, 100), score=0.9)
        result = detector.check(FRAME, offscreen)
        self.assertFalse(result.checked)
        self.assertTrue(result.live)

    def test_default_result_is_permissive(self):
        self.assertTrue(LivenessResult().live)
        self.assertFalse(LivenessResult().checked)


class EngineIntegrationTests(unittest.TestCase):
    """A spoofed face must not reset the countdown, and must not crash it."""

    def setUp(self) -> None:
        from test_presence import FRAME as ENGINE_FRAME
        from test_presence import OWNER, FakeClock, StubBackend, make_config, make_gallery

        self.clock = FakeClock()
        patcher = mock.patch.object(
            sys.modules["autolock.presence"].time, "monotonic", self.clock
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.engine_frame = ENGINE_FRAME
        self.owner_embedding = OWNER
        self.backend = StubBackend()
        self.make_config = make_config
        self.make_gallery = make_gallery

    def _engine(self, liveness):
        from autolock.presence import PresenceEngine

        cfg = self.make_config()
        engine = PresenceEngine(cfg, self.backend, self.make_gallery(), None)
        engine.liveness = liveness
        return engine

    def test_a_spoofed_owner_does_not_reset_the_clock(self):
        from autolock.presence import EV_FACE

        engine = self._engine(detector_with_net(StubNet(output=(0.95, 0.05))))
        self.backend.set(((10, 10, 80, 80), self.owner_embedding))

        result = engine.process(self.engine_frame)
        self.assertTrue(result.spoof_rejected)
        self.assertIsNone(result.owner, "a photo of you is not you")
        self.assertNotEqual(result.evidence, EV_FACE)

        self.clock.advance(6.0)
        self.assertTrue(engine.process(self.engine_frame).should_lock)

    def test_a_live_owner_still_resets_the_clock(self):
        from autolock.presence import EV_FACE

        engine = self._engine(detector_with_net(StubNet(output=(0.02, 0.98))))
        self.backend.set(((10, 10, 80, 80), self.owner_embedding))

        result = engine.process(self.engine_frame)
        self.assertFalse(result.spoof_rejected)
        self.assertEqual(result.evidence, EV_FACE)

    def test_a_broken_liveness_model_never_breaks_the_monitor(self):
        from autolock.presence import EV_FACE

        engine = self._engine(
            detector_with_net(StubNet(raises=RuntimeError("boom"), fail_after=1))
        )
        self.backend.set(((10, 10, 80, 80), self.owner_embedding))

        result = engine.process(self.engine_frame)  # must not raise
        self.assertFalse(result.spoof_rejected)
        self.assertEqual(result.evidence, EV_FACE, "recognition must still work")

    def test_no_liveness_detector_at_all_is_fine(self):
        from autolock.presence import EV_FACE

        engine = self._engine(None)
        self.backend.set(((10, 10, 80, 80), self.owner_embedding))
        self.assertEqual(engine.process(self.engine_frame).evidence, EV_FACE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

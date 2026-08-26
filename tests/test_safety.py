"""Tests for the guards that stop the locker locking you out of your machine."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolock import safety as safety_mod  # noqa: E402
from autolock.safety import (  # noqa: E402
    ArmingGate,
    Blindness,
    CircuitBreaker,
    PauseSwitch,
    SafetySupervisor,
)


def setUpModule() -> None:
    # These guards log loudly on purpose; that is noise in a test run.
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ClockedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = mock.patch.object(safety_mod.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)


class ArmingGateTests(ClockedTest):
    def test_disarmed_until_recognised(self):
        gate = ArmingGate(require_recognition=True, startup_grace_s=0.0)
        self.assertFalse(gate.check())
        self.assertIn("not been recognised", gate.check().reason)

        gate.note_recognised()
        self.assertTrue(gate.check())

    def test_recognition_waives_the_startup_grace(self):
        """Once you have been seen, waiting out the grace serves no purpose."""
        gate = ArmingGate(require_recognition=True, startup_grace_s=20.0)
        self.assertFalse(gate.check())  # grace, and not yet recognised

        gate.note_recognised()
        self.assertTrue(gate.check(), "recognition should end the grace immediately")

    def test_grace_still_covers_the_window_before_recognition(self):
        gate = ArmingGate(require_recognition=False, startup_grace_s=20.0)
        self.assertFalse(gate.check())
        self.assertIn("startup grace", gate.check().reason)

        self.clock.advance(21.0)
        self.assertTrue(gate.check())

    def test_unlocking_restarts_the_grace(self):
        gate = ArmingGate(require_recognition=True, startup_grace_s=20.0)
        gate.note_recognised()
        self.clock.advance(30.0)
        self.assertTrue(gate.check())

        gate.restart_grace()  # the user just typed their password
        self.assertFalse(gate.check(), "a fresh unlock gets the grace back")

        gate.note_recognised()  # ...until they are recognised again
        self.assertTrue(gate.check())

    def test_recognition_can_be_waived(self):
        gate = ArmingGate(require_recognition=False, startup_grace_s=0.0)
        self.assertTrue(gate.check())


class BlindnessTests(ClockedTest):
    def test_no_frames_forbids_locking(self):
        blind = Blindness()
        self.assertTrue(blind.check())

        blind.note_no_frame()
        self.assertFalse(blind.check())
        self.assertIn("blind", blind.check().reason)

    def test_regaining_sight_asks_for_a_countdown_restart(self):
        blind = Blindness(resume_reset_s=1.0)
        blind.note_no_frame()
        self.clock.advance(5.0)
        self.assertTrue(blind.note_frame(), "a long blindness must restart the countdown")
        self.assertTrue(blind.check())

    def test_a_single_dropped_frame_does_not_restart_anything(self):
        blind = Blindness(resume_reset_s=1.0)
        blind.note_no_frame()
        self.clock.advance(0.1)
        self.assertFalse(blind.note_frame())


class CircuitBreakerTests(ClockedTest):
    def test_trips_after_too_many_locks(self):
        breaker = CircuitBreaker(max_locks=3, window_s=120.0, pause_s=300.0)
        for _ in range(2):
            breaker.note_lock()
            self.clock.advance(5.0)
        self.assertTrue(breaker.check())

        breaker.note_lock()
        self.assertFalse(breaker.check())
        self.assertEqual(breaker.trips, 1)

    def test_recovers_after_the_pause(self):
        breaker = CircuitBreaker(max_locks=2, window_s=120.0, pause_s=300.0)
        breaker.note_lock()
        breaker.note_lock()
        self.assertFalse(breaker.check())

        self.clock.advance(301.0)
        self.assertTrue(breaker.check())

    def test_locks_spread_out_never_trip_it(self):
        breaker = CircuitBreaker(max_locks=3, window_s=120.0, pause_s=300.0)
        for _ in range(10):
            breaker.note_lock()
            self.clock.advance(121.0)  # each one falls outside the window
            self.assertTrue(breaker.check())
        self.assertEqual(breaker.trips, 0)

    def test_zero_disables_the_breaker(self):
        breaker = CircuitBreaker(max_locks=0)
        for _ in range(50):
            breaker.note_lock()
        self.assertTrue(breaker.check())


class PauseSwitchTests(ClockedTest):
    def test_file_presence_blocks_locking(self):
        with tempfile.TemporaryDirectory() as tmp:
            switch = PauseSwitch(Path(tmp) / "PAUSE", poll_s=0.0)
            self.assertTrue(switch.check())

            switch.engage()
            self.assertFalse(switch.check())
            self.assertTrue(switch.path.exists())

            switch.release()
            self.assertTrue(switch.check())
            self.assertFalse(switch.path.exists())

    def test_an_externally_created_file_is_noticed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PAUSE"
            switch = PauseSwitch(path, poll_s=0.0)
            path.write_text("stop", encoding="utf-8")
            self.assertFalse(switch.check())


class SupervisorTests(ClockedTest):
    def _supervisor(self, tmp: str, **kwargs) -> SafetySupervisor:
        options = {
            "pause_file": Path(tmp) / "PAUSE",
            "require_recognition": True,
            "startup_grace_s": 0.0,
            "max_locks": 3,
            "lock_window_s": 120.0,
            "breaker_pause_s": 300.0,
        }
        options.update(kwargs)
        return SafetySupervisor(**options)

    def test_a_fresh_start_never_locks(self):
        """The headline guarantee for autostart: boot cannot lock you out."""
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(tmp, startup_grace_s=20.0)
            for _ in range(100):
                self.assertFalse(supervisor.may_lock())
                self.clock.advance(1.0)

    def test_locking_allowed_as_soon_as_you_are_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(tmp, startup_grace_s=20.0)
            self.assertFalse(supervisor.may_lock())

            supervisor.note_recognised()
            self.assertTrue(supervisor.may_lock(), "no grace left to wait out once seen")

    def test_a_lock_loop_is_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(tmp)
            supervisor.note_recognised()
            self.assertTrue(supervisor.may_lock())

            for _ in range(3):
                supervisor.note_lock()
                self.clock.advance(1.0)

            self.assertFalse(supervisor.may_lock())
            self.assertIn("circuit breaker", supervisor.may_lock().reason)

    def test_blindness_outranks_being_armed(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(tmp)
            supervisor.note_recognised()
            self.assertTrue(supervisor.may_lock())

            supervisor.blindness.note_no_frame()
            self.assertFalse(supervisor.may_lock())

    def test_pause_outranks_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(tmp)
            supervisor.note_recognised()
            supervisor.pause.poll_s = 0.0
            self.assertTrue(supervisor.may_lock())

            supervisor.pause.engage()
            self.assertFalse(supervisor.may_lock())
            self.assertIn("paused", supervisor.may_lock().reason)


class AutostartTests(unittest.TestCase):
    def test_launch_command_is_windowless_and_previewless(self):
        from autolock import autostart

        command = autostart.launch_command()
        self.assertIn("run", command)
        self.assertIn("--no-preview", command)

    def test_extra_arguments_are_carried_through(self):
        from autolock import autostart

        command = autostart.launch_command(["--body-hold=60"])
        self.assertIn("--body-hold=60", command)

    def test_entry_path_is_defined_for_this_platform(self):
        from autolock import autostart

        self.assertIsNotNone(autostart.entry_path())


class LockerTests(unittest.TestCase):
    def test_a_locker_exists_for_this_platform(self):
        from autolock.lock import get_locker

        locker = get_locker()
        self.assertTrue(hasattr(locker, "lock"))
        self.assertIn(locker.is_locked(), (True, False, None))
        self.assertTrue(locker.describe())


if __name__ == "__main__":
    unittest.main(verbosity=2)

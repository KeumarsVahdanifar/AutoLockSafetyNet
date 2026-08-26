"""Safety net for the safety net.

A camera-driven auto-locker has one catastrophic failure mode: locking you out
of your own machine in a loop. You log in, it locks; you log in, it locks. That
is worse than never locking at all, so the monitor refuses to lock unless it is
genuinely confident, and gives up entirely if it starts misbehaving.

Four independent guards:

* `ArmingGate`    — never lock until you have actually been recognised once.
                    A cold boot, a camera that opens on a black frame, or a
                    template that no longer matches all leave it disarmed
                    rather than locking a machine it cannot see.
* `Blindness`     — a camera that is missing, busy or broken produces no
                    evidence either way. Being blind is never grounds to lock,
                    and regaining sight restarts the countdown.
* `CircuitBreaker`— if locks start firing repeatedly in a short window,
                    something is wrong. Stop locking and say so loudly.
* `PauseSwitch`   — a file on disk disables locking, so you can always stop it
                    from another machine, a file manager, or a shell.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    """Whether locking is permitted, and why not when it is refused."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class ArmingGate:
    """Holds the monitor disarmed until it has recognised you at least once.

    This is the single most important guard against a lockout loop. At login
    the camera is often still warming up, the room may be dark, and you may not
    be in front of it yet. Arming on first recognition means the worst case of
    every one of those situations is "it never locks", not "it locks forever".
    """

    def __init__(self, require_recognition: bool = True, startup_grace_s: float = 20.0) -> None:
        self.require_recognition = bool(require_recognition)
        self.startup_grace_s = float(startup_grace_s)
        self._started = time.monotonic()
        self._armed = not self.require_recognition
        self._warned = False

    @property
    def armed(self) -> bool:
        return self._armed

    def note_recognised(self) -> None:
        if not self._armed:
            log.info("Recognised you — monitor armed")
            self._armed = True

    def restart_grace(self) -> None:
        """Begin the startup grace again, e.g. after the session is unlocked."""
        self._started = time.monotonic()

    def check(self) -> Verdict:
        now = time.monotonic()
        if not self._armed:
            if not self._warned and (now - self._started) > 60.0:
                log.warning(
                    "Still not armed after 60s — nothing will be locked until you are "
                    "recognised once. Check lighting, camera, and `autolock test`."
                )
                self._warned = True
            return Verdict(False, "not armed: you have not been recognised yet")
        elapsed = now - self._started
        if elapsed < self.startup_grace_s:
            return Verdict(False, f"startup grace ({self.startup_grace_s - elapsed:.0f}s left)")
        return Verdict(True)


class Blindness:
    """Tracks stretches where the camera gave us nothing to look at."""

    def __init__(self, resume_reset_s: float = 1.0) -> None:
        self.resume_reset_s = float(resume_reset_s)
        self._blind_since: float | None = None

    @property
    def blind(self) -> bool:
        return self._blind_since is not None

    @property
    def blind_for(self) -> float:
        return 0.0 if self._blind_since is None else time.monotonic() - self._blind_since

    def note_no_frame(self) -> None:
        if self._blind_since is None:
            self._blind_since = time.monotonic()
            log.warning("No camera frames — locking suspended until sight returns")

    def note_frame(self) -> bool:
        """Record a good frame. True when the caller should restart the countdown.

        A camera that has just come back cannot be allowed to lock instantly on
        a stale deadline: you get the full timeout to be recognised again.
        """
        if self._blind_since is None:
            return False
        duration = time.monotonic() - self._blind_since
        self._blind_since = None
        if duration >= self.resume_reset_s:
            log.info("Camera back after %.1fs — countdown restarted", duration)
            return True
        return False

    def check(self) -> Verdict:
        if self.blind:
            return Verdict(False, f"camera blind for {self.blind_for:.0f}s")
        return Verdict(True)


class CircuitBreaker:
    """Stops locking when locks start firing far too often.

    Repeated locks in a short window mean the monitor is wrong about something —
    a bad template, a camera pointing at the wall — and every extra lock makes
    the machine harder to use. Tripping stops the damage and leaves an
    explanation in the log.
    """

    def __init__(self, max_locks: int = 3, window_s: float = 120.0, pause_s: float = 300.0) -> None:
        self.max_locks = int(max_locks)
        self.window_s = float(window_s)
        self.pause_s = float(pause_s)
        self._locks: deque[float] = deque()
        self._tripped_until = 0.0
        self.trips = 0

    @property
    def tripped(self) -> bool:
        return time.monotonic() < self._tripped_until

    def note_lock(self) -> None:
        now = time.monotonic()
        self._locks.append(now)
        while self._locks and now - self._locks[0] > self.window_s:
            self._locks.popleft()

        if self.max_locks > 0 and len(self._locks) >= self.max_locks:
            self._tripped_until = now + self.pause_s
            self.trips += 1
            self._locks.clear()
            log.error(
                "Circuit breaker tripped: %d locks within %.0fs. Locking is paused for "
                "%.0fs. Re-enrol (`autolock enroll --name <you> --append`) or check the "
                "camera — run `autolock test` to see what it is scoring.",
                self.max_locks,
                self.window_s,
                self.pause_s,
            )

    def check(self) -> Verdict:
        if self.tripped:
            remaining = self._tripped_until - time.monotonic()
            return Verdict(False, f"circuit breaker tripped ({remaining:.0f}s left)")
        return Verdict(True)

    def reset(self) -> None:
        self._locks.clear()
        self._tripped_until = 0.0


class PauseSwitch:
    """An escape hatch that needs no keyboard: a file on disk.

    Create the file and locking stops; delete it and it resumes. Useful when
    the preview window is not up, over a remote shell, or in the seconds after
    a lock when you would rather not be locked again.
    """

    def __init__(self, path: Path, poll_s: float = 1.0) -> None:
        self.path = Path(path)
        self.poll_s = float(poll_s)
        self._checked_at = 0.0
        self._present = False
        self._logged = False

    def active(self) -> bool:
        now = time.monotonic()
        if now - self._checked_at >= self.poll_s:
            self._checked_at = now
            try:
                present = self.path.exists()
            except OSError:
                present = False
            if present != self._present:
                log.warning(
                    "Pause file %s %s — locking %s",
                    self.path.name,
                    "created" if present else "removed",
                    "disabled" if present else "re-enabled",
                )
            self._present = present
        return self._present

    def check(self) -> Verdict:
        if self.active():
            return Verdict(False, f"paused by {self.path.name}")
        return Verdict(True)

    def engage(self) -> None:
        self.path.write_text(
            "Delete this file to re-enable AutoLock Safety Net.\n", encoding="utf-8"
        )
        self._checked_at = 0.0

    def release(self) -> None:
        self.path.unlink(missing_ok=True)
        self._checked_at = 0.0


class SafetySupervisor:
    """All four guards behind one question: may we lock right now?"""

    def __init__(
        self,
        pause_file: Path,
        require_recognition: bool = True,
        startup_grace_s: float = 20.0,
        max_locks: int = 3,
        lock_window_s: float = 120.0,
        breaker_pause_s: float = 300.0,
    ) -> None:
        self.arming = ArmingGate(require_recognition, startup_grace_s)
        self.blindness = Blindness()
        self.breaker = CircuitBreaker(max_locks, lock_window_s, breaker_pause_s)
        self.pause = PauseSwitch(pause_file)
        self._last_refusal = ""

    def may_lock(self) -> Verdict:
        for guard in (self.pause, self.breaker, self.blindness, self.arming):
            verdict = guard.check()
            if not verdict:
                if verdict.reason != self._last_refusal:
                    log.info("Lock withheld — %s", verdict.reason)
                    self._last_refusal = verdict.reason
                return verdict
        self._last_refusal = ""
        return Verdict(True)

    def note_lock(self) -> None:
        self.breaker.note_lock()

    def note_recognised(self) -> None:
        self.arming.note_recognised()

    def status(self) -> str:
        bits = [
            "armed" if self.arming.armed else "DISARMED",
            "blind" if self.blindness.blind else "seeing",
        ]
        if self.breaker.tripped:
            bits.append("BREAKER TRIPPED")
        if self.pause.active():
            bits.append("PAUSED")
        return ", ".join(bits)

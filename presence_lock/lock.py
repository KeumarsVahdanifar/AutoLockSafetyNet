"""Workstation locking and lock-state detection (Windows)."""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
_DESKTOP_SWITCHDESKTOP = 0x0100


def lock_workstation() -> bool:
    """Lock the session. Returns True when the call succeeded."""
    if not IS_WINDOWS:
        log.error("Locking is only implemented for Windows")
        return False
    try:
        return bool(ctypes.windll.user32.LockWorkStation())
    except Exception as exc:
        log.error("LockWorkStation failed: %s", exc)
        return False


def is_session_locked() -> bool:
    """True when the secure desktop is up (lock screen / UAC prompt).

    `OpenInputDesktop` cannot open the input desktop while the session is
    locked, which makes it a cheap, dependency-free lock probe. Anything
    unexpected is reported as "not locked" so the monitor keeps working.
    """
    if not IS_WINDOWS:
        return False
    try:
        user32 = ctypes.windll.user32
        handle = user32.OpenInputDesktop(0, False, _DESKTOP_SWITCHDESKTOP)
        if not handle:
            return True
        user32.CloseDesktop(handle)
        return False
    except Exception:
        return False

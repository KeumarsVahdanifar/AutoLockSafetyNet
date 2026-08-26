"""Locking the session, on Windows, macOS and Linux.

Each platform gets a small strategy object that knows how to lock and, where
possible, how to tell whether the session is already locked. `is_locked` may
return None, meaning "cannot tell" — callers must treat that as *not* a licence
to lock repeatedly, which is what the cooldown in the monitor is for.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

_DESKTOP_SWITCHDESKTOP = 0x0100


def _run(command: list[str], timeout: float = 5.0) -> bool:
    """Run a command, returning True on a zero exit status."""
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("command %s failed: %s", command[0], exc)
        return False


def _capture(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return completed.stdout if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------------
class Locker(ABC):
    """How to lock this platform's session."""

    name = "unsupported"

    @abstractmethod
    def lock(self) -> bool:
        """Lock the session. True if the call succeeded."""

    def is_locked(self) -> bool | None:
        """True/False if known, None if this platform cannot tell."""
        return None

    def describe(self) -> str:
        return self.name

    @property
    def available(self) -> bool:
        return True


class WindowsLocker(Locker):
    name = "windows"

    def lock(self) -> bool:
        try:
            return bool(ctypes.windll.user32.LockWorkStation())
        except Exception as exc:
            log.error("LockWorkStation failed: %s", exc)
            return False

    def is_locked(self) -> bool | None:
        """`OpenInputDesktop` cannot open the input desktop while locked.

        That makes it a cheap, dependency-free probe for the secure desktop
        (lock screen, UAC prompt).
        """
        try:
            user32 = ctypes.windll.user32
            handle = user32.OpenInputDesktop(0, False, _DESKTOP_SWITCHDESKTOP)
            if not handle:
                return True
            user32.CloseDesktop(handle)
            return False
        except Exception:
            return None


class MacLocker(Locker):
    name = "macos"

    _CGSESSION = (
        "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession"
    )

    def lock(self) -> bool:
        # Fast user switching back to the login window: a genuine lock that
        # works without any extra dependency.
        if os.path.exists(self._CGSESSION) and _run([self._CGSESSION, "-suspend"]):
            return True
        # Falls back to sleeping the display, which locks whenever "require
        # password after sleep" is set (System Settings > Lock Screen).
        if _run(["pmset", "displaysleepnow"]):
            log.debug("locked via pmset; requires 'require password' to be enabled")
            return True
        log.error("no working lock method found on macOS")
        return False

    def is_locked(self) -> bool | None:
        try:
            import Quartz  # type: ignore[import-not-found]  # pyobjc, optional
        except Exception:
            return None
        try:
            session = Quartz.CGSessionCopyCurrentDictionary()
            if not session:
                return None
            return bool(session.get("CGSSessionScreenIsLocked", False))
        except Exception:
            return None

    def describe(self) -> str:
        method = "CGSession" if os.path.exists(self._CGSESSION) else "pmset displaysleepnow"
        return f"macos ({method})"


class LinuxLocker(Locker):
    name = "linux"

    # Ordered by how reliably each one actually locks rather than just blanks.
    _CANDIDATES: tuple[tuple[str, list[str]], ...] = (
        ("loginctl", ["loginctl", "lock-session"]),
        ("xdg-screensaver", ["xdg-screensaver", "lock"]),
        ("gnome-screensaver-command", ["gnome-screensaver-command", "-l"]),
        ("dbus-send", [
            "dbus-send", "--session", "--dest=org.freedesktop.ScreenSaver",
            "--type=method_call", "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.Lock",
        ]),
        ("xscreensaver-command", ["xscreensaver-command", "-lock"]),
        ("swaylock", ["swaylock", "-f"]),
    )

    def __init__(self) -> None:
        self._methods = [
            (binary, command)
            for binary, command in self._CANDIDATES
            if shutil.which(binary) is not None
        ]
        self._preferred: list[str] | None = None

    @property
    def available(self) -> bool:
        return bool(self._methods)

    def lock(self) -> bool:
        # Stick with whatever worked last time instead of re-probing the list.
        if self._preferred and _run(self._preferred):
            return True
        for binary, command in self._methods:
            if _run(command):
                self._preferred = command
                log.debug("locked via %s", binary)
                return True
        log.error(
            "no working screen locker found. Install one of: %s",
            ", ".join(binary for binary, _ in self._CANDIDATES),
        )
        return False

    def is_locked(self) -> bool | None:
        session = os.environ.get("XDG_SESSION_ID")
        if not session or shutil.which("loginctl") is None:
            return None
        output = _capture(["loginctl", "show-session", session, "-p", "LockedHint"])
        if not output:
            return None
        return "yes" in output.strip().lower()

    def describe(self) -> str:
        if not self._methods:
            return "linux (no locker found)"
        return f"linux ({', '.join(binary for binary, _ in self._methods)})"


class UnsupportedLocker(Locker):
    name = "unsupported"

    @property
    def available(self) -> bool:
        return False

    def lock(self) -> bool:
        log.error("Locking is not implemented for platform %r", sys.platform)
        return False


# ----------------------------------------------------------------------
def build_locker() -> Locker:
    if IS_WINDOWS:
        return WindowsLocker()
    if IS_MACOS:
        return MacLocker()
    if IS_LINUX:
        return LinuxLocker()
    return UnsupportedLocker()


_LOCKER: Locker | None = None


def get_locker() -> Locker:
    global _LOCKER
    if _LOCKER is None:
        _LOCKER = build_locker()
    return _LOCKER


def lock_workstation() -> bool:
    return get_locker().lock()


def is_session_locked() -> bool:
    """Best-effort probe; False when the platform cannot tell."""
    return get_locker().is_locked() is True

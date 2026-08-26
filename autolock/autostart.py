"""Install and remove a "start at login" entry, on all three platforms.

Autostart is where an auto-locker is most dangerous: a monitor that starts
before you are seated, before the camera is awake, or with a template that no
longer matches will lock the machine you just logged into — over and over.

Three things prevent that, and all are applied here by default:

* the entry opens the control panel with monitoring already running, so the
  thing holding your lock key is visible and its pause button is one click
  away — an auto-locker you cannot see is one you cannot stop;
* the monitor's own arming gate means it will not lock until it has recognised
  you once (see `safety.py`);
* `uninstall` is a single command, and the pause file works even when you
  cannot reach a terminal.

`--headless` is available for anyone who would rather have the old invisible
background process.

`autolock autostart --install` prints exactly what it wrote and how to undo it.
"""

from __future__ import annotations

import logging
import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT
from .lock import IS_LINUX, IS_MACOS, IS_WINDOWS

log = logging.getLogger(__name__)

APP_ID = "AutoLockSafetyNet"
APP_NAME = "AutoLock Safety Net"


def launch_command(
    extra_args: list[str] | None = None, headless: bool = False
) -> list[str]:
    """The command an autostart entry should run.

    By default this opens the control panel with monitoring already running:
    an auto-locker you cannot see is an auto-locker you cannot stop, and the
    window is the fastest route to the pause button. `--headless` gives the
    invisible background process instead, for anyone who prefers it.

    A frozen build runs itself; a source checkout runs the current interpreter
    against this project. `pythonw.exe` on Windows suppresses the console
    window — the Tk window still appears.
    """
    args = (
        ["run", "--no-preview"] if headless else ["gui", "--start"]
    ) + list(extra_args or [])

    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return [sys.executable, *args]

    interpreter = sys.executable
    if IS_WINDOWS:
        windowed = Path(interpreter).with_name("pythonw.exe")
        if windowed.exists():
            interpreter = str(windowed)
    return [interpreter, "-m", "autolock", *args]


def _quote(parts: list[str]) -> str:
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


@dataclass
class InstallResult:
    ok: bool
    location: str
    message: str
    undo: str


# ----------------------------------------------------------------------
# Windows — Startup folder shortcut via a .cmd stub (no COM dependency)
# ----------------------------------------------------------------------
def _windows_startup_dir() -> Path:
    return (
        Path(os.environ["APPDATA"])
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def _windows_entry() -> Path:
    return _windows_startup_dir() / f"{APP_ID}.cmd"


def _install_windows(extra_args: list[str] | None, headless: bool) -> InstallResult:
    entry = _windows_entry()
    entry.parent.mkdir(parents=True, exist_ok=True)
    command = launch_command(extra_args, headless)
    # `start ""` detaches so the login sequence is never held up by us.
    script = (
        "@echo off\r\n"
        f"rem {APP_NAME} — delete this file to stop it starting at login.\r\n"
        f'cd /d "{PROJECT_ROOT}"\r\n'
        f'start "" {_quote(command)}\r\n'
    )
    entry.write_text(script, encoding="utf-8")
    return InstallResult(
        ok=True,
        location=str(entry),
        message=f"Installed a Startup entry for {APP_NAME}.",
        undo=f'del "{entry}"   (or: autolock autostart --uninstall)',
    )


def _uninstall_windows() -> InstallResult:
    entry = _windows_entry()
    existed = entry.exists()
    entry.unlink(missing_ok=True)
    return InstallResult(
        ok=True,
        location=str(entry),
        message="Removed the Startup entry." if existed else "No Startup entry was installed.",
        undo="",
    )


# ----------------------------------------------------------------------
# macOS — a per-user LaunchAgent
# ----------------------------------------------------------------------
def _macos_entry() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"com.{APP_ID.lower()}.plist"


def _install_macos(extra_args: list[str] | None, headless: bool) -> InstallResult:
    entry = _macos_entry()
    entry.parent.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": f"com.{APP_ID.lower()}",
        "ProgramArguments": launch_command(extra_args, headless),
        "WorkingDirectory": str(PROJECT_ROOT),
        "RunAtLoad": True,
        "KeepAlive": False,  # never respawn a monitor that is failing
        "StandardOutPath": str(PROJECT_ROOT / "logs" / "autostart.out.log"),
        "StandardErrorPath": str(PROJECT_ROOT / "logs" / "autostart.err.log"),
    }
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with entry.open("wb") as handle:
        plistlib.dump(plist, handle)

    subprocess.run(["launchctl", "unload", str(entry)], capture_output=True, check=False)
    loaded = subprocess.run(["launchctl", "load", str(entry)], capture_output=True, check=False)
    note = ""
    if loaded.returncode != 0:
        note = " (run `launchctl load` yourself, or just log out and back in)"
    return InstallResult(
        ok=True,
        location=str(entry),
        message=(
            f"Installed a LaunchAgent for {APP_NAME}{note}.\n"
            "  macOS will ask for Camera permission the first time it runs — grant it to\n"
            "  the terminal or the app bundle under System Settings > Privacy & Security."
        ),
        undo=f'launchctl unload "{entry}" && rm "{entry}"',
    )


def _uninstall_macos() -> InstallResult:
    entry = _macos_entry()
    existed = entry.exists()
    if existed:
        subprocess.run(["launchctl", "unload", str(entry)], capture_output=True, check=False)
    entry.unlink(missing_ok=True)
    return InstallResult(
        ok=True,
        location=str(entry),
        message="Removed the LaunchAgent." if existed else "No LaunchAgent was installed.",
        undo="",
    )


# ----------------------------------------------------------------------
# Linux — a systemd user unit, or an XDG autostart .desktop as a fallback
# ----------------------------------------------------------------------
def _systemd_entry() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "systemd" / "user" / f"{APP_ID.lower()}.service"


def _desktop_entry() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "autostart" / f"{APP_ID.lower()}.desktop"


def _install_linux(extra_args: list[str] | None, headless: bool) -> InstallResult:
    command = launch_command(extra_args, headless)

    if shutil.which("systemctl"):
        entry = _systemd_entry()
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(
            f"""[Unit]
Description={APP_NAME}
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
WorkingDirectory={PROJECT_ROOT}
ExecStart={_quote(command)}
# Deliberately not Restart=always: a monitor that keeps dying should stay dead
# rather than being resurrected into a lock loop.
Restart=on-failure
RestartSec=30
StartLimitBurst=3
StartLimitIntervalSec=300

[Install]
WantedBy=graphical-session.target
""",
            encoding="utf-8",
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        subprocess.run(
            ["systemctl", "--user", "enable", entry.name], capture_output=True, check=False
        )
        return InstallResult(
            ok=True,
            location=str(entry),
            message=(
                f"Installed a systemd user unit for {APP_NAME}.\n"
                f"  Start it now with: systemctl --user start {entry.name}\n"
                f"  Follow the log with: journalctl --user -u {entry.name} -f"
            ),
            undo=f'systemctl --user disable --now {entry.name} && rm "{entry}"',
        )

    entry = _desktop_entry()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        f"""[Desktop Entry]
Type=Application
Name={APP_NAME}
Exec={_quote(command)}
Path={PROJECT_ROOT}
X-GNOME-Autostart-enabled=true
Terminal=false
""",
        encoding="utf-8",
    )
    return InstallResult(
        ok=True,
        location=str(entry),
        message=f"Installed an XDG autostart entry for {APP_NAME}.",
        undo=f'rm "{entry}"',
    )


def _uninstall_linux() -> InstallResult:
    removed = []
    unit = _systemd_entry()
    if unit.exists():
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", unit.name],
            capture_output=True,
            check=False,
        )
        unit.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        removed.append(str(unit))

    desktop = _desktop_entry()
    if desktop.exists():
        desktop.unlink(missing_ok=True)
        removed.append(str(desktop))

    return InstallResult(
        ok=True,
        location=", ".join(removed),
        message=(
            ("Removed: " + ", ".join(removed))
            if removed
            else "No autostart entry was installed."
        ),
        undo="",
    )


# ----------------------------------------------------------------------
def entry_path() -> Path | None:
    if IS_WINDOWS:
        return _windows_entry()
    if IS_MACOS:
        return _macos_entry()
    if IS_LINUX:
        unit = _systemd_entry()
        return unit if unit.exists() or shutil.which("systemctl") else _desktop_entry()
    return None


def is_installed() -> bool:
    if IS_LINUX:
        return _systemd_entry().exists() or _desktop_entry().exists()
    entry = entry_path()
    return bool(entry and entry.exists())


def install(extra_args: list[str] | None = None, headless: bool = False) -> InstallResult:
    if IS_WINDOWS:
        return _install_windows(extra_args, headless)
    if IS_MACOS:
        return _install_macos(extra_args, headless)
    if IS_LINUX:
        return _install_linux(extra_args, headless)
    return InstallResult(False, "", f"Autostart is not supported on {sys.platform}.", "")


def uninstall() -> InstallResult:
    if IS_WINDOWS:
        return _uninstall_windows()
    if IS_MACOS:
        return _uninstall_macos()
    if IS_LINUX:
        return _uninstall_linux()
    return InstallResult(False, "", f"Autostart is not supported on {sys.platform}.", "")

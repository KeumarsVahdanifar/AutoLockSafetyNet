"""Desktop control panel (Tkinter, so it ships with Python everywhere).

The monitor runs on a worker thread and hands frames and state to the UI
through a queue; Tkinter is touched only from the main thread. Everything the
CLI can do is reachable here: enrol, arm, tune, autostart, and the pause
switch — which is the control that matters most when something goes wrong.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable

import cv2
import numpy as np

from . import __version__, autostart, models
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, ensure_dirs
from .identity import list_identities
from .lock import get_locker
from .presence import EV_FACE, EV_NONE, FrameResult
from .ui import draw_overlay

log = logging.getLogger(__name__)

PREVIEW_W, PREVIEW_H = 560, 420

# A calm dark palette; the accent colours match the overlay's box colours so
# the preview and the chrome agree with each other.
BG = "#15171c"
PANEL = "#1e2128"
PANEL_2 = "#252932"
FG = "#e6e8ee"
MUTED = "#8b93a7"
GREEN = "#64dc78"
AMBER = "#fabe3c"
RED = "#f04646"
BLUE = "#5aa9f0"


@dataclass
class Update:
    """One frame's worth of state, passed from the worker to the UI."""

    frame: np.ndarray | None = None
    result: FrameResult | None = None
    fps: float = 0.0
    safety: str = ""
    status: str = ""
    error: str = ""
    stopped: bool = False


class MonitorThread(threading.Thread):
    """Runs the presence pipeline off the UI thread."""

    def __init__(self, cfg: Config, arm: bool, updates: queue.Queue[Update]) -> None:
        super().__init__(daemon=True, name="autolock-monitor")
        self.cfg = cfg
        self.arm = arm
        self.updates = updates
        self._stop = threading.Event()
        self.app: Any = None

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        from .app import PresenceLockApp

        try:
            cfg = Config.from_dict(self.cfg.to_dict())
            cfg.preview = False  # the GUI draws the preview itself
            app = PresenceLockApp(cfg, arm=self.arm)
            self.app = app
        except Exception as exc:
            self.updates.put(Update(error=str(exc), stopped=True))
            return

        self.updates.put(Update(status="running"))
        budget = 1.0 / max(1.0, cfg.target_fps)
        fps = 0.0
        session_was_locked = False

        try:
            while not self._stop.is_set():
                started = time.monotonic()

                if app._session_locked():
                    if not session_was_locked:
                        session_was_locked = True
                        if cfg.release_camera_when_locked:
                            app.camera.release()
                        self.updates.put(
                            Update(status="session locked", safety=app.safety.status())
                        )
                    time.sleep(1.0)
                    continue

                if session_was_locked:
                    session_was_locked = False
                    app.engine.reset(present=True)
                    app.safety.arming.restart_grace()

                frame = app.camera.read()
                if frame is None:
                    app.safety.blindness.note_no_frame()
                    self.updates.put(
                        Update(status="no camera", safety=app.safety.status())
                    )
                    time.sleep(0.3)
                    continue

                if app.safety.blindness.note_frame():
                    app.engine.reset(present=True)

                result = app.engine.process(frame)
                if result.evidence == EV_FACE:
                    app.safety.note_recognised()

                if result.should_lock:
                    verdict = app.safety.may_lock()
                    if verdict:
                        app._do_lock(result.lock_reason)
                    else:
                        result.should_lock = False
                        result.withheld_reason = verdict.reason

                view = draw_overlay(
                    frame,
                    result,
                    timeout_s=cfg.absence_timeout_s,
                    backend_name=app.backend.name,
                    identities=app.gallery.names,
                    fps=fps,
                    dry_run=cfg.dry_run or not self.arm,
                    mirror=cfg.mirror_preview,
                    safety=app.safety.status(),
                )
                self.updates.put(
                    Update(frame=view, result=result, fps=fps, safety=app.safety.status())
                )

                elapsed = time.monotonic() - started
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-3))
                if elapsed < budget:
                    time.sleep(budget - elapsed)
        except Exception as exc:  # a crash must never look like "still watching"
            log.exception("Monitor thread failed")
            self.updates.put(Update(error=str(exc), stopped=True))
        finally:
            try:
                app.close()
            except Exception:
                pass
            self.updates.put(Update(status="stopped", stopped=True))


class AutoLockGUI:
    def __init__(self, root: tk.Tk, cfg: Config, config_path=DEFAULT_CONFIG_PATH) -> None:
        self.root = root
        self.cfg = cfg
        self.config_path = config_path
        self.updates: queue.Queue[Update] = queue.Queue(maxsize=4)
        self.monitor: MonitorThread | None = None
        self._photo: Any = None
        self._blank = self._placeholder()

        root.title(f"AutoLock Safety Net {__version__}")
        root.configure(bg=BG)
        root.minsize(900, 620)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._init_style()
        self._build()
        self._tick()

    # ------------------------------------------------------------------
    # Chrome
    # ------------------------------------------------------------------
    def _init_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, fieldbackground=PANEL_2)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 15, "bold"))
        style.configure(
            "Big.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 26, "bold")
        )
        style.configure(
            "TButton", background=PANEL_2, foreground=FG, borderwidth=0, padding=(12, 7)
        )
        style.map("TButton", background=[("active", "#333844"), ("disabled", "#22252c")])
        style.configure("Accent.TButton", background=BLUE, foreground="#0d1117")
        style.map("Accent.TButton", background=[("active", "#7dbcf5")])
        style.configure("Danger.TButton", background=RED, foreground="#ffffff")
        style.map("Danger.TButton", background=[("active", "#f66")])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(16, 8))
        style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL_2)],
            foreground=[("selected", FG)],
        )
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.configure("Horizontal.TScale", background=PANEL)

    def _placeholder(self) -> np.ndarray:
        canvas = np.full((PREVIEW_H, PREVIEW_W, 3), 26, dtype=np.uint8)
        cv2.putText(
            canvas, "camera preview", (PREVIEW_W // 2 - 90, PREVIEW_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1, cv2.LINE_AA,
        )
        return canvas

    def _build(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(14, 8))
        ttk.Label(header, text="AutoLock Safety Net", style="Title.TLabel").pack(side="left")
        self.lock_method = ttk.Label(
            header, text=get_locker().describe(), style="TLabel", foreground=MUTED
        )
        self.lock_method.pack(side="right")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # --- left: preview + status ---
        left = ttk.Frame(body, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.preview = tk.Label(left, bg=PANEL, bd=0)
        self.preview.pack(padx=12, pady=12)

        status = ttk.Frame(left, style="Panel.TFrame")
        status.pack(fill="x", padx=12, pady=(0, 12))

        self.countdown = ttk.Label(status, text="--", style="Big.TLabel")
        self.countdown.grid(row=0, column=0, sticky="w")
        self.state_label = ttk.Label(status, text="stopped", style="Panel.TLabel")
        self.state_label.grid(row=0, column=1, sticky="w", padx=(14, 0))
        self.safety_label = ttk.Label(status, text="", style="Muted.TLabel")
        self.safety_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.pack(fill="x", padx=12, pady=(0, 14))
        self.start_btn = ttk.Button(
            controls, text="Start monitoring", style="Accent.TButton", command=self.on_start
        )
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            controls, text="Stop", command=self.on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=6)
        self.test_btn = ttk.Button(controls, text="Test (never locks)", command=self.on_test)
        self.test_btn.pack(side="left")
        self.pause_btn = ttk.Button(controls, text="Pause locking", command=self.on_pause)
        self.pause_btn.pack(side="right")

        # --- right: tabs ---
        right = ttk.Notebook(body, width=330)
        right.pack(side="right", fill="both")
        right.add(self._tab_setup(right), text="Setup")
        right.add(self._tab_tuning(right), text="Tuning")
        right.add(self._tab_startup(right), text="Startup")

    # ------------------------------------------------------------------
    def _tab_setup(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame", padding=14)

        ttk.Label(tab, text="Enrolled identities", style="Panel.TLabel").pack(anchor="w")
        self.identities_label = ttk.Label(tab, text="none", style="Muted.TLabel", wraplength=290)
        self.identities_label.pack(anchor="w", pady=(2, 10))

        ttk.Label(tab, text="Your name", style="Panel.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar(value=(list_identities() or [""])[0])
        tk.Entry(
            tab, textvariable=self.name_var, bg=PANEL_2, fg=FG, insertbackground=FG,
            relief="flat", highlightthickness=0,
        ).pack(fill="x", pady=(2, 8), ipady=5)

        ttk.Button(tab, text="Enrol my face", style="Accent.TButton", command=self.on_enroll).pack(
            fill="x"
        )
        ttk.Button(tab, text="Add more poses", command=self.on_enroll_append).pack(
            fill="x", pady=(6, 0)
        )
        ttk.Label(
            tab,
            text=(
                "Enrolment opens a guided window and walks you through seven head "
                "poses. Add more poses after a haircut, new glasses, or a change in "
                "your desk lighting."
            ),
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).pack(anchor="w", pady=(10, 12))

        ttk.Button(tab, text="Download models", command=self.on_models).pack(fill="x")
        ttk.Button(tab, text="Run diagnostics", command=self.on_doctor).pack(fill="x", pady=(6, 0))

        self.refresh_identities()
        return tab

    def _tab_tuning(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame", padding=14)

        self.timeout_var = tk.DoubleVar(value=self.cfg.absence_timeout_s)
        self._slider(
            tab, "Lock after (seconds unrecognised)", self.timeout_var, 1, 30,
            "The countdown starts the moment you are not recognised.",
        )

        self.body_var = tk.DoubleVar(value=self.cfg.body_hold_s)
        self._slider(
            tab, "Head-down allowance (seconds)", self.body_var, 0, 180,
            "0 = strict. Above 0, your body being visible buys this much time "
            "before locking — a ceiling, not a reset.",
        )

        self.threshold_var = tk.DoubleVar(value=self.cfg.match_threshold or 0.0)
        self._slider(
            tab, "Match threshold (0 = backend default)", self.threshold_var, 0.0, 0.7,
            "Higher is stricter about who counts as you.", resolution=0.01,
        )

        self.unknown_var = tk.BooleanVar(value=self.cfg.lock_on_unknown)
        ttk.Checkbutton(
            tab, text="Lock when a stranger is at the desk", variable=self.unknown_var
        ).pack(anchor="w", pady=(8, 0))

        self.dry_var = tk.BooleanVar(value=self.cfg.dry_run)
        ttk.Checkbutton(
            tab, text="Dry run (log instead of locking)", variable=self.dry_var
        ).pack(anchor="w")

        ttk.Button(tab, text="Save settings", style="Accent.TButton", command=self.on_save).pack(
            fill="x", pady=(14, 0)
        )
        ttk.Label(
            tab, text=f"Saved to {self.config_path.name}", style="Muted.TLabel"
        ).pack(anchor="w", pady=(4, 0))
        return tab

    def _slider(self, parent, label, var, lo, hi, hint, resolution=1.0) -> None:
        ttk.Label(parent, text=label, style="Panel.TLabel").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x")
        value = ttk.Label(row, text=f"{var.get():g}", style="Panel.TLabel", width=5)
        value.pack(side="right")
        scale = tk.Scale(
            row, from_=lo, to=hi, orient="horizontal", variable=var, resolution=resolution,
            showvalue=False, bg=PANEL, fg=FG, troughcolor=PANEL_2, highlightthickness=0,
            bd=0, sliderrelief="flat", activebackground=BLUE,
            command=lambda _v: value.configure(text=f"{var.get():g}"),
        )
        scale.pack(side="left", fill="x", expand=True)
        ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=290, justify="left").pack(
            anchor="w"
        )

    def _tab_startup(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        ttk.Label(tab, text="Start at login", style="Panel.TLabel").pack(anchor="w")
        self.autostart_label = ttk.Label(tab, text="", style="Muted.TLabel", wraplength=290)
        self.autostart_label.pack(anchor="w", pady=(2, 10))

        ttk.Button(
            tab, text="Enable at login", style="Accent.TButton", command=self.on_autostart_install
        ).pack(fill="x")
        ttk.Button(tab, text="Disable at login", command=self.on_autostart_remove).pack(
            fill="x", pady=(6, 0)
        )

        ttk.Label(
            tab,
            text=(
                "Safe by design: at login the monitor stays disarmed until it has "
                "recognised you once, waits out a startup grace period, never locks "
                "while the camera is blind, and stops itself if locks start firing "
                "repeatedly.\n\n"
                "Emergency stop: create a file named PAUSE in the project folder, or "
                "press Pause locking above. Locking stops until it is deleted."
            ),
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

        self.refresh_autostart()
        return tab

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def collect_config(self) -> Config:
        cfg = Config.from_dict(self.cfg.to_dict())
        cfg.absence_timeout_s = float(self.timeout_var.get())
        cfg.body_hold_s = float(self.body_var.get())
        cfg.match_threshold = float(self.threshold_var.get())
        cfg.lock_on_unknown = bool(self.unknown_var.get())
        cfg.dry_run = bool(self.dry_var.get())
        cfg.identity = self.name_var.get().strip()
        return cfg.normalise()

    def on_save(self) -> None:
        cfg = self.collect_config()
        cfg.save(self.config_path)
        self.cfg = cfg
        messagebox.showinfo("Saved", f"Settings written to {self.config_path}")

    def _start(self, arm: bool) -> None:
        if self.monitor and self.monitor.is_alive():
            return
        if not list_identities():
            messagebox.showwarning(
                "Nobody enrolled",
                "Enrol your face first — the monitor has nobody to recognise.",
            )
            return
        self.cfg = self.collect_config()
        self.monitor = MonitorThread(self.cfg, arm, self.updates)
        self.monitor.start()
        self.start_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.state_label.configure(text="starting...", foreground=MUTED)

    def on_start(self) -> None:
        self._start(arm=True)

    def on_test(self) -> None:
        self._start(arm=False)

    def on_stop(self) -> None:
        if self.monitor:
            self.monitor.stop()
        self.stop_btn.configure(state="disabled")

    def on_pause(self) -> None:
        from .safety import PauseSwitch

        switch = PauseSwitch(PROJECT_ROOT / self.cfg.pause_file)
        if switch.active():
            switch.release()
            self.pause_btn.configure(text="Pause locking")
        else:
            switch.engage()
            self.pause_btn.configure(text="Resume locking")

    def _run_detached(self, target: Callable[[], None], label: str) -> None:
        """Run a blocking task off the UI thread, reporting failures."""

        def wrapper() -> None:
            try:
                target()
            except Exception as exc:
                # Bind the text now: `exc` is unbound once the except block
                # ends, so a lambda closing over it would raise NameError
                # later on the UI thread and swallow the real error.
                message = str(exc)
                log.exception("%s failed", label)
                self.root.after(0, lambda: messagebox.showerror(label, message))

        threading.Thread(target=wrapper, daemon=True, name=f"autolock-{label}").start()

    def on_enroll(self, append: bool = False) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Name needed", "Type the name to enrol under first.")
            return
        if self.monitor and self.monitor.is_alive():
            messagebox.showwarning("Stop first", "Stop the monitor before enrolling.")
            return

        from .enroll import enroll_interactive

        def task() -> None:
            enroll_interactive(self.collect_config(), name, append=append)
            self.root.after(0, self.refresh_identities)

        self._run_detached(task, "Enrolment")

    def on_enroll_append(self) -> None:
        self.on_enroll(append=True)

    def on_models(self) -> None:
        self._run_detached(lambda: models.ensure_all(), "Models")

    def on_doctor(self) -> None:
        from .cli import main as cli_main

        self._run_detached(lambda: cli_main(["doctor"]), "Diagnostics")
        messagebox.showinfo("Diagnostics", "Results are written to logs/autolock.log")

    def on_autostart_install(self) -> None:
        result = autostart.install()
        messagebox.showinfo("Start at login", f"{result.message}\n\n{result.location}")
        self.refresh_autostart()

    def on_autostart_remove(self) -> None:
        result = autostart.uninstall()
        messagebox.showinfo("Start at login", result.message)
        self.refresh_autostart()

    def refresh_identities(self) -> None:
        names = list_identities()
        self.identities_label.configure(text=", ".join(names) if names else "none yet")

    def refresh_autostart(self) -> None:
        if autostart.is_installed():
            self.autostart_label.configure(text=f"Enabled\n{autostart.entry_path()}")
        else:
            self.autostart_label.configure(text="Not enabled")

    # ------------------------------------------------------------------
    # UI loop
    # ------------------------------------------------------------------
    def _show(self, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk

        resized = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops it
        self.preview.configure(image=self._photo)

    def _tick(self) -> None:
        latest: Update | None = None
        try:
            while True:  # only the newest frame matters
                latest = self.updates.get_nowait()
                if latest.error or latest.stopped:
                    break
        except queue.Empty:
            pass

        if latest is not None:
            self._apply(latest)

        self.root.after(33, self._tick)

    def _apply(self, update: Update) -> None:
        if update.error:
            messagebox.showerror("Monitor stopped", update.error)
        if update.stopped:
            self.start_btn.configure(state="normal")
            self.test_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.state_label.configure(text="stopped", foreground=MUTED)
            self.countdown.configure(text="--")
            self._show(self._blank)
            return

        if update.frame is not None:
            self._show(update.frame)

        if update.safety:
            self.safety_label.configure(text=update.safety)

        result = update.result
        if result is None:
            if update.status:
                self.state_label.configure(text=update.status, foreground=AMBER)
            return

        self.countdown.configure(text=f"{result.seconds_to_lock:.1f}s")
        if result.evidence == EV_FACE:
            self.state_label.configure(text="recognised", foreground=GREEN)
        elif result.withheld_reason:
            self.state_label.configure(text=result.withheld_reason, foreground=AMBER)
        elif result.evidence == EV_NONE:
            self.state_label.configure(text="not recognised", foreground=RED)
        else:
            self.state_label.configure(text=result.evidence_label, foreground=AMBER)

    def on_close(self) -> None:
        if self.monitor and self.monitor.is_alive():
            self.monitor.stop()
            self.monitor.join(timeout=3.0)
        self.root.destroy()


def launch(cfg: Config | None = None, config_path=DEFAULT_CONFIG_PATH) -> int:
    ensure_dirs()
    root = tk.Tk()
    AutoLockGUI(root, cfg or Config.load(config_path), config_path)
    root.mainloop()
    return 0

"""Desktop control panel (Tkinter, so it ships with Python everywhere).

The monitor runs on a worker thread and hands frames and state to the UI
through a queue; Tkinter is touched only from the main thread. Everything the
CLI can do is reachable here: enrol, arm, tune, autostart, and the pause
switch — which is the control that matters most when something goes wrong.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable

import cv2
import numpy as np

from . import __version__, autostart, models
from .config import (
    DEFAULT_CONFIG_PATH,
    FIELD_SPECS,
    PROJECT_ROOT,
    Config,
    coerce_value,
    ensure_dirs,
    grouped_fields,
)
from .identity import list_identities
from .lock import get_locker
from .presence import EV_FACE, EV_NONE, FrameResult
from .ui import draw_overlay

log = logging.getLogger(__name__)

PREVIEW_W, PREVIEW_H = 560, 420

# A calm dark palette. The status accents match the overlay's box colours, so
# the preview and the chrome around it say the same thing in the same colour.
BG = "#101216"  # window ground
PANEL = "#181b21"  # cards
PANEL_2 = "#212630"  # inputs, wells
LINE = "#2b313c"  # hairline separators
FG = "#e8eaf0"
MUTED = "#7f8899"
GREEN = "#5ed17a"
AMBER = "#f5b73d"
RED = "#ef5350"
BLUE = "#5aa9f0"

# One type scale, used everywhere, so nothing is "nearly" the same size as
# something else. Segoe UI on Windows, with sane fallbacks elsewhere.
_FAMILY = {"win32": "Segoe UI", "darwin": "SF Pro Text"}.get(sys.platform, "DejaVu Sans")
FONT_TITLE = (_FAMILY, 16, "bold")
FONT_HERO = (_FAMILY, 34, "bold")
FONT_BODY = (_FAMILY, 10)
FONT_LABEL = (_FAMILY, 10)
FONT_SMALL = (_FAMILY, 9)
FONT_GROUP = (_FAMILY, 8, "bold")

PAD = 16  # the one spacing unit; everything is a multiple of it
SEARCH_PLACEHOLDER = "Search settings"


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
    def __init__(
        self,
        root: tk.Tk,
        cfg: Config,
        config_path=DEFAULT_CONFIG_PATH,
        start_monitoring: bool = False,
    ) -> None:
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

        if start_monitoring:
            # Launched at login: arm straight away, but only after the window
            # is actually on screen, so a failure is visible rather than a
            # message box behind nothing.
            root.after(400, self._autostart_monitoring)

    def _autostart_monitoring(self) -> None:
        if not list_identities():
            self.state_label.configure(
                text="enrol your face to begin", foreground=AMBER
            )
            return
        self.on_start()

    # ------------------------------------------------------------------
    # Chrome
    # ------------------------------------------------------------------
    def _init_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")  # the only built-in theme that honours colours
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=FG, fieldbackground=PANEL_2,
                        font=FONT_BODY, borderwidth=0, focuscolor=BLUE)

        # Surfaces
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Well.TFrame", background=PANEL_2)
        style.configure("Rule.TFrame", background=LINE)

        # Type
        style.configure("TLabel", background=BG, foreground=FG, font=FONT_BODY)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=FONT_LABEL)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONT_SMALL)
        style.configure("HeaderMuted.TLabel", background=BG, foreground=MUTED, font=FONT_SMALL)
        style.configure("Title.TLabel", background=BG, foreground=FG, font=FONT_TITLE)
        style.configure("Hero.TLabel", background=PANEL, foreground=FG, font=FONT_HERO)
        style.configure("Group.TLabel", background=PANEL, foreground=MUTED, font=FONT_GROUP)

        # Buttons: one quiet default, one accent, one danger. Generous hit areas.
        style.configure("TButton", background=PANEL_2, foreground=FG,
                        borderwidth=0, padding=(14, 9), font=FONT_BODY)
        style.map("TButton",
                  background=[("pressed", "#2c333f"), ("active", "#2a303b"),
                              ("disabled", "#1b1f26")],
                  foreground=[("disabled", "#4d5563")])
        style.configure("Accent.TButton", background=BLUE, foreground="#0b1017",
                        font=(_FAMILY, 10, "bold"))
        style.map("Accent.TButton",
                  background=[("pressed", "#4d97dc"), ("active", "#74b7f4"),
                              ("disabled", "#1b1f26")],
                  foreground=[("disabled", "#4d5563")])
        style.configure("Danger.TButton", background=PANEL_2, foreground=RED)
        style.map("Danger.TButton", background=[("active", "#332224")])

        # Tabs sit on the window ground, with the selected one lifted onto the
        # card colour so it reads as the front of a stack. clam draws a raised
        # client border and shifts the selected tab; both are removed here so
        # the tab strip and the panel below it read as one surface.
        try:
            style.layout("TNotebook", [])
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0),
                        bordercolor=BG, lightcolor=BG, darkcolor=BG)
        # clam draws each tab with a bevel; flattening its three border colours
        # is the only way to get a plain tab out of it.
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                        padding=(18, 11), borderwidth=0, font=FONT_BODY,
                        bordercolor=BG, lightcolor=BG, darkcolor=BG)
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", FG), ("active", FG)],
                  lightcolor=[("selected", PANEL)],
                  darkcolor=[("selected", PANEL)],
                  bordercolor=[("selected", PANEL)],
                  expand=[("selected", (0, 0, 0, 0))])

        style.configure("TCheckbutton", background=PANEL, foreground=FG,
                        font=FONT_LABEL, focuscolor=PANEL)
        style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", FG)])
        style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2,
                        foreground=FG, arrowcolor=MUTED, padding=6, borderwidth=0)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL_2)])
        style.configure("Vertical.TScrollbar", background="#39414f", troughcolor=PANEL_2,
                        borderwidth=0, arrowsize=13, bordercolor=PANEL_2,
                        lightcolor="#39414f", darkcolor="#39414f")
        style.map("Vertical.TScrollbar", background=[("active", "#4a5464")])

    def _placeholder(self) -> np.ndarray:
        """An empty state that says what to do, not just that nothing is here."""
        canvas = np.full((PREVIEW_H, PREVIEW_W, 3), 0x26, dtype=np.uint8)
        canvas[:] = (0x30, 0x26, 0x21)  # PANEL_2 in BGR
        centre = (PREVIEW_W // 2, PREVIEW_H // 2)
        cv2.circle(canvas, (centre[0], centre[1] - 26), 30, (0x4a, 0x42, 0x3c), 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (centre[0], centre[1] + 34), (46, 26), 0, 180, 360,
                    (0x4a, 0x42, 0x3c), 2, cv2.LINE_AA)
        for text, dy, shade in (("Camera is off", 84, (0x99, 0x88, 0x7f)),
                                ("Press Start monitoring", 108, (0x76, 0x68, 0x60))):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(canvas, text, (centre[0] - tw // 2, centre[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, shade, 1, cv2.LINE_AA)
        return canvas

    def _build(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=PAD, pady=(PAD, PAD // 2))
        title = ttk.Frame(header)
        title.pack(side="left")
        ttk.Label(title, text="AutoLock Safety Net", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title, text="locks when you are not the one in front of the camera",
            style="HeaderMuted.TLabel",
        ).pack(anchor="w")
        ttk.Label(header, text=get_locker().describe(), style="HeaderMuted.TLabel").pack(
            side="right", pady=(6, 0)
        )

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=PAD, pady=(PAD // 2, PAD))

        # --- left: the thing you actually look at ---
        left = ttk.Frame(body, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, PAD))

        self.preview = tk.Label(left, bg=PANEL_2, bd=0)
        self.preview.pack(padx=PAD, pady=(PAD, PAD - 4))

        # Status reads as one sentence: a number, then what it means.
        status = ttk.Frame(left, style="Panel.TFrame")
        status.pack(fill="x", padx=PAD)
        self.countdown = ttk.Label(status, text="--", style="Hero.TLabel")
        self.countdown.pack(side="left")
        wording = ttk.Frame(status, style="Panel.TFrame")
        wording.pack(side="left", padx=(PAD - 4, 0), pady=(6, 0))
        self.state_label = ttk.Label(wording, text="stopped", style="Panel.TLabel")
        self.state_label.pack(anchor="w")
        self.safety_label = ttk.Label(wording, text="not running", style="Muted.TLabel")
        self.safety_label.pack(anchor="w")

        ttk.Frame(left, style="Rule.TFrame", height=1).pack(
            fill="x", padx=PAD, pady=(PAD - 4, 0)
        )

        # Primary action on the left, destructive-ish one far from it.
        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.pack(fill="x", padx=PAD, pady=PAD)
        self.start_btn = ttk.Button(
            controls, text="Start monitoring", style="Accent.TButton", command=self.on_start
        )
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            controls, text="Stop", command=self.on_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.test_btn = ttk.Button(controls, text="Test", command=self.on_test)
        self.test_btn.pack(side="left", padx=(8, 0))
        self.pause_btn = ttk.Button(
            controls, text="Pause locking", style="Danger.TButton", command=self.on_pause
        )
        self.pause_btn.pack(side="right")

        # --- right: everything you configure ---
        right = ttk.Notebook(body, width=360)
        right.pack(side="right", fill="both")
        right.add(self._tab_setup(right), text="Setup")
        right.add(self._tab_settings(right), text="Settings")
        right.add(self._tab_startup(right), text="Startup")

    # ------------------------------------------------------------------
    def _section(self, parent, title: str, first: bool = False) -> ttk.Frame:
        """A titled block. Sections carry the hierarchy so labels do not have to."""
        ttk.Label(parent, text=title.upper(), style="Group.TLabel").pack(
            anchor="w", pady=(0 if first else PAD + 4, 6)
        )
        holder = ttk.Frame(parent, style="Panel.TFrame")
        holder.pack(fill="x")
        return holder

    def _entry(self, parent, textvariable) -> tk.Entry:
        entry = tk.Entry(
            parent, textvariable=textvariable, bg=PANEL_2, fg=FG, insertbackground=FG,
            relief="flat", highlightthickness=1, highlightbackground=PANEL_2,
            highlightcolor=BLUE, font=FONT_BODY,
        )
        entry.pack(fill="x", ipady=6)
        return entry

    def _add_placeholder(self, entry: tk.Entry, var: tk.StringVar, text: str) -> None:
        """Grey prompt text that clears on focus and returns when left empty."""
        state = {"showing": False}

        def show() -> None:
            if not var.get():
                state["showing"] = True
                entry.configure(fg=MUTED)
                var.set(text)

        def hide(_event=None) -> None:
            if state["showing"]:
                state["showing"] = False
                var.set("")
            entry.configure(fg=FG)

        def restore(_event=None) -> None:
            if not var.get():
                show()

        entry.bind("<FocusIn>", hide)
        entry.bind("<FocusOut>", restore)
        show()

    def _hint(self, parent, text: str, pad: tuple[int, int] = (6, 0)) -> ttk.Label:
        label = ttk.Label(
            parent, text=text, style="Muted.TLabel", wraplength=310, justify="left"
        )
        label.pack(anchor="w", pady=pad)
        return label

    def _tab_setup(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame", padding=PAD)

        who = self._section(tab, "Who this watches for", first=True)
        self.identities_label = ttk.Label(
            who, text="none", style="Panel.TLabel", wraplength=310
        )
        self.identities_label.pack(anchor="w", pady=(0, PAD - 4))

        ttk.Label(who, text="Name", style="Panel.TLabel").pack(anchor="w", pady=(0, 4))
        self.name_var = tk.StringVar(value=(list_identities() or [""])[0])
        self._entry(who, self.name_var)

        buttons = ttk.Frame(who, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(
            buttons, text="Enrol my face", style="Accent.TButton", command=self.on_enroll
        ).pack(side="left")
        ttk.Button(buttons, text="Add more poses", command=self.on_enroll_append).pack(
            side="left", padx=(8, 0)
        )
        self._hint(
            who,
            "Enrolment walks you through seven head poses. Add more after a haircut, "
            "new glasses, or a change in your desk lighting.",
        )

        maintenance = self._section(tab, "Maintenance")
        row = ttk.Frame(maintenance, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Download models", command=self.on_models).pack(side="left")
        ttk.Button(row, text="Run diagnostics", command=self.on_doctor).pack(
            side="left", padx=(8, 0)
        )
        self._hint(maintenance, "Diagnostics are written to logs/autolock.log.")

        self.refresh_identities()
        return tab

    # ------------------------------------------------------------------
    # Settings, generated from the field registry in config.py
    # ------------------------------------------------------------------
    def _tab_settings(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame")

        top = ttk.Frame(tab, style="Panel.TFrame", padding=(PAD, PAD, PAD, 10))
        top.pack(fill="x")
        self.search_var = tk.StringVar()
        search = self._entry(top, self.search_var)
        self._add_placeholder(search, self.search_var, SEARCH_PLACEHOLDER)
        self.search_var.trace_add("write", lambda *_: self._filter_settings())

        row = ttk.Frame(top, style="Panel.TFrame")
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(
            row, text="18 of 42 shown", style="Muted.TLabel", name="settingscount"
        ).pack(side="left")
        self.count_label = row.nametowidget("settingscount")
        self.advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row, text="Show advanced", variable=self.advanced_var,
            command=self._filter_settings,
        ).pack(side="right")

        ttk.Frame(tab, style="Rule.TFrame", height=1).pack(fill="x")

        # Scrollable body: Tkinter has no scrollable frame, so it is a Canvas
        # with a frame inside it and the scrollregion kept in sync.
        body = ttk.Frame(tab, style="Panel.TFrame")
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        self.settings_inner = ttk.Frame(canvas, style="Panel.TFrame")
        window = canvas.create_window((0, 0), window=self.settings_inner, anchor="nw")

        self.settings_inner.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        # The scrollbar is packed first: an expanding canvas packed before it
        # claims the whole width and leaves the scrollbar zero pixels wide.
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", on_wheel)  # Windows / macOS
        canvas.bind_all("<Button-4>", lambda _e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind_all("<Button-5>", lambda _e: canvas.yview_scroll(1, "units"))

        self._build_setting_rows(self.settings_inner)
        # Rows are packed as they are built; apply the filter once now so the
        # advanced ones start hidden rather than appearing until first toggled.
        self._filter_settings()

        ttk.Frame(tab, style="Rule.TFrame", height=1).pack(fill="x")
        footer = ttk.Frame(tab, style="Panel.TFrame", padding=PAD)
        footer.pack(fill="x")
        ttk.Button(
            footer, text="Save", style="Accent.TButton", command=self.on_save
        ).pack(side="left")
        ttk.Button(footer, text="Reset to defaults", command=self.on_reset).pack(
            side="left", padx=(8, 0)
        )
        self.save_hint = ttk.Label(
            footer, text=self.config_path.name, style="Muted.TLabel"
        )
        self.save_hint.pack(side="right", pady=(10, 0))
        return tab

    def _build_setting_rows(self, parent) -> None:
        """One row per Config field, laid out by group."""
        self.vars: dict[str, tk.Variable] = {}
        self.rows: dict[str, ttk.Frame] = {}
        self.group_headers: dict[str, ttk.Label] = {}
        self.empty_label = ttk.Label(parent, text="no settings match", style="Muted.TLabel")

        for group, keys in grouped_fields().items():
            header = ttk.Label(parent, text=group.upper(), style="Group.TLabel")
            header.pack(anchor="w", padx=PAD, pady=(PAD + 2, 2))
            self.group_headers[group] = header

            for key in keys:
                spec = FIELD_SPECS[key]
                current = getattr(self.cfg, key)
                row = ttk.Frame(parent, style="Panel.TFrame")
                row.pack(fill="x", padx=PAD, pady=(10, 0))
                self.rows[key] = row
                self._build_one_row(row, key, spec, current)

    def _build_one_row(self, row, key: str, spec, current) -> None:
        """One setting: label, control, then the sentence explaining it."""
        if isinstance(current, bool):
            # The checkbox carries its own label, so no separate caption.
            var = tk.BooleanVar(value=current)
            ttk.Checkbutton(row, text=spec.label, variable=var).pack(anchor="w")
        elif spec.choices:
            var = tk.StringVar(value=str(current))
            ttk.Label(row, text=spec.label, style="Panel.TLabel").pack(anchor="w", pady=(0, 4))
            ttk.Combobox(
                row, textvariable=var, values=list(spec.choices), state="readonly",
                font=FONT_BODY,
            ).pack(fill="x")
        elif isinstance(current, (int, float)) and spec.minimum is not None:
            var = tk.DoubleVar(value=float(current))
            head = ttk.Frame(row, style="Panel.TFrame")
            head.pack(fill="x")
            ttk.Label(head, text=spec.label, style="Panel.TLabel").pack(side="left")
            # The value sits beside its own label rather than under the slider,
            # so the eye reads "name: value" in one move.
            readout = ttk.Label(head, text=f"{current:g}", style="Panel.TLabel",
                                foreground=BLUE)
            readout.pack(side="right")
            tk.Scale(
                row, from_=spec.minimum, to=spec.maximum, orient="horizontal", variable=var,
                resolution=spec.step or 1, showvalue=False, bg=PANEL, fg=FG,
                troughcolor=PANEL_2, highlightthickness=0, bd=0, sliderrelief="flat",
                activebackground=BLUE, sliderlength=18, width=10,
                command=lambda _v, r=readout, v=var: r.configure(text=f"{v.get():g}"),
            ).pack(fill="x", pady=(2, 0))
        else:
            text = (
                ",".join(str(part) for part in current)
                if isinstance(current, tuple)
                else str(current)
            )
            var = tk.StringVar(value=text)
            ttk.Label(row, text=spec.label, style="Panel.TLabel").pack(anchor="w", pady=(0, 4))
            self._entry(row, var)

        self.vars[key] = var
        self._hint(row, spec.help, pad=(4, 0))

    def _matches_filter(self, key: str, needle: str, show_advanced: bool) -> bool:
        spec = FIELD_SPECS[key]
        if needle:
            # A search reaches advanced settings too, otherwise looking one up
            # by name silently finds nothing.
            return (
                needle in key.lower()
                or needle in spec.label.lower()
                or needle in spec.help.lower()
            )
        return show_advanced or not spec.advanced

    def _filter_settings(self) -> None:
        """Re-pack the visible rows, in registry order.

        Everything is unpacked and packed again rather than toggled in place:
        Tk appends re-packed widgets to the end, so hiding and showing rows
        would otherwise slowly shuffle the page out of order.
        """
        needle = self.search_var.get().strip().lower()
        if needle == SEARCH_PLACEHOLDER.lower():
            needle = ""  # the prompt text is not a search term
        show_advanced = bool(self.advanced_var.get())

        for header in self.group_headers.values():
            header.pack_forget()
        for row in self.rows.values():
            row.pack_forget()

        shown = 0
        for group, keys in grouped_fields().items():
            visible = [k for k in keys if self._matches_filter(k, needle, show_advanced)]
            if not visible:
                continue
            self.group_headers[group].pack(anchor="w", padx=PAD, pady=(PAD + 2, 2))
            for key in visible:
                self.rows[key].pack(fill="x", padx=PAD, pady=(10, 0))
            shown += len(visible)

        if shown:
            self.empty_label.pack_forget()
        else:
            self.empty_label.pack(pady=PAD * 2)
        self.count_label.configure(text=f"{shown} of {len(FIELD_SPECS)} shown")

    def _tab_startup(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, style="Panel.TFrame", padding=PAD)

        block = self._section(tab, "Start at login", first=True)
        self.autostart_label = ttk.Label(
            block, text="", style="Panel.TLabel", wraplength=310
        )
        self.autostart_label.pack(anchor="w", pady=(0, 10))

        buttons = ttk.Frame(block, style="Panel.TFrame")
        buttons.pack(fill="x")
        ttk.Button(
            buttons, text="Enable", style="Accent.TButton", command=self.on_autostart_install
        ).pack(side="left")
        ttk.Button(buttons, text="Disable", command=self.on_autostart_remove).pack(
            side="left", padx=(8, 0)
        )
        self._hint(
            block,
            "At login this window opens again, already monitoring. Nothing runs hidden: "
            "a locker you cannot see is a locker you cannot stop.",
        )

        guards = self._section(tab, "Why this cannot lock you out")
        for line in (
            "Stays disarmed until it has recognised you once.",
            "Never locks while the camera is missing or busy.",
            "Stops itself if locks start firing repeatedly.",
            "Grace period after every login, ended early by seeing you.",
        ):
            ttk.Label(
                guards, text=f"·  {line}", style="Muted.TLabel", wraplength=310,
                justify="left",
            ).pack(anchor="w", pady=(0, 3))

        escape = self._section(tab, "Emergency stop")
        self._hint(
            escape,
            "Press Pause locking, or create a file named PAUSE in the project folder — "
            "from a file manager, another machine, anywhere. Locking stops until it is "
            "deleted.",
            pad=(0, 0),
        )

        self.refresh_autostart()
        return tab

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def collect_config(self) -> tuple[Config, list[str]]:
        """Read every widget back into a Config, reporting anything unparseable."""
        cfg = Config.from_dict(self.cfg.to_dict())
        errors: list[str] = []

        for key, var in self.vars.items():
            raw = var.get()
            try:
                if isinstance(raw, bool):
                    setattr(cfg, key, raw)
                elif isinstance(raw, float) and isinstance(getattr(cfg, key), int):
                    setattr(cfg, key, int(round(raw)))  # sliders are DoubleVar
                elif isinstance(raw, (int, float)):
                    setattr(cfg, key, type(getattr(cfg, key))(raw))
                else:
                    setattr(cfg, key, coerce_value(cfg, key, str(raw)))
            except (ValueError, TypeError) as exc:
                errors.append(f"{FIELD_SPECS[key].label}: {exc}")

        return cfg.normalise(), errors

    def on_save(self) -> None:
        cfg, errors = self.collect_config()
        if errors:
            messagebox.showerror("Invalid settings", "\n".join(errors))
            return

        cfg.save(self.config_path)
        self.cfg = cfg

        if self.monitor and self.monitor.is_alive():
            # Settings are read when the pipeline is built, so a running
            # monitor has to be restarted for them to take effect.
            if messagebox.askyesno(
                "Restart monitor?",
                "Saved. The monitor is running with the old settings — restart it now?",
            ):
                arm = self.monitor.arm
                self.on_stop()
                self.monitor.join(timeout=5.0)
                self._start(arm=arm)
            return
        messagebox.showinfo("Saved", f"Settings written to {self.config_path}")

    def on_reset(self) -> None:
        if not messagebox.askyesno(
            "Reset settings", "Put every setting back to its built-in default?"
        ):
            return
        defaults = Config()
        for key, var in self.vars.items():
            value = getattr(defaults, key)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.DoubleVar):
                var.set(float(value))
            elif isinstance(value, tuple):
                var.set(",".join(str(part) for part in value))
            else:
                var.set(str(value))
        self._refresh_readouts()

    def _refresh_readouts(self) -> None:
        """Redraw the settings rows so slider readouts match their variables."""
        for child in list(self.settings_inner.winfo_children()):
            child.destroy()
        self.cfg = self.collect_config()[0]
        self._build_setting_rows(self.settings_inner)
        self._filter_settings()

    def _start(self, arm: bool) -> None:
        if self.monitor and self.monitor.is_alive():
            return
        if not list_identities():
            messagebox.showwarning(
                "Nobody enrolled",
                "Enrol your face first — the monitor has nobody to recognise.",
            )
            return
        cfg, errors = self.collect_config()
        if errors:
            messagebox.showerror("Invalid settings", "\n".join(errors))
            return
        self.cfg = cfg
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
            enroll_interactive(self.collect_config()[0], name, append=append)
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


def launch(
    cfg: Config | None = None,
    config_path=DEFAULT_CONFIG_PATH,
    start_monitoring: bool = False,
) -> int:
    ensure_dirs()
    root = tk.Tk()
    AutoLockGUI(root, cfg or Config.load(config_path), config_path, start_monitoring)
    try:
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass
    root.mainloop()
    return 0

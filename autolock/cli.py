"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__, models
from .config import DEFAULT_CONFIG_PATH, IDENTITY_DIR, PROJECT_ROOT, Config, ensure_dirs
from .logging_setup import quiet_third_party, setup_logging

log = logging.getLogger("autolock")


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autolock",
        description="Lock the workstation when the enrolled person is no longer at the desk.",
    )
    parser.add_argument("--version", action="version", version=f"autolock-safetynet {__version__}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="config JSON path")
    parser.add_argument("--log-level", dest="log_level", default=None, help="DEBUG/INFO/WARNING")

    sub = parser.add_subparsers(dest="command")

    # ---- shared camera/backend options ----
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera", dest="camera_index", type=int, default=None)
    common.add_argument(
        "--camera-api",
        dest="camera_api",
        choices=["auto", "dshow", "msmf", "avfoundation", "v4l2", "gstreamer", "any"],
    )
    common.add_argument("--backend", dest="backend", choices=["auto", "opencv", "insightface"])
    common.add_argument("--det-threshold", dest="det_threshold", type=float, default=None)
    common.add_argument("--min-face-px", dest="min_face_px", type=int, default=None)
    common.add_argument(
        "--no-rotation-retry", dest="rotation_retry", action="store_false", default=None
    )

    # ---- shared tuning options ----
    tuning = argparse.ArgumentParser(add_help=False)
    tuning.add_argument(
        "--identity", dest="identity", default=None, help="which enrolled name to use"
    )
    tuning.add_argument(
        "--timeout", dest="absence_timeout_s", type=float, default=None,
        help="seconds unrecognised before locking (default 3)",
    )
    tuning.add_argument("--threshold", dest="match_threshold", type=float, default=None)
    # Holds are 0 by default: only a recognised face keeps the session open.
    # Passing one buys a longer fuse for that evidence and switches on the
    # detector it needs.
    tuning.add_argument(
        "--body-hold", dest="body_hold_s", type=float, default=None,
        help="max seconds unrecognised while your body is visible "
             "(head down, turned away); enables the pose model",
    )
    tuning.add_argument(
        "--track-hold", dest="track_hold_s", type=float, default=None,
        help="max seconds unrecognised while an unscored face sits in your tracked box",
    )
    tuning.add_argument(
        "--motion-hold", dest="motion_hold_s", type=float, default=None,
        help="max seconds unrecognised while there is movement where you were",
    )

    # ---- run ----
    run = sub.add_parser(
        "run", parents=[common, tuning], help="start the presence monitor (default)"
    )
    run.add_argument("--fps", dest="target_fps", type=float, default=None)
    run.add_argument("--no-preview", dest="preview", action="store_false", default=None)
    run.add_argument(
        "--no-lock-on-unknown", dest="lock_on_unknown", action="store_false", default=None,
        help="do not lock just because an unrecognised face is at the desk",
    )
    run.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=None,
        help="log the lock decision instead of locking",
    )

    # ---- test ----
    sub.add_parser(
        "test",
        parents=[common, tuning],
        help="live view of scores, evidence and the countdown; never locks",
    )

    # ---- enroll ----
    enroll = sub.add_parser(parents=[common], name="enroll", help="train the monitor on your face")
    enroll.add_argument("--name", required=True, help="identity name, e.g. your first name")
    enroll.add_argument("--samples", type=int, default=8, help="samples per pose (default 8)")
    enroll.add_argument("--pose-timeout", type=float, default=25.0, help="seconds per pose")
    enroll.add_argument("--append", action="store_true", help="add to an existing template")
    enroll.add_argument("--from-images", type=Path, default=None, help="enrol from a photo folder")

    # ---- gui ----
    gui_cmd = sub.add_parser("gui", help="open the desktop control panel")
    gui_cmd.add_argument(
        "--start", dest="start_monitoring", action="store_true",
        help="begin monitoring as soon as the window opens (used by autostart)",
    )

    # ---- autostart ----
    auto = sub.add_parser("autostart", help="start at login (install / remove / status)")
    auto_group = auto.add_mutually_exclusive_group()
    auto_group.add_argument("--install", action="store_true", help="run at login")
    auto_group.add_argument("--uninstall", action="store_true", help="stop running at login")
    auto_group.add_argument("--status", action="store_true", help="report what is installed")
    auto.add_argument(
        "--headless", action="store_true",
        help="run invisibly in the background instead of opening the control panel",
    )
    auto.add_argument(
        "--arg", dest="extra_args", action="append", default=None, metavar="FLAG",
        help="extra flag for the installed command, e.g. --arg --body-hold=60",
    )

    # ---- liveness ----
    live = sub.add_parser(
        "liveness", parents=[common], help="anti-spoofing: calibrate, enable or disable"
    )
    live_group = live.add_mutually_exclusive_group()
    live_group.add_argument(
        "--test", action="store_true",
        help="show live anti-spoof scores so you can pick a threshold",
    )
    live_group.add_argument("--enable", action="store_true", help="turn the check on and save")
    live_group.add_argument("--disable", action="store_true", help="turn the check off and save")
    live.add_argument(
        "--threshold", dest="liveness_threshold", type=float, default=None,
        help="score below which a face counts as a spoof (default 0.55)",
    )
    live.add_argument("--seconds", type=float, default=30.0, help="how long --test runs")

    # ---- pause ----
    pause = sub.add_parser("pause", help="stop or resume locking without stopping the monitor")
    pause_group = pause.add_mutually_exclusive_group()
    pause_group.add_argument("--on", action="store_true", help="stop locking")
    pause_group.add_argument("--off", action="store_true", help="resume locking")

    # ---- identities ----
    sub.add_parser("identities", help="list enrolled identities")

    # ---- models ----
    models_cmd = sub.add_parser("models", help="download the model weights")
    models_cmd.add_argument("--force", action="store_true", help="re-download even if cached")

    # ---- doctor / config ----
    sub.add_parser("doctor", parents=[common], help="check the environment end to end")
    cfg_cmd = sub.add_parser("config", help="show, write or edit the config file")
    cfg_cmd.add_argument(
        "--write", action="store_true", help="write the full config to config.json"
    )
    cfg_cmd.add_argument(
        "--set", dest="settings", action="append", metavar="KEY=VALUE", default=None,
        help="set a value and save it, e.g. --set absence_timeout_s=3 "
             "(repeatable; creates config.json if absent)",
    )
    cfg_cmd.add_argument(
        "--unset", dest="unset", action="append", metavar="KEY", default=None,
        help="restore a key to its built-in default",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "config", "func", "name", "samples", "append", "from_images",
                       "force", "write", "pose_timeout"}
    }
    return cfg.apply_overrides(overrides).normalise()


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace, arm: bool = True) -> int:
    from .app import PresenceLockApp

    cfg = _config_from_args(args)
    if not arm:
        cfg.preview = True
    try:
        app = PresenceLockApp(cfg, arm=arm)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    return app.run()


def cmd_enroll(args: argparse.Namespace) -> int:
    from .enroll import enroll_from_images, enroll_interactive

    cfg = _config_from_args(args)
    name = args.name.strip().replace(" ", "_")
    if not name or any(ch in name for ch in '\\/:*?"<>|'):
        log.error("Invalid identity name: %r", args.name)
        return 2

    try:
        if args.from_images:
            enroll_from_images(cfg, name, args.from_images, append=args.append)
        else:
            enroll_interactive(
                cfg,
                name,
                samples_per_pose=args.samples,
                pose_timeout_s=args.pose_timeout,
                append=args.append,
            )
    except KeyboardInterrupt:
        log.warning("Enrolment cancelled — nothing was saved")
        return 1
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2

    log.info("Done. Check it with:  python main.py test")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch
    except ImportError as exc:  # tkinter missing on a stripped-down Linux python
        log.error(
            "The GUI needs tkinter and Pillow: %s\n"
            "  Debian/Ubuntu: sudo apt install python3-tk && pip install pillow\n"
            "  Fedora:        sudo dnf install python3-tkinter && pip install pillow",
            exc,
        )
        return 2
    return launch(
        Config.load(args.config), args.config, getattr(args, "start_monitoring", False)
    )


def cmd_autostart(args: argparse.Namespace) -> int:
    from . import autostart

    if args.uninstall:
        result = autostart.uninstall()
        log.info("%s", result.message)
        return 0 if result.ok else 1

    if args.install:
        result = autostart.install(args.extra_args, args.headless)
        if not result.ok:
            log.error("%s", result.message)
            return 1
        log.info("%s", result.message)
        log.info("  entry:   %s", result.location)
        log.info(
            "  command: %s",
            " ".join(autostart.launch_command(args.extra_args, args.headless)),
        )
        log.info(
            "  shows:   %s",
            "nothing — it runs invisibly; watch logs/autolock.log"
            if args.headless
            else "the control panel, already monitoring",
        )
        log.info("  remove:  %s", result.undo)
        log.info(
            "Safety at login: the monitor stays disarmed until it recognises you once, "
            "waits out startup_grace_s, never locks while the camera is blind, and trips "
            "a breaker if locks fire repeatedly. Emergency stop: create the file %s",
            Config.load(args.config).pause_file,
        )
        return 0

    # default: status
    if autostart.is_installed():
        entry = autostart.entry_path()
        log.info("Start at login: ENABLED (%s)", entry)
        try:
            installed = entry.read_text(encoding="utf-8") if entry else ""
        except OSError:
            installed = ""
        if installed:
            visible = "gui" in installed
            log.info(
                "  it will open: %s",
                "the control panel, already monitoring"
                if visible
                else "nothing — it runs invisibly; watch logs/autolock.log",
            )
    else:
        log.info("Start at login: not enabled")
    log.info(
        "Would run: %s",
        " ".join(autostart.launch_command(args.extra_args, args.headless)),
    )
    return 0


def cmd_liveness(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)

    if args.enable or args.disable:
        stored = Config.load(args.config)
        stored.require_liveness = bool(args.enable)
        if args.liveness_threshold is not None:
            stored.liveness_threshold = float(args.liveness_threshold)
        stored.save(args.config)
        log.info(
            "Liveness check %s (threshold %.2f). Saved to %s",
            "ENABLED" if args.enable else "disabled",
            stored.liveness_threshold,
            args.config,
        )
        if args.enable:
            log.info(
                "If your own face ever scores below the threshold you will be locked "
                "repeatedly — the circuit breaker will stop it after 3 locks in 60s, and "
                "`autolock liveness --disable` turns this back off."
            )
        return 0

    # --- calibration ---
    import cv2

    from .backends import build_backend
    from .camera import Camera
    from .liveness import LivenessDetector

    detector = LivenessDetector(threshold=cfg.liveness_threshold)
    if not detector.available:
        log.error("Anti-spoofing model is not usable: %s", detector.status)
        log.error("Everything else still works; the check simply cannot run here.")
        return 2

    backend = build_backend(cfg)
    camera = Camera(cfg.camera_index, cfg.camera_api, cfg.frame_width, cfg.frame_height)
    if not camera.open():
        log.error("Cannot open camera %d", cfg.camera_index)
        return 2

    log.info("Calibration: watch the score with your real face, then hold up a photo")
    log.info("  of yourself on a phone. Pick a threshold between the two clusters.")
    log.info("  ESC or Ctrl-C to stop (running %.0fs).", args.seconds)

    scores: list[float] = []
    deadline = time.monotonic() + args.seconds
    window = "Liveness calibration"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                time.sleep(0.1)
                continue
            faces = backend.detect(frame)
            if faces:
                score = detector.score(frame, faces[0])
                if score == score:  # not NaN
                    scores.append(score)
                    x, y, w, h = faces[0].bbox
                    live = score >= cfg.liveness_threshold
                    colour = (80, 220, 100) if live else (70, 70, 240)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                    cv2.putText(
                        frame,
                        f"liveness {score:.3f}  {'LIVE' if live else 'SPOOF'}",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
                    )
            if cfg.mirror_preview:
                frame = cv2.flip(frame, 1)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        backend.close()
        detector.close()

    if not scores:
        log.error("No faces were scored — nothing to calibrate from.")
        return 2

    array = sorted(scores)
    log.info("Scored %d frames:", len(array))
    log.info(
        "  min %.3f | p05 %.3f | median %.3f | p95 %.3f | max %.3f",
        array[0],
        array[max(0, int(0.05 * len(array)) - 1)],
        array[len(array) // 2],
        array[min(len(array) - 1, int(0.95 * len(array)))],
        array[-1],
    )
    log.info(
        "Set the threshold below your real-face scores and above your photo scores:\n"
        "  autolock liveness --enable --threshold <value>"
    )
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    from .safety import PauseSwitch

    cfg = Config.load(args.config)
    switch = PauseSwitch(PROJECT_ROOT / cfg.pause_file)

    if args.on:
        switch.engage()
        log.info("Locking paused. Resume with `autolock pause --off` or delete %s", switch.path)
    elif args.off:
        switch.release()
        log.info("Locking resumed.")
    else:
        log.info("Locking is %s (%s)", "PAUSED" if switch.active() else "active", switch.path)
    return 0


def cmd_identities(_args: argparse.Namespace) -> int:
    from .identity import Identity, list_identities

    names = list_identities()
    if not names:
        log.info("No identities enrolled yet. Run:  python main.py enroll --name <you>")
        return 0

    log.info("Enrolled identities in %s:", IDENTITY_DIR)
    for name in names:
        identity = Identity.load_by_name(name)
        stats = identity.intra_stats()
        poses = sorted(set(identity.poses))
        log.info(
            "  %-16s %3d samples | backend=%-11s threshold=%.3f | poses: %s | self-sim p05=%.3f",
            name,
            len(identity),
            identity.backend,
            identity.threshold,
            ", ".join(poses),
            stats["p05"],
        )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    paths = models.ensure_all(force=args.force)
    for key, path in paths.items():
        log.info("%-6s %s (%.1f MB)", key, path.name, path.stat().st_size / 1e6)
    missing = [k for k in models.REGISTRY if k not in paths]
    if missing:
        log.warning("Not downloaded: %s", ", ".join(missing))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import cv2

    from . import autostart
    from .backends import build_backend
    from .camera import Camera
    from .identity import list_identities
    from .lock import get_locker

    cfg = _config_from_args(args)
    problems = 0

    log.info("autolock-safetynet %s | python %s", __version__, sys.version.split()[0])
    log.info("opencv %s", cv2.__version__)

    for spec in models.REGISTRY.values():
        state = "ok" if models.is_present(spec) else "MISSING"
        if state == "MISSING" and spec.required:
            problems += 1
        log.info("model %-6s %-8s %s", spec.key, state, spec.purpose)

    try:
        backend = build_backend(cfg)
        log.info("backend ok: %s (dim=%d)", backend.name, backend.embedding_dim)
        backend.close()
    except Exception as exc:
        log.error("backend FAILED: %s", exc)
        problems += 1

    try:
        from .body import BodyDetector

        body = BodyDetector()
        log.info("body fallback: %s", "ok" if body.available else "unavailable")
        body.close()
    except Exception as exc:
        log.warning("body fallback unavailable: %s", exc)

    camera = Camera(cfg.camera_index, cfg.camera_api, cfg.frame_width, cfg.frame_height)
    if camera.open():
        frame = camera.read()
        log.info(
            "camera %d ok: %s",
            cfg.camera_index,
            f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "opened but no frame",
        )
        if frame is None:
            problems += 1
    else:
        log.error("camera %d FAILED to open", cfg.camera_index)
        problems += 1
    camera.release()

    names = list_identities()
    if names:
        log.info("identities: %s", ", ".join(names))
    else:
        log.error("identities: none enrolled — run `python main.py enroll --name <you>`")
        problems += 1

    locker = get_locker()
    if locker.available:
        state = locker.is_locked()
        log.info(
            "lock method: %s (session currently %s)",
            locker.describe(),
            {True: "locked", False: "active", None: "unknown"}[state],
        )
    else:
        log.error("no working lock method on this platform: %s", locker.describe())
        problems += 1

    log.info(
        "safety: %s | startup grace %.0fs | breaker %d locks/%.0fs | pause file %s",
        "disarmed until recognised" if cfg.require_initial_recognition else "arms immediately",
        cfg.startup_grace_s,
        cfg.max_locks_per_window,
        cfg.lock_window_s,
        PROJECT_ROOT / cfg.pause_file,
    )
    if autostart.is_installed():
        log.info("start at login: enabled (%s)", autostart.entry_path())
    else:
        log.info("start at login: not enabled (`autolock autostart --install`)")

    log.info("%s", "All good." if problems == 0 else f"{problems} problem(s) found.")
    return 0 if problems == 0 else 1


def cmd_config(args: argparse.Namespace) -> int:
    import json

    cfg = Config.load(args.config)
    changed: list[str] = []

    for assignment in args.settings or []:
        if "=" not in assignment:
            log.error("Expected KEY=VALUE, got %r", assignment)
            return 2
        key, _, raw = assignment.partition("=")
        key = key.strip()
        try:
            before = getattr(cfg, key)
            setattr(cfg, key, coerce_setting(cfg, key, raw.strip()))
        except AttributeError:
            log.error("Unknown setting %r. See `python main.py config` for the full list.", key)
            return 2
        except ValueError as exc:
            log.error("%s", exc)
            return 2
        changed.append(f"{key}: {before!r} -> {getattr(cfg, key)!r}")

    for key in args.unset or []:
        key = key.strip()
        if not hasattr(cfg, key):
            log.error("Unknown setting %r", key)
            return 2
        before = getattr(cfg, key)
        setattr(cfg, key, getattr(Config(), key))
        changed.append(f"{key}: {before!r} -> {getattr(cfg, key)!r} (default)")

    if changed:
        cfg.normalise()
        path = cfg.save(args.config)
        for line in changed:
            log.info("%s", line)
        log.info("Saved %s", path)
        return 0

    if args.write:
        path = cfg.save(args.config)
        log.info("Wrote %s", path)
        return 0

    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def coerce_setting(cfg: Config, key: str, raw: str):
    """Parse a CLI string into the type the config field already holds."""
    current = getattr(cfg, key)  # raises AttributeError for unknown keys

    if isinstance(current, bool):  # before int: bool is a subclass of int
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} expects a boolean, got {raw!r}")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{key} expects a whole number, got {raw!r}") from None
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"{key} expects a number, got {raw!r}") from None
    if isinstance(current, tuple):
        try:
            return tuple(int(part) for part in raw.split(",") if part.strip())
        except ValueError:
            raise ValueError(f"{key} expects comma-separated whole numbers, got {raw!r}") from None
    return raw


# ----------------------------------------------------------------------
COMMANDS = (
    "run", "test", "gui", "enroll", "identities", "models",
    "doctor", "config", "autostart", "pause", "liveness",
)
_GLOBAL_FLAGS_WITH_VALUE = ("--config", "--log-level")


def _with_default_command(argv: list[str]) -> list[str]:
    """Make `run` the implicit subcommand, after any global flags."""
    if any(token in ("-h", "--help", "--version") for token in argv):
        return argv

    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _GLOBAL_FLAGS_WITH_VALUE:
            index += 2
            continue
        if token.startswith("--") and token.split("=", 1)[0] in _GLOBAL_FLAGS_WITH_VALUE:
            index += 1
            continue
        break

    if index >= len(argv) or argv[index] not in COMMANDS:
        return argv[:index] + ["run"] + argv[index:]
    return argv


def main(argv: list[str] | None = None) -> int:
    quiet_third_party()
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(_with_default_command(raw))

    ensure_dirs()
    cfg_level = Config.load(args.config).log_level
    setup_logging(args.log_level or cfg_level)

    handlers = {
        "run": lambda a: cmd_run(a, arm=True),
        "test": lambda a: cmd_run(a, arm=False),
        "gui": cmd_gui,
        "enroll": cmd_enroll,
        "identities": cmd_identities,
        "models": cmd_models,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "autostart": cmd_autostart,
        "pause": cmd_pause,
        "liveness": cmd_liveness,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log.info("Interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

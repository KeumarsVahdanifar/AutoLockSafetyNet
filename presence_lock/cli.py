"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, models
from .config import DEFAULT_CONFIG_PATH, IDENTITY_DIR, Config, ensure_dirs
from .logging_setup import quiet_third_party, setup_logging

log = logging.getLogger("presence_lock")


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plock",
        description="Lock the workstation when the enrolled person is no longer at the desk.",
    )
    parser.add_argument("--version", action="version", version=f"presence-lock {__version__}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="config JSON path")
    parser.add_argument("--log-level", dest="log_level", default=None, help="DEBUG/INFO/WARNING")

    sub = parser.add_subparsers(dest="command")

    # ---- shared camera/backend options ----
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera", dest="camera_index", type=int, default=None)
    common.add_argument("--camera-api", dest="camera_api", choices=["dshow", "msmf", "any"])
    common.add_argument("--backend", dest="backend", choices=["auto", "opencv", "insightface"])
    common.add_argument("--det-threshold", dest="det_threshold", type=float, default=None)
    common.add_argument("--min-face-px", dest="min_face_px", type=int, default=None)
    common.add_argument(
        "--no-rotation-retry", dest="rotation_retry", action="store_false", default=None
    )

    # ---- run ----
    run = sub.add_parser("run", parents=[common], help="start the presence monitor (default)")
    run.add_argument("--identity", dest="identity", default=None, help="which enrolled name to use")
    run.add_argument("--timeout", dest="absence_timeout_s", type=float, default=None)
    run.add_argument("--threshold", dest="match_threshold", type=float, default=None)
    run.add_argument("--fps", dest="target_fps", type=float, default=None)
    run.add_argument("--no-preview", dest="preview", action="store_false", default=None)
    run.add_argument(
        "--no-body", dest="use_body_fallback", action="store_false", default=None,
        help="disable the head-down / turned-away pose fallback",
    )
    run.add_argument(
        "--no-motion", dest="use_motion_fallback", action="store_false", default=None
    )
    run.add_argument(
        "--no-lock-on-unknown", dest="lock_on_unknown", action="store_false", default=None,
        help="do not lock just because an unrecognised face is at the desk",
    )
    run.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=None,
        help="log the lock decision instead of locking",
    )

    # ---- test ----
    test = sub.add_parser(
        "test", parents=[common], help="live view of scores and evidence; never locks"
    )
    test.add_argument("--identity", dest="identity", default=None)
    test.add_argument("--timeout", dest="absence_timeout_s", type=float, default=None)
    test.add_argument("--threshold", dest="match_threshold", type=float, default=None)

    # ---- enroll ----
    enroll = sub.add_parser(parents=[common], name="enroll", help="train the monitor on your face")
    enroll.add_argument("--name", required=True, help="identity name, e.g. your first name")
    enroll.add_argument("--samples", type=int, default=8, help="samples per pose (default 8)")
    enroll.add_argument("--pose-timeout", type=float, default=25.0, help="seconds per pose")
    enroll.add_argument("--append", action="store_true", help="add to an existing template")
    enroll.add_argument("--from-images", type=Path, default=None, help="enrol from a photo folder")

    # ---- identities ----
    sub.add_parser("identities", help="list enrolled identities")

    # ---- models ----
    models_cmd = sub.add_parser("models", help="download the model weights")
    models_cmd.add_argument("--force", action="store_true", help="re-download even if cached")

    # ---- doctor / config ----
    sub.add_parser("doctor", parents=[common], help="check the environment end to end")
    cfg_cmd = sub.add_parser("config", help="show or write the config file")
    cfg_cmd.add_argument("--write", action="store_true", help="write defaults to config.json")

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "config", "func", "name", "samples", "append", "from_images",
                       "force", "write", "pose_timeout"}
    }
    return cfg.apply_overrides(overrides)


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

    log.info("Done. Check it with:  python plock.py test")
    return 0


def cmd_identities(_args: argparse.Namespace) -> int:
    from .identity import Identity, list_identities

    names = list_identities()
    if not names:
        log.info("No identities enrolled yet. Run:  python plock.py enroll --name <you>")
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

    from .backends import build_backend
    from .camera import Camera
    from .identity import list_identities
    from .lock import IS_WINDOWS, is_session_locked

    cfg = _config_from_args(args)
    problems = 0

    log.info("presence-lock %s | python %s", __version__, sys.version.split()[0])
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
        log.error("identities: none enrolled — run `python plock.py enroll --name <you>`")
        problems += 1

    if IS_WINDOWS:
        log.info("lock api ok (session currently %s)", "locked" if is_session_locked() else "active")
    else:
        log.error("locking is Windows-only; this platform is %s", sys.platform)
        problems += 1

    log.info("%s", "All good." if problems == 0 else f"{problems} problem(s) found.")
    return 0 if problems == 0 else 1


def cmd_config(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.write:
        path = cfg.save(args.config)
        log.info("Wrote %s", path)
        return 0
    import json

    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


# ----------------------------------------------------------------------
COMMANDS = ("run", "test", "enroll", "identities", "models", "doctor", "config")
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
        "enroll": cmd_enroll,
        "identities": cmd_identities,
        "models": cmd_models,
        "doctor": cmd_doctor,
        "config": cmd_config,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log.info("Interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

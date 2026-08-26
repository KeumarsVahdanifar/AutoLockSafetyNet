# AutoLock Safety Net

Locks your workstation when **you** — specifically you — stop sitting in front
of it.

Not "when a face disappears". Face recognition decides, so a colleague at your
desk cannot hold your session open, and the countdown only ever resets when
*your* face is recognised.

Windows · macOS · Linux · CLI and desktop GUI · everything runs locally.

```bash
pip install -r requirements.txt
python main.py gui            # or: python main.py enroll --name you && python main.py run
```

---

## What it does

| | naive webcam lockers | AutoLock Safety Net |
|---|---|---|
| detector | Haar cascade, frontal only | YuNet (or InsightFace SCRFD) |
| who counts | anybody with a face | only the enrolled identity |
| head tilted onto a hand | missed | rotated-frame retry (±35°, ±60°) |
| stranger at your desk | holds the session open | locks in 2 s |
| the countdown | resets on any face | resets only on *your* face |
| camera unplugged | often locks you out | never locks while blind |
| starts at login | locks you out at the login screen | disarmed until it recognises you |
| runaway locking | locks you out repeatedly | circuit breaker stops it |

---

## Install

### From source (all platforms)

```bash
git clone https://github.com/KeumarsVahdanifar/AutoLockSafetyNet
cd AutoLockSafetyNet
pip install -r requirements.txt

python main.py models      # ~45 MB of weights into models/
python main.py doctor      # camera, models, backend, identity, lock method
```

Or install it properly and get an `autolock` command anywhere:

```bash
pip install -e ".[gui]"
autolock gui
```

### Prebuilt executables

Grab a bundle from
[Releases](https://github.com/KeumarsVahdanifar/AutoLockSafetyNet/releases) —
no Python needed. They are **unsigned**, so:

* **Windows** — SmartScreen warns: *More info* → *Run anyway*.
* **macOS** — `xattr -dr com.apple.quarantine "AutoLock Safety Net.app"`.
* **Linux** — `chmod +x AutoLockSafetyNet/AutoLockSafetyNet`.

Building from source avoids all three.

### Platform notes

| | lock method | camera | extra setup |
|---|---|---|---|
| **Windows** | `LockWorkStation` | DirectShow | none |
| **macOS** | `CGSession -suspend`, else `pmset displaysleepnow` | AVFoundation | grant **Camera** permission; for `pmset` also enable *require password immediately* |
| **Linux** | first of `loginctl` / `xdg-screensaver` / `gnome-screensaver-command` / `dbus` / `xscreensaver` / `swaylock` | V4L2 | a screen locker must be installed; `sudo apt install python3-tk` for the GUI |

`python main.py doctor` prints exactly which method it found on your machine.

---

## Quick start

```bash
python main.py enroll --name kian    # 1. train it on your face
python main.py test                  # 2. watch the live scores, never locks
python main.py run                   # 3. arm it
```

Or do all of it from the GUI:

```bash
python main.py gui
```

The control panel has a live preview with the countdown, a **Setup** tab for
enrolment, a **Tuning** tab with sliders, and a **Startup** tab that installs
and removes the login entry. **Pause locking** is always one click away.

In the CLI preview window: **ESC** quit, **P** pause, **L** lock now.

---

## 1. Enrolment — training it on your face only

`enroll` walks you through seven head poses and collects several embeddings for
each:

| pose | why it matters |
|---|---|
| front | the baseline |
| left / right | you turn to talk to someone |
| up | leaning back, reading the top of the screen |
| **down** | looking at the keyboard |
| tilt | head resting on a hand |
| natural | typing, glancing around, leaning back |

Every sample is quality-gated (detector score, face size, sharpness, exposure)
and de-duplicated, so sitting still does not fill the template with copies of
one frame. Matching takes the **maximum** similarity across all stored samples
rather than the distance to an average face — an unusual pose is scored against
the enrolled sample that actually resembles it.

```bash
python main.py enroll --name kian                 # guided wizard
python main.py enroll --name kian --append        # add more poses later
python main.py enroll --name kian --from-images photos/
python main.py identities                         # what is enrolled
```

Re-run with `--append` after a haircut, new glasses, or a lighting change. It
is the cheapest fix for "it stopped recognising me".

---

## 2. The countdown

**Recognition is the only thing that resets the clock.** The moment a frame
goes by without your face being recognised, the countdown starts and keeps
falling. A face in shot, a body, movement — none of it stops the clock. At zero
the workstation locks.

Being *seen* is not enough. Being *recognised* is.

### The guarantee

At the shipped defaults, measured against the real state machine at 12 fps:

| what the camera sees | locks after |
|---|---|
| an unrecognised face (clear stranger) | **2.00 s** |
| a face too ambiguous to score either way | **2.92 s** |
| nobody at all | **2.92 s** |

Nothing in frame can outlast `absence_timeout_s`. `tests/test_presence.py` pins
this for both face bands at the real frame rate.

### Buying time for working head-down

Three weaker signals are computed and shown on the preview, each with a
**hold** — the maximum seconds you may go unrecognised while that evidence is
in shot. All three ship at `0`, holding nothing:

| signal | what it sees | config |
|---|---|---|
| tracked | a face in your tracked box that scores too low to confirm | `track_hold_s` |
| body | no face, but MediaPipe still sees your head and shoulders | `body_hold_s` |
| motion | movement where you last were | `motion_hold_s` |

A hold is a **ceiling, not a reset**: with `--body-hold 60`, looking down at
your keyboard gives you at most 60 seconds, and the countdown visibly falls
60 → 0 the whole time. Only being recognised puts it back to full.

```bash
python main.py run --body-hold 60      # up to 60 s head-down (needs mediapipe)
python main.py run                     # strict: 3 s, recognition or nothing
```

---

## 3. Start at login — without locking yourself out

This is the dangerous part of any auto-locker: a monitor that starts before you
are seated will lock the machine you just logged into, over and over.

```bash
python main.py autostart --install     # writes the login entry, prints how to undo it
python main.py autostart --status
python main.py autostart --uninstall
```

| platform | what gets written |
|---|---|
| Windows | `%APPDATA%\...\Startup\AutoLockSafetyNet.cmd` |
| macOS | `~/Library/LaunchAgents/com.autolocksafetynet.plist` |
| Linux | `~/.config/systemd/user/autolocksafetynet.service`, else an XDG `.desktop` |

### Four independent guards

1. **Disarmed until recognised.** It will not lock until it has recognised you
   at least once this session. At login the camera is still warming up and you
   may not be seated — the worst case of this guard is "it never locks", never
   "it locks forever". (`require_initial_recognition`)
2. **Startup grace.** No locking at all for the first 20 s after start, and the
   grace restarts every time the session is unlocked, so you are never locked
   straight back out. (`startup_grace_s`)
3. **Never lock while blind.** A camera that is missing, busy (Zoom, Teams) or
   broken produces no evidence either way. Being blind is not grounds to lock,
   and regaining sight restarts the countdown rather than firing on a stale one.
4. **Circuit breaker.** If 3 locks fire within 120 s, something is wrong — a bad
   template, a camera pointing at a wall. Locking stops for 5 minutes and says
   so in the log. (`max_locks_per_window`, `lock_window_s`, `breaker_pause_s`)

`systemd` is installed with `Restart=on-failure` and a start limit — never
`Restart=always` — so a monitor that keeps dying stays dead instead of being
resurrected into a lock loop. The macOS agent uses `KeepAlive=false` for the
same reason.

### Emergency stop

Works even with no terminal, no GUI, and no keyboard shortcut:

```bash
python main.py pause --on      # stop locking
python main.py pause --off     # resume
```

Or create a file named `PAUSE` in the project folder by any means — a file
manager, an SSH session, another machine. Locking stops until it is deleted.
The GUI's **Pause locking** button does the same thing.

**If you are ever locked out:** log in, and within the 20 s startup grace run
`python main.py pause --on`, or delete the autostart entry printed by
`autostart --status`.

---

## Commands

```bash
python main.py gui                    # desktop control panel
python main.py run  [--timeout 3] [--threshold 0.4] [--no-preview] [--dry-run]
                    [--body-hold 60] [--track-hold 20] [--motion-hold 10]
                    [--no-lock-on-unknown] [--fps 12] [--camera 0]
                    [--backend auto|opencv|insightface]
python main.py test                   # same pipeline, locking disabled
python main.py enroll --name <you> [--append] [--samples 12] [--from-images DIR]
python main.py identities
python main.py autostart [--install|--uninstall|--status] [--arg --body-hold=60]
python main.py pause [--on|--off]
python main.py models [--force]
python main.py doctor
python main.py config [--write] [--set KEY=VALUE] [--unset KEY]
```

`run` is the default, so `python main.py --timeout 5` works.
`python -m autolock ...` and (after `pip install -e .`) `autolock ...` are
equivalent.

---

## Configuration

Every value lives in `config.json`. Change one without editing any source:

```bash
python main.py config --set absence_timeout_s=3
python main.py config --set body_hold_s=60          # also enables the pose model
python main.py config --unset body_hold_s           # back to the built-in default
python main.py config                               # print the effective config
```

Values are type-checked against the field they target, so `--set
absence_timeout_s=abc` is refused rather than silently stored. CLI flags on
`run` override the file for that session without saving.

| key | default | meaning |
|---|---|---|
| `absence_timeout_s` | `3.0` | seconds after the last recognition before locking |
| `match_threshold` | `0.0` | cosine threshold; `0` = the backend's own (SFace `0.363`) |
| `match_margin` | `0.08` | below `threshold - margin` a face is a stranger |
| `confirm_frames` | `2` | consecutive matches before you count as recognised |
| `track_hold_s` / `body_hold_s` / `motion_hold_s` | `0` | ceiling on seconds unrecognised while that evidence is in shot |
| `lock_on_unknown` | `true` | lock when a stranger is at the desk and you are not |
| `unknown_confirm_s` | `2.0` | how long a stranger must persist to count |
| `require_initial_recognition` | `true` | stay disarmed until recognised once |
| `startup_grace_s` | `20.0` | no locking for this long after start or unlock |
| `max_locks_per_window` | `3` | circuit breaker; `0` disables it |
| `target_fps` | `12.0` | processing rate — the main CPU dial |
| `camera_api` | `auto` | per-platform capture backend |
| `dry_run` | `false` | log instead of locking |

### Tuning

| symptom | fix |
|---|---|
| locks while you are working | raise `absence_timeout_s`, or `enroll --append` in that pose |
| locks while you work head-down | `--body-hold 60`, and enrol the "down" pose |
| does not recognise you | lower `match_threshold` (try `0.32`), add samples, improve lighting |
| recognises other people | raise `match_threshold` (try `0.40`) |
| locks on you as an "unrecognised face" | raise `match_margin` (try `0.15`), or `--no-lock-on-unknown` |
| CPU too high | lower `target_fps`, raise `recognize_every` |

`python main.py test` shows the similarity on every box — that number is what
all the thresholds compare against.

---

## Face backends

| backend | needs | notes |
|---|---|---|
| `opencv` *(default)* | `opencv-contrib-python` | YuNet + SFace, CPU-friendly, works everywhere including Python 3.14 |
| `insightface` | `insightface` + `onnxruntime` | SCRFD + ArcFace, better at steep angles; needs a C++ toolchain on Windows, no Python 3.14 wheels yet |

`backend: auto` picks InsightFace when it imports, otherwise OpenCV. Templates
record which backend produced them and are ignored by the other — switching
backends means re-enrolling.

---

## Privacy

Everything runs locally. No image, embedding or event ever leaves the machine —
there is no network code beyond the one-time model download from GitHub and
Google's MediaPipe CDN.

Enrolment stores **embeddings only**: float vectors in `data/identity/*.npz`,
never photographs. `data/`, `models/`, `logs/`, `config.json` and `PAUSE` are
all git-ignored. Delete `data/identity/<name>.npz` to remove an identity
completely.

`logs/autolock.log` records identity names and lock events — clear it before
sharing the folder.

---

## Is it production ready?

For its intended job — a personal convenience lock on your own machine — yes,
and it is built to fail safe. Be clear about what it is:

**It decides when to _lock_, never when to _unlock_.** A false match costs you
a few seconds of screen time; it never grants access. That is why this is a
convenience lock, **not an authentication factor** — it is not a replacement
for your password, and a photograph on a phone would satisfy a 2D face matcher
like this one. It has no liveness detection.

Known limitations:

* one camera, one machine, no multi-monitor or multi-seat awareness;
* macOS lock-state detection needs `pyobjc`, and without it the app relies on
  its cooldown instead of knowing the screen is already locked;
* on Linux it needs a screen locker installed, and Wayland compositors vary;
* the release binaries are unsigned;
* recognition accuracy depends entirely on your enrolment and your lighting.

---

## Development

```bash
python -m unittest discover -s tests -v    # 58 tests, no camera or models needed
ruff check .
pyinstaller packaging/autolock.spec        # standalone bundle into dist/
```

CI runs the suite on Windows, macOS and Linux against Python 3.10 and 3.12.
Tagging `v*` builds and attaches an executable for each platform.

### Layout

```
main.py                      entry point (python -m autolock also works)
autolock/
  cli.py        commands and flags
  gui.py        Tkinter control panel
  config.py     the Config dataclass, paths, --set coercion targets
  app.py        the monitor loop
  presence.py   the countdown state machine   <- the interesting part
  safety.py     arming gate, blindness, circuit breaker, pause switch
  identity.py   templates, matching, quality stats
  enroll.py     the guided capture wizard
  lock.py       per-platform locking and lock-state probes
  autostart.py  login entries for Windows / macOS / Linux
  body.py       MediaPipe pose fallback
  backends/     YuNet+SFace, optional InsightFace, rotation helper
  camera.py     capture with per-platform backends and reconnect backoff
  ui.py         preview overlay
  models.py     weight download and cache
packaging/      PyInstaller spec
tests/          unit tests
```

## License

MIT — see [LICENSE](LICENSE).

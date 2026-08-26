# Presence Lock

Locks your Windows workstation when **you** stop sitting in front of it — not
when a face detector happens to lose your face.

The original version used a Haar cascade: it saw any frontal face, so a
colleague could hold your session open, and it lost *your* face the moment you
looked down at the keyboard, which locked the PC while you were still using it.
This version fixes both ends of that problem.

| | before | now |
|---|---|---|
| detector | Haar cascade, frontal only | YuNet (or InsightFace SCRFD) |
| who counts | anybody with a face | only the enrolled identity |
| head down / turned away | locks on you | held by pose + tracking fallbacks |
| head tilted | missed | rotated-frame retry |
| stranger at your desk | keeps the session open | optional immediate lock |
| config | edit the source | `config.json` + CLI flags |

## Install

```bash
pip install -r requirements.txt
python plock.py models      # ~45 MB of weights into models/
python plock.py doctor      # camera, models, backend, identity, lock API
```

## Quick start

```bash
python plock.py enroll --name kian    # 1. train it on your face
python plock.py test                  # 2. watch the live scores, never locks
python plock.py run                   # 3. arm it
```

In the preview window: **ESC** quit, **P** pause, **L** lock now.

## 1. Enrolment — training it on your face only

`enroll` walks you through seven head poses and collects several embeddings for
each:

| pose | why it matters |
|---|---|
| front | the baseline |
| left / right | you turn to talk to someone |
| up | leaning back, reading the top of the screen |
| **down** | looking at the keyboard — the pose that used to lock your PC |
| tilt | head resting on a hand |
| natural | typing, glancing around, leaning back |

Every sample is quality-gated (detector score, face size, sharpness, exposure)
and de-duplicated, so sitting still does not fill the template with copies of
one frame. Matching then takes the **maximum** similarity across all stored
samples rather than the distance to an average face — an unusual pose is scored
against the enrolled sample that actually resembles it.

```bash
python plock.py enroll --name kian                 # guided wizard
python plock.py enroll --name kian --append        # add more poses later
python plock.py enroll --name kian --samples 12    # denser template
python plock.py enroll --name kian --from-images photos/   # from a folder
python plock.py identities                         # what is enrolled
```

Re-run with `--append` after a haircut, new glasses, or a lighting change at
your desk. It is the cheapest fix for "it stopped recognising me".

## 2. Staying detected with your head down or turned away

Presence is decided from four layers of evidence, strongest first. Each weaker
layer may only **extend** a presence that face recognition already established,
and each one expires on its own clock measured from the last real recognition:

```
face     you were recognised in this frame                    -> resets everything
tracked  a face sits in your tracked box but scores too low   -> 30 s  (track_grace_s)
         to recognise: steep yaw, backlight, motion blur
body     no face at all, but MediaPipe still sees your head   -> 120 s (body_grace_s)
         and shoulders: head down over the keyboard, turned away
motion   nothing but movement in the region you last occupied -> 20 s  (motion_grace_s)
```

So looking down at your keyboard for two minutes keeps the session open; an
empty chair with a moving curtain behind it does not, because the graces run out
and only a real recognition resets them. After a lock, all weak evidence is
discarded — nothing but your face re-arms the monitor.

On top of that, when an upright detector finds nothing the frame is retried
rotated by ±35° and ±60°, which recovers a head tilted onto a hand.

## 3. Locking

- Locks after `absence_timeout_s` (default 3 s) with no evidence.
- `--lock-on-unknown` also locks when an unrecognised face is at the desk and
  you are not — off by default, because a steep pose can score low. The
  `match_margin` band keeps a badly-posed *you* out of the stranger bucket.
- While the session is locked the loop idles at 1 Hz and releases the camera, so
  the webcam LED goes out.
- `--dry-run` logs the decision instead of locking. Use it while tuning.

## Commands

```bash
python plock.py run [--timeout 3] [--threshold 0.4] [--no-preview] [--dry-run]
                    [--no-body] [--no-motion] [--lock-on-unknown] [--fps 12]
                    [--camera 0] [--backend auto|opencv|insightface]
python plock.py test          # same pipeline, locking disabled
python plock.py enroll --name <you>
python plock.py identities
python plock.py models [--force]
python plock.py doctor
python plock.py config [--write]     # dump / create config.json
```

`run` is the default, so `python plock.py --timeout 3` works too.
`python -m presence_lock ...` is equivalent to `python plock.py ...`.

## Configuration

`python plock.py config --write` writes every default to `config.json`; CLI
flags override the file. The knobs you are most likely to touch:

| key | default | meaning |
|---|---|---|
| `absence_timeout_s` | `3.0` | seconds of no evidence before locking |
| `match_threshold` | `0.0` | cosine threshold; `0` = the backend's own (SFace `0.363`) |
| `match_margin` | `0.08` | below `threshold - margin` a face is a stranger |
| `confirm_frames` | `2` | consecutive matches before you count as recognised |
| `track_grace_s` / `body_grace_s` / `motion_grace_s` | `30` / `120` / `20` | how long each weak layer may extend presence |
| `target_fps` | `12.0` | processing rate — the main CPU dial |
| `recognize_every` | `2` | run the embedding model every Nth frame |
| `lock_on_unknown` | `false` | lock when a stranger is at the desk and you are not |
| `rotation_retry` | `true` | retry detection on rotated frames |
| `preview` / `mirror_preview` | `true` | preview window, selfie view |
| `dry_run` | `false` | log instead of locking |

## Tuning

| symptom | fix |
|---|---|
| locks while you are working | raise `absence_timeout_s`, or `enroll --append` in that pose |
| does not recognise you | lower `match_threshold` (try `0.32`), add samples, improve lighting |
| recognises other people | raise `match_threshold` (try `0.40`), re-enrol with better light |
| CPU too high | lower `target_fps`, raise `recognize_every`, `--no-body` |
| stays open with the chair empty | lower `body_grace_s`, or `--no-motion` |

Run `python plock.py test` and watch the similarity printed on each box — that
number is what every threshold is compared against.

## Face backends

| backend | needs | notes |
|---|---|---|
| `opencv` *(default)* | `opencv-contrib-python` only | YuNet + SFace, CPU-friendly, works on Python 3.14 |
| `insightface` | `insightface` + `onnxruntime` | SCRFD + ArcFace, better at steep angles; needs a C++ toolchain on Windows and has no Python 3.14 wheels yet |

`backend: auto` picks InsightFace when it imports, otherwise OpenCV. Templates
record which backend produced them and are ignored by the other one — switching
backends means re-enrolling.

## Run it at login

Task Scheduler → Create Task → *Run only when user is logged on*:

```
Program:   pythonw.exe
Arguments: "C:\Users\<you>\Documents\Presence_Lock\plock.py" run --no-preview
Start in:  C:\Users\<you>\Documents\Presence_Lock
```

`pythonw.exe` plus `--no-preview` runs it with no console and no window. Logs
go to `logs/presence_lock.log`.

## Privacy

Everything runs locally; no image ever leaves the machine. Enrolment stores
**embeddings only** — float vectors in `data/identity/*.npz`, not photos. `data/`
and `models/` are git-ignored. Delete `data/identity/<name>.npz` to remove an
identity.

Treat this as a convenience lock, not an authentication factor: it decides when
to *lock*, never when to unlock, so the worst case of a false match is that the
screen stays on a moment longer.

## Tests

```bash
python -m unittest discover -s tests -v
```

20 tests over the state machine, identity store, rotation geometry, motion
sensor and overlay — no camera required.

## Layout

```
plock.py                      entry point
presence_lock/
  cli.py        commands and flags
  config.py     the Config dataclass and paths
  app.py        the monitor loop, key handling, lock-screen idling
  presence.py   the evidence state machine  <- the interesting part
  identity.py   templates, matching, quality stats
  enroll.py     the guided capture wizard
  body.py       MediaPipe pose fallback
  backends/     YuNet+SFace, optional InsightFace, rotation helper
  camera.py     capture with reconnect backoff
  ui.py         preview overlay
  models.py     weight download and cache
tests/          unit tests
```

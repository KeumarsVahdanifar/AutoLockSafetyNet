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
| head down / turned away | locks on you | recognised through it; opt-in `--body-hold` for the rest |
| head tilted | missed | rotated-frame retry |
| stranger at your desk | keeps the session open | locks in 2 s |
| countdown | resets on any face | resets only on *your* face |
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

## 2. The countdown

**Recognition is the only thing that resets the clock.** The moment a frame goes
by without your face being recognised, the countdown starts and keeps falling —
a face in shot, a body, movement, none of it stops the clock. At zero the
workstation locks.

Being *seen* is not enough. Being *recognised* is.

Detection still works hard to recognise you in awkward poses, and that is where
the enrolled poses earn their keep: the frame is retried rotated by ±35° and
±60° when an upright detector finds nothing, which recovers a head tilted onto a
hand, and a template enrolled with the "down" and "left/right" poses keeps
scoring you while you work.

### Buying time for working head-down

Three weaker signals are still computed and shown on the preview, and each has a
**hold** — the maximum seconds you may go unrecognised while that evidence is in
shot. All three ship at `0`, meaning they hold nothing:

| signal | what it sees | config | default |
|---|---|---|---|
| tracked | a face in your tracked box that scores too low to confirm | `track_hold_s` | `0` |
| body | no face at all, but MediaPipe still sees your head and shoulders | `body_hold_s` | `0` |
| motion | movement in the region you last occupied | `motion_hold_s` | `0` |

A hold is a **ceiling, not a reset**: with `--body-hold 60`, looking down at your
keyboard gives you at most 60 seconds before the lock, and the countdown visibly
falls 60 → 0 the whole time. Only looking back at the camera puts it back to
full.

```bash
python plock.py run --body-hold 60      # up to 60 s head-down, pose model on
python plock.py run                     # strict: 3 s, recognition or nothing
```

A confirmed stranger cancels every hold, and after a lock all weak evidence is
discarded — nothing but your face re-arms the monitor.

## 3. Locking

- Locks `absence_timeout_s` (default 3 s) after the last recognition.
- Locks after `unknown_confirm_s` (default 2 s) when an unrecognised face is at
  the desk and you are not. Seeing you resets that timer, so a colleague
  reading over your shoulder while you sit there is not an intruder.
- A confirmed stranger also cancels every weak layer, so their own body and
  movement cannot hold your session open — that applies even with
  `--no-lock-on-unknown`, where it just means the normal countdown runs.
- The `match_margin` band separates "too low to confirm" from "definitely
  someone else", so a badly-posed *you* is tracked rather than treated as an
  intruder. Widen it if your own steep poses trip the lock.
- While the session is locked the loop idles at 1 Hz and releases the camera, so
  the webcam LED goes out.
- `--dry-run` logs the decision instead of locking. Use it while tuning.

## Commands

```bash
python plock.py run [--timeout 3] [--threshold 0.4] [--no-preview] [--dry-run]
                    [--body-hold 60] [--track-hold 20] [--motion-hold 10]
                    [--no-lock-on-unknown] [--fps 12]
                    [--camera 0] [--backend auto|opencv|insightface]
python plock.py test          # same pipeline, locking disabled
python plock.py enroll --name <you>
python plock.py identities
python plock.py models [--force]
python plock.py doctor
python plock.py config [--write] [--set KEY=VALUE] [--unset KEY]
```

`run` is the default, so `python plock.py --timeout 3` works too.
`python -m presence_lock ...` is equivalent to `python plock.py ...`.

## The guarantee

At the shipped defaults, measured against the real state machine at 12 fps:

| what the camera sees | locks after |
|---|---|
| an unrecognised face (clear stranger) | **2.00 s** — the `unknown_confirm_s` path |
| a face too ambiguous to score either way | **2.92 s** — the countdown runs out |
| nobody at all | **2.92 s** |

Nothing but your recognised face resets the clock, so no face in frame can
outlast `absence_timeout_s`. The 0.08 s shortfall is the `confirm_frames`
debounce landing between frames. `tests/test_presence.py` pins this for both
face bands.

## Configuration

Every value lives in `config.json`. Change one without editing any source:

```bash
python plock.py config --set absence_timeout_s=3     # save a value
python plock.py config --set body_hold_s=60          # also enables the pose model
python plock.py config --unset body_hold_s           # back to the built-in default
python plock.py config                               # print the effective config
python plock.py config --write                       # materialise the whole file
```

Values are type-checked against the field they target, so `--set
absence_timeout_s=abc` is refused rather than silently stored. CLI flags on
`run` override the file for that session without saving.

The knobs you are most likely to touch:

| key | default | meaning |
|---|---|---|
| `absence_timeout_s` | `3.0` | seconds after the last recognition before locking |
| `match_threshold` | `0.0` | cosine threshold; `0` = the backend's own (SFace `0.363`) |
| `match_margin` | `0.08` | below `threshold - margin` a face is a stranger |
| `confirm_frames` | `2` | consecutive matches before you count as recognised |
| `track_hold_s` / `body_hold_s` / `motion_hold_s` | `0` / `0` / `0` | ceiling on seconds unrecognised while that evidence is in shot; `0` holds nothing |
| `evidence_hold_s` | `1.5` | how long a signal counts as active after its last sighting |
| `target_fps` | `12.0` | processing rate — the main CPU dial |
| `recognize_every` | `2` | run the embedding model every Nth frame |
| `lock_on_unknown` | `true` | lock when a stranger is at the desk and you are not |
| `unknown_confirm_s` | `2.0` | how long a stranger must persist to count |
| `rotation_retry` | `true` | retry detection on rotated frames |
| `preview` / `mirror_preview` | `true` | preview window, selfie view |
| `dry_run` | `false` | log instead of locking |

## Tuning

| symptom | fix |
|---|---|
| locks while you are working head-down | `--body-hold 60`, and `enroll --append` in that pose |
| locks while you are working | raise `absence_timeout_s`, or `enroll --append` in that pose |
| does not recognise you | lower `match_threshold` (try `0.32`), add samples, improve lighting |
| recognises other people | raise `match_threshold` (try `0.40`), re-enrol with better light |
| locks on you as an "unrecognised face" | raise `match_margin` (try `0.15`), `enroll --append` in that pose, or `--no-lock-on-unknown` |
| CPU too high | lower `target_fps`, raise `recognize_every`, leave `body_hold_s` at `0` |
| stays open with the chair empty | lower `body_hold_s` / `motion_hold_s` back toward `0` |

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

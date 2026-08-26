"""AutoLock Safety Net — lock the workstation when *you* stop sitting in front of it.

Layered presence detection:
  1. face detection + face recognition (only the enrolled identity counts),
  2. short-lived tracking for faces the recogniser cannot score (extreme pose),
  3. body/pose fallback for a head that is down or turned away,
  4. motion fallback inside the last known region.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]

#!/usr/bin/env python
"""Presence Lock entry point.

    python plock.py models                    download the model weights
    python plock.py enroll --name kian        train it on your face
    python plock.py test                      watch the scores, never locks
    python plock.py run                       arm the monitor
    python plock.py doctor                    check the environment
"""

from presence_lock.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

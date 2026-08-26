#!/usr/bin/env python
"""AutoLock Safety Net entry point.

    python main.py models                    download the model weights
    python main.py enroll --name kian        train it on your face
    python main.py test                      watch the scores, never locks
    python main.py run                       arm the monitor
    python main.py doctor                    check the environment
"""

from autolock.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

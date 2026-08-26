# PyInstaller spec — one bundle per platform.
#
#   pip install pyinstaller
#   pyinstaller packaging/autolock.spec
#
# Model weights are deliberately NOT bundled: they are ~45 MB, they are
# downloaded on first run into models/, and keeping them out means a rebuild
# does not have to re-ship them. The app fetches whatever is missing.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is injected by PyInstaller

hidden = collect_submodules("autolock")
datas = []
binaries = []

# MediaPipe hides its .tflite graphs in package data; without this the pose
# fallback silently fails inside a frozen build.
try:
    import mediapipe  # noqa: F401

    datas += collect_data_files("mediapipe", include_py_files=False)
    hidden += collect_submodules("mediapipe")
except ImportError:
    pass

a = Analysis(  # noqa: F821
    [str(PROJECT / "main.py")],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + ["PIL._tkinter_finder"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoLockSafetyNet",
    debug=False,
    strip=False,
    upx=False,
    # A console is genuinely useful here: the CLI subcommands are the primary
    # interface and `autostart` installs the windowed variant anyway.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=sys.platform == "darwin",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AutoLockSafetyNet",
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name="AutoLock Safety Net.app",
        bundle_identifier="com.autolocksafetynet",
        info_plist={
            # macOS refuses camera access without a stated purpose, and the
            # prompt shows this text verbatim.
            "NSCameraUsageDescription": (
                "AutoLock Safety Net uses the camera to recognise you and lock the "
                "screen when you are not there. Video never leaves this device."
            ),
            "LSUIElement": False,
            "CFBundleShortVersionString": "2.0.0",
        },
    )

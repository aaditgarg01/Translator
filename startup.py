"""Standard-library-only checks before importing native/ML dependencies.

On macOS an iCloud placeholder exists and has a size, but reading it can wait
indefinitely for hydration. Inspect metadata only; never open it to check.
"""

import os
from pathlib import Path
import sys

_DATALESS = 0x40000000  # Darwin SF_DATALESS; absent on Windows/Linux.
_SOURCE_DIRS = ("audio", "stream", "translate", "tts", "utils", "video", "whisper")


def require_local(path, *, recursive=False):
    """Reject cloud placeholders; leave missing-file checks to their consumer."""
    path = Path(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        return
    if getattr(info, "st_flags", 0) & _DATALESS:
        raise RuntimeError(
            f"iCloud has offloaded a required file or folder:\n  {path}\n"
            "Reading it would stall startup. In Finder choose Keep Downloaded and "
            "wait for the download to finish. For a lasting fix, keep the project "
            "outside iCloud Desktop/Documents (for example ~/Developer/ESA_Translator) "
            "and run python3.12 scripts/setup.py there to rebuild .venv. "
            "Do not copy the old .venv. Keep your data/faces folder and models."
        )
    if recursive and path.is_dir():
        with os.scandir(path) as entries:
            for entry in entries:
                require_local(entry.path, recursive=not entry.is_symlink())


def check_runtime(root=None):
    """Check local code/environment without importing any installed packages."""
    if sys.platform != "darwin":
        return
    root = Path(root) if root else Path(__file__).resolve().parent
    # Check OpenCV's bootstrap first: its tiny config files frequently get evicted.
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if sys.prefix != sys.base_prefix:
        packages = Path(sys.prefix) / "lib" / version / "site-packages"
        require_local(packages / "cv2" / "config.py")
        require_local(packages, recursive=True)
    require_local(root / "config.py")
    for directory in _SOURCE_DIRS:
        require_local(root / directory, recursive=True)


def cli_preflight():
    """Give CLI users immediate feedback before any potentially slow imports."""
    print("[Startup] Checking local application and Python environment…", flush=True)
    try:
        check_runtime()
    except (RuntimeError, OSError) as exc:
        print(f"[Startup] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None

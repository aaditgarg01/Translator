"""
Download Piper voice models from the official rhasspy/piper-voices repo on
Hugging Face into  models/piper/.

A Piper voice is two files: <voice>.onnx and <voice>.onnx.json.

Examples:
    python scripts/download_piper_voice.py en_US-amy-medium
    python scripts/download_piper_voice.py hi_IN-pratham-medium

Browse all voices here:
    https://huggingface.co/rhasspy/piper-voices/tree/main
"""

import os
import sys
import urllib.request
import re
import shutil
import ssl
import tempfile

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "piper")


def voice_url(voice: str) -> str:
    """Derive the repo path from the voice key, e.g.
    en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium.onnx
    """
    if not re.fullmatch(r"[a-z]{2,3}_[A-Z]{2}-[A-Za-z0-9_-]+-(?:x_low|low|medium|high)", voice):
        raise ValueError(f"Invalid Piper voice key: {voice!r}")
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(f"Voice key '{voice}' should look like <lang_REGION>-<name>-<quality>.")
    region, quality = parts[0], parts[-1]
    name = "-".join(parts[1:-1])
    lang = region.split("_")[0]
    return f"{BASE}/{lang}/{region}/{name}/{quality}/{voice}.onnx"


def _download(url: str, dest: str) -> None:
    """Stream to a temporary file, then replace only after a complete download."""
    import certifi
    print(f"  Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ESA-Translator"})
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = None
    try:
        with urllib.request.urlopen(req, timeout=60, context=context) as response:
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(dest), delete=False) as output:
                temporary = output.name
                shutil.copyfileobj(response, output)
            expected = response.headers.get("Content-Length")
            if expected is not None and os.path.getsize(temporary) != int(expected):
                raise IOError("Voice download was incomplete. Please retry.")
        os.replace(temporary, dest)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(f"    Saved {dest} ({os.path.getsize(dest):,} bytes)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    failed = False
    for voice in sys.argv[1:]:
        try:
            onnx_url = voice_url(voice)
            print(f"\nDownloading voice '{voice}' ...")
            _download(onnx_url, os.path.join(OUT_DIR, f"{voice}.onnx"))
            _download(onnx_url + ".json", os.path.join(OUT_DIR, f"{voice}.onnx.json"))
            print(f"OK: {voice} ready.")
        except Exception as exc:
            failed = True
            print(f"ERROR: Failed for '{voice}': {exc}\n   Check the name at "
                  f"https://huggingface.co/rhasspy/piper-voices/tree/main")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Download Piper voice models from the official rhasspy/piper-voices repo on
Hugging Face into  models/piper/.

A Piper voice is two files: <voice>.onnx and <voice>.onnx.json.

Examples:
    python scripts/download_piper_voice.py en_US-amy-medium
    python scripts/download_piper_voice.py ja_JP-test-medium
    python scripts/download_piper_voice.py hi_IN-pratham-medium

Browse all voices here:
    https://huggingface.co/rhasspy/piper-voices/tree/main
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "piper")


def voice_url(voice: str) -> str:
    """Derive the repo path from the voice key, e.g.
    en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium.onnx
    """
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(f"Voice key '{voice}' should look like <lang_REGION>-<name>-<quality>.")
    region, quality = parts[0], parts[-1]
    name = "-".join(parts[1:-1])
    lang = region.split("_")[0]
    return f"{BASE}/{lang}/{region}/{name}/{quality}/{voice}.onnx"


def _download(url: str, dest: str) -> None:
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"    → {dest}  ({os.path.getsize(dest):,} bytes)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    for voice in sys.argv[1:]:
        onnx_url = voice_url(voice)
        print(f"\nDownloading voice '{voice}' …")
        try:
            _download(onnx_url, os.path.join(OUT_DIR, f"{voice}.onnx"))
            _download(onnx_url + ".json", os.path.join(OUT_DIR, f"{voice}.onnx.json"))
            print(f"✅ {voice} ready.")
        except Exception as exc:
            print(f"❌ Failed for '{voice}': {exc}\n   Check the name at "
                  f"https://huggingface.co/rhasspy/piper-voices/tree/main")


if __name__ == "__main__":
    main()

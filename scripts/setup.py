"""Create an isolated, platform-native environment without changing system Python."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--japanese", action="store_true", help="Include optional Japanese TTS (Python 3.11/3.12)")
    parser.add_argument("--download-voice", action="store_true", help="Download the default voice for the selected setup")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.11, 3.12, or 3.13 (3.12 recommended).")
    if args.japanese and sys.version_info >= (3, 13):
        parser.error("Use Python 3.11 or 3.12 with --japanese.")
    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    requirements = "requirements-japanese.txt" if args.japanese else "requirements.txt"
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=root)
    subprocess.run([str(python), "-m", "pip", "install", "-r", requirements], check=True, cwd=root)
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=root)
    if args.download_voice:
        script = "download_sbv2_model.py" if args.japanese else "download_piper_voice.py"
        command = [str(python), str(root / "scripts" / script)]
        if not args.japanese:
            command.append("en_US-amy-medium")
        subprocess.run(command, check=True, cwd=root)
    target = "Japanese" if args.japanese else "English"
    print(f'\nReady. Run: "{python}" main.py --doctor --target {target}')
    print(f'Then run: "{python}" main.py --target {target}')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Setup failed (exit {exc.returncode}). Resolve the error above and rerun setup.", file=sys.stderr)
        raise SystemExit(exc.returncode)

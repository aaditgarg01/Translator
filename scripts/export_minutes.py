"""Rebuild printable meeting minutes from an autosaved journal after interruption."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meeting.recorder import recover


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        print(recover(args.journal, args.output_dir))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Minutes export failed: {exc}\n')

"""Generate a WAV (and optionally play it) without ASR, translation, or a camera."""

import argparse
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from tts.voices import SAMPLE_TEXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=list(config.PIPER_VOICES), default=config.TARGET_LANGUAGE)
    parser.add_argument('--text', help='Native-language sentence to synthesize')
    parser.add_argument('--output', type=Path, required=True, help='Destination WAV file')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--output-device', help='Speaker index or name, used with --play')
    args = parser.parse_args()
    from tts.piper_backend import PiperBackend
    backend = PiperBackend()
    backend.preload(args.language)
    started = time.perf_counter()
    pcm, rate = backend.synth(args.text or SAMPLE_TEXTS[args.language], args.language)
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)
    print(f'Saved {args.output.resolve()}: {len(pcm) / (2 * rate):.2f}s audio; synthesis {elapsed:.2f}s')
    if args.play:
        import numpy as np
        import sounddevice as sd
        device = args.output_device
        if device is not None and device.isdecimal():
            device = int(device)
        sd.play(np.frombuffer(pcm, dtype='<i2'), samplerate=rate, device=device, blocking=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Voice test failed: {exc}', file=sys.stderr)
        raise SystemExit(1)

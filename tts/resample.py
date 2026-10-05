"""
Sample-rate conversion.

Different TTS engines emit different rates (Piper 22 050 Hz, Style-Bert-VITS2
44 100 Hz), but the streamer opens its speaker / FFmpeg sink ONCE at a single
rate.  So every engine's PCM is resampled to OUTPUT_SAMPLE_RATE here before it
reaches the queue.

Quality ladder (best available wins): soxr → scipy → plain numpy linear.
All operate on raw int16 mono PCM bytes and return int16 mono PCM bytes.
"""

import numpy as np

try:
    import soxr
    _HAVE_SOXR = True
except Exception:
    _HAVE_SOXR = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample int16 mono PCM bytes from src_rate to dst_rate."""
    if not pcm or src_rate == dst_rate:
        return pcm

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    if _HAVE_SOXR:
        y = soxr.resample(x, src_rate, dst_rate)
    elif _HAVE_SCIPY:
        g = gcd(src_rate, dst_rate)
        y = resample_poly(x, dst_rate // g, src_rate // g)
    else:
        # plain linear interpolation — fine for speech
        n_out = int(round(len(x) * dst_rate / src_rate))
        if n_out <= 0:
            return b""
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        fp = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(fp, xp, x)

    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16).tobytes()

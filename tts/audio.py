"""Join speech sentences with consistent volume and click-free boundaries."""

import numpy as np


def join_sentences(parts, sample_rate, pause_ms=150):
    """Normalize once for the utterance, preserving relative sentence loudness."""
    if sample_rate <= 0 or pause_ms < 0:
        raise ValueError('Sample rate must be positive and sentence pause nonnegative')
    joined = []
    for part in parts:
        audio = np.asarray(part, dtype=np.float32).reshape(-1).copy()
        if not len(audio):
            continue
        if not np.isfinite(audio).all():
            raise ValueError('Speech model generated invalid audio samples')
        # A 3 ms ramp avoids a discontinuity when a model emits a nonzero edge.
        fade = min(round(sample_rate * 0.003), len(audio) // 2)
        if fade:
            audio[:fade] *= np.linspace(0, 1, fade)
            audio[-fade:] *= np.linspace(1, 0, fade)
        if joined and pause_ms:
            joined.append(np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32))
        joined.append(audio)
    if not joined:
        raise RuntimeError('Voice generated no speech. Check its pronunciation dependencies and input text.')
    audio = np.concatenate(joined)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-7:
        raise RuntimeError('Voice generated silent audio. Check the voice model and input language.')
    audio *= 0.95 / peak  # Headroom for resampling; avoid sentence-by-sentence pumping.
    return (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()

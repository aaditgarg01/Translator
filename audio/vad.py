"""
Voice Activity Detection.

Primary backend is Silero VAD (a tiny, fast neural model) which is far more
robust to background noise than a plain energy gate.  If Silero (or torch)
isn't available we transparently fall back to an RMS-energy detector so the
audio milestone can still be demoed.

The public object is a *segmenter*: you feed it fixed-size frames and it tells
you when a complete utterance has ended, handing back the buffered float32
audio ready for Whisper.
"""

import numpy as np

from config import (
    AUDIO_SAMPLE_RATE,
    VAD_FRAME_SIZE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_MAX_SPEECH_S,
    ENERGY_VAD_THRESHOLD,
)
from utils.logging_utils import get_logger

log = get_logger("VAD")


class SpeechSegmenter:
    """Accumulates frames and emits complete speech segments.

    Usage:
        seg = SpeechSegmenter(backend="silero")
        for frame in frames_of_512_samples:
            result = seg.process(frame)   # -> np.ndarray | None
            if result is not None:
                whisper(result)
    """

    def __init__(self, backend: str = "silero"):
        self.sr = AUDIO_SAMPLE_RATE
        self.frame_size = VAD_FRAME_SIZE
        self.frame_ms = 1000.0 * self.frame_size / self.sr

        self._silero = None
        self.backend = backend
        if backend == "silero":
            self._silero = self._load_silero()
            if self._silero is None:
                log.warning("Silero unavailable — falling back to energy VAD.")
                self.backend = "energy"

        # buffering state
        self._buf: list[np.ndarray] = []
        self._pad: list[np.ndarray] = []          # rolling pre-speech padding
        self._pad_frames = max(1, int(VAD_SPEECH_PAD_MS / self.frame_ms))
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    @property
    def speaking(self) -> bool:
        """True while an utterance is in progress (read by the overlay)."""
        return self._speaking

    # ── backend loading ──────────────────────────────────────────────
    def _load_silero(self):
        try:
            from silero_vad import load_silero_vad
            import torch  # noqa: F401  (ensures torch is present)
            model = load_silero_vad()
            log.info("Silero VAD loaded.")
            return model
        except Exception as exc:
            log.warning("Could not load silero-vad (%s); using energy VAD.", exc)
            return None

    # ── per-frame probability ────────────────────────────────────────
    def _speech_prob(self, frame: np.ndarray) -> float:
        if self.backend == "silero":
            import torch
            with torch.no_grad():
                t = torch.from_numpy(frame)
                return float(self._silero(t, self.sr).item())
        # energy backend → pseudo-probability
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return 1.0 if rms > ENERGY_VAD_THRESHOLD else 0.0

    # ── main entry ───────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame (float32, VAD_FRAME_SIZE samples).

        Returns a finished utterance as a float32 array, or None if the
        current utterance is still in progress (or there is just silence).
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        prob = self._speech_prob(frame)
        is_voice = prob >= VAD_THRESHOLD

        if is_voice:
            if not self._speaking:
                self._speaking = True
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._buf = list(self._pad)        # include pre-speech padding
                log.info("Speech detected")
            self._buf.append(frame)
            self._speech_ms += self.frame_ms
            self._silence_ms = 0.0
        else:
            if self._speaking:
                self._buf.append(frame)            # keep trailing silence
                self._silence_ms += self.frame_ms

        self._pad.append(frame)
        if len(self._pad) > self._pad_frames:
            self._pad.pop(0)

        if not self._speaking:
            return None

        ended = (
            self._silence_ms >= VAD_MIN_SILENCE_MS
            or len(self._buf) * self.frame_ms >= VAD_MAX_SPEECH_S * 1000.0
        )
        if not ended:
            return None

        # finalise utterance
        segment = np.concatenate(self._buf) if self._buf else None
        speech_ms = self._speech_ms
        self._reset()

        if segment is None or speech_ms < VAD_MIN_SPEECH_MS:
            return None
        return segment

    def _reset(self) -> None:
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._buf = []

    def reset(self) -> None:
        """Public reset — drop any in-progress utterance and clear VAD state.

        Used to discard whatever the mic captured while it was gated during TTS
        playback, so echo never becomes a (partial) utterance.
        """
        self._reset()
        self._pad = []
        if self.backend == "silero" and self._silero is not None:
            try:
                self._silero.reset_states()
            except Exception:
                pass

"""
Thread 2 — Audio.

Reads the microphone at 16 kHz mono, runs frames through the VAD segmenter,
and drops complete speech segments onto the audio queue.  It also flips the
shared ``speech_active`` flag so the video overlay can show who is talking.

This thread does nothing else — no recognition, no translation.  It just turns
sound into "here is a finished utterance".
"""

import threading
import numpy as np
import sounddevice as sd

from config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, VAD_FRAME_SIZE, VAD_BACKEND
from audio.vad import SpeechSegmenter
from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger("Audio")


class AudioCapture:
    """Owns one thread: mic → VAD → audio_queue."""

    def __init__(self, audio_queue, state):
        self.audio_queue = audio_queue
        self.state = state
        self.segmenter = SpeechSegmenter(backend=VAD_BACKEND)

        self._stream: sd.InputStream | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        def _callback(indata, _frames, _time, status):
            if status:
                log.debug("stream status: %s", status)
            self._handle_frame(indata[:, 0].copy())

        try:
            self._stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                blocksize=VAD_FRAME_SIZE,
                dtype="float32",
                callback=_callback,
            )
            self._stream.start()
            log.info("Microphone open @ %d Hz (%s VAD)", AUDIO_SAMPLE_RATE, self.segmenter.backend)
            while self._running:
                sd.sleep(100)
        except Exception as exc:
            self.error = str(exc)
            log.error("Microphone error: %s", exc)
        finally:
            self._running = False

    def _handle_frame(self, frame: np.ndarray) -> None:
        # Half-duplex: while our own TTS is playing (and a short tail after),
        # ignore the mic so we don't transcribe the speaker output (echo loop).
        if self.state.mic_muted:
            if self.segmenter.speaking:
                self.segmenter.reset()
            self.state.set_speech_active(False)
            return

        segment = self.segmenter.process(frame)
        # mirror VAD state for the overlay
        self.state.set_speech_active(self.segmenter.speaking)
        if segment is not None:
            dur = len(segment) / AUDIO_SAMPLE_RATE
            log.info("Utterance ready (%.1fs)", dur)
            put_drop_oldest(self.audio_queue, segment)

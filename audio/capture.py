"""Microphone capture with VAD processing outside PortAudio's callback."""

import queue
import threading

import numpy as np
import sounddevice as sd

from config import (
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_INPUT_DEVICE,
    VAD_FRAME_SIZE, VAD_BACKEND,
)
from audio.vad import SpeechSegmenter
from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger("Audio")


class AudioCapture:
    def __init__(self, audio_queue, state):
        self.audio_queue = audio_queue
        self.state = state
        self.segmenter = SpeechSegmenter(backend=VAD_BACKEND)
        self._frames = queue.Queue(maxsize=20)
        self._discontinuity = threading.Event()
        self._stream = None
        self._running = False
        self._thread = None
        self.error = None

    def _callback(self, indata, _frames, _time, status):
        # No model inference or logging on the real-time audio thread.
        if status:
            self._discontinuity.set()
        item = (indata[:, 0].copy(), self.state.mic_muted)
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            self._discontinuity.set()

    def start(self):
        if self._running:
            return
        self.error = None
        try:
            self._stream = sd.InputStream(
                device=AUDIO_INPUT_DEVICE, samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS, blocksize=VAD_FRAME_SIZE,
                dtype="float32", callback=self._callback,
            )
            self._stream.start()
            self._running = True
            self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.stop()
            raise RuntimeError(
                f"Cannot start microphone: {exc}. Check microphone privacy permissions "
                "and use --list-devices / --input-device to select a microphone."
            ) from exc
        log.info("Microphone open @ %d Hz (%s VAD)", AUDIO_SAMPLE_RATE, self.segmenter.backend)

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as exc:
                log.warning("Microphone close failed: %s", exc)
            finally:
                self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.state.set_speech_active(False)

    def _run(self):
        try:
            while self._running:
                try:
                    frame, captured_muted = self._frames.get(timeout=0.2)
                except queue.Empty:
                    if not self._stream.active:
                        raise RuntimeError("Microphone stopped or was disconnected.")
                    continue
                if self._discontinuity.is_set():
                    self._discontinuity.clear()
                    self.segmenter.reset()
                    self.state.set_speech_active(False)
                    # Discard stale audio after an overflow; keep latency bounded.
                    while True:
                        try:
                            self._frames.get_nowait()
                        except queue.Empty:
                            break
                    continue
                self._handle_frame(frame, captured_muted)
        except Exception as exc:
            if self._running:
                self.error = str(exc)
                log.error("Microphone processing failed: %s", exc)
        finally:
            self._running = False
            self.state.set_speech_active(False)

    def _handle_frame(self, frame: np.ndarray, captured_muted=False):
        # Remember mute state at capture time as well as processing time.
        if captured_muted or self.state.mic_muted:
            self.segmenter.reset()
            self.state.set_speech_active(False)
            return
        segment = self.segmenter.process(frame)
        self.state.set_speech_active(self.segmenter.speaking)
        if segment is not None:
            log.info("Utterance ready (%.1fs)", len(segment) / AUDIO_SAMPLE_RATE)
            put_drop_oldest(self.audio_queue, segment)

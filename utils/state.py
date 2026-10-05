"""
Shared, thread-safe state that does NOT belong on a queue.

Queues carry data that flows one way through the pipeline.  But a few facts are
"latest value wins" and read by several threads at once:

* is someone speaking right now?      (set by Audio, read by Video overlay)
* who is the active speaker?           (set by Video tracking + Whisper)
* what is the latest caption?          (set by Translate, drawn by Video)

A single lock-guarded object is the simplest correct way to share those.
"""

import threading
import time
from dataclasses import dataclass


@dataclass
class Caption:
    original: str = ""
    translated: str = ""
    speaker: str = ""
    ts: float = 0.0


class SharedState:
    """Latest-value cross-thread state, all access guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speech_active = False
        self._current_speaker = ""
        self._caption = Caption()
        self._fps = 0.0
        self._output_active = False       # TTS audio currently playing?
        self._mute_until = 0.0            # ignore mic until this time (echo tail)

    # ── speech / speaker ─────────────────────────────────────────────
    def set_speech_active(self, active: bool) -> None:
        with self._lock:
            self._speech_active = active

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def set_current_speaker(self, name: str) -> None:
        with self._lock:
            self._current_speaker = name

    @property
    def current_speaker(self) -> str:
        with self._lock:
            return self._current_speaker

    # ── caption ──────────────────────────────────────────────────────
    def set_caption(self, original: str, translated: str, speaker: str = "") -> None:
        with self._lock:
            self._caption = Caption(original, translated, speaker, time.time())

    def get_caption(self) -> Caption:
        with self._lock:
            return self._caption

    # ── microphone gating (echo / feedback prevention) ───────────────
    def set_output_active(self, active: bool) -> None:
        """Marked True while TTS audio is being played through the speaker."""
        with self._lock:
            self._output_active = active

    def mute_mic_for(self, seconds: float) -> None:
        """Keep the mic gated for a short tail after playback ends."""
        with self._lock:
            self._mute_until = time.time() + seconds

    @property
    def mic_muted(self) -> bool:
        """True while TTS is playing, or during the post-playback echo tail."""
        with self._lock:
            return self._output_active or time.time() < self._mute_until

    # ── fps (for the overlay) ────────────────────────────────────────
    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

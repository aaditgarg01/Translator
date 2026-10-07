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
from collections import Counter


@dataclass
class Caption:
    original: str = ""
    translated: str = ""
    speaker: str = ""
    ts: float = 0.0
    track_id: int | None = None


@dataclass
class AudioSegment:
    audio: object
    speaker: str = ""
    track_id: int | None = None


class SharedState:
    """Latest-value cross-thread state, all access guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speech_active = False
        self._current_speaker = ""
        self._speaker_votes = Counter()
        self._speaker_observations = 0
        self._caption = Caption()
        self._captions = {}
        self._fps = 0.0
        self._output_active = False       # TTS audio currently playing?
        self._mute_until = 0.0            # ignore mic until this time (echo tail)

    # ── speech / speaker ─────────────────────────────────────────────
    def set_speech_active(self, active: bool) -> None:
        with self._lock:
            if active and not self._speech_active:
                self._speaker_votes.clear()
                self._speaker_observations = 0
            self._speech_active = active
            if not active:
                self._current_speaker = ""

    def observe_speaker(self, name, track_id):
        with self._lock:
            self._current_speaker = name if self._speech_active else ""
            if self._speech_active:
                self._speaker_observations += 1
                if track_id is not None:
                    self._speaker_votes[(name, track_id)] += 1

    def take_utterance_speaker(self):
        """Freeze attribution before ASR/translation latency; uncertain stays blank."""
        with self._lock:
            result = ("", None)
            if self._speaker_votes:
                key, count = self._speaker_votes.most_common(1)[0]
                if count >= 2 and count >= self._speaker_observations * 0.6:
                    result = key
            self._speaker_votes.clear()
            self._speaker_observations = 0
            return result

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
    def set_caption(self, original: str, translated: str, speaker: str = "", track_id=None) -> None:
        with self._lock:
            self._caption = Caption(original, translated, speaker, time.time(), track_id)
            self._captions[track_id] = self._caption
            while len(self._captions) > 10:
                oldest = min(self._captions, key=lambda key: self._captions[key].ts)
                del self._captions[oldest]

    def get_captions(self, max_age=8):
        with self._lock:
            now = time.time()
            return [c for c in self._captions.values() if now - c.ts <= max_age]

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

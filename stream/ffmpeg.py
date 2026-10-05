"""
Thread 6 — Audio output / RTP.

Consumes raw PCM and sends it somewhere you can hear it.  Two sinks, chosen by
STREAM_MODE:

* "local" → default speaker via sounddevice (most reliable for a demo)
* "rtp"   → an FFmpeg process that encodes and streams RTP for VLC / OBS
* "both"  → both at once

Critically, the sinks are opened ONCE and reused forever.  We never restart
FFmpeg per sentence — the process stays alive and we keep writing PCM to its
stdin:

    pcm_queue  →  Stream  →  speaker  and/or  ffmpeg → rtp://host:port
"""

import os
import queue
import threading
import subprocess

import numpy as np

from config import (
    STREAM_MODE,
    RTP_HOST,
    RTP_PORT,
    RTP_CODEC,
    FFMPEG_BIN,
    SDP_OUTPUT,
    AUDIO_CHANNELS,
    MIC_MUTE_COOLDOWN_S,
)
from utils.logging_utils import get_logger

log = get_logger("Stream")


class AudioStreamer:
    """Owns one thread: pcm_queue → speaker / FFmpeg-RTP."""

    def __init__(self, pcm_queue, sample_rate: int, state=None):
        self.pcm_queue = pcm_queue
        self.sample_rate = sample_rate
        self.state = state
        self.mode = STREAM_MODE

        self._speaker = None
        self._ffmpeg: subprocess.Popen | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ── sinks ────────────────────────────────────────────────────────
    def _open_speaker(self) -> None:
        import sounddevice as sd
        self._speaker = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=AUDIO_CHANNELS,
            dtype="int16",
        )
        self._speaker.start()
        log.info("Local speaker open @ %d Hz.", self.sample_rate)

    def _open_ffmpeg(self) -> None:
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", str(AUDIO_CHANNELS),
            "-i", "pipe:0",
            "-acodec", RTP_CODEC,
            "-f", "rtp", f"rtp://{RTP_HOST}:{RTP_PORT}",
            "-sdp_file", SDP_OUTPUT,
        ]
        try:
            self._ffmpeg = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log.info("FFmpeg RTP → rtp://%s:%d  (SDP: %s)", RTP_HOST, RTP_PORT, SDP_OUTPUT)
            log.info("Open it in VLC:  vlc %s", SDP_OUTPUT)
        except FileNotFoundError:
            log.error("FFmpeg not found ('%s'). RTP disabled — install FFmpeg or set FFMPEG_BIN.", FFMPEG_BIN)
            self._ffmpeg = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        if self.mode in ("local", "both"):
            try:
                self._open_speaker()
            except Exception as exc:
                log.error("Could not open speaker: %s", exc)
        if self.mode in ("rtp", "both"):
            self._open_ffmpeg()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._speaker is not None:
            try:
                self._speaker.stop(); self._speaker.close()
            except Exception:
                pass
        if self._ffmpeg is not None:
            try:
                self._ffmpeg.stdin.close()
                self._ffmpeg.terminate()
            except Exception:
                pass

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        playing = False
        while self._running:
            try:
                pcm = self.pcm_queue.get(timeout=0.3)
            except queue.Empty:
                # nothing left to play → release the mic after a short echo tail
                if playing and self.state is not None:
                    self.state.set_output_active(False)
                    self.state.mute_mic_for(MIC_MUTE_COOLDOWN_S)
                    playing = False
                continue
            if not pcm:
                continue
            # gate the mic while we drive audio to the speaker (echo guard)
            if self.state is not None:
                self.state.set_output_active(True)
                playing = True
            self._write(pcm)

    def _write(self, pcm: bytes) -> None:
        if self._speaker is not None:
            try:
                self._speaker.write(pcm)
            except Exception as exc:
                log.error("Speaker write failed: %s", exc)
        if self._ffmpeg is not None and self._ffmpeg.stdin is not None:
            try:
                self._ffmpeg.stdin.write(pcm)
                self._ffmpeg.stdin.flush()
            except Exception as exc:
                log.error("FFmpeg write failed: %s", exc)
                self._ffmpeg = None
        log.debug("Sent %d bytes", len(pcm))

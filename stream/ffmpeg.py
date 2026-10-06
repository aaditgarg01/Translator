"""Local speaker and/or FFmpeg RTP output with bounded shutdown."""

import queue
import shutil
import subprocess
import tempfile
import threading

from config import (
    STREAM_MODE, RTP_HOST, RTP_PORT, RTP_CODEC, FFMPEG_BIN, SDP_OUTPUT,
    AUDIO_CHANNELS, AUDIO_OUTPUT_DEVICE, MIC_MUTE_COOLDOWN_S,
)
from utils.logging_utils import get_logger

log = get_logger("Stream")


class AudioStreamer:
    def __init__(self, pcm_queue, sample_rate: int, state=None):
        self.pcm_queue = pcm_queue
        self.sample_rate = sample_rate
        self.state = state
        self.mode = STREAM_MODE
        self._speaker = None
        self._ffmpeg = None
        self._ffmpeg_log = None
        self._running = False
        self._thread = None
        self._stopped = threading.Event()
        self.error = None

    def _open_speaker(self):
        import sounddevice as sd
        self._speaker = sd.RawOutputStream(
            device=AUDIO_OUTPUT_DEVICE, samplerate=self.sample_rate,
            channels=AUDIO_CHANNELS, dtype="int16",
        )
        try:
            self._speaker.start()
        except Exception:
            self._speaker.close()
            self._speaker = None
            raise
        log.info("Local speaker open @ %d Hz.", self.sample_rate)

    def _open_ffmpeg(self):
        if not shutil.which(FFMPEG_BIN):
            raise RuntimeError(f"FFmpeg not found: {FFMPEG_BIN}. Install FFmpeg or use --stream local.")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", str(AUDIO_CHANNELS),
            "-i", "pipe:0", "-acodec", RTP_CODEC,
            "-sdp_file", SDP_OUTPUT, "-f", "rtp", f"rtp://{RTP_HOST}:{RTP_PORT}",
        ]
        self._ffmpeg_log = tempfile.TemporaryFile()
        try:
            self._ffmpeg = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._ffmpeg_log, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._ffmpeg_log.close()
            self._ffmpeg_log = None
            raise
        log.info("FFmpeg RTP → rtp://%s:%d (SDP: %s)", RTP_HOST, RTP_PORT, SDP_OUTPUT)

    def start(self):
        if self._running:
            return
        if self.mode not in ("local", "rtp", "both"):
            raise ValueError(f"Invalid STREAM_MODE: {self.mode!r}")
        self.error = None
        self._stopped.clear()
        failures = []
        for enabled, opener in (
            (self.mode in ("local", "both"), self._open_speaker),
            (self.mode in ("rtp", "both"), self._open_ffmpeg),
        ):
            if enabled:
                try:
                    opener()
                except Exception as exc:
                    failures.append(str(exc))
                    log.warning("Audio output unavailable: %s", exc)
        if self._speaker is None and self._ffmpeg is None:
            self.stop()
            raise RuntimeError("No audio output is available. " + "; ".join(failures))
        self._running = True
        self._thread = threading.Thread(target=self._run, name="stream", daemon=True)
        self._thread.start()

    def _close_ffmpeg(self):
        proc, self._ffmpeg = self._ffmpeg, None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            if proc.stdin is not None:
                proc.stdin.close()
        if self._ffmpeg_log is not None:
            self._ffmpeg_log.close()
            self._ffmpeg_log = None

    def stop(self):
        self._running = False
        self._stopped.set()
        # Interrupt potentially blocked pipe/speaker writes before joining.
        proc = self._ffmpeg
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass  # It may have exited between poll() and terminate().
        if self._speaker is not None:
            try:
                self._speaker.abort()
            except Exception as exc:
                log.warning("Speaker abort failed: %s", exc)
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._close_ffmpeg()
        if self._speaker is not None:
            self._speaker.close()
            self._speaker = None
        if self.state is not None:
            self.state.set_output_active(False)

    def _run(self):
        try:
            while self._running:
                try:
                    pcm = self.pcm_queue.get(timeout=0.2)
                except queue.Empty:
                    self._check_ffmpeg()
                    continue
                if not pcm:
                    continue
                if self.state is not None:
                    self.state.set_output_active(True)
                try:
                    # Small chunks allow prompt shutdown and keep both sinks aligned.
                    block = max(2, int(self.sample_rate * 0.02) * 2 * AUDIO_CHANNELS)
                    for offset in range(0, len(pcm), block):
                        if not self._running:
                            break
                        self._write(pcm[offset:offset + block])
                    if self._speaker is not None and self._running:
                        self._stopped.wait(float(self._speaker.latency))
                finally:
                    if self.state is not None:
                        self.state.mute_mic_for(MIC_MUTE_COOLDOWN_S)
                        self.state.set_output_active(False)
        except Exception as exc:
            if self._running:
                self.error = str(exc)
                log.error("Audio output failed: %s", exc)
        finally:
            self._running = False

    def _check_ffmpeg(self):
        if self._ffmpeg is not None and self._ffmpeg.poll() is not None:
            detail = ""
            if self._ffmpeg_log is not None:
                self._ffmpeg_log.seek(0)
                detail = self._ffmpeg_log.read(4096).decode("utf-8", errors="replace").strip()
            self._close_ffmpeg()
            if self._speaker is None:
                raise RuntimeError(f"FFmpeg exited. {detail}")
            log.warning("RTP stopped; local playback continues. %s", detail)

    def _write(self, pcm):
        self._check_ffmpeg()
        if self._ffmpeg is not None:
            try:
                remaining = memoryview(pcm)
                while remaining:
                    written = self._ffmpeg.stdin.write(remaining)
                    if not written:
                        raise BrokenPipeError("FFmpeg pipe closed")
                    remaining = remaining[written:]
            except (BrokenPipeError, OSError) as exc:
                self._close_ffmpeg()
                if self._speaker is None:
                    raise RuntimeError(f"RTP output failed: {exc}") from exc
                log.warning("RTP output failed; local playback continues: %s", exc)
        if self._speaker is not None:
            try:
                self._speaker.write(pcm)
                return
            except Exception as exc:
                self._speaker.close()
                self._speaker = None
                if self._ffmpeg is None:
                    raise RuntimeError(f"Speaker output failed: {exc}") from exc
                log.warning("Speaker failed; RTP continues: %s", exc)
        # RTP-only writes must not finish the echo guard before audio plays.
        self._stopped.wait(len(pcm) / (2 * AUDIO_CHANNELS * self.sample_rate))

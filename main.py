"""
ESA project Real-Time Translator — entry point.

The main thread does almost nothing: it builds the queues and shared state,
starts each subsystem's thread, then drives the OpenCV display window.  All the
real work happens in six independent threads connected by queues:

    Video    : camera → face recognition → overlay
    Audio    : mic → Silero VAD → audio_q
    Whisper  : audio_q → transcript_q
    Translate: transcript_q → translation_q
    TTS      : translation_q → pcm_q   (Piper)
    Stream   : pcm_q → speaker / RTP

Each subsystem can be toggled in config.py (ENABLE_*) so you can bring up one
milestone at a time.  Press 'q' or ESC in the window (or Ctrl-C) to quit.
"""

import time
import signal

import config
from utils.queues import Pipeline
from utils.state import SharedState
from utils.logging_utils import get_logger

log = get_logger("Main")


class App:
    def __init__(self) -> None:
        self.pipe = Pipeline()
        self.state = SharedState()
        self.components = []          # started, in start order (stopped in reverse)
        self.video = None
        self._running = False

    # ── build ────────────────────────────────────────────────────────
    def build(self) -> None:
        pipe, state = self.pipe, self.state

        # Consumers first, producers last, so nothing overflows at startup.
        tts = None
        if config.ENABLE_TTS:
            from tts.engine import TTS
            tts = TTS(pipe.translation, pipe.pcm, state)

        if config.ENABLE_STREAM and tts is not None:
            from stream.ffmpeg import AudioStreamer
            # All engines resample to a fixed rate, so the streamer opens once.
            streamer = AudioStreamer(pipe.pcm, sample_rate=tts.sample_rate, state=state)
            self.components.append(("Stream", streamer))

        if tts is not None:
            self.components.append(("TTS", tts))

        if config.ENABLE_TRANSLATE:
            from translate.translator import Translator
            self.components.append(("Translate", Translator(pipe.transcript, pipe.translation, state)))

        if config.ENABLE_WHISPER:
            from whisper.engine import WhisperEngine
            self.components.append(("Whisper", WhisperEngine(pipe.audio, pipe.transcript, state)))

        if config.ENABLE_AUDIO:
            from audio.capture import AudioCapture
            self.components.append(("Audio", AudioCapture(pipe.audio, state)))

        if config.ENABLE_VIDEO:
            from video.camera import VideoCapture
            self.video = VideoCapture(state, config.CAMERA_INDEX)

    # ── run ──────────────────────────────────────────────────────────
    def start(self) -> None:
        log.info("Starting subsystems (target language: %s)…", config.TARGET_LANGUAGE)
        for name, comp in self.components:
            log.info("→ starting %s", name)
            comp.start()
        if self.video is not None:
            if not self.video.start():
                log.error("Video failed to start: %s", self.video.error)
                self.video = None
        self._running = True
        log.info("All subsystems running. Press 'q'/ESC in the window or Ctrl-C to quit.")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info("Shutting down…")
        if self.video is not None:
            self.video.stop()
        for name, comp in reversed(self.components):
            try:
                comp.stop()
            except Exception as exc:
                log.error("Error stopping %s: %s", name, exc)
        log.info("Bye.")

    # ── main-thread display loop ─────────────────────────────────────
    def loop(self) -> None:
        if self.video is None or not config.SHOW_WINDOW:
            # Headless: just idle until interrupted.
            while self._running:
                time.sleep(0.2)
            return

        import cv2
        while self._running:
            frame = self.video.get_frame()
            if frame is not None:
                cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):     # q or ESC
                break
        cv2.destroyAllWindows()


def main() -> None:
    app = App()

    def _sigint(_sig, _frm):
        app._running = False
    signal.signal(signal.SIGINT, _sigint)

    app.build()
    app.start()
    try:
        app.loop()
    finally:
        app.stop()


if __name__ == "__main__":
    main()

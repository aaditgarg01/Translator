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

import argparse
import os
import queue
import sys
import time

if __name__ == "__main__" and not any(a in sys.argv for a in ("-h", "--help")):
    from startup import cli_preflight
    cli_preflight()

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
        self.scene_queue = queue.Queue(maxsize=8)
        self._running = False
        self._started = []

    # ── build ────────────────────────────────────────────────────────
    def build(self) -> None:
        from utils.diagnostics import validate_config
        validate_config()
        pipe, state = self.pipe, self.state

        # Consumers first, producers last, so nothing overflows at startup.
        tts = None
        if config.ENABLE_TTS:
            log.info("Loading speech synthesis libraries…")
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
            log.info("Loading translation libraries…")
            from translate.translator import Translator
            self.components.append(("Translate", Translator(pipe.transcript, pipe.translation, state, self.scene_queue)))

        if config.ENABLE_WHISPER:
            log.info("Loading speech recognition libraries…")
            from whisper.engine import WhisperEngine
            self.components.append(("Whisper", WhisperEngine(pipe.audio, pipe.transcript, state)))

        if config.ENABLE_AUDIO:
            log.info("Loading microphone libraries…")
            from audio.capture import AudioCapture
            self.components.append(("Audio", AudioCapture(pipe.audio, state)))

        if config.ENABLE_VIDEO:
            log.info("Loading OpenCV and registered faces…")
            from video.camera import VideoCapture
            self.video = VideoCapture(state, config.CAMERA_INDEX, self.scene_queue if config.ENABLE_TRANSLATE else None)

    # ── run ──────────────────────────────────────────────────────────
    def start(self) -> None:
        log.info("Starting subsystems (target language: %s)…", config.TARGET_LANGUAGE)
        for name, comp in self.components:
            log.info("→ starting %s", name)
            # Track before start so partially opened resources are cleaned up.
            self._started.append((name, comp))
            comp.start()
        if self.video is not None:
            if not self.video.start():
                log.error("Video failed to start: %s", self.video.error)
                self.video = None
        self._running = True
        log.info("All subsystems running. Press 'q'/ESC in the window or Ctrl-C to quit.")

    def stop(self) -> None:
        self._running = False
        if not self._started and self.video is None:
            return
        log.info("Shutting down…")
        if self.video is not None:
            try:
                self.video.stop()
            except Exception as exc:
                log.error("Error stopping video: %s", exc)
            self.video = None
        for name, comp in reversed(self._started):
            try:
                comp.stop()
            except Exception as exc:
                log.error("Error stopping %s: %s", name, exc)
        self._started.clear()
        log.info("Bye.")

    def _check_health(self):
        if self.video is not None and self.video.error:
            raise RuntimeError(f"Video stopped: {self.video.error}")
        for name, comp in self._started:
            if getattr(comp, "error", None):
                raise RuntimeError(f"{name} stopped: {comp.error}")

    # ── main-thread display loop ─────────────────────────────────────
    def loop(self) -> None:
        if self.video is None or not config.SHOW_WINDOW:
            # Headless: just idle until interrupted.
            while self._running:
                self._check_health()
                time.sleep(0.2)
            return

        import cv2
        while self._running:
            self._check_health()
            frame = self.video.get_frame()
            if frame is not None:
                cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                self.video.toggle_debug()
            if key == ord("s"):
                self.video.save_debug_frame()
            if key == ord("t"):
                self.video.toggle_scene_text()
            if key in (ord("q"), 27):     # q or ESC
                break
        cv2.destroyAllWindows()


def _device(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local real-time speech translator for macOS and Windows")
    parser.add_argument("--doctor", action="store_true", help="Check installation without downloading models")
    parser.add_argument("--list-devices", action="store_true", help="List microphones and speakers")
    parser.add_argument("--target", choices=list(config.LANGUAGE_CODES))
    parser.add_argument("--source", choices=["auto", *config.LANGUAGE_CODES])
    parser.add_argument("--no-video", action="store_true", help="Run the speech pipeline without a camera")
    parser.add_argument("--vision-debug", action="store_true", help="Show landmarks, motion and processing times (D toggles)")
    parser.add_argument("--scene-text", action="store_true", help="Translate printed camera text (T toggles)")
    parser.add_argument("--ocr-source", choices=["English", "French", "German", "Spanish", "Portuguese"])
    parser.add_argument("--no-visual-speaker", action="store_true", help="Disable mouth-motion attribution")
    parser.add_argument("--no-tts", action="store_true", help="Translate to captions/logs without speech output")
    parser.add_argument("--stream", choices=["local", "rtp", "both"])
    parser.add_argument("--input-device", type=_device)
    parser.add_argument("--output-device", type=_device)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--cpu", action="store_true", help="Use CPU inference for all models")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for argument, setting in (
        ("target", "TARGET_LANGUAGE"), ("source", "SOURCE_LANGUAGE"),
        ("stream", "STREAM_MODE"), ("input_device", "AUDIO_INPUT_DEVICE"),
        ("output_device", "AUDIO_OUTPUT_DEVICE"), ("camera", "CAMERA_INDEX"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(config, setting, value)
    if args.target:
        config.SDP_OUTPUT = os.path.join(config.BASE_DIR, f"stream_{config.TARGET_LANGUAGE}.sdp")
    if args.no_video:
        config.ENABLE_VIDEO = False
    if args.vision_debug:
        config.SHOW_VISION_DEBUG = True
    if args.scene_text:
        config.ENABLE_SCENE_TEXT = True
    if args.ocr_source:
        config.OCR_SOURCE_LANGUAGE = args.ocr_source
    if args.no_visual_speaker:
        config.ENABLE_VISUAL_SPEAKER = False
    if args.no_tts:
        config.ENABLE_TTS = config.ENABLE_STREAM = False
    if args.cpu:
        config.WHISPER_DEVICE = config.NLLB_DEVICE = config.SBV2_DEVICE = "cpu"
        config.WHISPER_COMPUTE_TYPE = "int8"

    from utils.diagnostics import installation_issues, list_devices, run_doctor
    app = App()
    try:
        if args.list_devices:
            list_devices()
            return 0
        if args.doctor:
            return run_doctor()
        issues = installation_issues()
        if issues:
            raise RuntimeError("Setup is incomplete:\n  " + "\n  ".join(issues))
        app.build()
        app.start()
        app.loop()
        return 0
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 0
    except Exception as exc:
        log.error("%s", exc)
        log.info("Run python main.py --doctor for setup checks; use --cpu for GPU problems.")
        return 1
    finally:
        app.stop()
        cv2 = sys.modules.get("cv2")
        if cv2 is not None:
            # GUI calls belong on the main thread on macOS.
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

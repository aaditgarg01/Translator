import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
from main import App, parse_args
from utils.state import SharedState
from utils.queues import put_drop_oldest
from utils.diagnostics import validate_config, installation_issues
from audio.vad import SpeechSegmenter
from tts.engine import TTS
from tts.resample import resample_pcm
from translate.translator import Translator


class LifecycleTests(unittest.TestCase):
    def test_shutdown_after_interrupt_cleans_up_in_reverse_order(self):
        app = App()
        calls = []
        for name in ("Consumer", "Producer"):
            comp = Mock()
            comp.stop.side_effect = lambda name=name: calls.append(name)
            app.components.append((name, comp))
        app.start()
        app._running = False  # Original Ctrl-C path failed to clean up here.
        app.stop()
        app.stop()
        self.assertEqual(calls, ["Producer", "Consumer"])

    def test_partial_startup_is_cleaned_up(self):
        app = App()
        output, failing = Mock(), Mock()
        failing.start.side_effect = RuntimeError("missing voice")
        app.components = [("Output", output), ("TTS", failing)]
        with self.assertRaisesRegex(RuntimeError, "missing voice"):
            try:
                app.start()
            finally:
                app.stop()
        output.stop.assert_called_once()
        failing.stop.assert_called_once()

    def test_background_audio_failure_is_visible(self):
        app = App()
        app._started = [("Audio", Mock(error="device disconnected"))]
        with self.assertRaisesRegex(RuntimeError, "device disconnected"):
            app._check_health()

    def test_cli_device_names_and_indexes(self):
        args = parse_args(["--input-device", "2", "--output-device", "USB Audio", "--no-video"])
        self.assertEqual(args.input_device, 2)
        self.assertEqual(args.output_device, "USB Audio")
        self.assertTrue(args.no_video)


class AudioTests(unittest.TestCase):
    def test_callback_queues_audio_without_running_vad(self):
        from audio.capture import AudioCapture
        with patch("audio.capture.SpeechSegmenter") as factory:
            capture = AudioCapture(queue.Queue(), SharedState())
        capture._callback(np.ones((512, 1), dtype=np.float32), 512, None, None)
        factory.return_value.process.assert_not_called()
        self.assertEqual(capture._frames.qsize(), 1)

    def test_capture_time_mute_discards_echo_after_playback(self):
        from audio.capture import AudioCapture
        with patch("audio.capture.SpeechSegmenter") as factory:
            capture = AudioCapture(queue.Queue(), SharedState())
        capture._handle_frame(np.ones(512, dtype=np.float32), captured_muted=True)
        factory.return_value.process.assert_not_called()
        factory.return_value.reset.assert_called_once()

    def test_microphone_start_failure_closes_stream(self):
        from audio.capture import AudioCapture
        with patch("audio.capture.SpeechSegmenter"), patch("audio.capture.sd.InputStream") as stream:
            stream.return_value.start.side_effect = RuntimeError("permission denied")
            capture = AudioCapture(queue.Queue(), SharedState())
            with self.assertRaisesRegex(RuntimeError, "privacy permissions"):
                capture.start()
            stream.return_value.close.assert_called_once()

    def test_vad_does_not_duplicate_onset_frame(self):
        seg = SpeechSegmenter("energy")
        voice = np.ones(512, dtype=np.float32) * 0.1
        seg.process(voice)
        self.assertEqual(len(seg._buf), 1)

    def test_vad_emits_speech_and_resets_after_silence(self):
        seg = SpeechSegmenter("energy")
        for _ in range(12):
            self.assertIsNone(seg.process(np.ones(512, dtype=np.float32) * 0.1))
        result = None
        for _ in range(19):
            result = seg.process(np.zeros(512, dtype=np.float32))
        self.assertIsNotNone(result)
        self.assertFalse(seg.speaking)
        self.assertEqual(result.dtype, np.float32)

    def test_vad_caps_total_duration_including_short_pauses(self):
        seg = SpeechSegmenter("energy")
        with patch("audio.vad.VAD_MAX_SPEECH_S", 0.32), patch("audio.vad.VAD_MIN_SPEECH_MS", 0):
            result = None
            for i in range(10):
                result = seg.process(np.full(512, 0.1 if i % 2 == 0 else 0, dtype=np.float32))
            self.assertIsNotNone(result)
            self.assertLessEqual(len(result), 512 * 10)

    def test_queue_keeps_most_recent_speech(self):
        q = queue.Queue(maxsize=2)
        for value in (1, 2, 3):
            put_drop_oldest(q, value)
        self.assertEqual([q.get_nowait(), q.get_nowait()], [2, 3])

    def test_resampling_preserves_duration(self):
        source = np.zeros(22050, dtype=np.int16).tobytes()
        result = resample_pcm(source, 22050, 48000)
        self.assertEqual(len(result), 48000 * 2)


class TranslationTests(unittest.TestCase):
    def test_same_language_passes_through_without_inference(self):
        output = queue.Queue()
        translator = Translator(queue.Queue(), output)
        translator._handle({"text": "Hello", "language": translator.target_whisper})
        self.assertEqual(output.get_nowait()["text"], "Hello")

    def test_unsupported_source_is_not_mislabeled_as_translated(self):
        output = queue.Queue()
        translator = Translator(queue.Queue(), output)
        translator._handle({"text": "Hallo", "language": "unsupported"})
        self.assertTrue(output.empty())

    def test_tts_preload_failure_prevents_false_success(self):
        tts = TTS(queue.Queue(), queue.Queue())
        with patch.object(tts, "_backend", side_effect=FileNotFoundError("missing voice")):
            with self.assertRaisesRegex(RuntimeError, "missing voice"):
                tts.start()
        self.assertFalse(tts._running)

    def test_japanese_never_falls_back_to_english_voice(self):
        tts = TTS(queue.Queue(), queue.Queue())
        with patch("tts.sbv2_backend.SBV2Backend", side_effect=ImportError("missing SBV2")):
            with self.assertRaises(ImportError):
                tts._backend("sbv2")
        self.assertIsNone(tts._piper)

    def test_unknown_tts_route_is_an_error(self):
        tts = TTS(queue.Queue(), queue.Queue())
        with self.assertRaises(ValueError):
            tts._engine_for("unsupported")


class ConfigurationTests(unittest.TestCase):
    def test_invalid_target_has_readable_error(self):
        with patch.object(config, "TARGET_LANGUAGE", "missing"):
            with self.assertRaisesRegex(ValueError, "Unknown target"):
                validate_config()

    def test_missing_voice_reports_download_command(self):
        with patch("utils.diagnostics.os.path.isfile", return_value=False):
            issues = installation_issues()
        self.assertTrue(any("download_piper_voice.py en_US-amy-medium" in issue for issue in issues))

    def test_whisper_mps_rejected(self):
        with patch.object(config, "WHISPER_DEVICE", "mps"):
            with self.assertRaisesRegex(ValueError, "does not support MPS"):
                validate_config()


if __name__ == "__main__":
    unittest.main()

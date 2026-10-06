import io
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video.device import open_camera
from stream.ffmpeg import AudioStreamer
from tts.piper_backend import PiperBackend
from scripts.download_piper_voice import _download, voice_url
from utils.state import SharedState


class CameraTests(unittest.TestCase):
    def test_native_backend_for_each_platform(self):
        import cv2
        for platform, expected in (("darwin", cv2.CAP_AVFOUNDATION), ("win32", cv2.CAP_DSHOW), ("linux", cv2.CAP_ANY)):
            with self.subTest(platform=platform), patch("video.device.sys.platform", platform), patch("video.device.cv2.VideoCapture") as factory:
                factory.return_value.isOpened.return_value = True
                self.assertIs(open_camera(0), factory.return_value)
                factory.assert_called_once_with(0, expected)

    def test_failed_backend_is_released_before_fallback(self):
        first, fallback = Mock(), Mock()
        first.isOpened.return_value = False
        fallback.isOpened.return_value = True
        with patch("video.device.sys.platform", "darwin"), patch("video.device.cv2.VideoCapture", side_effect=[first, fallback]):
            self.assertIs(open_camera(0), fallback)
        first.release.assert_called_once()

    def test_all_failed_attempts_release_resources(self):
        camera = Mock()
        camera.isOpened.return_value = False
        with patch("video.device.sys.platform", "linux"), patch("video.device.cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(RuntimeError, "privacy permissions"):
                open_camera(0)
        camera.release.assert_called_once()


class StreamTests(unittest.TestCase):
    def test_no_working_output_fails_startup(self):
        stream = AudioStreamer(queue.Queue(), 48000)
        stream.mode = "local"
        with patch.object(stream, "_open_speaker", side_effect=RuntimeError("no device")):
            with self.assertRaisesRegex(RuntimeError, "No audio output"):
                stream.start()
        self.assertFalse(stream._running)

    def test_both_mode_keeps_local_audio_when_ffmpeg_missing(self):
        stream = AudioStreamer(queue.Queue(), 48000)
        stream.mode = "both"
        speaker = Mock()
        def open_speaker():
            stream._speaker = speaker
        with patch.object(stream, "_open_speaker", side_effect=open_speaker), patch.object(stream, "_open_ffmpeg", side_effect=RuntimeError("not found")):
            stream.start()
            self.assertTrue(stream._running)
            stream.stop()
        speaker.close.assert_called_once()

    def test_shutdown_waits_for_ffmpeg_and_unmutes(self):
        state = SharedState()
        state.set_output_active(True)
        stream = AudioStreamer(queue.Queue(), 48000, state)
        process = Mock()
        process.poll.return_value = None
        stream._ffmpeg = process
        stream.stop()
        process.wait.assert_called_once_with(timeout=2)
        process.stdin.close.assert_called_once()
        self.assertFalse(state.mic_muted)

    def test_rtp_only_is_paced_to_audio_duration(self):
        stream = AudioStreamer(queue.Queue(), 48000)
        stream._stopped = Mock()
        stream._write(bytes(1920))
        stream._stopped.wait.assert_called_once_with(0.02)

    def test_ffmpeg_failure_has_diagnostic(self):
        stream = AudioStreamer(queue.Queue(), 48000)
        stream._ffmpeg = Mock()
        stream._ffmpeg.poll.return_value = 1
        stream._ffmpeg_log = io.BytesIO(b"Unknown encoder")
        with self.assertRaisesRegex(RuntimeError, "Unknown encoder"):
            stream._check_ffmpeg()


class PiperTests(unittest.TestCase):
    def test_requires_voice_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "voice.onnx")
            Path(path).touch()
            with patch("tts.piper_backend.PIPER_VOICES", {"English": path}):
                backend = PiperBackend()
                self.assertFalse(backend.available_for("English"))
                with self.assertRaisesRegex(FileNotFoundError, ".json"):
                    backend._load_voice("English")
                Path(path + ".json").touch()
                self.assertTrue(backend.available_for("English"))

    def test_current_piper_chunk_api(self):
        voice = Mock(spec=["synthesize"])
        voice.synthesize.return_value = [Mock(audio_int16_bytes=b"\x01\x00"), Mock(audio_int16_bytes=b"\x02\x00")]
        backend = PiperBackend()
        with patch.object(backend, "_load_voice", return_value=voice):
            pcm, rate = backend.synth("hello", "English")
        self.assertEqual(pcm, b"\x01\x00\x02\x00")
        self.assertEqual(rate, 22050)


class DownloadTests(unittest.TestCase):
    def test_valid_voice_url(self):
        self.assertTrue(voice_url("en_US-amy-medium").endswith("/en/en_US/amy/medium/en_US-amy-medium.onnx"))

    def test_invalid_voice_cannot_escape_output_directory(self):
        for name in ("../../voice", "en_US-../../escape-medium", "not-a-voice"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                voice_url(name)

    def test_incomplete_download_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "voice.onnx"
            dest.write_bytes(b"existing")
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.headers = {"Content-Length": "100"}
            response.read.side_effect = [b"partial", b""]
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(IOError, "incomplete"):
                    _download("https://example.com/voice.onnx", str(dest))
            self.assertEqual(dest.read_bytes(), b"existing")
            self.assertEqual(os.listdir(directory), ["voice.onnx"])


if __name__ == "__main__":
    unittest.main()

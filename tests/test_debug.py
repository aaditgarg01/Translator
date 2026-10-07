import unittest
from unittest.mock import patch, Mock
import tempfile
import threading
from pathlib import Path

import numpy as np
from video.debug import draw_debug
from video.camera import VideoCapture
from utils.state import SharedState
from main import parse_args


class DebugTests(unittest.TestCase):
    def test_debug_draws_landmarks_and_metrics_without_mutating_state(self):
        state = SharedState()
        state.set_fps(29.8)
        state.record_latency('ASR', 120)
        points = np.tile([300,220],(68,1)).astype('float32')
        frame = np.zeros((480,640,3),np.uint8)
        draw_debug(frame,[dict(bbox=(250,160,100,120),landmarks=points,track_id=2,speaker_score=.05)],state,12,30)
        self.assertTrue(frame.any())
        self.assertEqual(state.get_latencies(),{'ASR':120})

    def test_metrics_snapshot_is_independent(self):
        state = SharedState();state.record_latency('Vision', 10)
        snapshot = state.get_latencies();snapshot['Vision']=999
        self.assertEqual(state.get_latencies()['Vision'],10)

    def test_save_is_explicit_and_missing_frame_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder, patch('video.camera.FaceRegistry'), patch('config.DATA_DIR',folder):
            video = VideoCapture(SharedState())
            self.assertIsNone(video.save_debug_frame())
            self.assertEqual(list(Path(folder).iterdir()),[])
            video._frame = np.zeros((40,60,3),np.uint8)
            saved = video.save_debug_frame()
            self.assertTrue(saved.is_file())

    def test_slow_vision_does_not_block_preview_capture(self):
        analysis_started, release, captured = threading.Event(), threading.Event(), threading.Event()
        cam = Mock()
        count = [0]
        def read():
            count[0] += 1
            if count[0] >= 5:
                captured.set()
            return True, np.zeros((120,160,3),np.uint8)
        cam.read.side_effect = read
        def identify(*args, **kwargs):
            analysis_started.set()
            release.wait(2)
            return []
        with patch('video.camera.FaceRegistry') as registry, patch('video.camera.open_camera',return_value=cam), patch('config.ENABLE_VISUAL_SPEAKER',False):
            registry.return_value.identify.side_effect = identify
            video = VideoCapture(SharedState())
            try:
                self.assertTrue(video.start())
                self.assertTrue(analysis_started.wait(2))
                self.assertTrue(captured.wait(1), 'Capture waited for blocked analysis')
            finally:
                release.set()
                video.stop()

    def test_debug_and_scene_cli_controls(self):
        args = parse_args(['--vision-debug','--scene-text','--ocr-source','English'])
        self.assertTrue(args.vision_debug and args.scene_text)


if __name__ == '__main__':
    unittest.main()

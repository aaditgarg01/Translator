import queue
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from video.scene_text import ordered_quad, rectify, word_crops, PolygonTracker, SceneText
from translate.translator import Translator
from utils.state import SharedState


class SceneTextTests(unittest.TestCase):
    def test_clear_word_spaces_are_preserved_before_crnn(self):
        line = np.full((50, 360, 3), 255, np.uint8)
        cv2.putText(line, 'HELLO WORLD', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.assertEqual(len(word_crops(line)), 2)

    def test_perspective_order_and_rectification(self):
        points = np.array([[110, 60], [10, 10], [110, 10], [10, 60]], np.float32)
        np.testing.assert_allclose(ordered_quad(points), [[10,10],[110,10],[110,60],[10,60]])
        frame = np.zeros((100, 150, 3), np.uint8)
        frame[10:61, 10:111] = (30, 80, 150)
        crop = rectify(frame, points)
        self.assertEqual(crop.shape, (32, 100, 3))
        np.testing.assert_allclose(crop[16,50], [30,80,150])

    def test_polygon_follows_camera_motion_and_expires(self):
        gray = np.random.default_rng(3).integers(0, 255, (120, 200), np.uint8)
        poly = np.array([[30,30],[140,30],[140,90],[30,90]], np.float32)
        tracker = PolygonTracker()
        tracker.reset(gray, [dict(polygon=poly, original='hello')], 0)
        shifted = cv2.warpAffine(gray, np.float32([[1,0,5],[0,1,3]]), (200,120))
        regions = tracker.update(shifted, .1)
        self.assertEqual(len(regions), 1)
        np.testing.assert_allclose(regions[0]['polygon'], poly + [5,3], atol=1)
        self.assertEqual(tracker.update(shifted, 4), [])

    def test_featureless_region_is_hidden_not_left_floating(self):
        gray = np.zeros((100,100), np.uint8)
        tracker = PolygonTracker()
        tracker.reset(gray, [dict(polygon=np.float32([[10,10],[90,10],[90,90],[10,90]]))], 0)
        self.assertEqual(tracker.update(gray, .1), [])

    def test_ocr_translation_does_not_replace_speech_caption_or_play_audio(self):
        state, output = SharedState(), queue.Queue()
        state.set_caption('spoken', 'spoken translation', 'Alice')
        translator = Translator(queue.Queue(), output, state)
        callback = Mock()
        with patch.object(translator, '_translate', return_value='camera translation'):
            translator._handle_scene(dict(text='hello', language='en', on_result=callback))
        callback.assert_called_once()
        self.assertEqual(state.get_caption().original, 'spoken')
        self.assertTrue(output.empty())

    def test_failed_translation_releases_pending_request(self):
        scene = SceneText(queue.Queue(), '/missing')
        scene._pending['hello'] = 1
        translator = Translator(queue.Queue(), queue.Queue())
        with patch.object(translator, '_translate', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                translator._handle_scene(dict(text='hello', language='en', on_result=scene._translated))
        self.assertNotIn('hello', scene._pending)
        self.assertNotIn('hello', scene._cache)

    def test_cache_is_bounded_and_frames_keep_only_latest(self):
        scene = SceneText(queue.Queue(), '/missing')
        for i in range(140):
            scene._translated(str(i), 'translation', 1)
        self.assertEqual(len(scene._cache), 128)
        scene._enabled = True
        for stamp in range(3):
            scene.submit(np.zeros((90,160,3), np.uint8), stamp)
        self.assertEqual(scene._frames.qsize(), 1)
        self.assertEqual(scene._frames.get_nowait()[1], 2)

    def test_scene_off_does_not_accept_frames(self):
        scene = SceneText(queue.Queue(), '/missing')
        scene.submit(np.zeros((90,160,3), np.uint8), 1)
        self.assertTrue(scene._frames.empty())


if __name__ == '__main__':
    unittest.main()

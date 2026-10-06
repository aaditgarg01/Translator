import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from video.speaker import FaceTracks, SpeakerDecision, compensated_motion
from utils.state import SharedState, AudioSegment
from translate.translator import Translator
from whisper.engine import WhisperEngine


def face(x, name='Unknown', identity=-1, motion=0.0, tid=1, width=100):
    return dict(bbox=(x, 10, width, 100), name=name, id=identity, language=None,
                confidence=0, motion=motion, track_id=tid)


class SpeakerTests(unittest.TestCase):
    def test_head_translation_and_rotation_do_not_count_as_speech(self):
        points = np.random.default_rng(5).uniform(10, 100, (68, 2)).astype('float32')
        transform = cv2.getRotationMatrix2D((50, 50), 5, 1.02)
        transform[:, 2] += [4, 7]
        moved = cv2.transform(points[None], transform)[0]
        self.assertLess(compensated_motion(points, moved, np.ones(68), 100, .1), .001)
        moved[48:, 1] += 3
        self.assertGreater(compensated_motion(points, moved, np.ones(68), 100, .1), .2)

    def test_smaller_moving_face_wins_over_large_silent_face(self):
        selector = SpeakerDecision(dwell=.15)
        for t in (0, .1, .2, .3):
            faces = [face(0, 'Silent', 0, 0, 1, 200), face(300, 'Talking', 1, .1, 2)]
            result = selector.update(faces, True, t)
        self.assertEqual(result, ('Talking', 2))

    def test_no_vad_or_two_moving_mouths_remain_uncertain(self):
        for vad, second in ((False, 0), (True, .09)):
            selector = SpeakerDecision(dwell=0)
            self.assertEqual(selector.update([face(0, motion=.1), face(150, motion=second, tid=2)], vad, 1), ('', None))

    def test_speaker_clears_on_silence_or_face_loss(self):
        selector = SpeakerDecision(dwell=0)
        self.assertEqual(selector.update([face(0, motion=.1)], True, 0)[1], 1)
        self.assertEqual(selector.update([], True, .1), ('', None))
        self.assertEqual(selector.update([face(0, motion=.1)], False, .2), ('', None))

    def test_track_matching_survives_detection_order_and_expires(self):
        tracker = FaceTracks()
        first = tracker.update([face(0), face(200)], 0)
        second = tracker.update([face(205), face(5)], .1)
        self.assertEqual([f['track_id'] for f in second], [first[1]['track_id'], first[0]['track_id']])
        third = tracker.update([face(5)], 2)
        self.assertNotEqual(third[0]['track_id'], first[0]['track_id'])

    def test_utterance_keeps_captured_speaker_through_delayed_translation(self):
        state = SharedState()
        state.set_speech_active(True)
        for _ in range(5):
            state.observe_speaker('Alice', 1)
        state.set_speech_active(False)
        name, tid = state.take_utterance_speaker()
        self.assertEqual((name, tid), ('Alice', 1))
        engine = WhisperEngine(queue.Queue(), queue.Queue(), state)
        engine.model = SimpleNamespace(transcribe=lambda *a, **k: (
            [SimpleNamespace(text='Hello')], SimpleNamespace(language='en', language_probability=1)))
        engine._transcribe(AudioSegment(np.zeros(16000, 'float32'), name, tid))
        item = engine.transcript_queue.get_nowait()
        state.set_current_speaker('Bob')
        translator = Translator(queue.Queue(), queue.Queue(), state)
        with patch.object(translator, '_translate', return_value='Translation'):
            translator._handle(item)
        self.assertEqual((state.get_caption().speaker, state.get_caption().track_id), ('Alice', 1))

    def test_mixed_or_mostly_uncertain_utterance_is_unassigned(self):
        state = SharedState()
        state.set_speech_active(True)
        for _ in range(2):
            state.observe_speaker('Alice', 1)
        for _ in range(5):
            state.observe_speaker('', None)
        self.assertEqual(state.take_utterance_speaker(), ('', None))


if __name__ == '__main__':
    unittest.main()

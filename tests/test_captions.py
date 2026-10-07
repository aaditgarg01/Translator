import unittest
from unittest.mock import patch
import time

import cv2
import numpy as np

from video.speaker import FaceTracks
from video.tracking import caption_anchor, draw_captions
from video.text import script_for, draw_label
from video.shaping import visual_runs
from utils.state import Caption, SharedState
from test_vision import face


class CaptionTests(unittest.TestCase):
    def test_output_scaling_does_not_corrupt_internal_tracks(self):
        tracker = FaceTracks()
        first = tracker.update([face(20)], 0)
        first[0]['bbox'] = (40,20,200,200)
        second = tracker.update([face(22)], .1)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])

    def test_name_requires_repeat_observations(self):
        tracker = FaceTracks()
        for index in range(3):
            result = tracker.update([face(20, 'Alice', 4)], index * .1)
            self.assertEqual(result[0]['name'], 'Alice' if index == 2 else 'Unknown')

    def test_ambiguous_crossing_resets_identity(self):
        tracker = FaceTracks()
        first = tracker.update([face(0), face(40)], 0)
        next_faces = tracker.update([face(20)], .1)
        self.assertNotIn(next_faces[0]['track_id'], [f['track_id'] for f in first])

    def test_flow_bridges_a_brief_missed_detection(self):
        image = np.random.default_rng(8).integers(0,255,(160,240),np.uint8)
        tracker = FaceTracks()
        first = tracker.update([face(20)], 0, image)
        moved = cv2.warpAffine(image, np.float32([[1,0,5],[0,1,3]]), (240,160))
        second = tracker.update([], .1, moved)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertTrue(second[0]['tracked_only'])
        self.assertAlmostEqual(second[0]['bbox'][0],25,delta=1)
        self.assertEqual(tracker.update([],1,moved),[])

    def test_caption_matches_track_even_when_current_speaker_changes(self):
        caption = Caption('hello','bonjour','Alice',time.time(),7)
        faces = [face(30, 'Alice', 0, tid=7), face(200,'Bob',1,tid=9)]
        self.assertEqual(caption_anchor(caption,faces)['track_id'],7)
        self.assertIsNone(caption_anchor(caption,[faces[1]]))

    def test_caption_history_keeps_people_separate_and_expires(self):
        state = SharedState()
        with patch('utils.state.time.time',return_value=10):
            state.set_caption('a','translated a','Alice',1)
            state.set_caption('b','translated b','Bob',2)
            self.assertEqual(len(state.get_captions()),2)
        with patch('utils.state.time.time',return_value=20):
            self.assertEqual(state.get_captions(),[])

    def test_arabic_direction_keeps_numbers_and_latin_in_order(self):
        runs = visual_runs('Alice: مرحباً 123')
        self.assertEqual(runs[0][1], 'ltr')
        self.assertIn(('123', 'ltr'), runs)
        self.assertTrue(any('مرحبا' in value.replace('ً','') and direction == 'rtl' for value,direction in runs))

    def test_script_selection_for_all_non_latin_targets(self):
        for text, script in [('नमस्ते','devanagari'),('مرحباً','arabic'),('こんにちは','cjk'),('你好','cjk'),('안녕하세요','cjk')]:
            self.assertEqual(script_for(text),script)

    def test_labels_and_unassigned_caption_stay_in_frame(self):
        frame = np.zeros((180,320,3),np.uint8)
        x,y,w,h = draw_label(frame,'A long caption near the edge',310,170,max_width=200)
        self.assertGreaterEqual(x,0);self.assertLessEqual(x+w,320)
        self.assertGreaterEqual(y,0);self.assertLessEqual(y+h,180)
        draw_captions(frame,[],[Caption('test','Unassigned','',time.time())])
        self.assertTrue(frame.any())


if __name__ == '__main__':
    unittest.main()

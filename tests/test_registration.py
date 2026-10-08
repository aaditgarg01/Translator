import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from video.tracking import FaceRegistry


class RegistrationTests(unittest.TestCase):
    def test_enrolment_preserves_old_histograms_without_original_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = np.random.default_rng(3).integers(0, 255, (100, 100), dtype=np.uint8)
            model = FaceRegistry._new_recognizer()
            model.train([original], np.array([7], dtype=np.int32))
            model.save(str(folder / 'model.yml'))
            (folder / 'metadata.json').write_text(json.dumps({
                'persons': {'7': {'name': 'Existing', 'language': 'English'}}, 'next_id': 8}))
            registry = FaceRegistry(directory)
            old_histogram = registry.recognizer.getHistograms()[0].copy()
            frame = np.random.default_rng(9).integers(0, 255, (100, 100, 3), dtype=np.uint8)
            with patch.object(registry, 'detect', return_value=(frame[:, :, 0], [(0, 0, 100, 100)])), \
                    patch.object(registry, '_retrain', side_effect=AssertionError('must not reread old photos')):
                pid, count = registry.register_person('New', 'Japanese', [frame])
            self.assertEqual((pid, count), (8, 1))
            reloaded = FaceRegistry(directory)
            self.assertEqual(set(reloaded.recognizer.getLabels().flatten()), {7, 8})
            np.testing.assert_array_equal(reloaded.recognizer.getHistograms()[0], old_histogram)
            self.assertEqual(reloaded.persons['7']['name'], 'Existing')
            self.assertTrue((folder / '8' / 'face_0.jpg').exists())

    def test_missing_model_and_training_photos_fail_before_enrolment(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'metadata.json').write_text(json.dumps({
                'persons': {'0': {'name': 'Existing', 'language': 'English'}}, 'next_id': 1}))
            registry = FaceRegistry(directory)
            with self.assertRaisesRegex(RuntimeError, 'restore its training photos'):
                registry.register_person('New', 'Japanese', [])
            self.assertEqual(registry.next_id, 1)
            self.assertEqual(list(registry.persons), ['0'])
            self.assertFalse((folder / '1').exists())


if __name__ == '__main__':
    unittest.main()

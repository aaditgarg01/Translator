import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tts.audio import join_sentences
from tts.piper_backend import PiperBackend
from tts.voices import voice_issues


class SentenceAudioTests(unittest.TestCase):
    def test_pause_and_sample_rate_determine_duration(self):
        pcm = join_sentences([np.ones(1000), np.ones(1000)], 1000, pause_ms=150)
        out = np.frombuffer(pcm, dtype='<i2')
        self.assertEqual(len(out), 2150)
        self.assertTrue(np.all(out[1000:1150] == 0))
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 0)

    def test_relative_sentence_volume_is_preserved(self):
        pcm = join_sentences([np.full(100, .2), np.full(100, .4)], 1000, pause_ms=0)
        out = np.frombuffer(pcm, dtype='<i2')
        self.assertAlmostEqual(out[50] / out[150], .5, places=3)
        self.assertLessEqual(np.max(np.abs(out)), 31129)

    def test_empty_silent_and_nonfinite_output_are_errors(self):
        for parts in ([], [np.zeros(100)], [np.array([np.nan])]):
            with self.subTest(parts=parts), self.assertRaises((ValueError, RuntimeError)):
                join_sentences(parts, 22050)

    def test_new_language_uses_its_profile(self):
        backend = PiperBackend()
        voice = Mock()
        voice.synthesize.return_value = [SimpleNamespace(audio_float_array=np.ones(100))]
        with patch.object(backend, '_load_voice', return_value=voice):
            pcm, rate = backend.synth('Bonjour.', 'French')
        settings = voice.synthesize.call_args.kwargs['syn_config']
        self.assertFalse(settings.normalize_audio)
        self.assertEqual(settings.speaker_id, 0)
        self.assertTrue(pcm)
        self.assertEqual(rate, 22050)

    def test_established_voice_settings_remain_unchanged(self):
        for language in ('English', 'Hindi', 'Japanese'):
            with self.subTest(language=language):
                voice = Mock(spec=['synthesize'])
                voice.synthesize.return_value = [SimpleNamespace(audio_int16_bytes=b'\x01\x00')]
                backend = PiperBackend()
                with patch.object(backend, '_load_voice', return_value=voice):
                    self.assertEqual(backend.synth('text', language)[0], b'\x01\x00')
                voice.synthesize.assert_called_once_with('text')

    def test_preload_detects_pronunciation_failure_before_live_input(self):
        backend = PiperBackend()
        voice = Mock()
        voice.phonemize.side_effect = ModuleNotFoundError('missing pronunciation package')
        with patch.object(backend, '_load_voice', return_value=voice), self.assertRaises(ModuleNotFoundError):
            backend.preload('Chinese')


class VoiceMetadataTests(unittest.TestCase):
    def check(self, metadata, language):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'voice.onnx'
            model.touch()
            Path(str(model) + '.json').write_text(json.dumps(metadata), encoding='utf-8')
            return voice_issues(str(model), language)

    def test_wrong_language_model_is_rejected(self):
        issues = self.check({'language': {'code': 'en_US'}}, 'French')
        self.assertTrue(any('en_US' in issue for issue in issues))

    def test_chinese_missing_dependencies_have_install_command(self):
        with patch('tts.voices.importlib.util.find_spec', return_value=None):
            issues = self.check({'phoneme_type': 'pinyin', 'language': {'code': 'zh_CN'}}, 'Chinese')
        self.assertTrue(any('piper-tts[zh]' in issue for issue in issues))

    def test_japanese_missing_dependencies_have_install_command(self):
        with patch('tts.voices.importlib.util.find_spec', return_value=None):
            issues = self.check({'phoneme_type': 'japanese'}, 'Japanese')
        self.assertTrue(any('piper-tts[ja]' in issue for issue in issues))


if __name__ == '__main__':
    unittest.main()

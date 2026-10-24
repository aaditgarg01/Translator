import queue
import threading
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from translate.translator import Translator
from tts.engine import TTS
from utils.state import SharedState
from video.camera import VideoCapture
from video.speaker import FaceTracks


def detected():
    return dict(bbox=(50,30,90,100),id=1,name='Alice',language='English',confidence=90)


class PerformanceTests(unittest.TestCase):
    def test_translation_cache_separates_languages_and_uses_lru(self):
        translator=Translator(queue.Queue(),queue.Queue())
        with patch('translate.translator.TRANSLATION_CACHE_SIZE',2),patch.object(translator,'_infer',side_effect=lambda t,s,d:f'{s}:{d}:{t}') as model:
            a=translator._translate('Hello','eng_Latn','jpn_Jpan')
            self.assertEqual(translator._translate('Hello','eng_Latn','jpn_Jpan'),a)
            translator._translate('Hello','eng_Latn','deu_Latn')
            translator._translate('Hello','eng_Latn','jpn_Jpan')
            translator._translate('Other','eng_Latn','jpn_Jpan')
            translator._translate('Hello','eng_Latn','deu_Latn')
        self.assertEqual(model.call_count,4)
        self.assertEqual(len(translator._cache),2)

    def test_failed_or_empty_translation_is_not_cached(self):
        translator=Translator(queue.Queue(),queue.Queue())
        with patch.object(translator,'_infer',side_effect=[RuntimeError('failed'),'','ok']) as model:
            with self.assertRaises(RuntimeError):translator._translate('x','a','b')
            self.assertEqual(translator._translate('x','a','b'),'')
            self.assertEqual(translator._translate('x','a','b'),'ok')
            self.assertEqual(model.call_count,3)

    def test_same_language_start_skips_translation_model(self):
        translator=Translator(queue.Queue(),queue.Queue())
        with patch('translate.translator.SOURCE_LANGUAGE',translator.target_name),patch.object(translator,'load') as load:
            try:translator.start()
            finally:translator.stop()
        load.assert_not_called()

    def test_cpu_int8_preparation_selects_supported_engine_once(self):
        translator=Translator(queue.Queue(),queue.Queue());translator.model=Mock();translator._device='cpu'
        backend=SimpleNamespace(engine='none',supported_engines=['qnnpack'])
        quantize=Mock(return_value='quantized model')
        fake=SimpleNamespace(backends=SimpleNamespace(quantized=backend),
                             ao=SimpleNamespace(quantization=SimpleNamespace(quantize_dynamic=quantize)),
                             nn=SimpleNamespace(Linear='linear'),qint8='int8')
        with patch('translate.translator.NLLB_CPU_INT8',True),patch.dict(sys.modules,{'torch':fake}):
            translator._prepare_cpu_model();translator._prepare_cpu_model()
        self.assertEqual(backend.engine,'qnnpack')
        self.assertEqual(translator.model,'quantized model')
        quantize.assert_called_once()

    def test_int8_failure_reloads_clean_float_model_and_gpu_skips_quantization(self):
        translator=Translator(queue.Queue(),queue.Queue());translator.model=Mock();translator._device='mps'
        with patch('translate.translator.NLLB_CPU_INT8',True):
            translator._prepare_cpu_model()
        self.assertFalse(translator._quantization_attempted)
        translator._device='cpu'
        import torch
        with patch('translate.translator.NLLB_CPU_INT8',True),patch('torch.ao.quantization.quantize_dynamic',side_effect=RuntimeError('unsupported')),patch('transformers.AutoModelForSeq2SeqLM.from_pretrained') as load:
            translator._prepare_cpu_model()
        load.assert_called_once()
        self.assertIs(translator.model,load.return_value.to.return_value.eval.return_value)

    def test_pcm_cache_is_bounded_and_separates_languages(self):
        tts=TTS(queue.Queue(),queue.Queue());backend=Mock()
        backend.synth.return_value=(b'\x01\x00'*100,48000)
        with patch('tts.engine.TTS_CACHE_MAX_BYTES',400),patch.object(tts,'_backend',return_value=backend):
            first=tts._synthesize('Hello','English')
            self.assertEqual(tts._synthesize('Hello','English'),first)
            tts._synthesize('Hello','Hindi')
            tts._synthesize('Other','English')
            self.assertEqual(tts._cache_bytes,400)
            tts._synthesize('Hello','English')
        self.assertEqual(backend.synth.call_count,4)

    def test_oversized_pcm_is_never_retained(self):
        tts=TTS(queue.Queue(),queue.Queue());backend=Mock()
        backend.synth.return_value=(b'\x01\x00'*100,48000)
        with patch('tts.engine.TTS_CACHE_MAX_BYTES',50),patch.object(tts,'_backend',return_value=backend):
            tts._synthesize('Hello','English')
        self.assertFalse(tts._pcm_cache)
        self.assertEqual(tts._cache_bytes,0)

    def test_detector_skip_preserves_reliable_flow_and_identity_votes(self):
        gray=np.random.default_rng(2).integers(0,255,(180,240),np.uint8)
        tracks=FaceTracks()
        first=tracks.update([detected()],0,gray)
        moved=cv2.warpAffine(gray,np.float32([[1,0,4],[0,1,2]]),(240,180))
        next_faces=tracks.update([],.067,moved,detection_frame=False)
        self.assertEqual(next_faces[0]['track_id'],first[0]['track_id'])
        self.assertFalse(next_faces[0]['tracked_only'])
        # Flow frames do not create fake recogniser votes.
        self.assertEqual(len(tracks.tracks[first[0]['track_id']]['votes']),1)
        self.assertEqual(next_faces[0]['name'],'Unknown')

    def test_offscreen_face_does_not_track_an_unrelated_image_region(self):
        gray=np.random.default_rng(5).integers(0,255,(180,240),np.uint8)
        tracker=FaceTracks();f=detected();f['bbox']=(-200,-200,50,50)
        tracker.update([f],0,gray)
        with patch('video.speaker.cv2.goodFeaturesToTrack') as features:
            self.assertEqual(tracker.update([],.1,gray,detection_frame=False),[])
        features.assert_not_called()

    def test_quiet_analysis_skips_landmarks_and_reduces_full_detection_calls(self):
        with patch('video.camera.FaceRegistry') as registry,patch('config.VISION_DETECT_INTERVAL',3),patch('config.VISION_MAX_FPS',1000):
            video=VideoCapture(SharedState())
        gray=np.random.default_rng(4).integers(0,255,(180,240),np.uint8)
        frame=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        registry.return_value.identify.return_value=[detected()]
        video.speaker=Mock()
        video._running=True
        count=[0]
        def wait(_timeout):
            if count[0]>=6:
                video._running=False;return True
            count[0]+=1
            video._raw=(frame,__import__('time').monotonic())
            return True
        video._ready=Mock();video._ready.wait.side_effect=wait
        with patch('config.VISION_DETECT_INTERVAL',3),patch('config.VISION_MAX_FPS',1000):
            video._analyze()
        self.assertIsNone(video.error)
        self.assertEqual(registry.return_value.identify.call_count,2)
        video.speaker.update.assert_not_called()
        self.assertEqual(video.speaker.reset.call_count,6)


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from meeting.recorder import MeetingRecorder, compile_minutes, recover, _atomic
from utils.state import AudioSegment, SharedState
from whisper.engine import WhisperEngine
from translate.translator import Translator
from main import App, parse_args


def utterance(text='We decided to test the camera.', **kwargs):
    return dict(text=text, language='en', speaker='Aadit', track_id=3, **kwargs)


class MinutesTests(unittest.TestCase):
    def test_transcript_translation_identity_and_printable_export(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = MeetingRecorder(folder, '<Demo> / Meeting')
            key = recorder.record_transcript(utterance(captured_at=recorder.started+10, duration_seconds=2))
            recorder.record_translation(key, 'カメラをテストすることにしました。', 'Japanese')
            path = recorder.close()
            data = json.loads((path.parent/'minutes.json').read_text(encoding='utf-8'))
            self.assertEqual(data['participants'], ['Aadit'])
            entry = data['transcript'][0]
            self.assertEqual(entry['elapsed_seconds'], 8)
            self.assertEqual(entry['track_id'], 3)
            self.assertEqual(entry['translation'], 'カメラをテストすることにしました。')
            self.assertTrue((path.parent/'minutes.md').is_file())
            self.assertIn('&lt;Demo&gt;', path.read_text(encoding='utf-8'))
            self.assertNotIn('<Demo>', path.read_text(encoding='utf-8'))
            self.assertEqual(len(data['decision_candidates']), 1)

    def test_autosave_precedes_exports_and_recovers_out_of_order_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = MeetingRecorder(folder)
            first = recorder.record_transcript(utterance('First point.'))
            second = recorder.record_transcript(utterance('Second point.'))
            recorder.record_translation(first, '最初のポイント。', 'Japanese')
            self.assertFalse((recorder.directory/'minutes.html').exists())
            recover(recorder.directory/'transcript.jsonl')
            data = json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))
            by_id = {e['id']:e for e in data['transcript']}
            self.assertEqual(by_id[first]['translation'], '最初のポイント。')
            self.assertIsNone(by_id[second]['translation'])
            recorder.close()

    def test_closing_and_mid_meeting_export_are_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = MeetingRecorder(folder)
            recorder.record_transcript(utterance())
            recorder.export()
            self.assertIsNone(json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))['meeting'].get('ended_at'))
            recorder.close();recorder.close()
            data = json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))
            self.assertIn('ended_at',data['meeting'])
            self.assertIsNone(recorder.record_transcript(utterance('Too late.')))

    def test_repeated_speech_is_preserved_and_summary_does_not_invent_actions(self):
        events = [dict(kind='session',title='Test')]
        for i in range(12):
            events.append(dict(kind='utterance',id=str(i),text=f'The camera topic number {i} needs review.',
                               speaker='Unassigned',language='en',elapsed_seconds=i))
        data = compile_minutes(events)
        self.assertEqual(len(data['transcript']), 12)
        self.assertEqual(len(data['summary']), 5)
        self.assertFalse(data['decision_candidates'])
        self.assertFalse(data['action_candidates'])
        self.assertFalse(data['participants'])
        self.assertTrue(all(e['text'] in [x['text'] for x in data['transcript']] for e in data['summary']))

    def test_action_and_decision_candidates_keep_evidence_without_assigning_owner(self):
        events = [dict(kind='utterance',id='a',text='Decision: release on Monday. Alice will test audio.',
                       speaker='Aadit',language='en',elapsed_seconds=5)]
        data = compile_minutes(events)
        self.assertEqual(data['decision_candidates'][0]['text'], 'Decision: release on Monday.')
        self.assertEqual(data['action_candidates'][0]['text'], 'Alice will test audio.')
        self.assertNotIn('owner', data['action_candidates'][0])

    def test_english_translation_can_supply_cues_for_non_english_source(self):
        events = [dict(kind='utterance',id='a',text='हमने परीक्षण करने का फैसला किया।',speaker='A',language='hi',elapsed_seconds=0),
                  dict(kind='translation',id='a',text='We decided to test.',language='English')]
        data = compile_minutes(events)
        self.assertEqual(len(data['decision_candidates']),1)
        self.assertEqual(data['transcript'][0]['text'],events[0]['text'])

    def test_recovery_tolerates_only_an_interrupted_last_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'transcript.jsonl'
            path.write_bytes(b'{"kind":"session","title":"Recovered"}\n{"kind":"utterance","text":"\xe3\x81')
            recovered = recover(path)
            self.assertTrue(recovered.exists())
            self.assertEqual(len(json.loads((Path(folder)/'minutes.json').read_text(encoding='utf-8'))['gaps']),1)
            path.write_text('{"kind":"session"}\nBROKEN\n')
            with self.assertRaises(ValueError):recover(path)

    def test_export_failure_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'minutes.md';path.write_text('previous minutes')
            with patch('meeting.recorder.os.replace',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):_atomic(path,'new minutes')
            self.assertEqual(path.read_text(encoding='utf-8'),'previous minutes')
            self.assertEqual(list(Path(folder).iterdir()),[path])

    def test_concurrent_writers_produce_valid_journal_and_separate_sessions(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = MeetingRecorder(folder,'../../meeting')
            other = MeetingRecorder(folder,'../../meeting')
            self.assertNotEqual(recorder.directory,other.directory)
            workers=[threading.Thread(target=lambda:[recorder.record_transcript(utterance()) for _ in range(10)]) for _ in range(3)]
            for worker in workers:worker.start()
            for worker in workers:worker.join()
            recorder.close();other.close()
            data=json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['transcript']),30)
            self.assertEqual(recorder.directory.parent,Path(folder))

    def test_asr_persists_all_recognised_utterances_before_droppable_translation_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=MeetingRecorder(folder)
            engine=WhisperEngine(queue.Queue(),queue.Queue(maxsize=1),meeting=recorder)
            engine.model=SimpleNamespace(transcribe=lambda *a,**k:([SimpleNamespace(text='We will test.')],SimpleNamespace(language='en',language_probability=1)))
            audio=AudioSegment(np.zeros(16000,dtype='float32'),'Alice',7,recorder.started+4)
            engine._transcribe(audio);engine._transcribe(audio)
            self.assertEqual(engine.transcript_queue.qsize(),1)
            item=engine.transcript_queue.get_nowait()
            translator=Translator(queue.Queue(),queue.Queue(),meeting=recorder)
            with patch.object(translator,'_translate',return_value='translated'):
                translator._handle(item)
            recorder.close()
            data=json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['transcript']),2)
            self.assertEqual(sum(e['translation'] is not None for e in data['transcript']),1)
            self.assertEqual({e['speaker'] for e in data['transcript']},{'Alice'})

    def test_app_exports_even_if_build_fails_and_stop_runs_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=MeetingRecorder(folder);app=App();app.meeting=recorder
            app.stop();app.stop()
            self.assertTrue((recorder.directory/'minutes.html').exists())

    def test_ocr_jobs_never_enter_minutes(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=MeetingRecorder(folder)
            translator=Translator(queue.Queue(),queue.Queue(),meeting=recorder)
            with patch.object(translator,'_translate',return_value='camera translation'):
                translator._handle_scene(dict(text='printed',language='en',on_result=lambda *a:None))
            recorder.close()
            self.assertEqual(json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))['transcript'],[])

    def test_minutes_cli(self):
        args=parse_args(['--minutes-only','--meeting-title','Weekly Review','--performance','eco'])
        self.assertTrue(args.minutes_only)
        self.assertEqual(args.meeting_title,'Weekly Review')

    def test_equal_capture_times_keep_journal_order(self):
        data=compile_minutes([dict(kind='utterance',id=key,text=value,speaker='A',language='en',elapsed_seconds=1)
                              for key,value in [('z','First.'),('a','Second.')]])
        self.assertEqual([e['text'] for e in data['transcript']],['First.','Second.'])

    def test_minutes_only_build_skips_translation_and_speech_models(self):
        import config
        with patch.object(config,'ENABLE_TRANSLATE',False),patch.object(config,'ENABLE_TTS',False),patch.object(config,'ENABLE_STREAM',False),patch.object(config,'ENABLE_VIDEO',False),patch('audio.capture.AudioCapture'),patch('whisper.engine.WhisperEngine'),patch('translate.translator.Translator') as translation,patch('tts.engine.TTS') as tts:
            app=App();app.build()
        translation.assert_not_called();tts.assert_not_called()
        self.assertEqual([name for name,_ in app.components],['Whisper','Audio'])

    def test_audio_queue_overflow_is_reported_as_a_recording_gap(self):
        from audio.capture import AudioCapture
        with tempfile.TemporaryDirectory() as folder:
            recorder=MeetingRecorder(folder)
            state=SharedState()
            with patch('audio.capture.SpeechSegmenter') as segmenter:
                segmenter.return_value.process.return_value=np.zeros(16000,dtype='float32')
                segmenter.return_value.speaking=False
                capture=AudioCapture(queue.Queue(maxsize=1),state,recorder)
                capture._handle_frame(np.zeros(512,dtype='float32'),captured_at=123)
                self.assertEqual(capture.audio_queue.queue[0].captured_at,123)
                capture._handle_frame(np.zeros(512,dtype='float32'),captured_at=124)
            recorder.close()
            data=json.loads((recorder.directory/'minutes.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['gaps']),1)
            self.assertIn('ASR input queue overflow',data['gaps'][0]['reason'])

    def test_journal_disk_error_reaches_application_health(self):
        from unittest.mock import Mock
        recorder=Mock();recorder.record_transcript.side_effect=OSError('disk full');recorder.gap.side_effect=OSError('disk full')
        engine=WhisperEngine(queue.Queue(),queue.Queue(),meeting=recorder)
        engine.audio_queue.put(np.zeros(16000,dtype='float32'))
        engine.model=SimpleNamespace(transcribe=lambda *a,**k:([SimpleNamespace(text='A test')],SimpleNamespace(language='en',language_probability=1)))
        engine._running=True;engine._run()
        self.assertIn('disk full',engine.error)
        app=App();app._started=[('Whisper',engine)]
        with self.assertRaisesRegex(RuntimeError,'disk full'):app._check_health()


if __name__=='__main__':unittest.main()

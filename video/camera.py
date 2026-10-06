"""Independent camera preview and bounded, newest-frame-only vision analysis."""

import threading
import time

import cv2
import numpy as np

import config
from video.device import open_camera
from video.tracking import FaceRegistry, annotate
from video.speaker import FaceTracks, VisualSpeaker
from utils.logging_utils import get_logger

log = get_logger('Video')


class VideoCapture:
    def __init__(self, state, camera_index=config.CAMERA_INDEX):
        self.state = state
        self.camera_index = camera_index
        self.registry = FaceRegistry()
        self.tracks = FaceTracks()
        self.speaker = None
        self._cam = None
        self._frame = None
        self._raw = None
        self._faces = []
        self._result_time = 0.0
        self.vision_ms = 0.0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._running = False
        self._thread = self._vision_thread = None
        self.error = None
        self.vision_status = 'Starting vision'

    def start(self):
        if self._running:
            return True
        self.error = None
        try:
            self._cam = open_camera(self.camera_index)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self._cam.set(cv2.CAP_PROP_FPS, config.VIDEO_TARGET_FPS)
            self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if config.ENABLE_VISUAL_SPEAKER:
                try:
                    self.speaker = VisualSpeaker(config.LANDMARK_MODEL, config.SPEAKER_MOTION_THRESHOLD)
                    self.vision_status = 'Mouth motion + audio VAD'
                except (FileNotFoundError, cv2.error) as exc:
                    self.vision_status = 'Speaker model unavailable (see terminal)'
                    log.warning('%s; speaker remains unassigned.', exc)
            else:
                self.vision_status = 'Visual speaker disabled'
            self._running = True
            self._stop.clear()
            self._vision_thread = threading.Thread(target=self._analyze, name='vision', daemon=True)
            self._thread = threading.Thread(target=self._run, name='camera', daemon=True)
            self._vision_thread.start()
            self._thread.start()
            log.info('Camera open (index %d); preview and vision run independently.', self.camera_index)
            return True
        except Exception as exc:
            self.error = str(exc)
            self.stop()
            return False

    def stop(self):
        self._running = False
        self._stop.set()
        self._ready.set()
        for thread in (self._thread, self._vision_thread):
            if thread is not None:
                thread.join(timeout=3)
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    def get_frame(self):
        with self._lock:
            return self._frame

    def _analyze(self):
        try:
            while self._running:
                self._ready.wait(0.2)
                with self._lock:
                    item, self._raw = self._raw, None
                    self._ready.clear()
                if item is None or not self._running:
                    continue
                frame, captured = item
                started = time.monotonic()
                scale = min(1.0, config.VISION_WIDTH / frame.shape[1])
                small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                faces = self.tracks.update(self.registry.identify(small), captured)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                name, tid = '', None
                if self.speaker is not None:
                    name, tid = self.speaker.update(gray, faces, self.state.speech_active, captured)
                # Never attribute speech from an analysis result that has fallen behind.
                if time.monotonic() - captured <= 0.5:
                    self.state.observe_speaker(name, tid)
                else:
                    self.state.observe_speaker('', None)
                for face in faces:
                    face['bbox'] = tuple(round(v / scale) for v in face['bbox'])
                    for key in ('landmarks', 'flow_from', 'flow_to'):
                        if key in face:
                            face[key] = face[key] / scale
                with self._lock:
                    self._faces, self._result_time = faces, captured
                    self.vision_ms = (time.monotonic() - started) * 1000
                self._stop.wait(max(0, 1 / config.VISION_MAX_FPS - (time.monotonic() - started)))
        except Exception as exc:
            self.error = f'Vision analysis failed: {exc}'
            log.exception(self.error)

    def _run(self):
        period = 1 / max(1, config.VIDEO_TARGET_FPS)
        fps_start, count = time.monotonic(), 0
        last_read = fps_start
        try:
            while self._running:
                started = time.monotonic()
                ok, frame = self._cam.read()
                if not ok:
                    if time.monotonic() - last_read > 2:
                        raise RuntimeError('Camera disconnected or stopped delivering frames')
                    self._stop.wait(0.02)
                    continue
                captured = last_read = time.monotonic()
                with self._lock:
                    self._raw = (frame, captured)
                    faces = self._faces if captured - self._result_time < 0.5 else []
                    self._ready.set()
                output = annotate(frame, faces, self.state)
                cv2.putText(output, self.vision_status, (15, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (210, 210, 210), 1)
                with self._lock:
                    self._frame = output
                count += 1
                if captured - fps_start >= 1:
                    self.state.set_fps(count / (captured - fps_start))
                    fps_start, count = captured, 0
                self._stop.wait(max(0, period - (time.monotonic() - started)))
        except Exception as exc:
            if self._running:
                self.error = str(exc)
                log.exception('Camera failed')

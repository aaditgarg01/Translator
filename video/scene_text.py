"""OpenCV DB detection, perspective-rectified CRNN OCR and cached translation.

OCR runs on a single newest-frame queue. The speech translator owns NLLB;
this module only submits low-priority text jobs and receives their results.
"""
from collections import OrderedDict
from pathlib import Path
import queue
import string
import threading
import time

import cv2
import numpy as np

from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger('SceneText')


def ordered_quad(points):
    """Order a convex text box TL, TR, BR, BL, including a tilted rectangle."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def rectify(frame, polygon, preserve_aspect=False):
    src = ordered_quad(polygon)
    width = 100
    if preserve_aspect:
        width = int(np.clip(32 * np.linalg.norm(src[1] - src[0]) / max(1, np.linalg.norm(src[3] - src[0])), 32, 1600))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, 31], [0, 31]], np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (width, 32))


def word_crops(line):
    """CRNN's vocabulary has no space: preserve clear word gaps before decoding."""
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.mean(ink > 0) > .5:
        ink = 255 - ink
    occupied = (ink > 0).any(axis=0)
    columns = np.flatnonzero(occupied)
    if len(columns) == 0:
        return []
    gaps = np.flatnonzero(np.diff(columns) >= max(5, line.shape[0] * .25))
    cuts = [0] + [int((columns[i] + columns[i+1]) / 2) for i in gaps] + [line.shape[1]]
    return [line[:, a:b] for a, b in zip(cuts, cuts[1:]) if b - a >= 4]


class SceneOCR:
    def __init__(self, directory):
        folder = Path(directory)
        detector = folder / 'text_detection_en_ppocrv3_2023may.onnx'
        recognizer = folder / 'text_recognition_CRNN_CH_2021sep.onnx'
        if not detector.is_file() or not recognizer.is_file():
            raise FileNotFoundError('Run python scripts/download_vision_models.py ocr')
        self.detector = cv2.dnn_TextDetectionModel_DB(str(detector))
        self.detector.setInputParams(scale=1 / 255, size=(640, 640),
                                     mean=(123.675, 116.28, 103.53))
        self.detector.setInputScale(tuple(1 / (255 * np.array([.229, .224, .225]))))
        self.detector.setBinaryThreshold(.3).setPolygonThreshold(.6)
        self.detector.setUnclipRatio(1.8).setMaxCandidates(100)
        self.recognizer = cv2.dnn_TextRecognitionModel(str(recognizer))
        self.recognizer.setDecodeType('CTC-greedy')
        self.recognizer.setVocabulary(list(string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation))
        self.recognizer.setInputParams(scale=1 / 127.5, size=(100, 32), mean=(127.5, 127.5, 127.5))

    def read(self, frame):
        polygons, scores = self.detector.detect(frame)
        result = []
        for poly, score in sorted(zip(polygons, scores), key=lambda x: float(x[1]), reverse=True)[:8]:
            poly = ordered_quad(poly)
            if abs(cv2.contourArea(poly)) < 120:
                continue
            crops = word_crops(rectify(frame, poly, preserve_aspect=True))
            text = ' '.join(self.recognizer.recognize(crop).strip() for crop in crops).strip()
            if len(text) >= 2 and any(c.isalpha() for c in text):
                result.append(dict(polygon=poly, original=text, confidence=float(score)))
        return result


class PolygonTracker:
    """Hide stale overlays when local feature tracking becomes unreliable."""
    def __init__(self):
        self.gray = None
        self.regions = []
        self.timestamp = 0.0

    def reset(self, gray, regions, timestamp):
        self.gray, self.timestamp = gray, timestamp
        self.regions = [dict(r, polygon=r['polygon'].copy()) for r in regions]

    def update(self, gray, now):
        if self.gray is None or self.gray.shape != gray.shape or now - self.timestamp > 3:
            self.regions = []
        elif self.regions:
            kept = []
            for region in self.regions:
                mask = np.zeros_like(self.gray)
                cv2.fillConvexPoly(mask, region['polygon'].astype('int32'), 255)
                points = cv2.goodFeaturesToTrack(self.gray, 40, .01, 3, mask=mask)
                if points is None or len(points) < 4:
                    continue
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None, maxLevel=3)
                if moved is None:
                    continue
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None, maxLevel=3)
                if back is None:
                    continue
                valid = (status.ravel() != 0) & (back_status.ravel() != 0)
                valid &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5
                if valid.sum() < 4:
                    continue
                matrix, inliers = cv2.estimateAffinePartial2D(points[valid], moved[valid], method=cv2.RANSAC)
                if matrix is None or inliers.sum() < 4:
                    continue
                scale = np.linalg.norm(matrix[:, 0])
                if not .7 < scale < 1.4:
                    continue
                polygon = cv2.transform(region['polygon'][None], matrix)[0]
                if not np.isfinite(polygon).all():
                    continue
                kept.append(dict(region, polygon=polygon))
            self.regions = kept
        self.gray = gray
        return self.regions


class SceneText:
    def __init__(self, translation_queue, directory, source='en'):
        self.queue, self.directory, self.source = translation_queue, directory, source
        self._frames = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._enabled = False
        self._epoch = 0
        self._result = None
        self._version = self._view_version = 0
        self._previous_texts = set()
        self._cache = OrderedDict()
        self._pending = {}
        self.tracker = PolygonTracker()  # Only accessed by the camera thread.
        self.status = 'Camera text off (T)'
        self.ocr_ms = 0.0
        self.translation_ms = 0.0
        self.last_regions = []

    @property
    def enabled(self):
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = enabled
            self._epoch += 1
            self._result = None
            self._previous_texts.clear()
            self.status = 'Looking for printed text...' if enabled else 'Camera text off (T)'
        if enabled and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._run, name='scene-ocr', daemon=True)
            self._thread.start()

    def submit(self, frame, captured):
        if self.enabled:
            small = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
            with self._lock:
                epoch = self._epoch
            put_drop_oldest(self._frames, (small, captured, epoch))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _translated(self, text, value, elapsed):
        with self._lock:
            self._pending.pop(text, None)
            if value:
                self._cache[text] = value
                self._cache.move_to_end(text)
                while len(self._cache) > 128:
                    self._cache.popitem(last=False)
                self.translation_ms = elapsed

    def _run(self):
        try:
            ocr = SceneOCR(self.directory)
            while not self._stop.is_set():
                try:
                    frame, captured, epoch = self._frames.get(timeout=.2)
                except queue.Empty:
                    continue
                if not self.enabled:
                    continue
                started = time.monotonic()
                regions = ocr.read(frame)
                now = time.monotonic()
                texts = {r['original'] for r in regions}
                with self._lock:
                    if epoch != self._epoch or not self._enabled:
                        continue
                    stable = texts & self._previous_texts
                    self._previous_texts = texts
                    for text in stable:
                        if text not in self._cache and now - self._pending.get(text, -100) > 10:
                            self._pending[text] = now
                            put_drop_oldest(self.queue, dict(kind='scene', text=text, language=self.source,
                                                            on_result=self._translated))
                    self._version += 1
                    self._result = (self._version, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), regions, captured)
                    self.ocr_ms = (now - started) * 1000
                    self.status = f'Camera text: {len(regions)} regions | source {self.source}'
                self._stop.wait(max(0, 1 - (time.monotonic() - started)))
        except Exception as exc:
            with self._lock:
                self.status = f'Camera text unavailable: {exc}'
                self._enabled = False
            log.error('%s', self.status)

    def draw(self, output, raw_frame, now):
        from video.text import draw_label
        with self._lock:
            enabled, result, cache = self._enabled, self._result, dict(self._cache)
            status = self.status
        if enabled:
            gray = cv2.cvtColor(cv2.resize(raw_frame, (640, round(raw_frame.shape[0] * 640 / raw_frame.shape[1]))), cv2.COLOR_BGR2GRAY)
            if result and result[0] != self._view_version:
                self._view_version = result[0]
                self.tracker.reset(result[1], result[2], result[3])
            regions = self.tracker.update(gray, now)
            factor = output.shape[1] / 640
            self.last_regions = regions
            for region in regions:
                polygon = (region['polygon'] * factor).astype('int32')
                cv2.polylines(output, [polygon], True, (255, 220, 0), 2)
                original = region['original']
                translated = cache.get(original)
                label = translated or f'{original} ...'
                x, y = polygon.min(axis=0)
                draw_label(output, label, int(x), int(y) - 28, max_width=350, color=(0, 230, 255))
        else:
            self.last_regions = []
            self.tracker.reset(None, [], now)
        draw_label(output, status, 15, 95, max_width=output.shape[1] - 30, size=15)

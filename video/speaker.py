"""Visual speaker attribution using LBF landmarks and head-compensated LK flow.

This is a motion heuristic gated by audio VAD, not lip reading. Ambiguous,
occluded, stale, or off-camera speakers deliberately remain unassigned.
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from startup import require_local


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)


class FaceTracks:
    """Short-lived spatial IDs; never reuse an ID after a track expires."""

    def __init__(self, max_age=0.5):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.gray = None

    def _predict(self, gray):
        if self.gray is None or gray is None or self.gray.shape != gray.shape:
            return set()
        tracked = set()
        for tid, old in self.tracks.items():
            x, y, w, h = map(int, old['face']['bbox'])
            mask = np.zeros_like(self.gray)
            cv2.rectangle(mask, (max(0, x), max(0, y)), (x + w, y + h), 255, -1)
            points = cv2.goodFeaturesToTrack(self.gray, 35, .02, 5, mask=mask)
            if points is None or len(points) < 6:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None)
            if moved is None:
                continue
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None)
            if back is None:
                continue
            valid = (status.ravel() != 0) & (back_status.ravel() != 0)
            valid &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5
            if valid.sum() < 6:
                continue
            matrix, inliers = cv2.estimateAffinePartial2D(points[valid], moved[valid], method=cv2.RANSAC)
            if matrix is None or inliers.sum() < 6 or not .8 < np.linalg.norm(matrix[:, 0]) < 1.25:
                continue
            corners = cv2.transform(np.float32([[[x,y],[x+w,y+h]]]), matrix)[0]
            left, top = corners.min(axis=0)
            right, bottom = corners.max(axis=0)
            left, top = max(0, left), max(0, top)
            right, bottom = min(gray.shape[1] - 1, right), min(gray.shape[0] - 1, bottom)
            if right - left < 25 or bottom - top < 25:
                continue
            old['face']['bbox'] = (float(left), float(top), float(right-left), float(bottom-top))
            tracked.add(tid)
        return tracked

    def update(self, faces, now, gray=None):
        self.tracks = {k: v for k, v in self.tracks.items() if now - v['seen'] <= self.max_age}
        predicted = self._predict(gray)
        self.gray = gray
        pairs = []
        for i, face in enumerate(faces):
            for tid, old in self.tracks.items():
                if face['id'] >= 0 and old['face']['id'] >= 0 and face['id'] != old['face']['id']:
                    continue
                overlap = iou(face['bbox'], old['face']['bbox'])
                if overlap > 0.25:
                    pairs.append((overlap, i, tid))
        ambiguous = set()
        for i in range(len(faces)):
            scores = sorted([score for score, index, _ in pairs if index == i], reverse=True)
            if len(scores) > 1 and scores[0] - scores[1] < .1:
                ambiguous.add(i)
        pairs = [pair for pair in pairs if pair[1] not in ambiguous]
        assigned, used = {}, set()
        for _, i, tid in sorted(pairs, reverse=True):
            if i not in assigned and tid not in used:
                assigned[i] = tid
                used.add(tid)
        result = []
        for i, face in enumerate(faces):
            tid = assigned.get(i)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            face = dict(face, track_id=tid)
            old = self.tracks.get(tid)
            votes = old['votes'] if old else deque(maxlen=5)
            votes.append((face['id'], face['name'], face['language']))
            identities = [v for v in votes if v[0] >= 0]
            if identities:
                choice = max(set(identities), key=identities.count)
                if identities.count(choice) >= 3:
                    face.update(id=choice[0], name=choice[1], language=choice[2])
                else:
                    face.update(id=-1, name='Unknown', language=None)
            face['tracked_only'] = False
            self.tracks[tid] = {'face': dict(face), 'seen': now, 'votes': votes}
            result.append(face)
        for tid in predicted - used - {f['track_id'] for f in result}:
            old = self.tracks[tid]
            if now - old['seen'] < .35 and not any(iou(old['face']['bbox'], f['bbox']) > .2 for f in result):
                result.append(dict(old['face'], tracked_only=True, motion=0.0, active_speaker=False))
        return result


def compensated_motion(before, after, valid, width, dt):
    """Return lip velocity after robustly subtracting head translation/rotation."""
    valid = np.asarray(valid, bool).reshape(-1)
    head = np.arange(27, 48)[valid[27:48]]
    lips = np.arange(48, 68)[valid[48:68]]
    if len(head) < 8 or len(lips) < 8 or not 0 < dt <= 0.5:
        return 0.0
    matrix, inliers = cv2.estimateAffinePartial2D(
        before[head], after[head], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None or inliers.sum() < 6:
        return 0.0
    expected = cv2.transform(before[None, lips], matrix)[0]
    residual = np.linalg.norm(after[lips] - expected, axis=1)
    # Subpixel flow noise is not evidence of speech.
    return float(max(0, np.median(residual) - 0.15) / max(width, 1) / dt)


class SpeakerDecision:
    def __init__(self, threshold=0.035, dwell=0.18):
        self.threshold, self.dwell = threshold, dwell
        self.history = {}
        self.candidate = None
        self.since = 0.0

    def update(self, faces, speech_active, now):
        visible = {f['track_id'] for f in faces}
        self.history = {k: v for k, v in self.history.items() if k in visible}
        for face in faces:
            history = self.history.setdefault(face['track_id'], deque())
            history.append((now, face.get('motion', 0.0)))
            while history and now - history[0][0] > 0.35:
                history.popleft()
            face['speaker_score'] = float(np.mean([s for _, s in history]))
            face['active_speaker'] = False
        ranked = sorted(faces, key=lambda f: f['speaker_score'], reverse=True)
        winner = None
        if speech_active and ranked and not ranked[0].get('tracked_only') and ranked[0]['speaker_score'] >= self.threshold:
            first = ranked[0]['speaker_score']
            second = ranked[1]['speaker_score'] if len(ranked) > 1 else 0.0
            if first >= second * 1.6 and first - second >= self.threshold * 0.5:
                winner = ranked[0]
        tid = winner['track_id'] if winner else None
        if tid != self.candidate:
            self.candidate, self.since = tid, now
        if winner is None or now - self.since < self.dwell:
            return '', None
        winner['active_speaker'] = True
        name = winner['name'] if winner['name'] != 'Unknown' else f"Person {tid}"
        return name, tid


class VisualSpeaker:
    def __init__(self, model_path, threshold=0.035):
        if not Path(model_path).is_file():
            raise FileNotFoundError('Missing mouth landmark model. Run python scripts/download_vision_models.py speaker')
        require_local(model_path)
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(model_path))
        self.previous = {}
        self.decision = SpeakerDecision(threshold)

    def update(self, gray, faces, speech_active, now):
        old, self.previous = self.previous, {}
        if faces:
            boxes = np.array([f['bbox'] for f in faces], dtype=np.int32)
            ok, shapes = self.facemark.fit(gray, boxes)
            if ok:
                for face, shape in zip(faces, shapes):
                    points = np.asarray(shape, np.float32).reshape(68, 2)
                    face['landmarks'] = points
                    face['motion'] = 0.0
                    previous = old.get(face['track_id'])
                    if previous is not None:
                        prev_gray, prev_points, timestamp = previous
                        dt = now - timestamp
                        if prev_gray.shape == gray.shape and 0 < dt <= 0.5:
                            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                                prev_gray, gray, prev_points[:, None], None,
                                winSize=(15, 15), maxLevel=2)
                            if moved is not None:
                                returned, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                                    gray, prev_gray, moved, None, winSize=(15, 15), maxLevel=2)
                                if returned is not None:
                                    valid = status.ravel().astype(bool) & reverse_status.ravel().astype(bool)
                                    valid &= np.linalg.norm(returned[:, 0] - prev_points, axis=1) < 1.5
                                    face['motion'] = compensated_motion(
                                        prev_points, moved[:, 0], valid, face['bbox'][2], dt)
                                    face['flow_from'], face['flow_to'] = prev_points[valid], moved[:, 0][valid]
                    self.previous[face['track_id']] = (gray, points, now)
        return self.decision.update(faces, speech_active, now)

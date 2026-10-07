"""
Face detection, recognition and speaker tracking (pure OpenCV).

* Detection  : Haar cascade (frontal face)
* Recognition: LBPH recogniser (opencv-contrib)
* Speaker attribution: video.speaker uses mouth flow, head compensation and VAD.

The registry persists faces to disk so registered people survive restarts.
Registration itself lives in ``register.py``.
"""

import os
import json
import shutil

import cv2
import numpy as np

from config import (
    FACES_DIR,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE,
    FACE_RECOGNITION_THRESHOLD,
    FACE_SAMPLE_SIZE,
)
from utils.logging_utils import get_logger
from video.text import draw_label

log = get_logger("Faces")


class FaceRegistry:
    """Enrol and recognise faces with OpenCV LBPH."""

    def __init__(self, data_dir: str = FACES_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # First try local xml file, then fallback to cv2.data (fixes Mac pip installation issues)
        local_cascade = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "haarcascade_frontalface_default.xml")
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Face detector could not load {cascade_path}. "
                "Reinstall opencv-contrib-python in your active environment."
            )
        self.recognizer = self._new_recognizer()

        self.persons: dict[str, dict] = {}     # {id: {"name","language"}}
        self.next_id = 0
        self.is_trained = False
        self._load()

    @staticmethod
    def _new_recognizer():
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "Face recognition requires opencv-contrib-python. Uninstall "
                "opencv-python/opencv-python-headless and reinstall opencv-contrib-python."
            )
        return cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8,
            threshold=float(FACE_RECOGNITION_THRESHOLD),
        )

    # ── detection / recognition ──────────────────────────────────────
    def detect(self, frame: np.ndarray, min_size=None):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
            minSize=min_size or FACE_DETECTION_MIN_SIZE,
        )
        return gray, rects

    def identify(self, frame: np.ndarray, min_size=None) -> list[dict]:
        gray, rects = self.detect(frame, min_size)
        results = []
        for (x, y, w, h) in rects:
            roi = cv2.resize(gray[y:y + h, x:x + w], FACE_SAMPLE_SIZE)
            entry = {"bbox": (int(x), int(y), int(w), int(h)),
                     "name": "Unknown", "language": None, "confidence": 0.0, "id": -1}
            if self.is_trained:
                label, dist = self.recognizer.predict(roi)
                if dist < FACE_RECOGNITION_THRESHOLD:
                    person = self.persons.get(str(label))
                    if person:
                        entry.update(name=person["name"], language=person["language"],
                                     confidence=round(max(0, 100 - dist), 1), id=label)
            results.append(entry)
        return results

    # ── registration ─────────────────────────────────────────────────
    def register_person(self, name: str, language: str, frames: list[np.ndarray]):
        person_id = self.next_id
        self.persons[str(person_id)] = {"name": name, "language": language}
        self.next_id += 1
        person_dir = os.path.join(self.data_dir, str(person_id))
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for frame in frames:
            gray, rects = self.detect(frame)
            for (x, y, w, h) in rects:
                roi = cv2.resize(gray[y:y + h, x:x + w], FACE_SAMPLE_SIZE)
                cv2.imwrite(os.path.join(person_dir, f"face_{saved}.jpg"), roi)
                saved += 1

        if saved == 0:
            del self.persons[str(person_id)]
            self.next_id -= 1
            shutil.rmtree(person_dir, ignore_errors=True)
            return None, 0

        self._retrain()
        self._save_metadata()
        return person_id, saved

    def get_all_persons(self) -> dict[int, dict]:
        return {int(k): v for k, v in self.persons.items()}

    # ── persistence ──────────────────────────────────────────────────
    def _retrain(self):
        faces, labels = [], []
        for pid in self.persons:
            pdir = os.path.join(self.data_dir, pid)
            if not os.path.isdir(pdir):
                continue
            for fn in sorted(os.listdir(pdir)):
                if fn.endswith(".jpg"):
                    img = cv2.imread(os.path.join(pdir, fn), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(cv2.resize(img, FACE_SAMPLE_SIZE))
                        labels.append(int(pid))
        if faces:
            self.recognizer = self._new_recognizer()
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(os.path.join(self.data_dir, "model.yml"))
            self.is_trained = True

    def _save_metadata(self):
        with open(os.path.join(self.data_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"persons": self.persons, "next_id": self.next_id}, f,
                      indent=2, ensure_ascii=False)

    def _load(self):
        meta = os.path.join(self.data_dir, "metadata.json")
        model = os.path.join(self.data_dir, "model.yml")
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.persons = data.get("persons", {})
            self.next_id = data.get("next_id", 0)
        if os.path.exists(model) and self.persons:
            self.recognizer.read(model)
            self.is_trained = True
            log.info("Loaded %d registered face(s).", len(self.persons))


# ── overlay rendering ────────────────────────────────────────────────

COLOR_UNKNOWN  = (110, 110, 110)
COLOR_KNOWN    = (234, 126, 102)     # indigo-ish in BGR
COLOR_SPEAKING = (0, 215, 255)       # gold


def annotate(frame: np.ndarray, faces: list[dict], state) -> np.ndarray:
    """Draw boxes, the active-speaker banner, live caption and status."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]
    speaking = state.speech_active if state else False
    speaker = state.current_speaker if state else ""

    for f in faces:
        x, y, w, h = f["bbox"]
        is_speaker = speaking and f.get("active_speaker", False)
        if is_speaker:
            color = COLOR_SPEAKING
        elif f["name"] == "Unknown":
            color = COLOR_UNKNOWN
        else:
            color = COLOR_KNOWN

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3 if is_speaker else 2)

        display_name = f["name"] if f["name"] != "Unknown" else f"Person {f.get('track_id', '?')}"
        label = display_name if not f["language"] else f"{display_name} ({f['language']})"
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        ly = y - 10 if y - 25 > 0 else y + h + tw[1] + 12
        ry1, ry2 = (ly - tw[1] - 6, ly + 6)
        cv2.rectangle(out, (x, ry1), (x + tw[0] + 12, ry2), color, -1)
        cv2.putText(out, label, (x + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

    # active-speaker banner: "Aadit is Speaking…"
    if speaking:
        banner = f"{speaker} is Speaking..." if speaker else "Speaker uncertain / off camera"
        cv2.putText(out, banner, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    COLOR_SPEAKING, 2)

    # status + FPS top-left
    status = "TRANSLATING..." if speaking else "LISTENING"
    scol = COLOR_SPEAKING if speaking else (0, 200, 100)
    cv2.putText(out, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scol, 2)
    if state:
        cv2.putText(out, f"FPS: {state.fps:4.1f}", (w_img - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Captions follow the track recorded with the utterance, not today's speaker.
    if state:
        draw_captions(out, faces, state.get_captions())
    return out


def caption_anchor(caption, faces):
    return next((f for f in faces if caption.track_id is not None and f.get('track_id') == caption.track_id), None)


def draw_captions(out, faces, captions):
    placed = []
    unassigned = []
    for caption in captions:
        face = caption_anchor(caption, faces)
        if face is None:
            unassigned.append(caption)
            continue
        x, y, w, h = map(int, face['bbox'])
        text = f'{caption.speaker}: {caption.translated}' if caption.speaker else caption.translated
        # Try below the face, then above it, keeping nearby bubbles separated.
        anchor_y = y + h + 24
        if anchor_y + 100 > out.shape[0]:
            anchor_y = max(115, y - 110)
        for px, py, pw, ph in placed:
            if abs(x - px) < 280 and abs(anchor_y - py) < ph:
                anchor_y = py - 100 if py >= 215 else py + ph + 4
        placed.append(draw_label(out, text, x, anchor_y, max_width=280, size=18, color=(240, 240, 255)))
    if unassigned:
        latest = max(unassigned, key=lambda cap: cap.ts)
        name = latest.speaker or 'Speaker unassigned'
        draw_label(out, f'{name}: {latest.translated}', 15, out.shape[0] - 90,
                   max_width=out.shape[1] - 30, size=20)

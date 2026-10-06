"""
Thread 1 — Video.

Runs continuously and independently of the speech pipeline:
    read webcam → detect & recognise faces → pick active speaker → annotate.

It never does speech recognition or translation.  The annotated frame is
written to a single shared slot; the MAIN thread reads that slot and calls
cv2.imshow (OpenCV windows must be driven from the main thread).
"""

import threading
import time

import cv2
import numpy as np
from video.device import open_camera

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, VIDEO_TARGET_FPS,
)
from video.tracking import FaceRegistry, annotate, pick_speaker
from utils.logging_utils import get_logger

log = get_logger("Video")


class VideoCapture:
    """Owns one thread: camera → face recognition → annotated frame slot."""

    def __init__(self, state, camera_index: int = CAMERA_INDEX):
        self.state = state
        self.camera_index = camera_index
        self.registry = FaceRegistry()

        self._cam: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._running:
            return True
        try:
            cam = open_camera(self.camera_index)
        except RuntimeError as exc:
            self.error = str(exc)
            log.error(self.error)
            return False
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cam = cam
        self._running = True
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._thread.start()
        log.info("Camera open (index %d).", self.camera_index)
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        frame_period = 1.0 / max(1, VIDEO_TARGET_FPS)
        fps_t0, fps_n = time.time(), 0
        while self._running and self._cam is not None:
            ok, frame = self._cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            faces = self.registry.identify(frame)
            # update active speaker only while the mic reports speech
            if self.state.speech_active:
                self.state.set_current_speaker(pick_speaker(faces))

            annotated = annotate(frame, faces, self.state)
            with self._lock:
                self._frame = annotated

            # fps bookkeeping
            fps_n += 1
            if fps_n >= 15:
                now = time.time()
                fps = fps_n / (now - fps_t0)
                self.state.set_fps(fps)
                log.debug("FPS: %.1f", fps)
                fps_t0, fps_n = now, 0

            time.sleep(frame_period)

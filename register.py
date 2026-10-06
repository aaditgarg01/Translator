"""
Register a participant's face so the live system can label them
("Aadit is Speaking").

Usage:
    python register.py --name Aadit --language English
    python register.py --name Yuki --language Japanese --camera 0

Look at the camera; the script grabs frames whenever exactly one face is in
view, then trains the recogniser.  Press 'q' to abort.
"""

import argparse
import time

import cv2

import config
from video.tracking import FaceRegistry
from video.device import open_camera
from utils.logging_utils import get_logger

log = get_logger("Register")


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a face for the ESA project translator.")
    ap.add_argument("--name", required=True, help="Person's display name, e.g. Aadit")
    ap.add_argument("--language", required=True, choices=list(config.LANGUAGE_CODES),
                    help="The language this person speaks")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--count", type=int, default=config.FACE_CAPTURE_COUNT)
    args = ap.parse_args()

    if args.count < 1:
        ap.error("--count must be positive")
    cam = open_camera(args.camera)

    try:
        registry = FaceRegistry()
        frames, last_grab = [], 0.0
        log.info("Capturing %d face frames for '%s' (%s). Look at the camera…",
                 args.count, args.name, args.language)

        while len(frames) < args.count:
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames; check its connection.")
            _, rects = registry.detect(frame)
            preview = frame.copy()

            now = time.time()
            if len(rects) == 1 and now - last_grab > 0.15:
                frames.append(frame.copy())
                last_grab = now
                (x, y, w, h) = rects[0]
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 215, 255), 2)
            elif len(rects) != 1:
                cv2.putText(preview, "Show exactly ONE face", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(preview, f"Captured {len(frames)}/{args.count}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            cv2.imshow("Register — press q to abort", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                log.info("Aborted.")
                return

    finally:
        cam.release()
        cv2.destroyAllWindows()

    pid, saved = registry.register_person(args.name, args.language, frames)
    if pid is None:
        log.error("Registration failed — no faces were detected in the captures.")
    else:
        log.info("Registered '%s' (id=%d) with %d face samples.", args.name, pid, saved)


if __name__ == "__main__":
    main()

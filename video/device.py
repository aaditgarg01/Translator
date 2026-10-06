"""Select a native camera backend without trying Windows APIs on macOS."""

import sys
import cv2


def open_camera(index: int):
    backend = {
        "win32": cv2.CAP_DSHOW,
        "darwin": cv2.CAP_AVFOUNDATION,
    }.get(sys.platform, cv2.CAP_ANY)
    for api in dict.fromkeys((backend, cv2.CAP_ANY)):
        cam = cv2.VideoCapture(index, api)
        if cam.isOpened():
            return cam
        cam.release()
    raise RuntimeError(
        f"Cannot open camera {index}. Close other camera apps, check camera "
        "privacy permissions, or run with --no-video."
    )

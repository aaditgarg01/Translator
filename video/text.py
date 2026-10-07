"""Cached Unicode text tiles composited into OpenCV BGR frames."""
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=16)
def font(size):
    candidates = [os.environ.get('TRANSLATOR_FONT', ''),
                  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                  '/System/Library/Fonts/Supplemental/Arial.ttf',
                  'C:/Windows/Fonts/arial.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=256)
def tile(text, size, width, color):
    face = font(size)
    lines, line = [], ''
    for char in str(text).replace('\n', ' ')[:400]:
        if face.getlength(line + char) > width - 16 and line:
            lines.append(line)
            line = ''
            if len(lines) == 3:
                break
        line += char
    if line and len(lines) < 3:
        lines.append(line)
    if not lines:
        lines = [' ']
    image = Image.new('RGB', (max(20, min(width, int(max(face.getlength(s) for s in lines)) + 16)), len(lines) * (size + 8) + 8))
    draw = ImageDraw.Draw(image)
    for i, value in enumerate(lines):
        draw.text((8, 4 + i * (size + 8)), value, font=face, fill=color[::-1])
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_label(frame, text, x, y, max_width=400, size=19, color=(255, 255, 255)):
    label = tile(text, size, max(20, min(max_width, frame.shape[1])), color)
    h, w = label.shape[:2]
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    h, w = min(h, frame.shape[0]), min(w, frame.shape[1])
    frame[y:y+h, x:x+w] = label[:h, :w]
    return (x, y, w, h)

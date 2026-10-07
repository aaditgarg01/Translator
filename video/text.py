"""Cached Unicode text tiles composited into OpenCV BGR frames."""
from functools import lru_cache
from pathlib import Path
import os
import unicodedata

import config

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from video.shaping import shaped, draw_shaped


def script_for(text):
    if any('\u0900' <= c <= '\u097f' for c in text):
        return 'devanagari'
    if any('\u0600' <= c <= '\u06ff' for c in text):
        return 'arabic'
    if any('\u3000' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in text):
        return 'cjk'
    return 'latin'


@lru_cache(maxsize=32)
def font(size, script='latin'):
    bundled = {'latin': 'NotoSans-Regular.ttf', 'arabic': 'NotoSansArabic-Regular.ttf',
               'devanagari': 'NotoSansDevanagari-Regular.ttf', 'cjk': 'NotoSansCJKjp-Regular.otf'}
    candidates = [os.environ.get('TRANSLATOR_FONT', ''),
                  str(Path(config.MODELS_DIR) / 'vision' / bundled[script]),
                  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf']
    if script == 'latin':
        candidates += ['/System/Library/Fonts/Supplemental/Arial.ttf', 'C:/Windows/Fonts/arial.ttf',
                       '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size) if script == 'latin' else None


def text_clusters(text):
    clusters = []
    for char in text:
        attach = unicodedata.category(char).startswith('M') or char == '\u200d'
        if clusters and (attach or clusters[-1].endswith('\u200d') or unicodedata.combining(clusters[-1][-1]) == 9):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


@lru_cache(maxsize=256)
def tile(text, size, width, color):
    face = font(size, script_for(text))
    if face is None:
        text = "Install subtitle fonts: scripts/download_vision_models.py fonts"
        face = font(size)
    complex_script = script_for(text) in {'arabic', 'devanagari'} and hasattr(face, 'path') and isinstance(face.path, str)
    measure = (lambda value: shaped(value, face.path, size)[1]) if complex_script else face.getlength
    lines, line = [], ''
    for char in text_clusters(str(text).replace('\n', ' ')[:400]):
        if measure(line + char) > width - 16 and line:
            lines.append(line)
            line = ''
            if len(lines) == 3:
                lines[-1] = lines[-1][:-3] + "..."
                break
        line += char
    if line and len(lines) < 3:
        lines.append(line)
    if not lines:
        lines = [' ']
    line_height = max(size + 8, sum(face.getmetrics()) + 4) if complex_script else size + 8
    image = Image.new('RGB', (max(20, min(width, int(max(measure(s) for s in lines)) + 16)), len(lines) * line_height + 8))
    draw = ImageDraw.Draw(image)
    for i, value in enumerate(lines):
        if complex_script:
            draw_shaped(image, (8, 4 + i * line_height), value, face, size, color[::-1])
        else:
            draw.text((8, 4 + i * line_height), value, font=face, fill=color[::-1])
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_label(frame, text, x, y, max_width=400, size=19, color=(255, 255, 255)):
    label = tile(text, size, max(20, min(max_width, frame.shape[1])), color)
    h, w = label.shape[:2]
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    h, w = min(h, frame.shape[0]), min(w, frame.shape[1])
    frame[y:y+h, x:x+w] = label[:h, :w]
    return (x, y, w, h)

"""Offline model smoke test: no microphone, camera, translation model or downloads."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import config
from video.scene_text import SceneOCR
from video.speaker import VisualSpeaker
from video.text import font, script_for, tile
from video.shaping import shaped
from tts.voices import SAMPLE_TEXTS


def main():
    VisualSpeaker(config.LANDMARK_MODEL)
    ocr = SceneOCR(Path(config.MODELS_DIR) / 'vision')
    image = np.full((360,640,3),255,np.uint8)
    cv2.putText(image,'HELLO WORLD',(65,160),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
    started = time.perf_counter()
    texts = [item['original'] for item in ocr.read(image)]
    if 'HELLO WORLD' not in texts:
        raise RuntimeError(f'OCR smoke check failed: {texts}')
    print(f'OCR: {texts}; {(time.perf_counter()-started)*1000:.0f} ms', flush=True)
    for language, text in SAMPLE_TEXTS.items():
        face = font(24, script_for(text))
        if face is None:
            raise RuntimeError(f'Missing {language} font')
        glyphs, _ = shaped(text, face.path, 24)
        if any(glyph == 0 for glyph, _, _ in glyphs):
            raise RuntimeError(f'Missing glyph in {language} sample')
        rendered = tile(text,24,560,(255,255,255))
        if not rendered.any():
            raise RuntimeError(f'Empty {language} text rendering')
    print('Landmark model loaded; OCR and all 10 language glyph/render checks passed.')


if __name__ == '__main__':
    main()

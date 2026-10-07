"""Portable complex-script rendering when Pillow wheels lack libraqm.

HarfBuzz shapes logical glyph runs; Unicode BiDi orders the runs; FreeType
rasterizes their glyph IDs. No system font-shaping library is required.
"""
from functools import lru_cache
import unicodedata

import freetype
import uharfbuzz as hb
from bidi import algorithm as bidi
from PIL import Image


@lru_cache(maxsize=12)
def hb_font(path, size):
    font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)
    return font


def visual_runs(text):
    # Ignore bidi presentation controls from model output; derive direction from text.
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    if not text:
        return []
    storage = bidi.get_empty_storage()
    storage['base_level'] = bidi.get_base_level(text)
    storage['base_dir'] = 'R' if storage['base_level'] else 'L'
    bidi.get_embedding_levels(text, storage)
    bidi.explicit_embed_and_overrides(storage)
    bidi.resolve_weak_types(storage)
    bidi.resolve_neutral_types(storage, False)
    bidi.resolve_implicit_levels(storage, False)
    runs = []
    for char in storage['chars']:
        if not runs or runs[-1][0] != char['level']:
            runs.append([char['level'], ''])
        runs[-1][1] += char['ch']
        char['run'] = len(runs) - 1
    bidi.reorder_resolved_levels(storage, False)
    seen, visual = set(), []
    for char in storage['chars']:
        index = char['run']
        if index not in seen:
            seen.add(index)
            level, value = runs[index]
            visual.append((value, 'rtl' if level % 2 else 'ltr'))
    return visual


@lru_cache(maxsize=1024)
def shaped(text, path, size):
    font = hb_font(path, size)
    output, pen = [], 0.0
    for value, direction in visual_runs(text):
        buffer = hb.Buffer()
        buffer.add_str(value)
        buffer.guess_segment_properties()
        buffer.direction = direction
        hb.shape(font, buffer)
        for info, position in zip(buffer.glyph_infos, buffer.glyph_positions):
            output.append((info.codepoint, pen + position.x_offset / 64, position.y_offset / 64))
            pen += position.x_advance / 64
    return output, pen


def draw_shaped(image, xy, text, face, size, color):
    glyphs, _ = shaped(text, face.path, size)
    raster = freetype.Face(face.path)
    raster.set_char_size(size * 64)
    baseline = xy[1] + raster.size.ascender / 64
    for glyph_id, x, y in glyphs:
        raster.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)
        slot = raster.glyph
        bitmap = slot.bitmap
        if not bitmap.width or not bitmap.rows:
            continue
        raw = bytes(bitmap.buffer)
        if abs(bitmap.pitch) != bitmap.width:
            raw = b''.join(raw[i * abs(bitmap.pitch):i * abs(bitmap.pitch) + bitmap.width] for i in range(bitmap.rows))
        mask = Image.frombytes('L', (bitmap.width, bitmap.rows), raw)
        image.paste(color, (round(xy[0] + x + slot.bitmap_left), round(baseline - y - slot.bitmap_top)), mask)

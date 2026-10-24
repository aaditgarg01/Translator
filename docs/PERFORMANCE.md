# Processing changes and local checks

The meeting-minutes release removes repeated work before changing model quality.
It does not claim a fixed device-independent FLOP, energy or FPS improvement.

## Default processing

- Full Haar detection and LBPH recognition run every third analysis frame when
  a recent face track exists. Forward/backward-checked optical flow bridges the
  scheduled skips; failed detections remain distinct from scheduled skips.
  Empty scenes continue to look for newly visible faces. Set
  `VISION_DETECT_INTERVAL=1` to restore full scans on every analysis frame.
- Face feature extraction operates on the face region instead of allocating a
  full-frame mask and calculating features across the image for every person.
- Landmark and mouth-flow estimation are skipped during silence unless debug
  is visible. Their histories reset so stale movement cannot select a speaker.
- Exact text/source/target translations use a 256-entry LRU cache; exact spoken
  responses use a 4 MiB PCM cache with at most 32 entries. Both are per process.
  Repeated recognised turns still appear separately in the meeting transcript.
- Preview GUI updates display new frames and yield between event checks.
  Whisper's CPU budget defaults to two threads to reduce worker contention.
- `--minutes-only` omits NLLB translation and TTS entirely. Minutes use standard
  library extractive scoring and text cues; no summarisation model is loaded.

## Eco preset

`--performance eco` selects base Whisper, beam size one, 480-pixel-wide vision
and at most 10 analysis FPS. It also requests dynamic int8 NLLB Linear layers
on CPU. Preview still requests 30 FPS. The preset can change ASR and translation
quality and should be rehearsed with the intended accents and languages.

CUDA/MPS retain their normal translation path. Unsupported CPU quantization
preparation reloads a clean float32 model. `--no-translation-int8` disables that
option; `--whisper-model small` keeps the larger ASR model. Quantization adds
startup preparation time and is not a promise of lower latency on all CPUs.

## Local observations on 24 October 2026

These are camera-free checks on this Apple Silicon Mac, not a live meeting or
camera FPS benchmark. Face processing used two OpenCV threads and a stationary
640 by 360 frame containing a local face crop. The baseline was commit 6f99658.
The fixture and participant photographs are not published.

| Check | Baseline or first call | Optimized or repeat | Scope |
|---|---:|---:|---|
| Detection and tracking over 60 frames | 561.40 ms and 60 full scans | 113.03 ms and 20 full scans | Both retained a visible track in all 60 frames; stationary input |
| English speech generation | 1182.486 ms including voice load | 0.005 ms cached repeat | Identical output bytes; no second synthesis or resampling |
| Exact English to Japanese translation | 88364.792 ms including model load | 0.014 ms cached repeat | Same text and language pair; no second NLLB inference |
| CPU int8 translation of the same example | Float32 call above | 30408.85 ms including load and quantization | Both produced カメラをテストする for We will test the camera |

The translation checks used two PyTorch threads and available local model
caches. First-call timings include model preparation and can vary sharply
with memory pressure and cache state. One matching translation does not
establish multilingual quality equivalence. Cache-hit timings describe avoided
work and do not predict latency for a new sentence.

## Verification

Run `python -m unittest discover -s tests -v` and
`python scripts/test_vision_models.py`. Tests cover cache bounds and language
separation, intact repeated meeting turns, detector cadence, flow-only identity
handling, silence gating, export recovery, and queue-gap notices. The vision
smoke check exercises the real landmark and OCR models and all ten output
scripts. Physical microphone/speaker behavior, moving participants, native
language quality, preview FPS and end-to-end latency require hardware rehearsal.

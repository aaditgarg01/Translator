# ESA Real-Time Translator

Local speech translation with optional face recognition and audio streaming.
Microphone → speech detection → faster-whisper → NLLB translation → Piper or
Style-Bert-VITS2 → speaker and/or FFmpeg RTP.

The source and target are selected in `config.py` or with `--source` and
`--target`. Ten language routes are configured: English, Hindi, Japanese,
Chinese, Korean, Spanish, French, German, Portuguese and Arabic. Voice models
and pronunciation assets must be downloaded for each target. Inference runs
locally after setup; voice quality varies by model, dialect and input text.

## Multilingual voice quality

Install pronunciation extras and the updated French/German voices in the active
environment (on Windows use `.\.venv\Scripts\python.exe` for `python`):

```bash
python -m pip install -r requirements-voices.txt
python scripts/download_piper_voice.py fr_FR-siwis-medium de_DE-thorsten-high
```

French now uses Siwis; German uses the high-quality Thorsten model. These are
single-speaker alternatives to the previous 125/236-speaker MLS models. Old
models are kept locally, so you can switch back in `PIPER_VOICES` for comparison.
The additional seven language routes use sentence pauses, short edge fades,
and one volume normalization per utterance. English, Hindi and Japanese keep
their established synthesis settings. `PIPER_PROFILES` controls speaker ID,
speaking duration (`length_scale`) and sentence pauses.

Test a voice independently from microphone capture and translation:

```bash
python scripts/test_voice.py --language German --output german-test.wav --play
python scripts/test_voice.py --language French --output french-test.wav --play
python scripts/test_voice.py --language Chinese --output chinese-test.wav
```

Use `--text` for your own native-language sentence and `--output-device` to
select a speaker with `--play`. Omit `--play` to save a WAV without opening an
audio device. Chinese may download a large pronunciation model and tokenizer on
first use; its voice resources are kept under `models/piper/_resources`.
Startup now checks pronunciation before listening, so missing extras are
reported immediately. `--doctor` validates voice language and installed extras
without loading the voice or downloading assets.

If the WAV sounds smooth but live playback stutters, check the output-device
selection and watch for "Speaker buffer underrun" warnings. Playback uses
larger blocks and a robust output latency setting. This is distinct from
pronunciation quality, which depends on the model and native-language text.

## OpenCV feature 1: visual active-speaker detection

```bash
python scripts/download_vision_models.py speaker
python main.py --source English --target Japanese
```

The camera now uses 68 mouth/face landmarks and Lucas–Kanade optical flow,
subtracts head motion, and combines mouth motion with microphone VAD. A smaller
speaking face can win over a larger silent face. Two moving mouths, a missing
face or weak evidence produce **Speaker uncertain / off camera**. This is a
heuristic, not lip reading: chewing, occlusion and extreme angles can confuse it.
Speaker identity is saved with each utterance, so translation delays cannot
assign it to somebody who spoke later. Preview capture runs independently from
vision analysis with one pending frame, targeting 30 FPS without building a backlog.

Use `--no-visual-speaker` to disable attribution. A missing landmark model leaves
speakers unassigned and prints setup instructions; speech translation still works.
Tune `SPEAKER_MOTION_THRESHOLD` with your webcam and lighting. Local synthetic
motion tests and a real-model smoke test are included in the verification; live
multi-person accuracy and actual preview FPS require a camera test.

Model: [LBF / GSOC2017 implementation](https://github.com/kurnianggoro/GSOC2017),
pinned and checksum-verified by the downloader. Review upstream model/data terms
before redistribution. Models stay local under `models/vision`.

See [the OpenCV roadmap](docs/OPENCV_ROADMAP.md) for the next features.

## OpenCV feature 2: camera text translation

```bash
python -m pip install -r requirements.txt
python scripts/download_vision_models.py ocr
python main.py --scene-text --ocr-source English --target German
```

Press **T** in the camera window to toggle camera text. OpenCV DNN detects printed
text, corrects perspective, recognizes words and tracks the translated overlay
with optical flow. Text must repeat across two scans before translation. Results
are cached; camera text never speaks aloud or replaces conversation captions.
The existing translation worker gives queued speech priority, but a camera-text
translation already in progress must finish before the next speech translation.

The bundled detector is trained on English text. Its CRNN vocabulary covers
ASCII Latin letters, digits and punctuation, **not** accents, Devanagari, Arabic,
Japanese, Chinese or Korean input. `--ocr-source` sets the language for plain
Latin text; English is the tested starting point. Translation output can use any
configured target. OCR works best on clear horizontal or moderately tilted print;
handwriting, extreme perspective and blur are outside the tested scope.

The OpenCV Zoo models are Apache-2.0 licensed and pinned with SHA-256 checksums.
No images leave the device. Models download only through the setup script.

## OpenCV feature 3: participant tracks and anchored Unicode captions

```bash
python scripts/download_vision_models.py fonts
```

Participants keep a spatial track ID while visible. Optical flow bridges brief
missed detections; registered names need repeated recognition before appearing.
Ambiguous overlaps reset identity instead of forcing a match. This is short-term
tracking, not reliable re-identification after a person leaves the camera.

Each person's recent translation appears beside their own tracked face and
expires after eight seconds. If that track has disappeared, the caption moves
to the bottom with its original speaker label; it is never attached to another
person. Unregistered faces display `Person 1`, `Person 2`, etc.

Noto fonts support Latin, Hindi, Arabic, Japanese, Chinese and Korean output.
The downloader installs the fonts and their SIL Open Font License files locally.
Pillow, HarfBuzz, Unicode BiDi and FreeType render cached text tiles into the OpenCV image. Set
`TRANSLATOR_FONT` to an installed font path to override the bundled fonts.

## Quick start: macOS

Install Python **3.12** (3.11 also works). From this project folder:

```bash
python3.12 scripts/setup.py --download-voice
source .venv/bin/activate
python main.py --doctor
python main.py --no-video
```

Use `python3` instead if it already points to Python 3.12. Allow microphone and
camera access for the terminal/app running Python in **System Settings →
Privacy & Security**. Restart that terminal/app after changing permissions.

Apple Silicon runs faster-whisper on CPU; NLLB can use Apple's MPS backend.
Use `--cpu` if accelerated inference fails. Intel Macs need dependency wheels
compatible with their macOS/Python version; this is not covered by hosted CI.

## Quick start: Windows

Install Python **3.12**. In PowerShell, from this project folder:

```powershell
py -3.12 scripts/setup.py --download-voice
.\.venv\Scripts\python.exe main.py --doctor
.\.venv\Scripts\python.exe main.py --no-video
```

These commands work without changing PowerShell's script execution policy.
Allow desktop apps to access the microphone and camera under **Settings →
Privacy & security**. CPU inference works without CUDA; GPU acceleration is
optional. If CUDA libraries fail to load, use `--cpu`.

Do not copy a virtual environment between macOS and Windows. Run setup on each
computer. Python 3.11–3.13 are covered by core CI. The optional Style-Bert-VITS2 backend
still uses a separate Python 3.11/3.12 setup.

## Commands and devices

The examples below assume your virtual environment is active. On Windows you
can use `.\.venv\Scripts\python.exe` in place of `python`.

```bash
python main.py --list-devices
python main.py --doctor --no-video
python main.py --no-video --input-device 1 --output-device 3
python main.py --source Hindi --target English
python main.py --target Hindi --no-video
python main.py --no-tts                  # translated text in logs/camera captions
python main.py --cpu                     # CPU inference for every model
python main.py --camera 1
```

Device names (in quotes) also work in place of device indexes. `--doctor` checks
installed modules, voice files, OpenCV and audio settings without downloading
models or recording. It does not certify inference or camera permissions.
The first normal run can take several minutes to download/load Whisper and NLLB.
Subsequent runs reuse the Hugging Face cache.

Press **Ctrl-C** to quit, or **q / Escape** in the camera window. Startup failures
and shutdown both release opened devices and stop FFmpeg. Required model or
output failures produce an error and a nonzero exit code.

## Voices

### English and Hindi (Piper)

```bash
python scripts/download_piper_voice.py en_US-amy-medium
python scripts/download_piper_voice.py hi_IN-pratham-medium
```

Both the `.onnx` model and matching `.onnx.json` file are required in
`models/piper/`. Downloads use certificate verification and temporary files so
an interrupted download does not replace an existing complete file.

### Japanese and Chinese (Piper)

```bash
python -m pip install -r requirements-voices.txt
python scripts/download_piper_voice.py ja_JP-hi_fi_captain-medium zh_CN-chaowen-medium
python main.py --target Japanese --source English --no-video
```

Japanese requires Piper's OpenJTalk extension. Chinese uses g2pW-based
pronunciation, including its model and tokenizer on first use. A missing engine
never substitutes an English voice. The legacy Style-Bert-VITS2 backend remains
available through `requirements-japanese.txt` and `download_sbv2_model.py`;
using it also requires setting that language's `TTS_ROUTES` entry to `sbv2`.

## Face registration

```bash
python register.py --name Aadit --language English
python main.py
```

Show one face until capture finishes. Camera selection uses AVFoundation on
macOS and DirectShow on Windows, with a default-backend fallback. Audio-only
mode works without a camera. New registrations are ignored by Git; existing
tracked sample data is retained.

## VLC / OBS streaming

FFmpeg is optional for local playback. Install it for RTP:

```bash
brew install ffmpeg                  # macOS, with Homebrew installed
winget install Gyan.FFmpeg            # Windows
```

```bash
python main.py --stream both          # local speaker + RTP
python main.py --stream rtp           # RTP only
```

Open the generated `stream_English.sdp` (or the selected language's file) in
VLC. It appears after FFmpeg receives audio. If VLC blocks RTP, launch it with
`--demux=rtp`. Configure `RTP_HOST`, `RTP_PORT`, `RTP_CODEC`, and `FFMPEG_BIN` in
`config.py`. The default destination is `127.0.0.1:5004`.

`both` mode keeps the working output if one fails and logs the failure. `rtp`
mode fails if FFmpeg is unavailable. FFmpeg error details are reported instead
of discarded. RTP playback latency depends on the receiver; headphones avoid
feedback from remote speaker audio.

## Troubleshooting

| Symptom | Action |
|---|---|
| Wrong Python / missing modules | Use the environment created by `scripts/setup.py`; `--doctor` prints its interpreter path. |
| Cannot open microphone/speaker | Check OS privacy permissions, run `--list-devices`, and select a device. Try a built-in or USB device rather than a Bluetooth hands-free profile. |
| No speech output | Run `--doctor`; check both voice files exist and the right speaker is selected. |
| OpenCV has no `face` or cannot load its cascade | Remove conflicting `opencv-python` / `opencv-python-headless` packages, then reinstall `opencv-contrib-python` from the relevant requirements file. |
| CUDA / MPS error | Start with `--cpu`; get the CPU pipeline working before configuring GPU acceleration. |
| English input is repeated in English | That is expected with the English target. Choose another target to translate. |
| Missing Japanese package/model | Use the optional Japanese setup above with Python 3.11/3.12. |
| Download fails | Check connectivity/proxy certificates and retry; do not disable TLS verification. |

## Architecture and configuration

`config.py` controls language routing, models, camera, devices, VAD and queue
sizes. Command-line options override settings for one run. `ENABLE_*` switches
allow subsystem testing; disabling a stage does not automatically bypass it.

| Stage | File |
|---|---|
| Camera and overlay | `video/camera.py`, `video/tracking.py` |
| Microphone and speech detection | `audio/capture.py`, `audio/vad.py` |
| Transcription | `whisper/engine.py` |
| Translation | `translate/translator.py` |
| Speech synthesis | `tts/engine.py`, `tts/piper_backend.py`, `tts/sbv2_backend.py` |
| Audio output | `stream/ffmpeg.py` |

VAD runs on a worker thread, outside the microphone callback. Bounded queues
drop stale work to limit latency. Microphone capture is gated during speech
playback and a short echo tail. Unsupported source languages are skipped.

Limitations: speaker attribution uses the largest recognized face, not lip
synchronization. OpenCV's overlay font displays a placeholder for Japanese and
Hindi captions; full text remains in logs and TTS. Speech recognition and
translation can make errors. Consult the individual model licenses before
redistributing models or using them commercially.

## Tests

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Tests cover cleanup, failed startup, audio callbacks and echo gating, speech
segmentation, translation routing, Piper APIs, camera backend selection,
streaming failures, resampling and incomplete downloads. They do not require
model downloads or access to a camera/microphone. GitHub Actions installs the
core dependencies and runs them on macOS and Windows with Python 3.11–3.13.
A live hardware test is still required on each target computer.

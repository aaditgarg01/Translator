# ESA Real-Time Translator

Local speech translation with optional face recognition and audio streaming.
Microphone → speech detection → faster-whisper → NLLB translation → Piper or
Style-Bert-VITS2 → speaker and/or FFmpeg RTP.

The default target is **English**, the source is detected automatically, and
output plays through your local speaker. English, Hindi and Japanese are the
configured languages. Models download during setup/first use; inference then
runs locally from the downloaded files.

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
computer. Python 3.13 is supported for the core English/Hindi setup; use 3.11
or 3.12 for the supported Japanese setup.

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

### Japanese (optional Style-Bert-VITS2)

Run setup with Python 3.11 or 3.12:

```bash
# macOS
python3.12 scripts/setup.py --japanese --download-voice
# Windows
py -3.12 scripts/setup.py --japanese --download-voice
```

Then use that environment:

```bash
python main.py --target Japanese --doctor
python main.py --target Japanese --no-video
```

Alternatively, install `requirements-japanese.txt` and run
`scripts/download_sbv2_model.py` manually. The optional requirements retain
NumPy 1.x and a compatible OpenCV release for the Japanese native dependencies.
BERT assets download on first use. Japanese uses a Japanese voice; a missing
engine never silently substitutes an English voice. Japanese setup requires
additional native dependencies and is not part of the core CI matrix.

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

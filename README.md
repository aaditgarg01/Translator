# ESA project Real-Time Translator

> Live face-aware speech translation. A person speaks → their face is
> recognised → speech is transcribed → translated → spoken back in the target
> language, and streamed to VLC / OBS. Everything runs locally.

This is the fresh, **multithreaded queue-based** rebuild. Six independent
threads are connected by queues — nothing waits for anything else, so one slow
translation never freezes the camera or the mic.

```
                 CAMERA ──► Video thread ──► overlay window ("Aadit is Speaking")
                                              (face detect + recognise + track)

 MIC ──► Audio thread ──► [audio_q] ──► Whisper ──► [transcript_q] ──►
        (Silero VAD)                    (GPU ASR)

      ──► Translate ──► [translation_q] ──► Piper TTS ──► [pcm_q] ──►
          (NLLB-200)                        (local)

      ──► Stream thread ──► speaker  and/or  FFmpeg → RTP → VLC / OBS
```

Every arrow is a `queue.Queue`. See [`main.py`](main.py) for the wiring.

---

## Architecture

| Thread | Class | File | Job |
|---|---|---|---|
| 1 Video | `VideoCapture` | [video/camera.py](video/camera.py) | webcam, face recognition, speaker overlay |
| 2 Audio | `AudioCapture` | [audio/capture.py](audio/capture.py) | mic @16 kHz, Silero VAD → speech segments |
| 3 Whisper | `WhisperEngine` | [whisper/engine.py](whisper/engine.py) | faster-whisper ASR (GPU) |
| 4 Translate | `Translator` | [translate/translator.py](translate/translator.py) | NLLB-200 → target language |
| 5 TTS | `TTS` | [tts/engine.py](tts/engine.py) | routed TTS → raw PCM (models loaded **once**) |
| 6 Stream | `AudioStreamer` | [stream/ffmpeg.py](stream/ffmpeg.py) | speaker / FFmpeg RTP (started **once**) |

TTS is multi-engine: [tts/sbv2_backend.py](tts/sbv2_backend.py) (Style-Bert-VITS2,
Japanese/Chinese) and [tts/piper_backend.py](tts/piper_backend.py) (Piper,
English/Hindi), unified to one sample rate by [tts/resample.py](tts/resample.py).

Shared, lock-guarded state (speaking flag, active speaker, live caption) lives
in [utils/state.py](utils/state.py); the four queues in [utils/queues.py](utils/queues.py).

**Workload balance** — GPU: OpenCV + faster-whisper. CPU: Silero VAD, NLLB,
Piper, FFmpeg.

---

## Setup

> **Use Python 3.12 or 3.11 — not 3.13.** Style-Bert-VITS2 depends on
> `pyopenjtalk-dict`, which only ships prebuilt Windows wheels up to cp312. On
> 3.13 pip tries to compile it from C++ source and fails. Create the venv with
> `py -3.12 -m venv venv`.

### 1. Environment
```bash
py -3.12 -m venv venv        # Windows (3.12 or 3.11)
# python -m venv venv        # if your default python is already 3.12/3.11
venv\Scripts\activate            # Windows
pip install -r requirements.txt
# GPU (recommended): install CUDA Torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. FFmpeg (only for RTP streaming)
```bash
winget install Gyan.FFmpeg       # Windows
```
Not needed if you keep `STREAM_MODE = "local"`.

### 3. TTS models for the target language

TTS is **routed by language** (`TTS_ROUTES` in config.py):

| Target language | Engine | Why |
|---|---|---|
| Japanese / Chinese | Style-Bert-VITS2 | Piper has no Japanese voice; SBV2 is the most natural |
| English / Hindi | Piper | fast, fully local |

Engines emit different sample rates; everything is resampled to
`OUTPUT_SAMPLE_RATE` (48 kHz) so the streamer opens once.

Download whatever your target needs:
```bash
# Japanese (default target) → Style-Bert-VITS2 voice
python scripts/download_sbv2_model.py

# English / Hindi → Piper voices
python scripts/download_piper_voice.py en_US-amy-medium
python scripts/download_piper_voice.py hi_IN-pratham-medium
```
The Japanese/Chinese BERT models SBV2 needs download automatically on first run.

---

## Usage — build it up one milestone at a time

The `ENABLE_*` flags in [config.py](config.py) let you verify each subsystem in
isolation before combining them.

**Milestone 1 — speech pipeline (no camera).** Set `ENABLE_VIDEO = False`:
```bash
python main.py
```
Speak → watch the `[Audio] [Whisper] [Translate] [TTS] [Stream]` logs and hear
the translation.

**Milestone 2 — computer vision.** Register faces, then enable video:
```bash
python register.py --name Aadit --language English
python register.py --name Yuki --language Japanese
# set ENABLE_VIDEO = True
python main.py
```
The window shows bounding boxes and **"Aadit is Speaking…"** while he talks.

**Milestone 3 — production.** Flip everything on, set `STREAM_MODE = "rtp"`,
and open the generated `stream.sdp` in VLC (`Media ▸ Open File ▸ stream.sdp`)
or add it as a source in OBS.

Press **q** or **ESC** in the window (or Ctrl-C) to quit.

---

## Configuration cheatsheet ([config.py](config.py))

| Setting | Default | Purpose |
|---|---|---|
| `TARGET_LANGUAGE` | `"Japanese"` | everything is translated into this |
| `ENABLE_*` | `True` | turn subsystems on/off for milestone testing |
| `VAD_BACKEND` | `"silero"` | `"energy"` falls back if Silero/torch missing |
| `WHISPER_MODEL_SIZE` | `"medium"` | `tiny`→`large-v3` accuracy vs speed |
| `STREAM_MODE` | `"local"` | `"local"` / `"rtp"` / `"both"` |
| `TTS_ROUTES` | — | language → TTS engine (`sbv2` / `piper`) |
| `OUTPUT_SAMPLE_RATE` | `48000` | unified rate fed to speaker / RTP |

---

## Notes & limitations

- The OpenCV overlay font is Latin-only; non-Latin captions (JA/HI) show a
  placeholder on screen but are spoken correctly by the TTS engine.
- Style-Bert-VITS2 covers Japanese / English / Chinese — **not Hindi**. Hindi
  output is handled by Piper. If you need a Hindi target, it routes to Piper
  automatically via `TTS_ROUTES`.
- Speaker attribution = the largest recognised face while the mic detects
  speech (no lip-sync). Good enough for a 1-on-1 / small-room demo.
- The old Streamlit prototype (`c_s_2.py`, `map_voice.py`, `src/`) is kept for
  reference and is independent of this new package.

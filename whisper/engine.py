"""
Thread 3 — Whisper.

Consumes audio segments, runs faster-whisper (GPU when available), and puts a
transcript dict onto the transcript queue:

    audio_queue  →  WhisperEngine  →  transcript_queue
                                       {"text", "language", "confidence"}

faster-whisper is CTranslate2-optimised, so this is the GPU-heavy stage that
runs comfortably alongside the CPU stages (VAD, translation, TTS).
"""

import threading
import queue
import time

import numpy as np

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    SOURCE_LANGUAGE,
    LANGUAGE_CODES,
)
from utils.queues import put_drop_oldest
from utils.state import AudioSegment
from utils.logging_utils import get_logger

log = get_logger("Whisper")


class WhisperEngine:
    """Owns one thread: audio_queue → faster-whisper → transcript_queue."""

    def __init__(self, audio_queue, transcript_queue, state=None):
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.state = state
        self.model = None
        self._running = False
        self._thread: threading.Thread | None = None

        # fixed source language if the user pinned one
        self._forced_lang = None
        if SOURCE_LANGUAGE and SOURCE_LANGUAGE.lower() != "auto":
            codes = LANGUAGE_CODES.get(SOURCE_LANGUAGE)
            self._forced_lang = codes["whisper"] if codes else None

    # ── model ────────────────────────────────────────────────────────
    def load(self) -> None:
        if self.model is not None:
            return
        # Import torch FIRST: on Windows this loads PyTorch's bundled CUDA
        # libraries (cublas64_12.dll, cuDNN) and registers their DLL directory,
        # which is how CTranslate2 finds cuBLAS for GPU inference.
        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except (ImportError, OSError):
            pass

        from faster_whisper import WhisperModel

        device = WHISPER_DEVICE
        compute = WHISPER_COMPUTE_TYPE
        if device == "auto":
            device = "cuda" if cuda_ok else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        # Try the requested (device, compute), then fall back gracefully — some
        # GPUs/CTranslate2 builds can't do float16, so we degrade rather than crash.
        candidates = [(device, compute)]
        if device == "cuda":
            for c in ("int8_float16", "int8", "float32"):
                if (device, c) not in candidates:
                    candidates.append((device, c))
            candidates.append(("cpu", "int8"))      # last resort: CPU
        else:
            for c in ("int8", "float32"):
                if (device, c) not in candidates:
                    candidates.append((device, c))

        last_err = None
        for dev, comp in candidates:
            try:
                log.info("Loading Whisper '%s' on %s (%s) …", WHISPER_MODEL_SIZE, dev, comp)
                model = WhisperModel(WHISPER_MODEL_SIZE, device=dev, compute_type=comp)
                # Warm-up inference: forces CUDA/cuBLAS to actually load now, so a
                # broken GPU backend (e.g. missing cublas64_12.dll) is rejected
                # here at startup instead of crashing on the first utterance.
                import numpy as np
                seg, _ = model.transcribe(np.zeros(8000, dtype=np.float32), beam_size=1)
                list(seg)
                self.model = model
                log.info("Whisper ready (%s, %s).", dev, comp)
                return
            except Exception as exc:
                last_err = exc
                log.warning("  %s/%s unavailable (%s) — trying next.", dev, comp, exc)
        raise RuntimeError(f"Could not initialise Whisper on any backend: {last_err}")

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self.load()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="whisper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        while self._running:
            try:
                segment = self.audio_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._transcribe(segment)
            except Exception as exc:
                log.error("Transcription error: %s", exc)

    def _transcribe(self, audio: np.ndarray) -> None:
        speaker, track_id = "", None
        if isinstance(audio, AudioSegment):
            speaker, track_id, audio = audio.speaker, audio.track_id, audio.audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        started = time.monotonic()
        segments, info = self.model.transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE,
            language=self._forced_lang,
            vad_filter=False,           # we already segmented with Silero
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if self.state is not None:
            self.state.record_latency("ASR", (time.monotonic() - started) * 1000)
        if len(text) < 2:
            return

        result = {
            "text": text,
            "language": info.language,
            "confidence": round(info.language_probability * 100, 1),
            "speaker": speaker,
            "track_id": track_id,
        }
        log.info("[%s %.0f%%] %s", info.language, result["confidence"], text)
        put_drop_oldest(self.transcript_queue, result)

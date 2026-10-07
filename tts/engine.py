"""
Thread 5 — TTS (multi-engine, routed by language).

Consumes translated text and synthesises it to raw 16-bit PCM, then resamples
to OUTPUT_SAMPLE_RATE and hands it to the streamer:

    translation_queue  →  TTS  →  pcm_queue   (int16 mono @ OUTPUT_SAMPLE_RATE)

The engine is chosen per language by config.TTS_ROUTES:
    Configured languages → Piper (Japanese/Chinese need pronunciation extras)
    Optional custom route → Style-Bert-VITS2

Because everything is translated INTO the target language, in practice one
engine is used per run — but routing keeps it correct if you go bidirectional.
Models are loaded once and reused (never relaunched per sentence).
"""

import threading
import queue
import time

from config import TTS_ROUTES, TARGET_LANGUAGE, OUTPUT_SAMPLE_RATE
from tts.resample import resample_pcm
from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger("TTS")


class TTS:
    """Owns one thread: translation_queue → routed engine → pcm_queue."""

    def __init__(self, translation_queue, pcm_queue, state=None):
        self.translation_queue = translation_queue
        self.pcm_queue = pcm_queue
        self.state = state
        self.sample_rate = OUTPUT_SAMPLE_RATE      # fixed; streamer opens once
        self._piper = None
        self._sbv2 = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ── backends ─────────────────────────────────────────────────────
    def _engine_for(self, language: str) -> str:
        if language not in TTS_ROUTES:
            raise ValueError(f"No TTS route configured for {language!r}.")
        return TTS_ROUTES[language]

    def _backend(self, engine: str):
        if engine == "sbv2":
            if self._sbv2 is None:
                from tts.sbv2_backend import SBV2Backend
                self._sbv2 = SBV2Backend()
            return self._sbv2
        if engine != "piper":
            raise ValueError(f"Unknown TTS engine: {engine!r}")
        if self._piper is None:
            from tts.piper_backend import PiperBackend
            self._piper = PiperBackend()
        return self._piper

    # ── lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        """Preload the engine for the target language to avoid first-word lag."""
        engine = self._engine_for(TARGET_LANGUAGE)
        log.info("Target '%s' → %s engine.", TARGET_LANGUAGE, engine)
        try:
            backend = self._backend(engine)
            if hasattr(backend, "preload"):
                backend.preload(TARGET_LANGUAGE)
            elif engine == "piper":
                backend._load_voice(TARGET_LANGUAGE)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {engine} for {TARGET_LANGUAGE}: {exc}. "
                "Run python main.py --doctor for setup instructions."
            ) from exc

    def start(self) -> None:
        if self._running:
            return
        self.load()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        while self._running:
            try:
                item = self.translation_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            text = item.get("text", "").strip()
            language = item.get("tgt_name", TARGET_LANGUAGE)
            if not text:
                continue
            try:
                pcm = self._synthesize(text, language)
                if pcm:
                    log.info("Generated %d bytes (%s)", len(pcm), language)
                    put_drop_oldest(self.pcm_queue, pcm)
            except Exception as exc:
                log.error("Synthesis error (%s): %s", language, exc)

    def _synthesize(self, text: str, language: str) -> bytes:
        started = time.monotonic()
        engine = self._engine_for(language)
        backend = self._backend(engine)
        pcm, src_rate = backend.synth(text, language)
        if not pcm:
            return b""
        result = resample_pcm(pcm, src_rate, OUTPUT_SAMPLE_RATE)
        if self.state is not None:
            self.state.record_latency("TTS", (time.monotonic() - started) * 1000)
        return result

"""
Thread 4 — Translation.

Consumes transcripts, translates each into the configured TARGET_LANGUAGE with
NLLB-200, and forwards the result for speech synthesis:

    transcript_queue  →  Translator  →  translation_queue
                                          {"text", "src_lang", "tgt_lang", "original"}

It also writes the latest original/translated pair into the shared caption so
the video overlay can render live subtitles.
"""

import threading
import queue
import time

from config import NLLB_MODEL_NAME, NLLB_DEVICE, LANGUAGE_CODES, TARGET_LANGUAGE, whisper_to_name
from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger("Translate")


class Translator:
    """Owns one thread: transcript_queue → NLLB → translation_queue."""

    def __init__(self, transcript_queue, translation_queue, state=None, scene_queue=None):
        self.transcript_queue = transcript_queue
        self.translation_queue = translation_queue
        self.state = state
        self.scene_queue = scene_queue
        self.tokenizer = None
        self.model = None
        self._device = None
        self._running = False
        self._thread: threading.Thread | None = None

        self.target_name = TARGET_LANGUAGE
        self.target_nllb = LANGUAGE_CODES[TARGET_LANGUAGE]["nllb"]
        self.target_whisper = LANGUAGE_CODES[TARGET_LANGUAGE]["whisper"]

    # ── model ────────────────────────────────────────────────────────
    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        log.info("Loading NLLB '%s' …", NLLB_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        if NLLB_DEVICE != "auto":
            self._device = NLLB_DEVICE
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        try:
            self.model.to(self._device).eval()
        except RuntimeError:
            if NLLB_DEVICE != "auto" or self._device == "cpu":
                raise
            log.warning("NLLB device %s failed; falling back to CPU.", self._device)
            self._device = "cpu"
            self.model.to("cpu").eval()
        log.info("NLLB ready on %s. Target language: %s", self._device, self.target_name)

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self.load()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="translate", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ── thread body ──────────────────────────────────────────────────
    def _run(self) -> None:
        while self._running:
            try:
                item = self.transcript_queue.get(timeout=0.1)
            except queue.Empty:
                if self.scene_queue is None:
                    continue
                try:
                    item = self.scene_queue.get_nowait()
                except queue.Empty:
                    continue
            try:
                if item.get("kind") == "scene":
                    self._handle_scene(item)
                else:
                    self._handle(item)
            except Exception as exc:
                log.error("Translation error: %s", exc)

    def _handle_scene(self, item):
        started = time.monotonic()
        translated = None
        try:
            source = self._resolve_nllb(item['language'])
            if source is None:
                raise ValueError('Unsupported OCR source language')
            translated = item['text'] if source == self.target_nllb else self._translate(item['text'], source, self.target_nllb)
        finally:
            item['on_result'](item['text'], translated, (time.monotonic() - started) * 1000)

    def _handle(self, item: dict) -> None:
        text = item["text"]
        src_whisper = item["language"]
        src_nllb = self._resolve_nllb(src_whisper)

        # already in the target language → pass through untouched
        if src_whisper == self.target_whisper or src_nllb == self.target_nllb:
            translated = text
        elif src_nllb is None:
            log.warning("Unsupported source language '%s' — skipping.", src_whisper)
            return
        else:
            translated = self._translate(text, src_nllb, self.target_nllb)

        log.info("%s → %s", whisper_to_name(src_whisper), translated)

        if self.state is not None:
            self.state.set_caption(original=text, translated=translated,
                                   speaker=item.get("speaker", ""), track_id=item.get("track_id"))

        put_drop_oldest(self.translation_queue, {
            "text": translated,
            "original": text,
            "src_lang": src_whisper,
            "tgt_lang": self.target_whisper,
            "tgt_name": self.target_name,
        })

    def _translate(self, text: str, src_nllb: str, tgt_nllb: str) -> str:
        import torch
        self.tokenizer.src_lang = src_nllb
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
        with torch.no_grad():
            try:
                out = self.model.generate(**inputs, forced_bos_token_id=tgt_id,
                                          max_new_tokens=256, num_beams=1)
            except RuntimeError:
                if NLLB_DEVICE != "auto" or self._device == "cpu":
                    raise
                log.warning("NLLB inference on %s failed; retrying on CPU.", self._device)
                self._device = "cpu"
                self.model.to("cpu").eval()
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                out = self.model.generate(**inputs, forced_bos_token_id=tgt_id,
                                          max_new_tokens=256, num_beams=1)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def _resolve_nllb(whisper_code: str) -> str | None:
        for codes in LANGUAGE_CODES.values():
            if codes["whisper"] == whisper_code:
                return codes["nllb"]
        return None

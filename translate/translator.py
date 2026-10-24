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
from collections import OrderedDict

from config import NLLB_MODEL_NAME, NLLB_DEVICE, NLLB_CPU_INT8, LANGUAGE_CODES, TARGET_LANGUAGE, SOURCE_LANGUAGE, TRANSLATION_CACHE_SIZE, whisper_to_name
from utils.queues import put_drop_oldest
from utils.logging_utils import get_logger

log = get_logger("Translate")


class Translator:
    """Owns one thread: transcript_queue → NLLB → translation_queue."""

    def __init__(self, transcript_queue, translation_queue, state=None, scene_queue=None, meeting=None):
        self.transcript_queue = transcript_queue
        self.translation_queue = translation_queue
        self.state = state
        self.scene_queue = scene_queue
        self.meeting = meeting
        self._cache = OrderedDict()
        self.cache_hits = 0
        self._quantization_attempted = False
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
        self._prepare_cpu_model()
        log.info("NLLB ready on %s. Target language: %s", self._device, self.target_name)

    def _prepare_cpu_model(self):
        if self._device != 'cpu' or not NLLB_CPU_INT8 or self._quantization_attempted:
            return
        import torch
        self._quantization_attempted = True
        engines = torch.backends.quantized.supported_engines
        if torch.backends.quantized.engine == 'none':
            engine = next((e for e in ('x86', 'fbgemm', 'qnnpack') if e in engines), None)
            if engine is None:
                log.warning('CPU int8 is unavailable in this PyTorch build; using float32.')
                return
            torch.backends.quantized.engine = engine
        log.info('Preparing CPU int8 translation (lower memory, quality may vary)…')
        try:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
            log.info('NLLB Linear layers use CPU int8.')
        except (RuntimeError, NotImplementedError, AttributeError) as exc:
            # In-place quantization may already have replaced some modules.
            log.warning('CPU int8 unavailable (%s); reloading a clean float32 model.', exc)
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to('cpu').eval()

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        # Same-language sessions need no NLLB until a differently sourced OCR job arrives.
        if SOURCE_LANGUAGE != self.target_name:
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
        started = time.monotonic()
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
            self.state.record_latency("Translation", (time.monotonic() - started) * 1000)
            self.state.set_caption(original=text, translated=translated,
                                   speaker=item.get("speaker", ""), track_id=item.get("track_id"))

        if self.meeting:
            self.meeting.record_translation(item.get('meeting_id'), translated, self.target_name)

        put_drop_oldest(self.translation_queue, {
            "text": translated,
            "original": text,
            "src_lang": src_whisper,
            "tgt_lang": self.target_whisper,
            "tgt_name": self.target_name,
        })

    def _translate(self, text: str, src_nllb: str, tgt_nllb: str) -> str:
        key = (text, src_nllb, tgt_nllb)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        value = self._infer(text, src_nllb, tgt_nllb)
        if TRANSLATION_CACHE_SIZE > 0 and value.strip():
            self._cache[key] = value
            while len(self._cache) > TRANSLATION_CACHE_SIZE:
                self._cache.popitem(last=False)
        return value

    def _infer(self, text: str, src_nllb: str, tgt_nllb: str) -> str:
        import torch
        self.load()
        self.tokenizer.src_lang = src_nllb
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
        with torch.inference_mode():
            try:
                out = self.model.generate(**inputs, forced_bos_token_id=tgt_id,
                                          max_new_tokens=256, num_beams=1)
            except RuntimeError:
                if NLLB_DEVICE != "auto" or self._device == "cpu":
                    raise
                log.warning("NLLB inference on %s failed; retrying on CPU.", self._device)
                self._device = "cpu"
                self.model.to("cpu").eval()
                self._prepare_cpu_model()
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

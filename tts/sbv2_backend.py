"""
Style-Bert-VITS2 backend — the most natural open Japanese TTS (also handles
English and Chinese).  Used for Japanese/Chinese targets.

The model + the per-language BERT model are loaded ONCE and reused.  BERT models
download automatically from Hugging Face on first use; the voice model itself is
fetched with scripts/download_sbv2_model.py.

synth(text, language) -> (pcm_int16_bytes, sample_rate)
"""

import os

import numpy as np

from config import (
    SBV2_DEVICE, SBV2_ASSETS_DIR, SBV2_MODEL_NAME,
    SBV2_MODEL_FILE, SBV2_CONFIG_FILE, SBV2_STYLE_FILE, SBV2_STYLE, SBV2_BERT,
)
from utils.logging_utils import get_logger

log = get_logger("TTS")

DEFAULT_RATE = 44_100


class SBV2Backend:
    def __init__(self):
        self._model = None
        self._loaded_bert: set[str] = set()
        self._device = self._resolve_device()
        self._fp32_done = False     # net_g cast to float32 yet?

    @staticmethod
    def _resolve_device() -> str:
        if SBV2_DEVICE != "auto":
            return SBV2_DEVICE
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def available_for(self, language: str) -> bool:
        return language in SBV2_BERT and self._model_files_present()

    def _model_files_present(self) -> bool:
        base = os.path.join(SBV2_ASSETS_DIR, SBV2_MODEL_NAME)
        return all(os.path.exists(os.path.join(base, f))
                   for f in (SBV2_MODEL_FILE, SBV2_CONFIG_FILE, SBV2_STYLE_FILE))

    # ── lazy loading ─────────────────────────────────────────────────
    def _lang_enum(self, language: str):
        from style_bert_vits2.constants import Languages
        return {"Japanese": Languages.JP, "English": Languages.EN,
                "Chinese": Languages.ZH}.get(language, Languages.JP)

    def _ensure_bert(self, language: str) -> None:
        if language in self._loaded_bert:
            return
        from style_bert_vits2.nlp import bert_models
        name = SBV2_BERT[language]
        lang = self._lang_enum(language)
        log.info("Loading BERT for %s (%s) …", language, name)
        model = bert_models.load_model(lang, name)
        bert_models.load_tokenizer(lang, name)
        # FIX: cast BERT to float32 on ALL devices (not just CPU).
        # On CUDA the model loads as fp16 and its features feed the generator
        # as fp16, causing "Half ... bias ... float" mismatch.
        try:
            model.float()
        except Exception as exc:
            log.debug("BERT fp32 cast skipped: %s", exc)
        self._loaded_bert.add(language)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self._model_files_present():
            raise FileNotFoundError(
                f"Style-Bert-VITS2 model not found under "
                f"{os.path.join(SBV2_ASSETS_DIR, SBV2_MODEL_NAME)}. "
                f"Run: python scripts/download_sbv2_model.py"
            )
        from style_bert_vits2.tts_model import TTSModel
        base = os.path.join(SBV2_ASSETS_DIR, SBV2_MODEL_NAME)
        log.info("Loading Style-Bert-VITS2 '%s' on %s …", SBV2_MODEL_NAME, self._device)
        self._model = TTSModel(
            model_path=os.path.join(base, SBV2_MODEL_FILE),
            config_path=os.path.join(base, SBV2_CONFIG_FILE),
            style_vec_path=os.path.join(base, SBV2_STYLE_FILE),
            device=self._device,
        )
        self._model.load()
        # FIX: cast net_g to float32 immediately after load on ALL devices.
        # RTX cards materialise net_g as fp16 which causes bias dtype mismatch.
        try:
            import torch
            if hasattr(self._model, 'net_g') and self._model.net_g is not None:
                self._model.net_g = self._model.net_g.to(torch.float32)
                self._fp32_done = True
                log.info("Cast net_g to float32 on %s after model load.", self._device)
        except Exception as exc:
            log.debug("Early net_g fp32 cast failed (will retry in synth): %s", exc)
        # Also try the standard force path in case net_g was ready
        self._force_fp32()
        log.info("Style-Bert-VITS2 ready.")

    def preload(self, language: str) -> None:
        self._ensure_model()
        self._ensure_bert(language)

    def _force_fp32(self) -> bool:
        """Cast the generator to float32 on ALL devices (CPU and CUDA).

        RTX / CUDA devices load net_g as fp16 by default, which causes a
        'Half ... bias ... float' RuntimeError on inference.  This method is
        safe to call repeatedly; it no-ops once already done or if net_g
        isn't loaded yet.
        """
        if self._fp32_done:
            return self._fp32_done
        net_g = getattr(self._model, "net_g", None) if self._model else None
        if net_g is None:
            return False
        try:
            import torch
            self._model.net_g = net_g.to(torch.float32)
            self._fp32_done = True
            log.info("Forced Style-Bert-VITS2 generator to float32 on %s.", self._device)
        except Exception as exc:
            log.debug("fp32 cast failed: %s", exc)
        return self._fp32_done

    # ── synthesis ────────────────────────────────────────────────────
    def synth(self, text: str, language: str) -> tuple[bytes, int]:
        self._ensure_model()
        self._ensure_bert(language)
        lang = self._lang_enum(language)

        self._force_fp32()                       # cast if net_g is ready
        try:
            sr, audio = self._infer(text, lang)
        except RuntimeError as exc:
            # First call materialises net_g as fp16 → "Half ... bias ... float".
            # Now that net_g exists, cast to fp32 and retry once.
            msg = str(exc)
            if ("Half" in msg or "should be the same" in msg) and not self._fp32_done:
                if self._force_fp32():
                    log.info("Recovered fp16 error — retrying in float32.")
                    sr, audio = self._infer(text, lang)
                else:
                    raise
            else:
                raise
        return self._to_pcm16(audio), int(sr or DEFAULT_RATE)

    def _infer(self, text: str, lang):
        try:
            return self._model.infer(text=text, language=lang, style=SBV2_STYLE)
        except TypeError:
            # older signatures may not accept the style kwarg
            return self._model.infer(text=text, language=lang)

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.floating):
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak <= 1.0:                      # normalised float → scale up
                audio = audio * 32767.0
            audio = np.clip(audio, -32768, 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        return audio.tobytes()
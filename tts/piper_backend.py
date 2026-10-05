"""
Piper TTS backend — fast, fully-local synthesis for the languages Piper
supports (English, Hindi, …).  Used for every routed language except Japanese
/ Chinese, which go to Style-Bert-VITS2.

Voices are loaded lazily and cached, so the model for a language is loaded once
and reused for every sentence (never relaunched per utterance).

synth(text, language) -> (pcm_int16_bytes, sample_rate)
"""

import os

from config import PIPER_VOICES
from utils.logging_utils import get_logger

log = get_logger("TTS")

DEFAULT_RATE = 22_050


class PiperBackend:
    def __init__(self):
        self._voices: dict[str, object] = {}     # language → PiperVoice
        self._rates: dict[str, int] = {}

    def available_for(self, language: str) -> bool:
        path = PIPER_VOICES.get(language)
        return bool(path and os.path.exists(path))

    def _load_voice(self, language: str):
        if language in self._voices:
            return self._voices[language]
        path = PIPER_VOICES.get(language)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Piper voice for '{language}' not found at {path}. "
                f"Run: python scripts/download_piper_voice.py <voice-name>"
            )
        from piper import PiperVoice
        voice = PiperVoice.load(path)
        rate = int(getattr(getattr(voice, "config", None), "sample_rate", DEFAULT_RATE))
        self._voices[language] = voice
        self._rates[language] = rate
        log.info("Piper voice loaded for %s (%s, %d Hz).", language, os.path.basename(path), rate)
        return voice

    def synth(self, text: str, language: str) -> tuple[bytes, int]:
        voice = self._load_voice(language)
        rate = self._rates.get(language, DEFAULT_RATE)
        # support both old and new piper-tts APIs
        if hasattr(voice, "synthesize_stream_raw"):
            pcm = b"".join(voice.synthesize_stream_raw(text))
        else:
            chunks = []
            for chunk in voice.synthesize(text):
                data = getattr(chunk, "audio_int16_bytes", None)
                if data is None and hasattr(chunk, "audio_int16_array"):
                    data = chunk.audio_int16_array.tobytes()
                if data:
                    chunks.append(data)
            pcm = b"".join(chunks)
        return pcm, rate

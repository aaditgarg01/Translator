"""
Piper TTS backend — fast, fully-local synthesis for the languages Piper
supports, including the Japanese and Chinese pronunciation extensions.

Voices are loaded lazily and cached, so the model for a language is loaded once
and reused for every sentence (never relaunched per utterance).

synth(text, language) -> (pcm_int16_bytes, sample_rate)
"""

import os
from pathlib import Path

from config import PIPER_VOICES, PIPER_PROFILES, MODELS_DIR
from tts.audio import join_sentences
from tts.voices import voice_issues, SAMPLE_TEXTS
from utils.logging_utils import get_logger

log = get_logger("TTS")

DEFAULT_RATE = 22_050


class PiperBackend:
    def __init__(self):
        self._voices: dict[str, object] = {}     # language → PiperVoice
        self._rates: dict[str, int] = {}

    def available_for(self, language: str) -> bool:
        path = PIPER_VOICES.get(language)
        return bool(path and os.path.isfile(path) and os.path.isfile(path + ".json"))

    def _load_voice(self, language: str):
        if language in self._voices:
            return self._voices[language]
        path = PIPER_VOICES.get(language)
        if not self.available_for(language):
            raise FileNotFoundError(
                f"Piper voice for '{language}' requires both {path} and its .json file. "
                f"Run: python scripts/download_piper_voice.py <voice-name>"
            )
        issues = voice_issues(path, language)
        if issues:
            raise RuntimeError("; ".join(issues))
        from piper import PiperVoice
        voice = PiperVoice.load(path, download_dir=Path(MODELS_DIR) / "piper" / "_resources")
        speaker = PIPER_PROFILES.get(language, {}).get("speaker_id")
        if speaker is not None and not 0 <= speaker < voice.config.num_speakers:
            raise ValueError(f"Invalid speaker_id {speaker} for {language} voice")
        rate = int(getattr(getattr(voice, "config", None), "sample_rate", DEFAULT_RATE))
        self._voices[language] = voice
        self._rates[language] = rate
        log.info("Piper voice loaded for %s (%s, %d Hz).", language, os.path.basename(path), rate)
        return voice

    def preload(self, language):
        voice = self._load_voice(language)
        log.info("Checking %s pronunciation (first use may download language assets)...", language)
        phonemes = voice.phonemize(SAMPLE_TEXTS.get(language, "Hello."))
        if not any(phonemes):
            raise RuntimeError(f"{language} phonemizer produced no speech symbols")

    def synth(self, text: str, language: str) -> tuple[bytes, int]:
        voice = self._load_voice(language)
        rate = self._rates.get(language, DEFAULT_RATE)
        profile = PIPER_PROFILES.get(language)
        if profile:
            from piper import SynthesisConfig
            settings = {key: value for key, value in profile.items() if key != "sentence_pause_ms"}
            chunks = voice.synthesize(text, syn_config=SynthesisConfig(normalize_audio=False, **settings))
            pcm = join_sentences(
                (chunk.audio_float_array for chunk in chunks), rate,
                pause_ms=profile.get("sentence_pause_ms", 150),
            )
            return pcm, rate
        # Established English/Hindi/Japanese settings remain unchanged.
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
        if not pcm:
            raise RuntimeError(f"{language} voice produced no audio; check input and pronunciation dependencies")
        return pcm, rate

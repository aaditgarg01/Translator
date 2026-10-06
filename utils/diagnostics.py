"""Offline configuration and installation checks; never download models."""

import importlib.util
import os
import platform
import shutil
import sys

import config


def validate_config():
    if config.TARGET_LANGUAGE not in config.LANGUAGE_CODES:
        raise ValueError(f"Unknown target language: {config.TARGET_LANGUAGE}")
    if config.SOURCE_LANGUAGE != "auto" and config.SOURCE_LANGUAGE not in config.LANGUAGE_CODES:
        raise ValueError(f"Unknown source language: {config.SOURCE_LANGUAGE}")
    if config.STREAM_MODE not in {"local", "rtp", "both"}:
        raise ValueError("STREAM_MODE must be local, rtp, or both")
    if config.AUDIO_SAMPLE_RATE != 16000 or config.VAD_FRAME_SIZE != 512 or config.AUDIO_CHANNELS != 1:
        raise ValueError("The speech pipeline requires 16000 Hz, 512-sample VAD frames, and mono audio")
    if config.OUTPUT_SAMPLE_RATE <= 0:
        raise ValueError("OUTPUT_SAMPLE_RATE must be positive")
    if config.WHISPER_DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("WHISPER_DEVICE must be auto, cpu, or cuda (Whisper does not support MPS)")
    if config.NLLB_DEVICE not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("NLLB_DEVICE must be auto, cpu, cuda, or mps")
    if not 1 <= config.RTP_PORT <= 65534:
        raise ValueError("RTP_PORT must be between 1 and 65534")
    if config.ENABLE_TTS and config.TTS_ROUTES.get(config.TARGET_LANGUAGE) not in {"piper", "sbv2"}:
        raise ValueError(f"No supported TTS route for {config.TARGET_LANGUAGE}")
    for name in ("AUDIO_QUEUE_MAX", "TRANSCRIPT_QUEUE_MAX", "TRANSLATION_QUEUE_MAX", "PCM_QUEUE_MAX"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")


def installation_issues():
    """Return actionable errors for the enabled stages without loading ML models."""
    issues = []
    try:
        validate_config()
    except ValueError as exc:
        return [str(exc)]
    required = []
    if config.ENABLE_AUDIO or (config.ENABLE_STREAM and config.ENABLE_TTS and config.STREAM_MODE != "rtp"):
        required.append(("sounddevice", "sounddevice"))
    if config.ENABLE_VIDEO:
        required.append(("cv2", "opencv-contrib-python"))
    if config.ENABLE_WHISPER:
        required.append(("faster_whisper", "faster-whisper"))
    if config.ENABLE_TRANSLATE:
        required.extend([("torch", "torch"), ("transformers", "transformers"), ("sentencepiece", "sentencepiece")])
    if config.ENABLE_TTS:
        engine = config.TTS_ROUTES[config.TARGET_LANGUAGE]
        if engine == "piper":
            required.append(("piper", "piper-tts"))
            voice = config.PIPER_VOICES.get(config.TARGET_LANGUAGE)
            if not voice or not all(os.path.isfile(voice + suffix) for suffix in ("", ".json")):
                name = os.path.basename(voice).removesuffix(".onnx") if voice else "<voice-name>"
                issues.append(f"Missing Piper voice/config. Run: python scripts/download_piper_voice.py {name}")
        else:
            required.append(("style_bert_vits2", "requirements-japanese.txt"))
            base = os.path.join(config.SBV2_ASSETS_DIR, config.SBV2_MODEL_NAME)
            if not all(os.path.isfile(os.path.join(base, f)) for f in (
                config.SBV2_MODEL_FILE, config.SBV2_CONFIG_FILE, config.SBV2_STYLE_FILE,
            )):
                issues.append("Missing Japanese voice. Run: python scripts/download_sbv2_model.py")
    for module, package in required:
        if importlib.util.find_spec(module) is None:
            command = f"-r {package}" if package.endswith(".txt") else package
            issues.append(f"Missing {module}. Run: python -m pip install {command}")
    if config.ENABLE_TTS and config.ENABLE_STREAM and config.STREAM_MODE == "rtp" and not shutil.which(config.FFMPEG_BIN):
        issues.append("FFmpeg is missing. Install it or use --stream local.")
    return issues


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("Select an index with --input-device N / --output-device N.")


def run_doctor():
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Target: {config.TARGET_LANGUAGE}; output: {config.STREAM_MODE}")
    issues = installation_issues()
    if config.ENABLE_VIDEO and importlib.util.find_spec("cv2"):
        try:
            import cv2
            if not hasattr(cv2, "face"):
                issues.append("OpenCV lacks face recognition. Remove opencv-python/headless and reinstall opencv-contrib-python.")
            local = os.path.join(config.BASE_DIR, "haarcascade_frontalface_default.xml")
            cascade = local if os.path.isfile(local) else os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if cv2.CascadeClassifier(cascade).empty():
                issues.append("OpenCV face cascade is missing; reinstall opencv-contrib-python.")
        except Exception as exc:
            issues.append(f"OpenCV could not load: {exc}")
    if importlib.util.find_spec("sounddevice"):
        try:
            import sounddevice as sd
            if config.ENABLE_AUDIO:
                sd.check_input_settings(device=config.AUDIO_INPUT_DEVICE, channels=1, dtype="float32", samplerate=config.AUDIO_SAMPLE_RATE)
            if config.ENABLE_TTS and config.ENABLE_STREAM and config.STREAM_MODE in {"local", "both"}:
                sd.check_output_settings(device=config.AUDIO_OUTPUT_DEVICE, channels=1, dtype="int16", samplerate=config.OUTPUT_SAMPLE_RATE)
        except Exception as exc:
            issues.append(f"Audio device settings: {exc}. Run --list-devices and select a working device.")
    if config.ENABLE_STREAM and config.STREAM_MODE == "both" and not shutil.which(config.FFMPEG_BIN):
        print("WARNING: FFmpeg missing; only local playback will be available.")
    for issue in issues:
        print(f"ERROR: {issue}")
    if not issues:
        print("Setup checks passed. Model inference and camera/microphone permissions are checked when you start.")
    return 1 if issues else 0

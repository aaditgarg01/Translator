"""Voice metadata checks that do not load models or download assets."""

import importlib.util
import json
from pathlib import Path

from config import LANGUAGE_CODES

SAMPLE_TEXTS = {
    'English': 'Hello. How are you today?',
    'Hindi': 'नमस्ते। आज आप कैसे हैं?',
    'Japanese': 'こんにちは。今日はお元気ですか。',
    'Chinese': '你好。你今天怎么样？',
    'Korean': '안녕하세요. 오늘은 어떠세요?',
    'Spanish': 'Hola. ¿Cómo estás hoy?',
    'French': 'Bonjour. Comment allez-vous aujourd’hui ?',
    'German': 'Guten Tag. Wie geht es Ihnen heute?',
    'Portuguese': 'Olá. Como você está hoje?',
    'Arabic': 'مرحباً. كيف حالك اليوم؟',
}


def voice_issues(path, language):
    if not path or not all(Path(path + suffix).is_file() for suffix in ('', '.json')):
        name = Path(path).stem if path else '<voice-name>'
        return [f'Missing Piper voice/config. Run: python scripts/download_piper_voice.py {name}']
    try:
        data = json.loads(Path(path + '.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return [f'Cannot read voice configuration for {language}: {exc}']
    issues = []
    code = data.get('language', {}).get('code', '')
    expected = LANGUAGE_CODES.get(language, {}).get('whisper')
    if code and expected and code.split('_')[0] != expected:
        issues.append(f'{language} is configured with a {code} voice. Choose a voice for {expected}.')
    phoneme_type = data.get('phoneme_type', 'espeak')
    dependencies = {
        'japanese': (['pyopenjtalk'], 'ja'),
        'pinyin': (['unicode_rbnf', 'sentence_stream', 'g2pw', 'transformers'], 'zh'),
    }
    if phoneme_type in dependencies:
        modules, extra = dependencies[phoneme_type]
        missing = [module for module in modules if importlib.util.find_spec(module) is None]
        if missing:
            issues.append(f'{language} pronunciation dependencies missing: {", ".join(missing)}. '
                          f'Run: python -m pip install "piper-tts[{extra}]>=1.8,<2"')
    return issues

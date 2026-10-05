"""
Tiny per-thread logger so every subsystem reports its status in a
consistent, greppable format:

    14:03:21 [Video    ] FPS: 61
    14:03:21 [Audio    ] Speech detected
    14:03:22 [Whisper  ] Hello everyone
    14:03:22 [Translate] こんにちは皆さん
    14:03:22 [TTS      ] Generated 64000 bytes
    14:03:22 [Stream   ] Sent 64000 bytes

Each component grabs one logger via ``get_logger("Video")`` and the name is
padded so the columns line up in the terminal.
"""

import logging
import sys

from config import LOG_LEVEL

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(name)-9s] %(message)s",
    datefmt="%H:%M:%S",
)

# We log translated text (Japanese, Hindi, …).  The default Windows console is
# cp1252 and would raise UnicodeEncodeError, so force stdout to UTF-8 once.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def get_logger(name: str) -> logging.Logger:
    """Return a standalone logger, e.g. get_logger('Whisper')."""
    logger = logging.getLogger(name.strip())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
    return logger

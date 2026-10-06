"""
Every arrow in the architecture diagram is a queue.  Bundling them in one
object keeps ``main.py`` clean and makes the data flow obvious:

    audio_q  -->  Whisper  -->  transcript_q  -->  Translate
                 -->  translation_q  -->  TTS  -->  pcm_q  -->  Stream

Queues are bounded so a slow stage applies back-pressure instead of letting
memory grow without limit.  Nothing waits for anything else — each thread just
``get()``s from its input queue and ``put()``s onto its output queue.
"""

from dataclasses import dataclass, field
from queue import Queue, Empty, Full

from config import (
    AUDIO_QUEUE_MAX,
    TRANSCRIPT_QUEUE_MAX,
    TRANSLATION_QUEUE_MAX,
    PCM_QUEUE_MAX,
)


@dataclass
class Pipeline:
    """The four hand-off queues that connect the worker threads."""

    audio: "Queue"       = field(default_factory=lambda: Queue(maxsize=AUDIO_QUEUE_MAX))
    transcript: "Queue"  = field(default_factory=lambda: Queue(maxsize=TRANSCRIPT_QUEUE_MAX))
    translation: "Queue" = field(default_factory=lambda: Queue(maxsize=TRANSLATION_QUEUE_MAX))
    pcm: "Queue"         = field(default_factory=lambda: Queue(maxsize=PCM_QUEUE_MAX))


def put_drop_oldest(q: "Queue", item) -> None:
    """Put onto a bounded queue, dropping the oldest item if it is full.

    Real-time audio should never block the producer — if a downstream stage
    falls behind we'd rather lose the stalest item than freeze the mic.
    """
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass

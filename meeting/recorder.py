"""Append-only transcript journal and reviewable minutes; no extra ML model."""

from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from uuid import uuid4

_STOP = set('a an and are as at be been but by for from has have i in is it its of on or our that the their this to was we were will with you your'.split())
_DECISION = re.compile(r'\b(?:we (?:have )?(?:agreed|decided)\b|it was (?:agreed|decided)\b|decision\s*:|the decision is\b)', re.I)
_ACTION = re.compile(r'\b(?:action item\b|to[ -]?do\s*:|we need to\b|\w+ will\b)', re.I)


def _utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc).isoformat(timespec='milliseconds')


def _atomic(path, content):
    """A failed export leaves the previous usable snapshot in place."""
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(content)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def compile_minutes(events):
    """Replay a journal, including translations that arrived after later speech."""
    meta = next((dict(e) for e in events if e.get('kind') == 'session'), {})
    entries, translations, gaps = {}, {}, []
    for event in events:
        kind = event.get('kind')
        if kind == 'utterance':
            entries[event['id']] = dict(event)
        elif kind == 'translation':
            translations[event['id']] = event
        elif kind == 'gap':
            gaps.append(event)
        elif kind == 'ended':
            meta['ended_at'] = event['at']
    for key, item in entries.items():
        item['translation'] = translations.get(key, {}).get('text')
        item['target_language'] = translations.get(key, {}).get('language')
        item['translation_status'] = ('complete' if key in translations else
                                      'pending_or_unavailable' if meta.get('target') else 'not_requested')
    transcript = sorted(entries.values(), key=lambda e: e['elapsed_seconds'])
    candidates = []
    for item in transcript:
        # Prefer English for English-only cue extraction when the source differs.
        value = item['text']
        if item.get('language') != 'en' and item.get('target_language') == 'English' and item.get('translation'):
            value = item['translation']
        for sentence in re.split(r'(?<=[.!?])\s+|(?<=[。！？।])|\n+', value):
            sentence = sentence.strip()
            if sentence:
                candidates.append(dict(text=sentence, entry_id=item['id'], speaker=item['speaker'],
                                       elapsed_seconds=item['elapsed_seconds']))
    # Bounded vocabulary per utterance and linear scoring; exact source sentences.
    tokens = [set(re.findall(r'[^\W\d_]+', c['text'].casefold())) - _STOP for c in candidates]
    counts = Counter(word for words in tokens for word in words)
    scores = [sum(math.log1p(counts[w]) for w in words) / math.sqrt(max(1, len(words))) for words in tokens]
    selected, seen = [], set()
    for index in sorted(range(len(candidates)), key=lambda i: (-scores[i], i)):
        key = candidates[index]['text'].casefold()
        if key not in seen:
            seen.add(key)
            selected.append(index)
        if len(selected) >= 5:
            break
    def cues(pattern):
        found, seen = [], set()
        for candidate in candidates:
            key = candidate['text'].casefold()
            if pattern.search(candidate['text']) and key not in seen:
                found.append(candidate);seen.add(key)
        return found
    return dict(schema_version=1, meeting=meta,
                participants=sorted({e['speaker'] for e in transcript if e['speaker'] != 'Unassigned'}),
                summary=[candidates[i] for i in sorted(selected)],
                decision_candidates=cues(_DECISION), action_candidates=cues(_ACTION),
                transcript=transcript, gaps=gaps,
                notes=['Summary points quote recognised speech; review them against what was said.',
                       'Decision and action cues are English heuristics, not confirmed assignments. Owners and due dates are not inferred.',
                       'Microphone gating during playback, queue overload, interruptions, and ASR errors can leave gaps.'])


def _stamp(seconds):
    seconds = int(max(0, seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def render_markdown(data):
    meta = data['meeting']
    lines = [f'# {meta.get("title", "Meeting")}', '', f'Started UTC: {meta.get("at", "Unknown")}',
             f'Ended UTC: {meta.get("ended_at", "In progress")}',
             'Participants: ' + (', '.join(data['participants']) or 'No identified speakers'), '']
    for heading, key in [('Summary', 'summary'), ('Decisions for review', 'decision_candidates'), ('Actions for review', 'action_candidates')]:
        lines += [f'## {heading}', '']
        lines += [f'- [{_stamp(e["elapsed_seconds"])}] {e["speaker"]}: {e["text"]}' for e in data[key]] or ['- None identified.']
        lines.append('')
    lines += ['## Transcript', '']
    for item in data['transcript']:
        lines += [f'### {_stamp(item["elapsed_seconds"])} {item["speaker"]}', item['text']]
        if item['translation']:
            lines += [f'Translation ({item["target_language"]}): {item["translation"]}']
        elif item['translation_status'] == 'pending_or_unavailable':
            lines += ['Translation pending or unavailable; original transcript retained.']
        lines.append('')
    lines += ['## Recording notes', ''] + ['- ' + n for n in data['notes']]
    lines += [f'- Recording gap at {_stamp(g["elapsed_seconds"])}: {g["reason"]}' for g in data['gaps']]
    return '\n'.join(lines) + '\n'


def render_html(data):
    meta = data['meeting'];parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>' + escape(meta.get('title', 'Meeting')) + '</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 24px;color:#172b3a}h1,h2{line-height:1.2}section{margin:32px 0}li{margin:12px 0}article{border-top:1px solid #ddd;padding:16px 0}small{color:#526675}.translation{color:#244f65}p{white-space:pre-wrap}@media print{body{margin:0;max-width:none}article{break-inside:avoid}}</style>',
        '<h1>' + escape(meta.get('title', 'Meeting')) + '</h1>',
        '<small>Started UTC: ' + escape(meta.get('at', 'Unknown')) + '<br>Ended UTC: ' + escape(meta.get('ended_at', 'In progress')) + '</small>',
        '<p>Participants: ' + escape(', '.join(data['participants']) or 'No identified speakers') + '</p>']
    for title, key in [('Summary', 'summary'), ('Decisions for review', 'decision_candidates'), ('Actions for review', 'action_candidates')]:
        parts += ['<section><h2>' + title + '</h2><ul>']
        parts += ['<li>' + escape(f'[{_stamp(e["elapsed_seconds"])}] {e["speaker"]}: {e["text"]}') + '</li>' for e in data[key]] or ['<li>None identified.</li>']
        parts += ['</ul></section>']
    parts += ['<section><h2>Transcript</h2>']
    for e in data['transcript']:
        parts += ['<article><strong>' + escape(_stamp(e['elapsed_seconds']) + ' ' + e['speaker']) + '</strong><p>' + escape(e['text']) + '</p>']
        if e['translation']:
            parts += ['<p class="translation">' + escape(str(e['target_language']) + ': ' + e['translation']) + '</p>']
        elif e['translation_status'] == 'pending_or_unavailable':
            parts += ['<p class="translation">Translation pending or unavailable; original transcript retained.</p>']
        parts += ['</article>']
    parts += ['</section><section><h2>Recording notes</h2><ul>']
    parts += ['<li>' + escape(n) + '</li>' for n in data['notes']]
    parts += ['<li>' + escape(_stamp(g['elapsed_seconds']) + ': ' + g['reason']) + '</li>' for g in data['gaps']]
    return '\n'.join(parts + ['</ul></section></html>'])


def export_events(events, directory):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
    data = compile_minutes(events)
    _atomic(directory / 'minutes.json', json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    _atomic(directory / 'minutes.md', render_markdown(data))
    _atomic(directory / 'minutes.html', render_html(data))
    return directory / 'minutes.html'


def recover(journal, directory=None):
    """An interrupted final JSONL line is ignored; malformed complete lines fail."""
    path = Path(journal);raw = path.read_bytes();events = []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise
            events.append(dict(kind='gap', elapsed_seconds=0, reason='Interrupted final journal entry was not recovered.'))
    if not any(e.get('kind') == 'session' for e in events):
        raise ValueError('This file does not contain a meeting session.')
    return export_events(events, directory or path.parent)


class MeetingRecorder:
    def __init__(self, directory, title='Meeting', source='auto', target=None):
        self.started = time.time()
        slug = re.sub(r'[^\w-]+', '-', title, flags=re.UNICODE).strip('-')[:60] or 'meeting'
        self.directory = Path(directory).expanduser() / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + slug + '-' + uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock();self._closed = False;self._export_lock = threading.Lock()
        self._journal = (self.directory / 'transcript.jsonl').open('a', encoding='utf-8')
        self._append(dict(kind='session', schema_version=1, title=title, at=_utc(self.started), source=source, target=target))

    def _append(self, event):
        with self._lock:
            if self._closed:
                return False
            self._journal.write(json.dumps(event, ensure_ascii=False) + '\n')
            self._journal.flush()
            return True

    def record_transcript(self, item):
        key = uuid4().hex
        ended = item.get('captured_at') or time.time()
        duration = item.get('duration_seconds', 0)
        event = dict(kind='utterance', id=key, at=_utc(ended - duration),
                     elapsed_seconds=round(max(0, ended - duration - self.started), 3),
                     duration_seconds=duration, speaker=item.get('speaker') or 'Unassigned',
                     track_id=item.get('track_id'), language=item['language'], text=item['text'],
                     language_confidence=item.get('confidence'))
        return key if self._append(event) else None

    def record_translation(self, key, text, language):
        if key and text.strip():
            self._append(dict(kind='translation', id=key, text=text, language=language))

    def gap(self, reason):
        self._append(dict(kind='gap', elapsed_seconds=round(time.time() - self.started, 3), reason=reason))

    def export(self):
        with self._export_lock:
            with self._lock:
                if not self._closed:
                    self._journal.flush()
                events = [json.loads(line) for line in (self.directory / 'transcript.jsonl').read_text(encoding='utf-8').splitlines()]
            return export_events(events, self.directory)

    def close(self):
        with self._lock:
            if not self._closed:
                self._append(dict(kind='ended', at=_utc()))
                self._journal.flush();os.fsync(self._journal.fileno());self._journal.close()
                self._closed = True
        return self.export()

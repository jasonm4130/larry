"""The immutable per-turn ``[speaker: name]`` identity tag — one format, one parser.

The tag is the single carrier of a turn's frozen speaker identity across the
pipeline: ``SpeakerTagProcessor`` (speaker_id.py) writes it onto
``TranscriptionFrame.text`` right after STT, it rides unchanged through
``WhisperHallucinationFilter`` (whose strip is non-mutating) into the LLM
context, and the scoped Mem0 store + the conversation log read it back — so the
tag, Mem0 retrieval/store, and the log all bind to the *same* identity that was
snapshotted at the turn boundary, never a live mutable value read later.

Kept dependency-free (only ``re``) so both the torch-backed speaker_id module
and the Mem0/log side can import it without pulling the other's heavy deps.
"""

import re

# Matches a leading "[speaker: <name>]" tag and captures the name + the rest.
# DOTALL so a (defensively) multi-line transcript still yields its full body.
_TAG_RE = re.compile(r"^\[speaker:\s*([^\]]*)\]\s*(.*)$", re.DOTALL)


def format_speaker_tag(name: str, text: str) -> str:
    """Prefix *text* with this turn's speaker tag (e.g. ``[speaker: jason] hi``)."""
    return f"[speaker: {name}] {text}"


def parse_speaker_tag(text: str) -> tuple[str | None, str]:
    """Split a tagged transcript into ``(speaker, clean_text)``.

    Returns ``(None, text)`` unchanged when there is no ``[speaker: …]`` prefix
    (e.g. the streaming-STT path that never tags), so callers can distinguish
    "no tag present" from an explicit ``[speaker: unknown]``.
    """
    m = _TAG_RE.match(text)
    if m is None:
        return None, text
    return m.group(1).strip(), m.group(2)

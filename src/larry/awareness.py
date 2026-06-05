"""Pure time-of-day and recency helpers for Larry's prompt context.

All functions are I/O-free; callers inject `hour`/`now` so logic is
fully unit-testable. See
docs/superpowers/specs/2026-06-05-time-and-recency-awareness-design.md.
"""

from __future__ import annotations

import datetime


def time_register(hour: int) -> str:
    """Return the coarse time-of-day register string for *hour* (0–23).

    Four buckets, matching the inline logic previously in
    pipeline._load_system_prompt (pipeline.py:227-237). Single source of truth.
    """
    if hour < 9:
        return (
            "It is early morning. Shorter replies. "
            "You are not enjoying this any more than the humans."
        )
    if hour < 16:
        return "It is mid-day. Standard register."
    if hour < 18:
        return "It is late afternoon. The pretense of patience drops."
    return "It is evening. The office is empty. You are still on. You note this."


def recency_phrase(last_seen: str | None, now: datetime.datetime) -> str | None:
    """Map a stored ISO-8601 timestamp to a coarse English recency phrase.

    Returns ``None`` for the first-time case (``last_seen`` is ``None`` or the
    speaker was just enrolled). The caller decides whether to emit "first time"
    text or nothing. All other cases return a non-empty string.

    Buckets (coarse, by design — see spec):
      same calendar day  → "earlier today"
      previous day       → "yesterday"
      2–6 days           → "N days ago"
      7+ days            → "a while ago"
    """
    if last_seen is None:
        return None

    try:
        seen_dt = datetime.datetime.fromisoformat(last_seen)
    except ValueError:
        return None

    # Normalise both ends to UTC date for calendar-day comparisons.
    if seen_dt.tzinfo is None:
        seen_dt = seen_dt.replace(tzinfo=datetime.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.UTC)

    now_date = now.date()
    seen_date = seen_dt.date()
    delta_days = (now_date - seen_date).days

    if delta_days < 0:
        return "earlier today"
    if delta_days == 0:
        return "earlier today"
    if delta_days == 1:
        return "yesterday"
    if delta_days < 7:
        return f"{delta_days} days ago"
    return "a while ago"

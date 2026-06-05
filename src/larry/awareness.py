"""Pure time-of-day and recency helpers for Larry's prompt context.

All functions are I/O-free; callers inject `hour`/`now` so logic is
fully unit-testable. See
docs/superpowers/specs/2026-06-05-time-and-recency-awareness-design.md.
"""

from __future__ import annotations


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

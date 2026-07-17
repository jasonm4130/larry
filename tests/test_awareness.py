"""Tests for larry.awareness — pure time and recency helpers."""

import datetime
from datetime import datetime as dt

import pytest

from larry import awareness


@pytest.mark.parametrize(
    "hour,expected_fragment",
    [
        # Early-morning bucket: 0 through 8 inclusive (hour < 9)
        (0, "early morning"),
        (8, "early morning"),
        # Mid-day bucket: 9 through 15 inclusive (hour < 16)
        (9, "mid-day"),
        (15, "mid-day"),
        # Late-afternoon bucket: 16 through 17 inclusive (hour < 18)
        (16, "late afternoon"),
        (17, "late afternoon"),
        # Evening bucket: 18 and up
        (18, "evening"),
        (23, "evening"),
    ],
)
def test_time_register_bucket(hour: int, expected_fragment: str):
    result = awareness.time_register(hour)
    assert expected_fragment in result


def test_time_register_exhaustive_coverage():
    """Every hour 0-23 returns a non-empty string, no hour raises."""
    for hour in range(24):
        result = awareness.time_register(hour)
        assert isinstance(result, str) and result


def test_time_register_returns_string_not_multiline():
    """Each bucket is a single prose line (no embedded newlines that break context layout)."""
    for hour in range(24):
        assert "\n" not in awareness.time_register(hour)


# ---------------------------------------------------------------------------
# recency_phrase
# ---------------------------------------------------------------------------

_NOW = dt(2026, 6, 5, 14, 0, 0, tzinfo=datetime.UTC)


def test_recency_phrase_none_input_returns_none():
    """Unknown speaker (no stored timestamp) → None; caller decides phrasing."""
    assert awareness.recency_phrase(None, _NOW) is None


def test_recency_phrase_same_day():
    last = dt(2026, 6, 5, 9, 30, 0, tzinfo=datetime.UTC).isoformat()
    assert awareness.recency_phrase(last, _NOW) == "earlier today"


def test_recency_phrase_yesterday():
    last = dt(2026, 6, 4, 20, 0, 0, tzinfo=datetime.UTC).isoformat()
    assert awareness.recency_phrase(last, _NOW) == "yesterday"


@pytest.mark.parametrize("days_ago", [2, 3, 6])
def test_recency_phrase_n_days_ago(days_ago: int):
    last = (_NOW - datetime.timedelta(days=days_ago)).isoformat()
    result = awareness.recency_phrase(last, _NOW)
    assert result == f"{days_ago} days ago"


def test_recency_phrase_seven_days_is_weeks():
    last = (_NOW - datetime.timedelta(days=7)).isoformat()
    result = awareness.recency_phrase(last, _NOW)
    assert result == "a while ago"


def test_recency_phrase_many_weeks():
    last = (_NOW - datetime.timedelta(days=30)).isoformat()
    result = awareness.recency_phrase(last, _NOW)
    assert result == "a while ago"


def test_recency_phrase_midnight_boundary():
    """A timestamp from 23:59 yesterday with now at 00:01 today is 'yesterday'."""
    now_midnight = dt(2026, 6, 5, 0, 1, 0, tzinfo=datetime.UTC)
    last = dt(2026, 6, 4, 23, 59, 0, tzinfo=datetime.UTC).isoformat()
    assert awareness.recency_phrase(last, now_midnight) == "yesterday"


def test_recency_phrase_six_to_seven_boundary():
    """6 days ago → 'N days ago'; 7 days ago → 'a while ago'."""
    last_6 = (_NOW - datetime.timedelta(days=6)).isoformat()
    last_7 = (_NOW - datetime.timedelta(days=7)).isoformat()
    assert awareness.recency_phrase(last_6, _NOW) == "6 days ago"
    assert awareness.recency_phrase(last_7, _NOW) == "a while ago"


def test_recency_phrase_future_timestamp_treated_as_now():
    """A last_seen one day in the future (clock skew) returns 'earlier today', not '-1 days ago'."""
    future = (_NOW + datetime.timedelta(days=1)).isoformat()
    result = awareness.recency_phrase(future, _NOW)
    assert result == "earlier today"


# --------------------------------------------------------------------------
# effective_recency_line: the recency line shown to the LLM must agree with
# THIS turn's speaker snapshot. On an unconfirmed ('unknown') turn the stored
# line still names the last confirmed speaker, so it must be suppressed —
# otherwise the prompt says "You are speaking with Alice" while an unproven
# voice is talking (Codex P2). Named turns pass the stored line through.
# --------------------------------------------------------------------------


def test_effective_recency_line_suppressed_on_unknown_turn():
    stored = "You are speaking with Alice. Last with you yesterday."
    assert awareness.effective_recency_line(stored, "unknown") is None


def test_effective_recency_line_passes_through_on_named_turn():
    stored = "You are speaking with Alice. Last with you yesterday."
    assert awareness.effective_recency_line(stored, "alice") == stored


def test_effective_recency_line_none_stays_none():
    assert awareness.effective_recency_line(None, "unknown") is None
    assert awareness.effective_recency_line(None, "bob") is None

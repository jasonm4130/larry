"""Tests for larry.awareness — pure time and recency helpers."""

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

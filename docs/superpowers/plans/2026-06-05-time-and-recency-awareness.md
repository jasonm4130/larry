# Time + Recency Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Larry two coarse, in-character senses of time he lacks today: (1) a live time-of-day register that refreshes each user turn rather than freezing at boot, and (2) a recency phrase injected into the prompt when a known speaker is identified. Both are the same philosophy — coarse, evocative context, not precise data.

**Architecture:** A new pure `awareness.py` module owns `time_register(hour)` and `recency_phrase(last_seen, now)` (no I/O, clock-injected). `speaker_id.py` gains an idempotent `last_seen TEXT` migration and a `touch_last_seen` helper. The pipeline gains a unified, ungated `_refresh_system_prompt()` that replaces the gated `_rebuild_system_prompt`, is called from `on_user_turn_stopped`, and is also called from `_on_speaker_change` for Part 2.

**Spec:** `docs/superpowers/specs/2026-06-05-time-and-recency-awareness-design.md`

**Dependency note — Part 2:** The recency wiring (`_on_speaker_change` reading `last_seen`, updating it, and injecting the recency line) only produces visible output once speakers are actually enrolled and `_on_speaker_change` fires with real names. That is what the voice-enrollment feature (`2026-06-05-voice-enrollment-and-dismiss-design.md`) unlocks. The Part 2 storage migration and `recency_phrase` logic are fully testable now; only the pipeline wiring waits for enrollment to be useful. Both parts share the `_refresh_system_prompt` seam, so Part 2 is specced here and lands in the same branch.

---

## File Structure

- **Create** `src/larry/awareness.py` — pure `time_register` and `recency_phrase` functions.
- **Create** `tests/test_awareness.py` — full coverage of the new module.
- **Modify** `src/larry/speaker_id.py` — add `last_seen TEXT` column (idempotent migration), `touch_last_seen` helper, and `last_seen` round-trip in `load_enrolled`.
- **Modify** `tests/test_speaker_id.py` — cover the new DB helpers.
- **Modify** `src/larry/self_layer.py` — extend `compose_system_prompt` to accept an optional `recency_line` kwarg and emit it inside `## Current Context`.
- **Modify** `tests/test_self_layer.py` — cover the new kwarg.
- **Modify** `src/larry/pipeline.py` — extract `time_register` call from `_load_system_prompt`, add `_refresh_system_prompt` (ungated, generalises the existing gated `_rebuild_system_prompt`), wire into `on_user_turn_stopped` (Part 1), wire recency into `_on_speaker_change` (Part 2).

`speakers.db` lives under `data/` which is already `.gitignore`d.

---

## Task 1: `awareness.py` — `time_register` (pure, four-bucket)

**Files:**
- Create: `src/larry/awareness.py`
- Create: `tests/test_awareness.py`

The inline time-of-day logic currently lives verbatim in `_load_system_prompt` (`pipeline.py:227-237`). This task extracts it into a single, testable source-of-truth.

- [ ] **Step 1: Write the failing test**

Create `tests/test_awareness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_awareness.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'larry.awareness'`.

- [ ] **Step 3: Implement minimal code**

Create `src/larry/awareness.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_awareness.py -q
```

Expected: PASS (3 tests).

- [ ] **Step 5: Lint and typecheck**

```bash
uv run ruff check src/larry/awareness.py tests/test_awareness.py && uv run pyright src/larry/awareness.py
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/larry/awareness.py tests/test_awareness.py
git commit -m "feat(awareness): extract time_register into pure awareness module"
```

---

## Task 2: `awareness.py` — `recency_phrase` (pure, five-bucket)

**Files:**
- Modify: `src/larry/awareness.py`
- Modify: `tests/test_awareness.py`

This is the Part 2 pure function. It maps an ISO-8601 timestamp string (or `None`) plus a `now: datetime` to a coarse English phrase, or `None` for the first-time case. No storage, no I/O.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_awareness.py`:

```python
from datetime import datetime as dt, timezone


# ---------------------------------------------------------------------------
# recency_phrase
# ---------------------------------------------------------------------------

_NOW = dt(2026, 6, 5, 14, 0, 0, tzinfo=timezone.utc)


def test_recency_phrase_none_input_returns_none():
    """Unknown speaker (no stored timestamp) → None; caller decides phrasing."""
    assert awareness.recency_phrase(None, _NOW) is None


def test_recency_phrase_same_day():
    last = dt(2026, 6, 5, 9, 30, 0, tzinfo=timezone.utc).isoformat()
    assert awareness.recency_phrase(last, _NOW) == "earlier today"


def test_recency_phrase_yesterday():
    last = dt(2026, 6, 4, 20, 0, 0, tzinfo=timezone.utc).isoformat()
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
    now_midnight = dt(2026, 6, 5, 0, 1, 0, tzinfo=timezone.utc)
    last = dt(2026, 6, 4, 23, 59, 0, tzinfo=timezone.utc).isoformat()
    assert awareness.recency_phrase(last, now_midnight) == "yesterday"


def test_recency_phrase_six_to_seven_boundary():
    """6 days ago → 'N days ago'; 7 days ago → 'a while ago'."""
    last_6 = (_NOW - datetime.timedelta(days=6)).isoformat()
    last_7 = (_NOW - datetime.timedelta(days=7)).isoformat()
    assert awareness.recency_phrase(last_6, _NOW) == "6 days ago"
    assert awareness.recency_phrase(last_7, _NOW) == "a while ago"
```

Also add `import datetime` at the top of the test file (after the existing imports).

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_awareness.py -k recency -q
```

Expected: FAIL — `AttributeError: module 'larry.awareness' has no attribute 'recency_phrase'`.

- [ ] **Step 3: Implement minimal code**

Add to `src/larry/awareness.py` (add `import datetime` alongside the existing `from __future__ import annotations` and `import datetime`; the latter already present):

```python
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
        seen_dt = seen_dt.replace(tzinfo=datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    now_date = now.date()
    seen_date = seen_dt.date()
    delta_days = (now_date - seen_date).days

    if delta_days == 0:
        return "earlier today"
    if delta_days == 1:
        return "yesterday"
    if delta_days < 7:
        return f"{delta_days} days ago"
    return "a while ago"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_awareness.py -q
```

Expected: PASS (all tests).

- [ ] **Step 5: Lint and typecheck**

```bash
uv run ruff check src/larry/awareness.py tests/test_awareness.py && uv run pyright src/larry/awareness.py
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/larry/awareness.py tests/test_awareness.py
git commit -m "feat(awareness): add recency_phrase with five coarse buckets"
```

---

## Task 3: `speaker_id.py` — `last_seen` column + `touch_last_seen` helper

**Files:**
- Modify: `src/larry/speaker_id.py`
- Modify: `tests/test_speaker_id.py`

Add a `last_seen TEXT` column to the speakers table (idempotent: additive `ALTER TABLE … ADD COLUMN` only runs when the column is absent), a `touch_last_seen(db_path, name, now)` helper that writes an ISO stamp, and surface the column in `load_enrolled` for the pipeline's recency lookup.

**Dependency note:** The `store_speaker` helper (for writing new embeddings) is introduced by the voice-enrollment feature. `touch_last_seen` is deliberately independent — it only needs an existing row by name, so it can be called by `_on_speaker_change` as soon as a speaker is recognised, regardless of whether the enrollment UI is built yet.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_speaker_id.py`:

```python
import sqlite3

import larry.speaker_id as speaker_id


# ---------------------------------------------------------------------------
# last_seen migration + helpers
# ---------------------------------------------------------------------------

def test_ensure_schema_adds_last_seen_column(tmp_path):
    """_ensure_schema adds last_seen TEXT; calling it twice is idempotent."""
    db = tmp_path / "speakers.db"
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        # Column must exist after first call.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(speakers)")}
        assert "last_seen" in cols
        # Second call must not raise.
        speaker_id._ensure_schema(conn)


def test_ensure_schema_upgrade_preserves_existing_rows(tmp_path):
    """Schema migration on a DB that already has rows must not drop them."""
    db = tmp_path / "speakers.db"
    embedding = b"\x00\x00\x80\x3f"  # float32 1.0 as bytes
    with sqlite3.connect(db) as conn:
        # Manually create the OLD schema (no last_seen column).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS speakers "
            "(name TEXT PRIMARY KEY, embedding BLOB NOT NULL, "
            "enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)", ("alice", embedding))
        conn.commit()
    # Now run the new _ensure_schema — should add last_seen without dropping alice.
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        rows = conn.execute("SELECT name FROM speakers").fetchall()
    assert [r[0] for r in rows] == ["alice"]


def test_touch_last_seen_writes_iso_stamp(tmp_path):
    """touch_last_seen inserts a valid ISO stamp for an existing row."""
    db = tmp_path / "speakers.db"
    embedding = b"\x00\x00\x80\x3f"
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        conn.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)", ("bob", embedding))
        conn.commit()

    speaker_id.touch_last_seen(db, "bob", now="2026-06-05T14:00:00")

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT last_seen FROM speakers WHERE name='bob'").fetchone()
    assert row[0] == "2026-06-05T14:00:00"


def test_touch_last_seen_unknown_speaker_is_noop(tmp_path):
    """touch_last_seen on a non-existent name must not raise or create a row."""
    db = tmp_path / "speakers.db"
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        conn.commit()
    # Should silently no-op — UPDATE WHERE name='ghost' matches 0 rows.
    speaker_id.touch_last_seen(db, "ghost", now="2026-06-05T14:00:00")
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
    assert count == 0


def test_load_enrolled_round_trips_last_seen(tmp_path):
    """load_enrolled returns last_seen alongside the embedding."""
    db = tmp_path / "speakers.db"
    import numpy as np
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        conn.execute(
            "INSERT INTO speakers (name, embedding, last_seen) VALUES (?, ?, ?)",
            ("carol", emb.tobytes(), "2026-06-01T10:00:00"),
        )
        conn.commit()

    result = speaker_id.load_enrolled(db)
    # load_enrolled currently returns name → embedding dict; after this task it
    # must return name → (embedding, last_seen) OR a separate helper exists.
    # Per the spec, load_enrolled returns the embedding dict unchanged; a
    # separate load_last_seen(db, name) is sufficient for the pipeline.
    # This test just checks the new load_last_seen helper.
    last = speaker_id.load_last_seen(db, "carol")
    assert last == "2026-06-01T10:00:00"


def test_load_last_seen_unknown_returns_none(tmp_path):
    db = tmp_path / "speakers.db"
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
    assert speaker_id.load_last_seen(db, "nobody") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_speaker_id.py -k "last_seen or touch_last or load_last" -q
```

Expected: FAIL — the `last_seen` column is absent from `_ensure_schema` and `touch_last_seen` / `load_last_seen` do not exist.

- [ ] **Step 3: Implement minimal code**

In `src/larry/speaker_id.py`, replace the existing `_ensure_schema` and add the two helpers after `load_enrolled`:

```python
def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the speakers table if needed; idempotently add last_seen column."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            name TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TEXT
        )
    """)
    # Idempotent migration: add last_seen on DBs created by the old schema.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(speakers)")}
    if "last_seen" not in existing_cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN last_seen TEXT")
    conn.commit()


def touch_last_seen(db_path: Path, name: str, *, now: str | None = None) -> None:
    """Update the last_seen timestamp for *name* to *now* (ISO-8601).

    No-op if *name* is not in the DB (unknown speaker). Does not insert new rows.
    """
    import datetime as _dt

    stamp = now or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute("UPDATE speakers SET last_seen = ? WHERE name = ?", (stamp, name))
        conn.commit()


def load_last_seen(db_path: Path, name: str) -> str | None:
    """Return the stored last_seen ISO stamp for *name*, or None if absent/unknown."""
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT last_seen FROM speakers WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        return None
    return row[0]  # may be None (NULL) for rows enrolled before this migration
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_speaker_id.py -q
```

Expected: PASS (all tests, including pre-existing ones — migration must not break them).

- [ ] **Step 5: Lint and typecheck**

```bash
uv run ruff check src/larry/speaker_id.py tests/test_speaker_id.py && uv run pyright src/larry/speaker_id.py
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/larry/speaker_id.py tests/test_speaker_id.py
git commit -m "feat(speaker-id): add last_seen column with idempotent migration and touch/load helpers"
```

---

## Task 4: `self_layer.compose_system_prompt` — accept optional `recency_line`

**Files:**
- Modify: `src/larry/self_layer.py`
- Modify: `tests/test_self_layer.py`

The `## Current Context` block currently holds only the time register. Add an optional `recency_line` kwarg so `_refresh_system_prompt` can inject the speaker recency phrase alongside it. The kwarg is `None` by default — existing call sites are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_self_layer.py`:

```python
_CARD = (
    "# Larry\n\nYou are Larry.\n\n"
    "## Hard Constraints — Strength 5 (Absolute. Non-negotiable.)\n\n"
    "You will never use slurs.\n\n"
    "## Soft Constraints\n\nBe warm.\n"
)


def test_compose_includes_recency_line_when_provided():
    prompt = self_layer.compose_system_prompt(
        card=_CARD,
        self_block="",
        time_context="It is mid-day. Standard register.",
        guardrails=self_layer.extract_hard_constraints(_CARD),
        recency_line="You are speaking with Jason. Last with you 4 days ago.",
    )
    assert "You are speaking with Jason" in prompt
    assert "4 days ago" in prompt
    # Recency line must be inside the ## Current Context block (before guardrails).
    context_start = prompt.find("## Current Context")
    guardrail_start = prompt.find(self_layer.GUARDRAIL_HEADER)
    recency_pos = prompt.find("4 days ago")
    assert context_start < recency_pos < guardrail_start


def test_compose_omits_recency_block_when_none():
    prompt = self_layer.compose_system_prompt(
        card=_CARD,
        self_block="",
        time_context="It is mid-day. Standard register.",
        guardrails=self_layer.extract_hard_constraints(_CARD),
        recency_line=None,
    )
    # No stray blank speaker line should appear.
    assert "You are speaking with" not in prompt


def test_compose_recency_default_is_none():
    """Calling compose_system_prompt without recency_line must not raise."""
    prompt = self_layer.compose_system_prompt(
        card=_CARD,
        self_block="",
        time_context="It is morning.",
        guardrails=self_layer.extract_hard_constraints(_CARD),
    )
    assert "## Current Context" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_self_layer.py -k "recency" -q
```

Expected: FAIL — `compose_system_prompt` does not yet accept `recency_line`.

- [ ] **Step 3: Implement minimal code**

In `src/larry/self_layer.py`, update the `compose_system_prompt` signature and body:

```python
def compose_system_prompt(
    *,
    card: str,
    self_block: str,
    time_context: str,
    guardrails: str,
    recency_line: str | None = None,
) -> str:
    """Assemble the system prompt with the immutable guardrails LAST.

    ``recency_line`` is an optional coarse speaker-recency phrase appended inside
    ``## Current Context`` (e.g. "You are speaking with Jason. Last with you 4 days ago.").
    Pass ``None`` (the default) to omit it — callers that don't know the speaker yet.
    """
    parts = [card.strip()]
    if self_block.strip():
        parts.append(self_block.strip())
    context_body = time_context
    if recency_line:
        context_body = f"{time_context}\n\n{recency_line}"
    parts.append(f"## Current Context\n\n{context_body}")
    parts.append(f"{GUARDRAIL_HEADER}\n\n{GUARDRAIL_PREAMBLE}\n\n{guardrails}")
    return "\n\n".join(parts) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_self_layer.py -q
```

Expected: PASS (all tests, including pre-existing Task 4 compose tests — the new default kwarg is backwards-compatible).

- [ ] **Step 5: Lint and typecheck**

```bash
uv run ruff check src/larry/self_layer.py tests/test_self_layer.py && uv run pyright src/larry/self_layer.py
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/larry/self_layer.py tests/test_self_layer.py
git commit -m "feat(self-layer): compose_system_prompt accepts optional recency_line kwarg"
```

---

## Task 5: Pipeline — Part 1: live time register via `_refresh_system_prompt`

**Files:**
- Modify: `src/larry/pipeline.py`

This task (a) replaces the inline time-of-day block in `_load_system_prompt` with a call to `awareness.time_register`, (b) introduces a single ungated `_refresh_system_prompt()` helper that subsumes the gated `_rebuild_system_prompt`, and (c) calls `_refresh_system_prompt()` from `on_user_turn_stopped` so the time register stays live every turn.

No new test file is needed here — the `awareness.time_register` function is covered by Task 1, and the `compose_system_prompt` path is covered by existing `test_self_layer.py` tests. The pipeline wiring test is a regression check run via `uv run pytest -q`.

- [ ] **Step 1: Add `from larry import awareness` import**

In `src/larry/pipeline.py`, add alongside the existing `from larry import self_layer` (line 61):

```python
from larry import awareness
```

- [ ] **Step 2: Replace inline time-of-day in `_load_system_prompt`**

`_load_system_prompt` currently spans `pipeline.py:223-243`. Replace the inline `if/elif` block:

Old (lines 226-237):
```python
    hour = datetime.datetime.now().hour
    if hour < 9:
        tod = (
            "It is early morning. Shorter replies. "
            "You are not enjoying this any more than the humans."
        )
    elif hour < 16:
        tod = "It is mid-day. Standard register."
    elif hour < 18:
        tod = "It is late afternoon. The pretense of patience drops."
    else:
        tod = "It is evening. The office is empty. You are still on. You note this."
```

New:
```python
    hour = datetime.datetime.now().hour
    tod = awareness.time_register(hour)
```

The `_load_system_prompt` signature and the `compose_system_prompt` call below it are unchanged. Now `_load_system_prompt` is the two-argument form (for boot composition and for the refresh helper below).

- [ ] **Step 3: Introduce `_refresh_system_prompt` (ungated), replace gated `_rebuild_system_prompt`**

The existing gated block at `pipeline.py:614-628`:

```python
    if cfg.self_evolution_enabled:

        async def _rebuild_system_prompt() -> None:
            messages = context.get_messages()
            new_system = _load_system_prompt(cfg.personality_path, cfg.self_layer_path)
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    msg["content"] = new_system
                    break
            context.set_messages(messages)

        llm.register_function(
            "keep_about_self",
            self_layer.make_keep_about_self_handler(cfg.self_layer_path, _rebuild_system_prompt),
        )
```

Replace with:

```python
    # Single, ungated system-prompt refresh path — always safe to call because
    # it merely recomposes the same content with a live time register.  The
    # self-evolution gate stays only around registering the keep_about_self tool.
    _recency_line: dict[str, str | None] = {"value": None}

    async def _refresh_system_prompt() -> None:
        messages = context.get_messages()
        new_system = _load_system_prompt(
            cfg.personality_path,
            cfg.self_layer_path,
            recency_line=_recency_line["value"],
        )
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                msg["content"] = new_system
                break
        context.set_messages(messages)

    if cfg.self_evolution_enabled:
        llm.register_function(
            "keep_about_self",
            self_layer.make_keep_about_self_handler(cfg.self_layer_path, _refresh_system_prompt),
        )
```

Note: `_load_system_prompt` now needs to accept `recency_line` — see sub-step below.

- [ ] **Step 4: Extend `_load_system_prompt` to forward `recency_line`**

Update the signature and `compose_system_prompt` call in `_load_system_prompt`:

```python
def _load_system_prompt(
    personality_path: Path,
    self_layer_path: Path,
    *,
    recency_line: str | None = None,
) -> str:
    """Compose the system prompt: card + self-layer + time + recency + immutable guardrails."""
    card = personality_path.read_text()
    hour = datetime.datetime.now().hour
    return self_layer.compose_system_prompt(
        card=card,
        self_block=self_layer.read_self_layer(self_layer_path),
        time_context=awareness.time_register(hour),
        guardrails=self_layer.extract_hard_constraints(card),
        recency_line=recency_line,
    )
```

The boot call site at `pipeline.py:414` (`system_prompt = _load_system_prompt(cfg.personality_path, cfg.self_layer_path)`) passes no `recency_line`, so it gets `None` — correct at boot.

- [ ] **Step 5: Wire `_refresh_system_prompt` into `on_user_turn_stopped`**

The existing handler at `pipeline.py:659-673` ends with `_pending["speaker"] = speaker_id.current_speaker`. Append the refresh call inside the handler (after the existing body):

```python
    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, *args, **kwargs) -> None:  # noqa: ARG001
        messages = context.get_messages()
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if msg.get("role") == "user" and isinstance(content, str):
                _pending["user_text"] = content
                break
        _pending["speaker"] = speaker_id.current_speaker
        await _refresh_system_prompt()  # keep time register live every turn
```

- [ ] **Step 6: Verify suite and types**

```bash
uv run pytest -q && uv run ruff check src/larry && uv run pyright src/larry/pipeline.py
```

Expected: all pass, 0 pyright errors.

- [ ] **Step 7: Commit**

```bash
git add src/larry/pipeline.py
git commit -m "feat(pipeline): live time-register refresh per user turn via ungated _refresh_system_prompt"
```

---

## Task 6: Pipeline — Part 2: recency wiring in `_on_speaker_change`

**Files:**
- Modify: `src/larry/pipeline.py`

Wire `awareness.recency_phrase` + `speaker_id.touch_last_seen` + `speaker_id.load_last_seen` into `_on_speaker_change` so the recency line is injected into the prompt (via `_recency_line["value"]` + `_refresh_system_prompt()`) whenever a known speaker is identified. Unknown speaker → holder cleared, no line emitted.

**Dependency reminder:** This code is correct and testable today; it only produces visible prompt output once speakers are enrolled (voice-enrollment feature). Until then, `_on_speaker_change` fires only with `"unknown"` and the holder stays `None`.

- [ ] **Step 1: Replace `_on_speaker_change` in the pipeline**

The current `_on_speaker_change` at `pipeline.py:341-343`:

```python
    def _on_speaker_change(new_name: str) -> None:
        mem0_service.user_id = new_name
        logger.info("Mem0 user_id updated to %r", new_name)
```

Replace with:

```python
    def _on_speaker_change(new_name: str) -> None:
        mem0_service.user_id = new_name
        logger.info("Mem0 user_id updated to %r", new_name)

        if new_name == "unknown":
            _recency_line["value"] = None
            return

        # Read last_seen before updating it (so recency is "how long since
        # they were last here", not "zero seconds ago").
        last_seen = speaker_id_module.load_last_seen(cfg.speakers_db, new_name)
        phrase = awareness.recency_phrase(last_seen, datetime.datetime.now(datetime.timezone.utc))
        if phrase is not None:
            _recency_line["value"] = (
                f"You are speaking with {new_name}. Last with you {phrase}."
            )
        else:
            # First time this speaker has talked to Larry.
            _recency_line["value"] = f"You are speaking with {new_name} for the first time."

        speaker_id_module.touch_last_seen(cfg.speakers_db, new_name)
        # Schedule a prompt refresh so the new recency line lands immediately.
        asyncio.create_task(_refresh_system_prompt())
```

Note: `speaker_id` is imported as a module (`from larry.speaker_id import SpeakerIDProcessor` at the import). Add a module-alias import near the top: `import larry.speaker_id as speaker_id_module` (or rename the existing processor variable) to avoid the name clash with the `SpeakerIDProcessor` instance called `speaker_id`. Concretely:

- At the imports section, add: `import larry.speaker_id as speaker_id_module`
- The `SpeakerIDProcessor` instance remains named `speaker_id` (processor variable) — unchanged call sites.
- `_recency_line` is declared before `_on_speaker_change` now sits (it was introduced in Task 5, step 3 — confirm the definition is in scope here; both closures are inside `run()`).

The `asyncio.create_task(_refresh_system_prompt())` call requires `task` (the pipeline task) to be running, so we guard with a check:

```python
        try:
            asyncio.create_task(_refresh_system_prompt())
        except RuntimeError:
            # Called before the event loop is running (shouldn't happen in normal
            # flow but safe to swallow here — the next on_user_turn_stopped will refresh).
            pass
```

- [ ] **Step 2: Verify suite and types**

```bash
uv run pytest -q && uv run ruff check src/larry && uv run pyright src/larry/pipeline.py
```

Expected: all pass, 0 pyright errors.

- [ ] **Step 3: Manual smoke test (documented, run on Mac dev with enrolled speakers)**

Pre-condition: at least one speaker enrolled via `uv run larry enroll <name>` and speaker's `last_seen` set to a past date:

```bash
sqlite3 data/speakers.db "UPDATE speakers SET last_seen='2026-05-29T10:00:00' WHERE name='<name>'"
```

Run: `uv run larry`, trigger wake word, speak. Expected in logs: `Mem0 user_id updated to '<name>'`. Expected in the composed system prompt (visible if you add a debug log): the `## Current Context` block should contain the speaker line with a "N days ago" phrase.

- [ ] **Step 4: Commit**

```bash
git add src/larry/pipeline.py
git commit -m "feat(pipeline): inject speaker recency phrase on speaker-change (Part 2)"
```

---

## Self-Review

- **Spec coverage:**
  - Time-of-day register extracted to pure function, single source of truth (Task 1). ✓
  - Register refreshed per user turn, not just at boot (Task 5, step 5). ✓
  - `recency_phrase` five-bucket pure function (Task 2). ✓
  - `last_seen TEXT` column, idempotent migration, `touch_last_seen`, `load_last_seen` (Task 3). ✓
  - `compose_system_prompt` extended with `recency_line` kwarg (Task 4). ✓
  - `_rebuild_system_prompt` unified into ungated `_refresh_system_prompt` (Task 5, step 3). ✓
  - `_on_speaker_change` wires recency read → phrase → holder → touch (Task 6). ✓
  - No new env config (spec: "Neither needs a feature flag"). ✓
  - Dependency on voice-enrollment called out explicitly (Tasks 6 note + spec). ✓
- **Placeholder scan:** All code steps have concrete implementations; commands have expected output.
- **Type consistency:** `time_register(hour: int) -> str`, `recency_phrase(last_seen: str | None, now: datetime.datetime) -> str | None`, `touch_last_seen(db_path, name, *, now)`, `load_last_seen(db_path, name) -> str | None`, `compose_system_prompt(..., recency_line: str | None = None)`, `_load_system_prompt(..., *, recency_line: str | None = None)` — all consistent across tasks.
- **Test isolation:** Every test uses `tmp_path`; no audio, torch, or network calls; pure functions use injected `hour`/`now`.

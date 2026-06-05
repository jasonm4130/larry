# Time + Recency Awareness — Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Author:** Jason + Claude (brainstormed)

## Intent

Give Larry two coarse, in-character senses of time he lacks today:

1. **Live time-of-day.** He already gets a time-of-day *register* ("it is
   evening, the office is empty, you are still on") — but it is computed once at
   boot and then frozen, so over a long session it goes stale. Make it refresh
   each user turn so it tracks the real clock. Stays coarse (the existing four
   buckets) — no exact "it's 4:47" clock, by choice.
2. **Last-spoke recency.** When Larry recognises a known speaker, he should know
   roughly how long it's been since they last talked to him ("back already?" /
   "it's been a week") — a coarse bucket, not a precise timestamp. Surfaced as
   context he *may* use, never forced.

Both are the same philosophy: coarse, evocative context injected into the prompt,
not precise data.

## Background / Why this matters

`_load_system_prompt` (pipeline.py:224) computes the hour once and bakes a
time-of-day register into the system prompt via
`self_layer.compose_system_prompt(time_context=...)`. The prompt is only
recomposed when the self-layer changes (`_rebuild_system_prompt`, gated behind
`self_evolution_enabled`), so Larry's sense of time is frozen at boot and can sit
hours stale. Nothing anywhere records when a given speaker was last heard.

## Dependency

- **Part 1 (time register)** is independent — ships anytime.
- **Part 2 (recency)** only does anything once speakers are actually enrolled and
  `_on_speaker_change` fires with real names. That is what the voice-enrollment
  feature (`2026-06-05-voice-enrollment-and-dismiss-design.md`) unlocks; before
  it, every turn is "unknown" and there is no one to be recent about. So Part 2
  lands after voice-enrollment. Both parts are specced together here because they
  share the prompt-composition seam and the coarse-context philosophy.

## Non-Goals

- An exact wall-clock or calendar date in the prompt (chose coarse register).
- A queryable `check_time()` tool (rejected — he should reference time
  spontaneously, not only on demand).
- Precise "last seen at 14:32 on Tuesday" — recency is bucketed, not exact.
- New env configuration. The time register strictly improves existing behaviour;
  recency only activates for known speakers. Neither needs a feature flag.

## Architecture

### New module: `src/larry/awareness.py`

Two pure, unit-testable functions (no I/O, no clock reads — `now`/`hour`
injected), mirroring the `self_layer.py` module-per-concern style:

- `time_register(hour: int) -> str` — the four-bucket time-of-day string,
  extracted verbatim from the inline logic now in `_load_system_prompt`
  (early morning <9 / mid-day <16 / late afternoon <18 / evening). Single source
  of truth, testable at the boundaries.
- `recency_phrase(last_seen: str | None, now: datetime) -> str | None` — maps a
  stored ISO timestamp to a coarse bucket relative to `now`:
  - `None` (never seen / just enrolled) → `None` (caller emits a "first time"
    line or nothing)
  - same calendar day → "earlier today"
  - previous calendar day → "yesterday"
  - 2–6 days → "N days ago"
  - 7+ days → "weeks ago" (or "a while ago")
  Returns `None` for the first-time case so the caller decides phrasing.

### Prompt composition: unfreeze + unify (pipeline.py)

The system prompt splits into a cached-static part and a recomputed-dynamic part:

- **Cache the personality card** — read `personality_path` once at boot (the only
  real disk cost). The self-layer and the dynamic context are recomposed on
  refresh; re-reading the small self-layer file per refresh is cheap.
- **`_refresh_system_prompt()`** — a single, **ungated** helper that recomposes
  the system message and writes it into `context` message[0] using the existing
  get_messages → mutate role=="system" → set_messages pattern. It assembles the
  `## Current Context` block from `time_register(now.hour)` plus the current
  recency line (if any), and calls `compose_system_prompt(...)`.
- **The existing `_rebuild_system_prompt`** (self-layer path) collapses into a
  call to `_refresh_system_prompt()` — one refresh path, not two. The
  self-evolution gate stays only around *registering the `keep_about_self`
  tool*, not around prompt refresh (refresh is harmless when self-evolution is
  off — it just recomposes the same self-layer plus a live time register).

### Part 1 wiring — refresh per user turn

Call `_refresh_system_prompt()` from the existing `on_user_turn_stopped` handler
(pipeline.py:654), which fires before the LLM processes the turn — so the live
register is in place for that very turn. The register changes ~4×/day; per-turn
refresh is cheap (string assembly + one small file read) and keeps it always
current with no cadence logic.

### Part 2 wiring — recency on speaker change

Extend the existing `_on_speaker_change(new_name)` handler (pipeline.py:340):

1. If `new_name` is a known speaker (not "unknown"), read their stored
   `last_seen` from `speakers.db`, compute `recency_phrase(last_seen, now)`, and
   stash the resulting line (e.g. "You are speaking with Jason. Last with you
   4 days ago." / first-time variant) in a small mutable holder in the `run()`
   closure (same pattern as `_pending`).
2. Update that speaker's `last_seen` to now in `speakers.db`.
3. Call `_refresh_system_prompt()` so the recency line lands in `## Current
   Context` immediately.

When the speaker is "unknown", the holder is cleared and no speaker line is
emitted. The recency line sits alongside the time register:

```
## Current Context

It is late afternoon. The pretense of patience drops.

You are speaking with Jason. Last with you 4 days ago.
```

### Storage: `last_seen` in `speakers.db` (speaker_id.py)

- `_ensure_schema` gains a `last_seen TEXT` column on the speakers table, added
  idempotently (additive migration; existing rows get NULL → treated as
  first-time by `recency_phrase`).
- A small helper writes `last_seen` for a name (reuse / sit beside the
  `store_speaker` helper introduced by the voice-enrollment work). `load_enrolled`
  round-trips the new column.

## Testing (TDD, Mac, no hardware)

All logic is pure or SQLite-only — no audio, no torch, no network:

1. **`time_register(hour)`** — correct bucket at each boundary hour
   (8/9, 15/16, 17/18, 18+); exhaustive over 0–23.
2. **`recency_phrase(last_seen, now)`** — `None` → None (first time); same-day →
   "earlier today"; yesterday; 2–6 days → "N days ago"; 7+ → weeks bucket.
   Boundary cases around midnight and the 7-day edge.
3. **`speakers.db` migration** — `_ensure_schema` adds `last_seen`; a row stored
   without it loads as `None`; storing then reloading round-trips an ISO stamp;
   schema upgrade on a pre-existing DB doesn't drop rows.
4. **Prompt composition** — `## Current Context` contains the live register; with
   a recency line set it includes the speaker line; with none it omits it.

## Open Questions

None outstanding — coarseness (chose register over exact clock), refresh cadence
(per user turn), recency buckets, storage location, and "no new config" are all
decided.

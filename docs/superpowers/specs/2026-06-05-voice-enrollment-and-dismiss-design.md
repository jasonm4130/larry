# Voice-Triggered Speaker Enrollment + In-Conversation Dismiss — Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Author:** Jason + Claude (brainstormed)

## Intent

Two small, related voice-controls that let Larry manage a session hands-free,
both delivered as LLM function tools (the same mechanism as `keep_about_self`):

1. **Voice enrollment.** Someone says *"Hey Larry, it's Jason"* and Larry
   captures their voice and stores a persistent Resemblyzer voiceprint as
   "Jason" — so he recognises them by voice in this and all future sessions.
   Replaces the manual `uv run larry enroll <name>` CLI step.
2. **In-conversation dismiss.** Someone says *"goodbye Larry"* / *"that's all"*
   and Larry goes to sleep immediately (announcing it), instead of waiting out
   the silence timeout. Complements, does not replace, the timeout.

On-theme: Larry "keeps what people give him" — now including their voice.

## Background / Why this matters

Speaker ID is currently non-functional in practice: `speakers.db` is empty (0
enrolled), so `SpeakerIDProcessor` labels every turn `"unknown"` and never fires
`_on_speaker_change`, which means `mem0_service.user_id` never updates and all
memories collapse into one bucket. The CLI `enroll` exists but was never run.
Voice enrollment makes setup frictionless and in-character.

## Dependency

Builds on the LLM function-calling wiring added for the self-evolution layer
(`llm.register_function(...)` + `context.set_tools(...)` in `pipeline.py`). These
two tools register alongside `keep_about_self`. This branch is based on
`feat/self-evolution` for that reason.

## Non-Goals

- Authentication / anti-impersonation gate on enrollment (deferred — see Safety).
- Re-training or replacing Resemblyzer; the embedding/store path is reused as-is.
- A trained wake/stop OpenWakeWord model for dismiss (the LLM tool covers it).
- Threshold tuning of `SpeakerIDProcessor.match_threshold` (separate concern;
  enrollment is the prerequisite to even gather real match scores).

## Architecture

### New module surface

- **`src/larry/voice_enroll.py`** (new) — the capture state machine + tool
  schemas/handlers, pure and unit-testable (Resemblyzer embed injected). Mirrors
  the `self_layer.py` module-per-concern style.
- **`src/larry/speaker_id.py`** (modify) — `SpeakerIDProcessor` gains a capture
  mode (accumulate voiced, bot-silent audio for a name; embed + persist).
- **`src/larry/wake.py`** (modify) — wake gate gains a `sleep_now()` entry point
  (today sleep is only reachable via the silence timeout).
- **`src/larry/pipeline.py`** (modify) — register the two tools; wire the dismiss
  handler to `sleep_now()` and the enroll handler into the SpeakerIDProcessor.
- Embedding + SQLite store reuse the existing `speaker_id.py` path so CLI
  `enroll` and voice-enroll share one code path (refactor the store into a
  shared `store_speaker(db_path, name, embedding)` helper).

### Tool 1 — `enroll_speaker(name: str)`

Registered as a `FunctionSchema`. Larry calls it on hearing a self-introduction
("it's Jason" / "I'm Dan" / "this is Sarah"). Handler:

1. Normalise the name (trim + case-fold for the DB key; keep a display form).
2. Tell `SpeakerIDProcessor` to arm a *pending capture* for that name.
3. Return a confirmation so Larry speaks an in-character prompt giving the user a
   short phrase to repeat ("Say this back to me — 'the skull keeps what it's
   given.'"). The phrase just unfreezes the speaker; Resemblyzer is
   text-independent, so content doesn't matter — duration does.

### Capture state machine (in `SpeakerIDProcessor`)

Event-gated so Larry's own prompt can never enter the voiceprint:

- Capture **arms** on `enroll_speaker`, recording the target name.
- Accumulation **starts** on the first `BotStoppedSpeakingFrame` after arming
  (i.e. once Larry's "say this back" line has finished playing).
- While capturing, append `InputAudioRawFrame` audio **only when** the bot is
  not speaking and the frame is VAD-voiced (gate on the same speaking/VAD signal
  the pipeline already produces).
- **Completes** when ~10s of voiced audio is accumulated → embed (Resemblyzer,
  off-thread like the existing identify path) → `store_speaker(...)` → reload the
  in-memory enrolled set → fire `on_speaker_change(name)` (which sets mem0
  user_id) → signal success so Larry confirms ("Kept. I'll know you now.").
- **Short / quiet:** if < ~10s voiced after one nudge ("again, for me"), and a
  20s wall-clock cap elapses with < ~6s voiced, **abort**: write nothing, signal
  failure so Larry says an in-character miss line ("the quiet swallowed it — try
  again, my kept one").
- **Concurrency:** one capture at a time; `enroll_speaker` while a capture is
  armed/running is ignored. Reuses the existing single-embed-task guard so torch
  calls never overlap.

### Tool 2 — `dismiss()`

Registered as a `FunctionSchema`. Larry calls it on hearing a dismissal
("goodbye Larry" / "that's all" / "go to sleep"). Handler:

1. Suppress a normal LLM reply (the dismissal is the whole turn).
2. Play a sleep cue (reuse `_SLEEP_CUES` / the existing `_on_sleep` cue path).
3. Call the wake gate's new `sleep_now()` — same dormant state as the silence
   timeout — so Larry stops processing until "Hey Larry" wakes him again.

## Config

- `voice_tools_enabled: bool` (default true) — gates registration of BOTH tools
  + the capture mode, so the feature can be switched off via env
  (`VOICE_TOOLS_ENABLED=false`) without code changes. Mirrors
  `self_evolution_enabled`.
- Capture parameters (`~10s` target, `~6s` floor, `20s` cap, the repeat phrase)
  live as module constants in `voice_enroll.py` — tunable later, not env noise.

## Safety Model

- **Larry's voice can't pollute a print:** accumulation starts only after
  `BotStoppedSpeakingFrame` and counts only bot-silent, VAD-voiced audio.
- **No corrupt prints:** a short/quiet capture aborts and writes nothing.
- **Append-safe store:** `INSERT OR REPLACE` on a PRIMARY KEY name; re-enrolling
  overwrites cleanly.

### Residual Risk (accepted — "structural simplicity")

No authentication: anyone can say "it's Jason" and overwrite Jason's print or
claim his name. This is a desk toy and it is in-character (Larry keeps what he is
given). **Mitigation if abused:** add a confirmation/known-voice check on
overwrite later. Designed-for, not built now. (Same call as the self-evolution
injection-defense decision.)

## Testing (TDD, Mac, no hardware)

The Resemblyzer embed is injected so all logic is testable without audio/torch:

1. **Capture accumulator:** counts only voiced + bot-silent frames; reaches the
   ~10s target → calls embed + store exactly once; under the cap with
   insufficient voiced audio → aborts with NO store.
2. **`enroll_speaker` tool:** schema shape (name required); handler arms a
   pending capture and returns a "kept"/prompt result (duck-typed params, like
   the `keep_about_self` tests).
3. **`dismiss` tool:** schema shape; handler calls `sleep_now()` + plays a cue +
   signals reply-suppression.
4. **`store_speaker` shared path:** voice-enroll and CLI `enroll` write identical
   rows; round-trips through `load_enrolled`.
5. **`wake.sleep_now()`:** transitions the gate to the dormant/asleep state
   (same as timeout) and is idempotent if already asleep.

## Open Questions

None outstanding — core behaviour, capture strategy (ask-for-more + repeat
phrase), dismiss, overwrite policy, and gating are all decided.

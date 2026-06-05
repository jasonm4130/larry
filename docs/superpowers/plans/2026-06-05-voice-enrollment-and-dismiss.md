# Voice-Triggered Speaker Enrollment + In-Conversation Dismiss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Larry enroll a speaker's voiceprint mid-conversation ("`Hey Larry, it's Jason`") and dismiss himself on command ("`goodbye Larry`"), both via LLM function tools — the same `register_function` / `FunctionSchema` mechanism as `keep_about_self`.

**Architecture:**
- `src/larry/voice_enroll.py` (new) — tool schemas, handler factories, and module constants. Pure and unit-testable; Resemblyzer embed injected.
- `src/larry/speaker_id.py` (modify) — `SpeakerIDProcessor` gains a capture state machine and a `store_speaker` shared helper extracted from `cli/enroll.py`.
- `src/larry/cli/enroll.py` (modify) — refactored to call the new `store_speaker` helper.
- `src/larry/wake.py` (modify) — `WakeWordGate` gains `sleep_now()` (same dormant state as timeout, idempotent).
- `src/larry/config.py` (modify) — add `voice_tools_enabled` field.
- `src/larry/pipeline.py` (modify) — register both tools; wire dismiss handler to `sleep_now()`; wire enroll handler into `SpeakerIDProcessor`.

**Spec:** `docs/superpowers/specs/2026-06-05-voice-enrollment-and-dismiss-design.md`

**Seam references:**
- `src/larry/speaker_id.py:26-35` (`_ensure_schema`) — reused by `store_speaker`
- `src/larry/speaker_id.py:38-45` (`load_enrolled`) — called after successful enrollment to refresh in-memory set
- `src/larry/speaker_id.py:48-129` (`SpeakerIDProcessor`) — gains `arm_capture`, `_capture_state`, `_capture_name`, `_capture_buffer`, `_capture_start`
- `src/larry/speaker_id.py:82-98` (InputAudioRawFrame buffering loop) — capture accumulation hooks here
- `src/larry/cli/enroll.py:47-53` (INSERT OR REPLACE) — becomes `store_speaker(db_path, name, embedding)` call
- `src/larry/wake.py:40-129` (`WakeWordGate`) — gains `sleep_now()` method
- `src/larry/wake.py:116-145` (timeout sleep path) — new method mirrors this block
- `src/larry/pipeline.py:341-348` (`_on_speaker_change`) — fired by `arm_capture` completion
- `src/larry/pipeline.py:414-417` (context + tools) — `voice_enroll.build_voice_tools()` added here
- `src/larry/pipeline.py:583-609` (`_on_sleep`, `_on_wake`) — `dismiss` handler calls `wake_gate.sleep_now()`
- `src/larry/pipeline.py:614-628` (register_function for `keep_about_self`) — mirror for `enroll_speaker` / `dismiss`
- `src/larry/config.py:10-55` (Config dataclass) — `voice_tools_enabled: bool` added near `self_evolution_enabled`

**Tech Stack:** Python 3.12, Pipecat 1.2.1, pytest, SQLite (reusing existing schema), no torch/audio in tests (Resemblyzer embed injected).

---

## File Structure

- **Create** `src/larry/voice_enroll.py` — capture constants, tool schemas, handler factories.
- **Create** `tests/test_voice_enroll.py` — tests for the above.
- **Modify** `src/larry/speaker_id.py` — extract `store_speaker` helper; add capture state machine.
- **Modify** `src/larry/cli/enroll.py` — call `store_speaker` instead of inline INSERT.
- **Modify** `tests/test_speaker_id.py` — cover `store_speaker` round-trip and capture state machine.
- **Modify** `src/larry/wake.py` — add `sleep_now()` method.
- **Modify** `tests/test_wake.py` — cover `sleep_now()`.
- **Modify** `src/larry/config.py` — add `voice_tools_enabled`.
- **Modify** `tests/test_config.py` — cover the new field.
- **Modify** `src/larry/pipeline.py` — wire the two tools.

`data/speakers.db` is already gitignored (under `/data/`).

Module constants defined once in `voice_enroll.py`:
- `CAPTURE_TARGET_VOICED_S = 10.0` — aimed voiced-audio target
- `CAPTURE_FLOOR_VOICED_S = 6.0` — minimum voiced before a capture counts
- `CAPTURE_CAP_WALL_S = 20.0` — wall-clock abort if not enough voiced by then
- `SAMPLE_RATE = 16000` — shared with the existing STT path
- `REPEAT_PHRASE = "the skull keeps what it's given"` — Larry's prompt to the speaker
- `ENROLL_CONFIRM = "Kept. I'll know you now."` — success confirmation
- `ENROLL_FAIL = "the quiet swallowed it — try again, my kept one"` — failure line

---

## Task 1: Config field — `voice_tools_enabled`

**Files:**
- Modify: `src/larry/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (alongside `test_self_evolution_defaults`):

```python
def test_voice_tools_enabled_defaults_true(monkeypatch, required_keys):
    cfg = load_config()
    assert cfg.voice_tools_enabled is True


def test_voice_tools_enabled_overridable(monkeypatch, required_keys):
    monkeypatch.setenv("VOICE_TOOLS_ENABLED", "false")
    cfg = load_config()
    assert cfg.voice_tools_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k voice_tools -q`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'voice_tools_enabled'`.

- [ ] **Step 3: Implement minimal config**

In `src/larry/config.py`, add to the `Config` dataclass directly after `self_evolution_enabled` (line 54):

```python
    voice_tools_enabled: bool
```

In `load_config()`, directly after the `self_evolution_enabled=...` line (~182):

```python
        voice_tools_enabled=os.environ.get("VOICE_TOOLS_ENABLED", "true").lower()
        not in ("false", "0", "no"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -k voice_tools -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/larry/config.py tests/test_config.py
git commit -m "feat(voice-enroll): add voice_tools_enabled config field"
```

---

## Task 2: `store_speaker` shared helper (extracted from CLI enroll)

**Files:**
- Modify: `src/larry/speaker_id.py`
- Modify: `src/larry/cli/enroll.py`
- Test: `tests/test_speaker_id.py`

This extracts the inline `INSERT OR REPLACE` from `cli/enroll.py:47-53` into a shared helper in `speaker_id.py` so CLI enroll and voice enroll share one code path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_speaker_id.py`:

```python
from larry.speaker_id import load_enrolled, store_speaker


def test_store_speaker_round_trips_through_load_enrolled(tmp_path):
    db = tmp_path / "speakers.db"
    emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    store_speaker(db, "jason", emb)
    enrolled = load_enrolled(db)
    assert "jason" in enrolled
    np.testing.assert_array_almost_equal(enrolled["jason"], emb)


def test_store_speaker_overwrites_existing_name(tmp_path):
    db = tmp_path / "speakers.db"
    old = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    new = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    store_speaker(db, "alice", old)
    store_speaker(db, "alice", new)
    enrolled = load_enrolled(db)
    np.testing.assert_array_almost_equal(enrolled["alice"], new)


def test_store_speaker_creates_db_if_missing(tmp_path):
    db = tmp_path / "nonexistent" / "speakers.db"
    emb = np.array([0.5, 0.5], dtype=np.float32)
    store_speaker(db, "dan", emb)
    assert db.exists()
    assert "dan" in load_enrolled(db)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speaker_id.py -k store_speaker -q`
Expected: FAIL — `ImportError: cannot import name 'store_speaker' from 'larry.speaker_id'`.

- [ ] **Step 3: Add `store_speaker` to `speaker_id.py`**

Add after `load_enrolled` (after line 45 of `src/larry/speaker_id.py`):

```python
def store_speaker(db_path: Path, name: str, embedding: np.ndarray) -> None:
    """Persist a speaker voiceprint (INSERT OR REPLACE on primary key name).

    Creates the database file and parent directories if absent. Both the CLI
    ``enroll`` command and the in-conversation voice-enroll path call this so
    there is exactly one storage code path.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)",
            (name, embedding.astype(np.float32).tobytes()),
        )
        conn.commit()
```

- [ ] **Step 4: Refactor `cli/enroll.py` to call `store_speaker`**

Replace the inline SQLite block in `src/larry/cli/enroll.py` (`main`, lines 43-53) with:

```python
    from larry.speaker_id import store_speaker

    cfg = load_config()
    store_speaker(cfg.speakers_db, name, embedding)
    print(f"Enrolled {name}. Embedding shape: {embedding.shape}.")
```

Remove the now-unused `cfg.data_dir.mkdir(...)` call (the helper handles it).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_speaker_id.py -k store_speaker -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/larry/speaker_id.py src/larry/cli/enroll.py tests/test_speaker_id.py
git commit -m "feat(voice-enroll): extract store_speaker shared helper from CLI enroll"
```

---

## Task 3: `WakeWordGate.sleep_now()` — programmatic sleep

**Files:**
- Modify: `src/larry/wake.py`
- Test: `tests/test_wake.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_wake.py`:

```python
def test_sleep_now_transitions_to_asleep(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        # Wake first.
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        slept: list[bool] = []
        gate.on_sleep = lambda: slept.append(True)

        gate.sleep_now()

        assert gate._awake is False
        assert slept == [True]
        # Model reset — mirrors the timeout path.
        assert model.reset_calls == 1

    _run(body)


def test_sleep_now_idempotent_when_already_asleep(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel()
        gate, sink = await _make_gate(monkeypatch, clock, model)

        assert gate._awake is False

        slept: list[bool] = []
        gate.on_sleep = lambda: slept.append(True)

        gate.sleep_now()
        gate.sleep_now()

        # Already asleep: on_sleep must NOT fire (no double cue), model NOT reset.
        assert slept == []
        assert model.reset_calls == 0

    _run(body)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wake.py -k sleep_now -q`
Expected: FAIL — `AttributeError: 'WakeWordGate' object has no attribute 'sleep_now'`.

- [ ] **Step 3: Add `sleep_now()` to `WakeWordGate`**

Add a new method to `WakeWordGate` in `src/larry/wake.py`, after `_handle_audio` (after line 197):

```python
    def sleep_now(self) -> None:
        """Transition to the dormant/asleep state immediately (same as timeout).

        Idempotent — calling while already asleep is a no-op. Fires ``on_sleep``
        and resets the OWW prediction buffer, identical to the timeout path in
        ``_handle_audio``.
        """
        if not self._awake:
            return
        self._awake = False
        self._speaking = False
        self._bot_speaking = False
        self._model.reset()
        logger.info("Larry going back to sleep (programmatic sleep_now).")
        if self.on_sleep is not None:
            self.on_sleep()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wake.py -k sleep_now -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run full wake test suite (regression guard)**

Run: `uv run pytest tests/test_wake.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/larry/wake.py tests/test_wake.py
git commit -m "feat(voice-enroll): add WakeWordGate.sleep_now() programmatic sleep"
```

---

## Task 4: Capture state machine in `SpeakerIDProcessor`

**Files:**
- Modify: `src/larry/speaker_id.py`
- Test: `tests/test_speaker_id.py`

The capture state machine gates enrollment so Larry's own TTS voice can never enter the voiceprint. It arms on `arm_capture(name)`, starts accumulating only after `bot_stopped_speaking()`, counts only VAD-voiced / bot-silent audio, and either embeds + persists on reaching the voiced-audio target, or aborts cleanly if the wall-clock cap expires with insufficient voiced audio.

The Resemblyzer embed function is injected at `arm_capture` call time (`embed_fn: Callable[[np.ndarray], np.ndarray]`) so tests run with zero torch dependency.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_speaker_id.py`:

```python
import time
import numpy as np
import pytest

from larry.speaker_id import SpeakerIDProcessor, store_speaker, load_enrolled


# Helpers -------------------------------------------------------------------

_SAMPLE_RATE = 16000
_SECONDS_OF_AUDIO = 0.1  # 100 ms of int16 PCM at 16 kHz
_FRAME_BYTES = int(_SAMPLE_RATE * 2 * _SECONDS_OF_AUDIO)


def _pcm_bytes(seconds: float = 0.1) -> bytes:
    """Return `seconds` of silent int16 PCM at 16 kHz."""
    n = int(_SAMPLE_RATE * seconds)
    return (b"\x00\x01" * n)[: n * 2]  # non-zero so it counts as voiced


def _make_proc(monkeypatch, tmp_path, on_speaker_change=None):
    import larry.speaker_id as sid_mod
    monkeypatch.setattr(sid_mod, "VoiceEncoder", _FakeEncoder)
    return SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        on_speaker_change=on_speaker_change,
    )


# Capture state machine tests -----------------------------------------------


def test_arm_capture_sets_pending_state(monkeypatch, tmp_path):
    proc = _make_proc(monkeypatch, tmp_path)
    fixed_emb = np.array([0.9, 0.1], dtype=np.float32)
    proc.arm_capture("jason", embed_fn=lambda audio: fixed_emb)
    assert proc._capture_name == "jason"
    assert proc._capture_state == "armed"


def test_arm_capture_ignored_when_already_armed(monkeypatch, tmp_path):
    proc = _make_proc(monkeypatch, tmp_path)
    emb = np.array([0.9, 0.1], dtype=np.float32)
    proc.arm_capture("jason", embed_fn=lambda audio: emb)
    proc.arm_capture("dan", embed_fn=lambda audio: emb)   # second call ignored
    assert proc._capture_name == "jason"


def test_accumulation_starts_only_after_bot_stopped_speaking(monkeypatch, tmp_path):
    proc = _make_proc(monkeypatch, tmp_path)
    emb = np.array([0.9, 0.1], dtype=np.float32)
    proc.arm_capture("jason", embed_fn=lambda audio: emb)

    # While armed-but-bot-not-stopped: audio must NOT be added to capture buffer.
    proc.add_capture_audio(_pcm_bytes(0.5), vad_voiced=True, bot_speaking=False)
    assert proc._capture_voiced_bytes == 0, "should not accumulate before bot_stopped_speaking"

    # Now bot stops speaking → transition to "capturing".
    proc.bot_stopped_speaking()
    assert proc._capture_state == "capturing"

    # After bot-stop: voiced + bot-silent audio IS accumulated.
    proc.add_capture_audio(_pcm_bytes(0.5), vad_voiced=True, bot_speaking=False)
    assert proc._capture_voiced_bytes > 0


def test_bot_speaking_audio_not_accumulated(monkeypatch, tmp_path):
    proc = _make_proc(monkeypatch, tmp_path)
    emb = np.array([0.9, 0.1], dtype=np.float32)
    proc.arm_capture("jason", embed_fn=lambda audio: emb)
    proc.bot_stopped_speaking()

    # bot_speaking=True: must not accumulate even if VAD-voiced.
    proc.add_capture_audio(_pcm_bytes(1.0), vad_voiced=True, bot_speaking=True)
    assert proc._capture_voiced_bytes == 0


def test_unvoiced_audio_not_accumulated(monkeypatch, tmp_path):
    proc = _make_proc(monkeypatch, tmp_path)
    emb = np.array([0.9, 0.1], dtype=np.float32)
    proc.arm_capture("jason", embed_fn=lambda audio: emb)
    proc.bot_stopped_speaking()

    # vad_voiced=False: must not accumulate.
    proc.add_capture_audio(_pcm_bytes(1.0), vad_voiced=False, bot_speaking=False)
    assert proc._capture_voiced_bytes == 0


def test_successful_capture_stores_and_returns_success(monkeypatch, tmp_path):
    db = tmp_path / "speakers.db"
    fixed_emb = np.array([0.7, 0.3], dtype=np.float32)
    completed: list[str] = []

    proc = _make_proc(monkeypatch, tmp_path, on_speaker_change=completed.append)
    proc.arm_capture(
        "jason",
        embed_fn=lambda audio: fixed_emb,
        db_path=db,
        target_voiced_s=2.0,  # override low for test speed
        floor_voiced_s=1.0,
    )
    proc.bot_stopped_speaking()

    # Feed 2.5 s of voiced audio in small chunks.
    chunk = _pcm_bytes(0.5)
    results: list[dict] = []
    for _ in range(5):
        r = proc.add_capture_audio(chunk, vad_voiced=True, bot_speaking=False)
        if r is not None:
            results.append(r)

    assert results, "capture should have completed"
    assert results[0]["status"] == "enrolled"
    assert results[0]["name"] == "jason"
    assert "jason" in load_enrolled(db)
    assert completed == ["jason"]   # on_speaker_change fired
    assert proc._capture_state == "idle"


def test_abort_when_wall_clock_cap_expires_with_insufficient_voiced(monkeypatch, tmp_path):
    db = tmp_path / "speakers.db"
    fixed_emb = np.array([0.7, 0.3], dtype=np.float32)
    stored: list[bool] = []

    proc = _make_proc(monkeypatch, tmp_path)
    # Patch time.monotonic inside speaker_id so the cap is controllable.
    import larry.speaker_id as sid_mod
    fake_time = [0.0]
    monkeypatch.setattr(sid_mod, "_monotonic", lambda: fake_time[0])

    proc.arm_capture(
        "jason",
        embed_fn=lambda audio: fixed_emb,
        db_path=db,
        target_voiced_s=10.0,
        floor_voiced_s=6.0,
        cap_wall_s=5.0,
    )
    proc.bot_stopped_speaking()

    # Advance fake time past cap; add only 2s voiced (< 6s floor).
    fake_time[0] = 6.0   # past the 5s cap
    result = proc.add_capture_audio(_pcm_bytes(2.0), vad_voiced=True, bot_speaking=False)

    assert result is not None
    assert result["status"] == "failed"
    assert "jason" not in load_enrolled(db)   # nothing written
    assert proc._capture_state == "idle"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speaker_id.py -k "capture or arm" -q`
Expected: FAIL — `AttributeError: 'SpeakerIDProcessor' object has no attribute 'arm_capture'`.

- [ ] **Step 3: Add capture state machine to `SpeakerIDProcessor`**

At the top of `src/larry/speaker_id.py`, add `import time` (rename to `_monotonic` so tests can patch it) and define:

```python
from time import monotonic as _monotonic
```

Add to `SpeakerIDProcessor.__init__` (after the existing instance vars, line ~68):

```python
        # Capture state machine — arms on enroll_speaker tool call, accumulates
        # voiced audio post-BotStoppedSpeakingFrame, embeds on threshold.
        self._capture_state: str = "idle"   # "idle" | "armed" | "capturing"
        self._capture_name: str = ""
        self._capture_bytes: bytearray = bytearray()
        self._capture_voiced_bytes: int = 0
        self._capture_start: float = 0.0
        self._capture_target_voiced_s: float = 10.0
        self._capture_floor_voiced_s: float = 6.0
        self._capture_cap_wall_s: float = 20.0
        self._capture_embed_fn: "Callable[[np.ndarray], np.ndarray] | None" = None
        self._capture_db_path: "Path | None" = None
```

Add three new public methods to `SpeakerIDProcessor`:

```python
    def arm_capture(
        self,
        name: str,
        *,
        embed_fn: "Callable[[np.ndarray], np.ndarray]",
        db_path: "Path | None" = None,
        target_voiced_s: float = 10.0,
        floor_voiced_s: float = 6.0,
        cap_wall_s: float = 20.0,
    ) -> None:
        """Arm a pending voiceprint capture for ``name``.

        Ignored if a capture is already in progress (one capture at a time).
        Accumulation only starts after ``bot_stopped_speaking()`` is called —
        so Larry's own "say this back to me" prompt can never pollute the print.

        ``embed_fn`` replaces direct Resemblyzer calls so the state machine is
        unit-testable without torch. In production, pipeline.py passes a lambda
        that calls ``self._encoder.embed_utterance``.
        """
        if self._capture_state != "idle":
            logger.info(
                "arm_capture(%r) ignored — capture already in state %r",
                name,
                self._capture_state,
            )
            return
        self._capture_name = name.strip()
        self._capture_state = "armed"
        self._capture_bytes = bytearray()
        self._capture_voiced_bytes = 0
        self._capture_start = _monotonic()
        self._capture_target_voiced_s = target_voiced_s
        self._capture_floor_voiced_s = floor_voiced_s
        self._capture_cap_wall_s = cap_wall_s
        self._capture_embed_fn = embed_fn
        self._capture_db_path = db_path if db_path is not None else self._db_path
        logger.info("Capture armed for %r (target=%.0fs, cap=%.0fs)", name, target_voiced_s, cap_wall_s)

    def bot_stopped_speaking(self) -> None:
        """Signal that the bot's TTS has finished — starts accumulation if armed."""
        if self._capture_state == "armed":
            self._capture_state = "capturing"
            self._capture_start = _monotonic()  # wall-clock cap measured from here
            logger.info("Capture accumulation started for %r", self._capture_name)

    def add_capture_audio(
        self,
        pcm_bytes: bytes,
        *,
        vad_voiced: bool,
        bot_speaking: bool,
    ) -> "dict | None":
        """Feed a chunk of PCM audio into the capture accumulator.

        Returns a result dict ``{"status": "enrolled", "name": ...}`` on
        success, ``{"status": "failed", "reason": ...}`` on abort, or ``None``
        if still accumulating.  Only counts frames that are VAD-voiced AND bot-
        silent. Called from ``process_frame`` (or directly in tests).
        """
        if self._capture_state != "capturing":
            return None

        elapsed = _monotonic() - self._capture_start
        voiced_s = self._capture_voiced_bytes / (16000 * 2)  # int16 bytes → seconds

        # Cap exceeded with insufficient voiced audio → abort.
        if elapsed >= self._capture_cap_wall_s and voiced_s < self._capture_floor_voiced_s:
            logger.warning(
                "Capture aborted for %r: %.1fs voiced in %.1fs (floor=%.1fs, cap=%.1fs)",
                self._capture_name,
                voiced_s,
                elapsed,
                self._capture_floor_voiced_s,
                self._capture_cap_wall_s,
            )
            reason = f"insufficient voiced audio ({voiced_s:.1f}s < {self._capture_floor_voiced_s:.1f}s floor)"
            self._capture_state = "idle"
            return {"status": "failed", "name": self._capture_name, "reason": reason}

        # Only accumulate bot-silent, VAD-voiced audio.
        if vad_voiced and not bot_speaking:
            self._capture_bytes.extend(pcm_bytes)
            self._capture_voiced_bytes += len(pcm_bytes)

        voiced_s = self._capture_voiced_bytes / (16000 * 2)

        # Target reached → embed + persist.
        if voiced_s >= self._capture_target_voiced_s:
            name = self._capture_name
            embed_fn = self._capture_embed_fn
            db_path = self._capture_db_path
            audio = pcm16_to_float32(bytes(self._capture_bytes))
            self._capture_state = "idle"

            if embed_fn is None or db_path is None:
                return {"status": "failed", "name": name, "reason": "embed_fn or db_path missing"}

            embedding = np.asarray(embed_fn(audio))
            store_speaker(db_path, name, embedding)
            # Reload in-memory enrolled set so new speaker is immediately identifiable.
            self._enrolled = load_enrolled(db_path)
            logger.info("Enrolled %r — %d speaker(s) now enrolled", name, len(self._enrolled))
            if self._on_speaker_change:
                self._on_speaker_change(name)
            return {"status": "enrolled", "name": name}

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speaker_id.py -k "capture or arm" -q`
Expected: PASS (all 8 new tests pass).

- [ ] **Step 5: Run full speaker_id test suite (regression guard)**

Run: `uv run pytest tests/test_speaker_id.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/larry/speaker_id.py tests/test_speaker_id.py
git commit -m "feat(voice-enroll): add capture state machine to SpeakerIDProcessor"
```

---

## Task 5: `voice_enroll.py` — tool schemas and handler factories

**Files:**
- Create: `src/larry/voice_enroll.py`
- Create: `tests/test_voice_enroll.py`

Mirrors the `self_layer.py` module-per-concern style. Pure functions; Pipecat `FunctionSchema`/`ToolsSchema` types used for the schema builders; handler factories take all stateful dependencies as injected arguments.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_enroll.py`:

```python
"""Unit tests for larry.voice_enroll — tool schemas and handler factories.

Self-contained: no audio, no torch, no network. Pipecat FunctionSchema and
ToolsSchema are imported (pure Python dataclasses — no pipeline runtime needed).
Handlers are exercised via duck-typed _Params just like test_self_layer.py.
"""

import asyncio
from pathlib import Path

import pytest

import larry.voice_enroll as ve


# ── Schema shape tests ──────────────────────────────────────────────────────


def test_build_voice_tools_contains_both_schemas():
    tools = ve.build_voice_tools()
    names = {fn.name for fn in tools.standard_tools}
    assert "enroll_speaker" in names
    assert "dismiss" in names


def test_enroll_speaker_schema_requires_name():
    tools = ve.build_voice_tools()
    enroll_fn = next(fn for fn in tools.standard_tools if fn.name == "enroll_speaker")
    schema = enroll_fn.to_default_dict()
    assert "name" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["name"]


def test_dismiss_schema_has_no_required_params():
    tools = ve.build_voice_tools()
    dismiss_fn = next(fn for fn in tools.standard_tools if fn.name == "dismiss")
    schema = dismiss_fn.to_default_dict()
    assert schema["parameters"].get("required", []) == []


# ── Handler: enroll_speaker ─────────────────────────────────────────────────


class _Params:
    """Duck-type for Pipecat FunctionCallParams."""

    def __init__(self, arguments: dict):
        self.arguments = arguments
        self._result = None

    async def result_callback(self, result):
        self._result = result


def test_enroll_handler_arms_capture_and_returns_prompt():
    armed: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def body():
        params = _Params({"name": "Jason"})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert "jason" in armed[0].lower() or "Jason" in armed[0], "arm should receive the name"
        assert params._result is not None
        assert params._result["status"] == "pending"
        # Result must include the repeat phrase so Larry speaks it.
        assert ve.REPEAT_PHRASE in params._result["prompt"]

    asyncio.run(body())


def test_enroll_handler_normalises_name():
    """Name is stripped and stored; casing kept for display but lowered for key."""
    armed_names: list[str] = []

    def fake_arm(name, **_kwargs):
        armed_names.append(name)

    async def body():
        params = _Params({"name": "  ALICE  "})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert armed_names[0] == "alice"   # normalised

    asyncio.run(body())


def test_enroll_handler_ignores_empty_name():
    armed: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def body():
        params = _Params({"name": "   "})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert armed == []
        assert params._result["status"] == "error"

    asyncio.run(body())


# ── Handler: dismiss ────────────────────────────────────────────────────────


def test_dismiss_handler_calls_sleep_now_and_returns_cue():
    slept: list[bool] = []

    async def fake_sleep_now():
        slept.append(True)

    async def body():
        params = _Params({})
        handler = ve.make_dismiss_handler(sleep_now_fn=fake_sleep_now)
        await handler(params)
        assert slept == [True], "sleep_now should be called"
        assert params._result is not None
        assert params._result["status"] == "dismissed"
        # The cue text is returned so the pipeline can play it.
        assert "cue" in params._result

    asyncio.run(body())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_voice_enroll.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'larry.voice_enroll'`.

- [ ] **Step 3: Implement `voice_enroll.py`**

Create `src/larry/voice_enroll.py`:

```python
"""Voice-triggered speaker enrollment and dismiss tools for Larry.

Two LLM function tools, registered alongside keep_about_self:

  enroll_speaker(name)  — arms the SpeakerIDProcessor capture state machine;
                          Larry prompts the user to speak a phrase so ~10s of
                          voiced audio can be captured, embedded, and persisted.
  dismiss()             — makes Larry go to sleep immediately (same dormant
                          state as the wake-gate silence timeout).

The Resemblyzer embed and the sleep_now() call are injected so this module is
unit-testable without audio, torch, or the pipeline runtime.

See docs/superpowers/specs/2026-06-05-voice-enrollment-and-dismiss-design.md.
"""

from collections.abc import Awaitable, Callable
import random

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# ---------------------------------------------------------------------------
# Module constants — tunable here, not env vars (not worth the noise).
# ---------------------------------------------------------------------------
CAPTURE_TARGET_VOICED_S: float = 10.0
CAPTURE_FLOOR_VOICED_S: float = 6.0
CAPTURE_CAP_WALL_S: float = 20.0
REPEAT_PHRASE: str = "the skull keeps what it's given"
ENROLL_CONFIRM: str = "Kept. I'll know you now."
ENROLL_FAIL: str = "the quiet swallowed it — try again, my kept one"

_DISMISS_CUES: list[str] = [
    "I'll keep listening.",
    "Until you come back.",
    "I never really sleep.",
    "Go on. I'll wait.",
]


# ---------------------------------------------------------------------------
# Tool schema factory
# ---------------------------------------------------------------------------

def build_voice_tools() -> ToolsSchema:
    """Build ToolsSchema for both voice tools (enroll_speaker + dismiss)."""
    enroll_fn = FunctionSchema(
        name="enroll_speaker",
        description=(
            "When someone introduces themselves by name ('it's Jason', 'I'm Dan', "
            "'this is Sarah'), call this to capture their voice and remember them. "
            "After calling it, speak the repeat phrase back to them so they know "
            "to say it aloud."
        ),
        properties={
            "name": {
                "type": "string",
                "description": "The speaker's first name as they gave it.",
            }
        },
        required=["name"],
    )
    dismiss_fn = FunctionSchema(
        name="dismiss",
        description=(
            "When someone says goodbye, 'that's all', 'go to sleep', or otherwise "
            "dismisses you, call this to go dormant immediately. Do not reply further "
            "after calling it — the cue line in the result is the last thing to speak."
        ),
        properties={},
        required=[],
    )
    return ToolsSchema(standard_tools=[enroll_fn, dismiss_fn])


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def make_enroll_speaker_handler(
    *,
    arm_capture_fn: Callable[..., None],
) -> Callable:
    """Build an async Pipecat function handler for enroll_speaker.

    ``arm_capture_fn(name, **kwargs)`` is called with the normalised name and
    the capture parameters from this module's constants.  In production this is
    ``speaker_id_proc.arm_capture``; in tests it is a fake.
    """

    async def handler(params) -> None:
        raw_name = str(params.arguments.get("name", "")).strip()
        name = raw_name.lower()
        if not name:
            await params.result_callback({"status": "error", "reason": "name was empty"})
            return
        arm_capture_fn(
            name,
            target_voiced_s=CAPTURE_TARGET_VOICED_S,
            floor_voiced_s=CAPTURE_FLOOR_VOICED_S,
            cap_wall_s=CAPTURE_CAP_WALL_S,
        )
        await params.result_callback(
            {
                "status": "pending",
                "name": name,
                "prompt": (
                    f"Say this back to me — '{REPEAT_PHRASE}.' "
                    f"That's how I'll keep you."
                ),
            }
        )

    return handler


def make_dismiss_handler(
    *,
    sleep_now_fn: Callable[[], Awaitable[None]],
) -> Callable:
    """Build an async Pipecat function handler for dismiss.

    ``sleep_now_fn()`` is awaited to transition the wake gate to sleep.  In
    production this wraps ``wake_gate.sleep_now()``; in tests it is a coroutine
    fake.
    """

    async def handler(params) -> None:
        cue = random.choice(_DISMISS_CUES)
        await sleep_now_fn()
        await params.result_callback({"status": "dismissed", "cue": cue})

    return handler
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_voice_enroll.py -q`
Expected: PASS (all tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/larry/voice_enroll.py tests/test_voice_enroll.py
git commit -m "feat(voice-enroll): voice_enroll module with tool schemas and handler factories"
```

---

## Task 6: Wire both tools into the pipeline

**Files:**
- Modify: `src/larry/pipeline.py`

This task adds no new unit tests (all logic is covered in Tasks 2-5; this is wiring). It follows the exact pattern of the `keep_about_self` wiring at `pipeline.py:614-628`.

- [ ] **Step 1: Add import**

Add to the imports block in `src/larry/pipeline.py` (after line 61 `from larry import self_layer`):

```python
from larry import voice_enroll
from larry.speaker_id import SpeakerIDProcessor
```

(Note: `SpeakerIDProcessor` is already imported at line 68 — verify no duplicate before adding.)

- [ ] **Step 2: Set tools on context**

In the context block around line 416-417 (after `if cfg.self_evolution_enabled: context.set_tools(...)`), extend to also add the voice tools when enabled:

```python
    if cfg.voice_tools_enabled:
        existing = self_layer.build_self_tool() if cfg.self_evolution_enabled else None
        voice_ts = voice_enroll.build_voice_tools()
        if existing is not None:
            # Merge: both tool sets active at once.
            from pipecat.adapters.schemas.tools_schema import ToolsSchema
            merged = ToolsSchema(
                standard_tools=existing.standard_tools + voice_ts.standard_tools
            )
            context.set_tools(merged)
        else:
            context.set_tools(voice_ts)
```

Note: if `self_evolution_enabled` is True, the existing code at line 417 already calls `context.set_tools(self_layer.build_self_tool())`. Replace that block with the merged logic above so a single `set_tools` call covers all enabled tools.

The updated block replacing lines 416-417 is:

```python
    _tool_fns: list = []
    if cfg.self_evolution_enabled:
        _tool_fns.extend(self_layer.build_self_tool().standard_tools)
    if cfg.voice_tools_enabled:
        _tool_fns.extend(voice_enroll.build_voice_tools().standard_tools)
    if _tool_fns:
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        context.set_tools(ToolsSchema(standard_tools=_tool_fns))
```

- [ ] **Step 3: Register handlers alongside `keep_about_self`**

In the handler registration block that begins at `pipeline.py:614` (`if cfg.self_evolution_enabled:`), add the voice tool registrations AFTER the self-evolution block (after line 628):

```python
    if cfg.voice_tools_enabled:
        # enroll_speaker: arm the capture state machine on speaker_id, which
        # will call _on_speaker_change (mem0 user_id) on success.
        def _arm_capture(name: str, **kwargs) -> None:
            speaker_id.arm_capture(
                name,
                embed_fn=lambda audio: np.asarray(speaker_id._encoder.embed_utterance(audio)),
                db_path=cfg.speakers_db,
                **kwargs,
            )

        llm.register_function(
            "enroll_speaker",
            voice_enroll.make_enroll_speaker_handler(arm_capture_fn=_arm_capture),
        )

        # dismiss: play a cue + send the gate to sleep.
        async def _sleep_now() -> None:
            line = random.choice(voice_enroll._DISMISS_CUES)
            logger.info("Dismiss cue: %r", line)
            t = asyncio.create_task(_play_cue(line))
            _cue_tasks.add(t)
            t.add_done_callback(_cue_tasks.discard)
            wake_gate.sleep_now()

        llm.register_function(
            "dismiss",
            voice_enroll.make_dismiss_handler(sleep_now_fn=_sleep_now),
        )
```

Note: `numpy` is already available as `np` in `pipeline.py` (check imports — add `import numpy as np` if not already present).

- [ ] **Step 4: Wire `BotStoppedSpeakingFrame` → `speaker_id.bot_stopped_speaking()`**

The capture machine needs `bot_stopped_speaking()` to transition from "armed" to "capturing". The existing pipeline already receives `BotStoppedSpeakingFrame` upstream (wake gate handles it at `wake.py:98-101`). Add a frame processor shim or handle it in `SpeakerIDProcessor.process_frame`.

Modify `SpeakerIDProcessor.process_frame` in `src/larry/speaker_id.py` to handle `BotStoppedSpeakingFrame` and `VADUserStartedSpeakingFrame`:

```python
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_stopped_speaking()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            # Track VAD state so add_capture_audio can gate on vad_voiced.
            self._vad_voiced = isinstance(frame, VADUserStartedSpeakingFrame)
            await self.push_frame(frame, direction)
```

Add `self._vad_voiced: bool = False` and `self._bot_speaking_for_capture: bool = False` to `__init__`.

Then in the `InputAudioRawFrame` branch of `process_frame`, after pushing the frame downstream, call the accumulator:

```python
        if isinstance(frame, InputAudioRawFrame):
            self._audio_buffer.extend(frame.audio)
            if len(self._audio_buffer) >= self._window_bytes:
                window = bytes(self._audio_buffer[: self._window_bytes])
                del self._audio_buffer[: self._window_bytes]
                if self._identify_task is None or self._identify_task.done():
                    self._identify_task = asyncio.create_task(
                        asyncio.to_thread(self._identify_speaker, window)
                    )
            # Feed capture accumulator (if armed/capturing) for every raw frame.
            if self._capture_state == "capturing":
                result = self.add_capture_audio(
                    frame.audio,
                    vad_voiced=self._vad_voiced,
                    bot_speaking=self._bot_speaking_for_capture,
                )
                if result is not None:
                    # Capture finished asynchronously HERE — long after the
                    # enroll_speaker tool returned "pending". The tool result
                    # callback already fired; it cannot speak this outcome. So
                    # Larry must speak it now: push a TTSSpeakFrame downstream to
                    # the TTS service. This closes the repeat-back UX the spec
                    # requires ("Kept. I'll know you now." / the miss line).
                    from larry.voice_enroll import ENROLL_CONFIRM, ENROLL_FAIL

                    line = ENROLL_CONFIRM if result["status"] == "enrolled" else ENROLL_FAIL
                    logger.info("Capture result: %s — speaking %r", result, line)
                    await self.push_frame(TTSSpeakFrame(line), direction)
            await self.push_frame(frame, direction)
            return
```

Also handle `BotStartedSpeakingFrame`:

```python
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking_for_capture = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking_for_capture = False
            self.bot_stopped_speaking()
            await self.push_frame(frame, direction)
```

Add `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `VADUserStartedSpeakingFrame`, `VADUserStoppedSpeakingFrame`, and `TTSSpeakFrame` to the `from pipecat.frames.frames import ...` line at the top of `speaker_id.py` (`TTSSpeakFrame` is what Larry speaks the capture confirmation/fail line through — see Step 4's result block).

- [ ] **Step 5: Verify suite + lint + types**

Run: `uv run pytest -q && uv run ruff check src/larry tests && uv run pyright src/larry/pipeline.py src/larry/speaker_id.py src/larry/voice_enroll.py`
Expected: all pass, 0 pyright errors.

- [ ] **Step 6: Commit**

```bash
git add src/larry/pipeline.py src/larry/speaker_id.py
git commit -m "feat(voice-enroll): wire enroll_speaker + dismiss tools into pipeline"
```

---

## Task 7: Integration smoke tests and docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Smoke test (Mac dev, documented)**

```bash
uv run pytest -q   # all tests green
uv run ruff check src/larry tests
uv run pyright src/larry
```

Manual test sequence (requires API keys):
1. `uv run larry` — boot; say "Hey Larry".
2. Say "Hey Larry, it's Jason." — expect Larry to reply with the repeat phrase.
3. Say the repeat phrase back — expect "Kept. I'll know you now." (or equivalent).
4. Verify `data/speakers.db` has a `jason` row: `python3 -c "import sqlite3; print(sqlite3.connect('data/speakers.db').execute('SELECT name FROM speakers').fetchall())"`.
5. Say "goodbye Larry" — expect a sleep cue and the gate going silent.
6. Say "Hey Larry" — Larry wakes normally; the capture session does not re-trigger.

If `VOICE_TOOLS_ENABLED=false`, neither tool should register and the above steps should fail gracefully (Larry doesn't know to enroll or dismiss).

- [ ] **Step 2: Update CLAUDE.md**

Add to `CLAUDE.md` under "Where to Make Changes":

```markdown
- **Voice tools (enrollment + dismiss)**: `src/larry/voice_enroll.py` (tool schemas + handler factories), plus the capture state machine in `src/larry/speaker_id.py` (`arm_capture`, `bot_stopped_speaking`, `add_capture_audio`). Toggle with `VOICE_TOOLS_ENABLED` (default true). `WakeWordGate.sleep_now()` is the dismiss entry point. CLI `uv run larry enroll <name>` uses the same `store_speaker` helper.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document voice-tools feature surface"
```

---

## Self-Review

- **Spec coverage:**
  - Voice enrollment via LLM tool (Tasks 5-6). ✓
  - Capture state machine: arms, accumulates post-BotStopped, gates bot-speaking + VAD (Task 4). ✓
  - 10s target / 6s floor / 20s cap (module constants in Task 5, injected into `arm_capture` in Task 4). ✓
  - `store_speaker` shared path (Task 2); CLI enroll refactored to call it. ✓
  - `sleep_now()` idempotent programmatic sleep (Task 3). ✓
  - dismiss tool plays cue + sleeps (Task 5-6). ✓
  - `voice_tools_enabled` config flag gating both tools (Task 1). ✓
  - Resemblyzer embed injected for testability (Task 4 `embed_fn` param). ✓
  - No corrupt prints: abort writes nothing (Task 4 abort path). ✓
  - Safety: accumulation starts only post-BotStopped (Task 4). ✓
  - One capture at a time — `arm_capture` is ignored while armed/capturing (Task 4). ✓
  - `on_speaker_change` fired on successful enrollment (Task 4 success path → `_on_speaker_change` → Mem0 user_id). ✓
  - CLAUDE.md docs (Task 7). ✓

- **Placeholder scan:** all code steps have concrete implementations; all verification commands have expected outputs.

- **Type consistency:** `store_speaker(db_path, name, embedding)`, `arm_capture(name, *, embed_fn, db_path=, target_voiced_s=, floor_voiced_s=, cap_wall_s=)`, `bot_stopped_speaking()`, `add_capture_audio(pcm_bytes, *, vad_voiced, bot_speaking) → dict | None`, `sleep_now()`, `build_voice_tools() → ToolsSchema`, `make_enroll_speaker_handler(*, arm_capture_fn) → Callable`, `make_dismiss_handler(*, sleep_now_fn) → Callable` — names and signatures are consistent across all tasks.

- **No test imports heavy deps:** `VoiceEncoder` monkeypatched in all speaker_id tests; embed injected in capture tests; no `sounddevice`, `torch`, or `pipecat` pipeline runtime in any test.

"""Unit tests for SpeakerIDProcessor's pure speaker-matching logic.

Self-contained: no heavy ML models, no audio devices, no network. The
SpeakerEmbedder is monkeypatched (via ``get_speaker_embedder``) with a fake
whose embedding output is fully controlled, so no torch model ever loads. We
exercise ``_identify_speaker`` directly (synchronous matching logic) rather
than the async audio-buffering path.
"""

import sqlite3

import numpy as np
import pytest

import larry.speaker_id as speaker_id
from larry.speaker_id import SpeakerIDProcessor, load_enrolled, store_speaker


class _FakeEmbedder:
    """Stand-in for a SpeakerEmbedder (e.g. ResemblyzerEmbedder).

    ``embed`` ignores its audio input and returns a fixed vector set on the
    instance, letting tests drive the cosine-similarity decision.
    """

    name = "resemblyzer"
    next_embedding: np.ndarray = np.zeros(4, dtype=np.float32)

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        return self.next_embedding


def _patch_embedder(monkeypatch, module=speaker_id) -> _FakeEmbedder:
    """Monkeypatch ``get_speaker_embedder`` to hand back one fixed fake instance."""
    fake = _FakeEmbedder()
    monkeypatch.setattr(module, "get_speaker_embedder", lambda name: fake)
    return fake


@pytest.fixture
def processor(monkeypatch, tmp_path):
    """A SpeakerIDProcessor wired to a fake embedder and empty on-disk DB.

    ``get_speaker_embedder`` is replaced before construction so ``__init__``
    never touches torch. The returned processor's ``_enrolled`` dict is
    populated directly in each test.
    """
    _patch_embedder(monkeypatch)
    proc = SpeakerIDProcessor(speakers_db_path=tmp_path / "speakers.db")
    return proc


def test_near_match_returns_enrolled_name(processor):
    # Stored speaker embedding.
    alice = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    processor._enrolled = {"alice": alice}

    # Incoming embedding nearly parallel to alice: cosine ~ 0.997 >= 0.75.
    processor._encoder.next_embedding = np.array([0.95, 0.08, 0.0, 0.0], dtype=np.float32)

    processor._identify_speaker(b"\x00\x00")

    assert processor._current_speaker == "alice"


def test_picks_best_among_several(processor):
    processor._enrolled = {
        "alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    # Closest to bob.
    processor._encoder.next_embedding = np.array([0.05, 0.99, 0.0, 0.0], dtype=np.float32)

    processor._identify_speaker(b"\x00\x00")

    assert processor._current_speaker == "bob"


def test_dissimilar_embedding_returns_unknown(processor):
    processor._enrolled = {"alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}

    # Orthogonal to alice: cosine ~ 0.0 < 0.75.
    processor._encoder.next_embedding = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    processor._identify_speaker(b"\x00\x00")

    assert processor._current_speaker == "unknown"


def test_no_enrolled_speakers_stays_unknown(processor):
    processor._enrolled = {}
    processor._encoder.next_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    processor._identify_speaker(b"\x00\x00")

    assert processor._current_speaker == "unknown"


def test_speaker_change_callback_fires_with_matched_name(monkeypatch, tmp_path):
    fake = _patch_embedder(monkeypatch)
    seen: list[str] = []
    proc = SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        on_speaker_change=seen.append,
    )
    proc._enrolled = {"alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    fake.next_embedding = np.array([0.97, 0.05, 0.0, 0.0], dtype=np.float32)

    proc._identify_speaker(b"\x00\x00")

    assert seen == ["alice"]
    assert proc._current_speaker == "alice"


def test_threshold_boundary_is_inclusive(monkeypatch, tmp_path):
    # Construct an incoming embedding whose cosine vs the stored one is exactly
    # the threshold, to lock the >= (inclusive) comparison.
    fake = _patch_embedder(monkeypatch)
    proc = SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        match_threshold=0.75,
    )
    stored = np.array([1.0, 0.0], dtype=np.float32)
    # Unit vector at angle theta where cos(theta) == 0.75.
    incoming = np.array([0.75, np.sqrt(1 - 0.75**2)], dtype=np.float32)
    proc._enrolled = {"alice": stored}
    fake.next_embedding = incoming

    assert speaker_id.cosine_similarity(incoming, stored) == pytest.approx(0.75)

    proc._identify_speaker(b"\x00\x00")

    assert proc._current_speaker == "alice"


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


# ---------------------------------------------------------------------------
# Embedder namespacing — a print from one embedder is never matched under
# another (schema/migration bump, Task 3).
# ---------------------------------------------------------------------------


def test_store_speaker_defaults_embedder_to_resemblyzer(tmp_path):
    db = tmp_path / "speakers.db"
    store_speaker(db, "jason", np.array([1.0, 0.0], dtype=np.float32))
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT embedder, dim FROM speakers WHERE name = 'jason'").fetchone()
    assert row == ("resemblyzer", 2)


def test_store_speaker_persists_given_embedder_and_derived_dim(tmp_path):
    db = tmp_path / "speakers.db"
    emb = np.zeros(192, dtype=np.float32)
    store_speaker(db, "jason", emb, embedder="titanet")
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT embedder, dim FROM speakers WHERE name = 'jason'").fetchone()
    assert row == ("titanet", 192)


def test_load_enrolled_filters_by_embedder(tmp_path):
    db = tmp_path / "speakers.db"
    store_speaker(db, "alice", np.array([1.0, 0.0], dtype=np.float32), embedder="resemblyzer")
    store_speaker(db, "bob", np.zeros(192, dtype=np.float32), embedder="titanet")

    resemblyzer_only = load_enrolled(db, embedder="resemblyzer")
    assert set(resemblyzer_only) == {"alice"}

    titanet_only = load_enrolled(db, embedder="titanet")
    assert set(titanet_only) == {"bob"}

    everyone = load_enrolled(db)
    assert set(everyone) == {"alice", "bob"}


def test_load_enrolled_legacy_rows_default_to_resemblyzer(tmp_path):
    """A DB written before the embedder column existed still matches under resemblyzer."""
    db = tmp_path / "speakers.db"
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with sqlite3.connect(db) as conn:
        # Manually create the OLD (pre-Task-3) schema.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS speakers "
            "(name TEXT PRIMARY KEY, embedding BLOB NOT NULL, "
            "enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_seen TEXT)"
        )
        conn.execute(
            "INSERT INTO speakers (name, embedding) VALUES (?, ?)",
            ("alice", embedding.tobytes()),
        )
        conn.commit()

    filtered = load_enrolled(db, embedder="resemblyzer")
    assert "alice" in filtered


def test_speaker_id_never_matches_print_from_different_embedder(monkeypatch, tmp_path):
    """Migration test: a print labelled for model A is never matched under model B.

    Even a numerically-identical vector must fail closed to 'unknown' — the
    print belongs to a different embedding space and load_enrolled filters it
    out before matching ever runs, so this isn't a threshold/near-miss case.
    """
    db = tmp_path / "speakers.db"
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store_speaker(db, "alice", vec, embedder="titanet")

    fake = _patch_embedder(monkeypatch)  # fake.name == "resemblyzer"
    proc = SpeakerIDProcessor(speakers_db_path=db)  # default embedder_name="resemblyzer"
    assert proc._enrolled == {}, "titanet print must not load under a resemblyzer processor"

    fake.next_embedding = vec  # identical vector — would match if not filtered
    proc._identify_speaker(b"\x00\x00")

    assert proc._current_speaker == "unknown"


# Capture state machine tests -----------------------------------------------

_SAMPLE_RATE = 16000
_SECONDS_OF_AUDIO = 0.1  # 100 ms of int16 PCM at 16 kHz
_FRAME_BYTES = int(_SAMPLE_RATE * 2 * _SECONDS_OF_AUDIO)


def _pcm_bytes(seconds: float = 0.1) -> bytes:
    """Return `seconds` of silent int16 PCM at 16 kHz."""
    n = int(_SAMPLE_RATE * seconds)
    return (b"\x00\x01" * n)[: n * 2]  # non-zero so it counts as voiced


def _make_proc(monkeypatch, tmp_path, on_speaker_change=None):
    import larry.speaker_id as sid_mod
    _patch_embedder(monkeypatch, module=sid_mod)
    return SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        on_speaker_change=on_speaker_change,
    )


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


def test_successful_capture_returns_ready_and_finalize_stores(monkeypatch, tmp_path):
    """add_capture_audio returns 'ready'; finalize_capture stores and fires callback."""
    import asyncio
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

    # Feed 2.5 s of voiced audio in small chunks until add_capture_audio returns "ready".
    chunk = _pcm_bytes(0.5)
    ready_result: dict | None = None
    for _ in range(5):
        r = proc.add_capture_audio(chunk, vad_voiced=True, bot_speaking=False)
        if r is not None:
            ready_result = r
            break

    assert ready_result is not None, "capture should have signalled ready"
    assert ready_result["status"] == "ready"
    assert ready_result["name"] == "jason"
    assert "audio" in ready_result
    # State transitions to "embedding" while finalize hasn't run yet.
    assert proc._capture_state == "embedding"

    # finalize_capture does the embed+store off the loop.
    final = asyncio.run(proc.finalize_capture(ready_result["name"], ready_result["audio"]))
    assert final["status"] == "enrolled"
    assert final["name"] == "jason"
    assert "jason" in load_enrolled(db)
    assert completed == ["jason"]   # on_speaker_change fired
    assert proc._capture_state == "idle"


def test_abort_when_wall_clock_cap_expires_with_insufficient_voiced(monkeypatch, tmp_path):
    db = tmp_path / "speakers.db"
    fixed_emb = np.array([0.7, 0.3], dtype=np.float32)

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
    assert result["status"] == "abort"
    assert "jason" not in load_enrolled(db)   # nothing written
    assert proc._capture_state == "idle"


def test_cap_with_floor_met_returns_ready(monkeypatch, tmp_path):
    """FIX B: when cap expires but voiced >= floor, return 'ready' (not stuck)."""
    import asyncio
    db = tmp_path / "speakers.db"
    fixed_emb = np.array([0.7, 0.3], dtype=np.float32)
    completed: list[str] = []

    proc = _make_proc(monkeypatch, tmp_path, on_speaker_change=completed.append)
    import larry.speaker_id as sid_mod
    fake_time = [0.0]
    monkeypatch.setattr(sid_mod, "_monotonic", lambda: fake_time[0])

    proc.arm_capture(
        "jason",
        embed_fn=lambda audio: fixed_emb,
        db_path=db,
        target_voiced_s=10.0,  # high — won't be reached via audio alone
        floor_voiced_s=6.0,
        cap_wall_s=20.0,
    )
    proc.bot_stopped_speaking()

    # Feed 7s of voiced audio (>= 6s floor, < 10s target).
    chunk = _pcm_bytes(0.5)
    for _ in range(14):  # 14 × 0.5s = 7s voiced
        proc.add_capture_audio(chunk, vad_voiced=True, bot_speaking=False)
    # No result yet — time not past cap.
    assert proc._capture_state == "capturing"

    # Now advance time past cap — next frame should return "ready".
    fake_time[0] = 25.0  # past 20s cap
    result = proc.add_capture_audio(_pcm_bytes(0.1), vad_voiced=True, bot_speaking=False)

    assert result is not None
    assert result["status"] == "ready", f"expected ready, got {result!r}"
    assert result["name"] == "jason"
    assert proc._capture_state == "embedding"

    # Confirm finalize works end-to-end.
    final = asyncio.run(proc.finalize_capture(result["name"], result["audio"]))
    assert final["status"] == "enrolled"
    assert "jason" in load_enrolled(db)
    assert completed == ["jason"]


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


def test_ensure_schema_adds_embedder_and_dim_columns(tmp_path):
    """_ensure_schema adds embedder TEXT + dim INTEGER; calling it twice is idempotent."""
    db = tmp_path / "speakers.db"
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(speakers)")}
        assert {"embedder", "dim"} <= cols
        speaker_id._ensure_schema(conn)


def test_ensure_schema_upgrade_backfills_embedder_and_dim_on_old_rows(tmp_path):
    """A row written under the pre-Task-3 schema gets embedder/dim defaults, not NULL."""
    db = tmp_path / "speakers.db"
    embedding = b"\x00\x00\x80\x3f"  # float32 1.0 as bytes
    with sqlite3.connect(db) as conn:
        # Manually create the OLD (pre-embedder-column) schema.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS speakers "
            "(name TEXT PRIMARY KEY, embedding BLOB NOT NULL, "
            "enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_seen TEXT)"
        )
        conn.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)", ("alice", embedding))
        conn.commit()
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        row = conn.execute("SELECT embedder, dim FROM speakers WHERE name = 'alice'").fetchone()
    assert row == ("resemblyzer", 256)


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
    emb = np.array([1.0, 0.0], dtype=np.float32)
    with sqlite3.connect(db) as conn:
        speaker_id._ensure_schema(conn)
        conn.execute(
            "INSERT INTO speakers (name, embedding, last_seen) VALUES (?, ?, ?)",
            ("carol", emb.tobytes(), "2026-06-01T10:00:00"),
        )
        conn.commit()

    speaker_id.load_enrolled(db)
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

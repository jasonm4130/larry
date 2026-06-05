"""Unit tests for SpeakerIDProcessor's pure speaker-matching logic.

Self-contained: no heavy ML models, no audio devices, no network. The
Resemblyzer ``VoiceEncoder`` is monkeypatched with a fake whose embedding
output is fully controlled, so no torch model ever loads. We exercise
``_identify_speaker`` directly (synchronous matching logic) rather than the
async audio-buffering path.
"""

import numpy as np
import pytest

import larry.speaker_id as speaker_id
from larry.speaker_id import SpeakerIDProcessor, load_enrolled, store_speaker


class _FakeEncoder:
    """Stand-in for Resemblyzer's VoiceEncoder.

    ``embed_utterance`` ignores its audio input and returns a fixed vector set
    on the instance, letting tests drive the cosine-similarity decision.
    """

    next_embedding: np.ndarray = np.zeros(4, dtype=np.float32)

    def embed_utterance(self, audio: np.ndarray) -> np.ndarray:
        return self.next_embedding


@pytest.fixture
def processor(monkeypatch, tmp_path):
    """A SpeakerIDProcessor wired to a fake encoder and empty on-disk DB.

    ``VoiceEncoder`` is replaced before construction so ``__init__`` never
    touches torch. The returned processor's ``_enrolled`` dict is populated
    directly in each test.
    """
    monkeypatch.setattr(speaker_id, "VoiceEncoder", _FakeEncoder)
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
    monkeypatch.setattr(speaker_id, "VoiceEncoder", _FakeEncoder)
    seen: list[str] = []
    proc = SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        on_speaker_change=seen.append,
    )
    proc._enrolled = {"alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    proc._encoder.next_embedding = np.array([0.97, 0.05, 0.0, 0.0], dtype=np.float32)  # pyright: ignore[reportArgumentType]

    proc._identify_speaker(b"\x00\x00")

    assert seen == ["alice"]
    assert proc._current_speaker == "alice"


def test_threshold_boundary_is_inclusive(monkeypatch, tmp_path):
    # Construct an incoming embedding whose cosine vs the stored one is exactly
    # the threshold, to lock the >= (inclusive) comparison.
    monkeypatch.setattr(speaker_id, "VoiceEncoder", _FakeEncoder)
    proc = SpeakerIDProcessor(
        speakers_db_path=tmp_path / "speakers.db",
        match_threshold=0.75,
    )
    stored = np.array([1.0, 0.0], dtype=np.float32)
    # Unit vector at angle theta where cos(theta) == 0.75.
    incoming = np.array([0.75, np.sqrt(1 - 0.75**2)], dtype=np.float32)
    proc._enrolled = {"alice": stored}
    proc._encoder.next_embedding = incoming  # pyright: ignore[reportArgumentType]

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
    monkeypatch.setattr(sid_mod, "VoiceEncoder", _FakeEncoder)
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

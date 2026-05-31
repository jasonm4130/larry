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
from larry.speaker_id import SpeakerIDProcessor


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

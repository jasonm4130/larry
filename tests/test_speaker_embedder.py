"""Unit tests for the SpeakerEmbedder interface and its Resemblyzer impl.

Self-contained: torch never loads. ``VoiceEncoder`` is monkeypatched with a
fake whose embedding output is fully controlled.
"""

import numpy as np
import pytest

import larry.speaker_embedder as speaker_embedder
from larry.speaker_embedder import ResemblyzerEmbedder, SpeakerEmbedder, get_speaker_embedder


class _FakeVoiceEncoder:
    """Stand-in for Resemblyzer's VoiceEncoder — no torch load."""

    def embed_utterance(self, audio: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_resemblyzer_embedder_satisfies_speaker_embedder_protocol(monkeypatch):
    monkeypatch.setattr(speaker_embedder, "VoiceEncoder", _FakeVoiceEncoder)
    embedder = ResemblyzerEmbedder()
    assert isinstance(embedder, SpeakerEmbedder)


def test_resemblyzer_embedder_name_is_resemblyzer(monkeypatch):
    monkeypatch.setattr(speaker_embedder, "VoiceEncoder", _FakeVoiceEncoder)
    embedder = ResemblyzerEmbedder()
    assert embedder.name == "resemblyzer"


def test_resemblyzer_embedder_embed_returns_ndarray(monkeypatch):
    monkeypatch.setattr(speaker_embedder, "VoiceEncoder", _FakeVoiceEncoder)
    embedder = ResemblyzerEmbedder()
    out = embedder.embed(np.zeros(16000, dtype=np.float32))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_almost_equal(out, [0.1, 0.2, 0.3])


def test_get_speaker_embedder_resemblyzer_returns_resemblyzer_impl(monkeypatch):
    monkeypatch.setattr(speaker_embedder, "VoiceEncoder", _FakeVoiceEncoder)
    embedder = get_speaker_embedder("resemblyzer")
    assert isinstance(embedder, ResemblyzerEmbedder)
    assert embedder.name == "resemblyzer"


def test_get_speaker_embedder_unknown_name_raises():
    with pytest.raises(ValueError, match="titanet"):
        get_speaker_embedder("titanet")

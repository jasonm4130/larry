"""Unit tests for the SpeakerEmbedder interface and its impls.

Two tiers:
  - Offline: torch/onnx/model never load. ``VoiceEncoder`` and the CAM++ model
    load/session are monkeypatched with fakes whose output is fully controlled.
  - Model-gated: real CAM++ embeddings, run only when the ~28 MB onnx model is
    already cached at ``data/models/`` (``uv run larry fetch-models``); skipped
    otherwise so the suite never depends on network reachability.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

import larry.speaker_embedder as speaker_embedder
from larry.speaker_embedder import (
    CamPlusPlusEmbedder,
    ResemblyzerEmbedder,
    SpeakerEmbedder,
    _compute_fbank,
    get_speaker_embedder,
)

FIXTURES = Path(__file__).parent / "fixtures"
_MODEL_DIR = Path("data") / "models"
_MODEL_FILE = _MODEL_DIR / "voxceleb_CAM++_LM.onnx"


# --- Resemblyzer (offline) ----------------------------------------------------


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


# --- CAM++ (offline: fake model session) --------------------------------------


class _FakeSession:
    """Stand-in for an onnxruntime InferenceSession — returns a fixed [1, 512]."""

    def __init__(self, out: np.ndarray) -> None:
        self._out = out

    def run(self, output_names, input_feed):
        return [self._out]


def _patch_fake_campplus(monkeypatch, out: np.ndarray) -> None:
    monkeypatch.setattr(speaker_embedder, "_ensure_model_cached", lambda model_dir=None: Path("x"))
    monkeypatch.setattr(speaker_embedder, "_make_session", lambda path: _FakeSession(out))


def _voiced_noise(n: int = 16000, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.05).astype(np.float32)


def test_get_speaker_embedder_campplus_returns_impl(monkeypatch):
    _patch_fake_campplus(monkeypatch, np.ones((1, 512), dtype=np.float32))
    embedder = get_speaker_embedder("wespeaker_campplus")
    assert isinstance(embedder, CamPlusPlusEmbedder)
    assert isinstance(embedder, SpeakerEmbedder)
    assert embedder.name == "wespeaker_campplus"


def test_campplus_embed_is_512d_and_unit_norm(monkeypatch):
    out = np.arange(512, dtype=np.float32).reshape(1, 512)
    _patch_fake_campplus(monkeypatch, out)
    vec = CamPlusPlusEmbedder().embed(_voiced_noise())
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_campplus_embed_is_deterministic(monkeypatch):
    _patch_fake_campplus(monkeypatch, np.linspace(-1, 1, 512, dtype=np.float32).reshape(1, 512))
    embedder = CamPlusPlusEmbedder()
    audio = _voiced_noise()
    np.testing.assert_array_equal(embedder.embed(audio), embedder.embed(audio))


def test_compute_fbank_shape_and_cmn():
    mat = _compute_fbank(_voiced_noise(16000))
    assert mat.ndim == 2 and mat.shape[1] == 80
    # Per-utterance cepstral-mean normalization → each mel bin is ~zero-mean over time.
    assert np.allclose(mat.mean(axis=0), 0.0, atol=1e-4)


# --- CAM++ (model-gated: real onnx embeddings) --------------------------------


@pytest.fixture(scope="session")
def campplus_model_dir() -> Path:
    if not _MODEL_FILE.exists():
        pytest.skip("CAM++ model not cached — run `uv run larry fetch-models` to enable")
    return _MODEL_DIR


def _read_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def test_campplus_matches_reference_golden(campplus_model_dir):
    """larry's kaldi-native-fbank pipeline reproduces WeSpeaker's torchaudio-reference
    embedding. A window-type/CMN/scaling regression drops this cosine below the gate."""
    embedder = CamPlusPlusEmbedder(campplus_model_dir)
    got = embedder.embed(_read_wav_mono16k(FIXTURES / "speech_sp0307.wav"))
    golden = np.load(FIXTURES / "speech_sp0307_campplus_golden.npy")
    cos = float(np.dot(got, golden))
    assert cos >= 0.9995, f"cosine {cos:.6f} vs reference golden — featurization drift"


def test_campplus_discriminates(campplus_model_dir):
    """Non-degeneracy smoke test: same speaker (two halves) scores far above the same
    speaker vs noise. (True cross-speaker calibration happens on-device, not here.)"""
    embedder = CamPlusPlusEmbedder(campplus_model_dir)
    wav = _read_wav_mono16k(FIXTURES / "speech_sp0307.wav")
    half = len(wav) // 2
    e_h1 = embedder.embed(wav[:half])
    e_h2 = embedder.embed(wav[half:])
    e_noise = embedder.embed(_voiced_noise(3 * 16000, seed=1234))

    same = float(np.dot(e_h1, e_h2))
    speaker_vs_noise = float(np.dot(e_h1, e_noise))
    assert same > 0.6, f"same-speaker halves cosine {same:.3f} unexpectedly low"
    assert speaker_vs_noise < 0.3, (
        f"speaker-vs-noise cosine {speaker_vs_noise:.3f} unexpectedly high"
    )
    assert same - speaker_vs_noise > 0.3

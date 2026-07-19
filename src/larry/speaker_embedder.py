"""Model-agnostic speaker-embedding interface.

``SpeakerEmbedder`` abstracts the voice-embedding model behind a single
``embed()`` method so ``SpeakerIDProcessor`` and the voice-enroll capture path
never call a specific model's SDK directly. Swapping embedders means adding an
impl plus a branch in ``get_speaker_embedder``, with no change anywhere else.
Different embedders are different vector spaces (and different dims — CAM++ is
512-d vs Resemblyzer's 256-d), so voiceprints are namespaced by embedder name in
the speakers DB (see ``speaker_id.py``'s schema) — a print from one embedder is
never cosine-matched under another.

Two impls ship today:
  - ``wespeaker_campplus`` (default) — WeSpeaker CAM++ (512-d, ~0.71% EER), an
    ONNX model run via onnxruntime with 80-dim Kaldi-fbank features. Far better
    far-field cosine separation than Resemblyzer (~4.5% EER).
  - ``resemblyzer`` — the original GE2E VoiceEncoder (256-d, ~4.5% EER), kept
    selectable via ``SPEAKER_EMBEDDER=resemblyzer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from resemblyzer import VoiceEncoder

if TYPE_CHECKING:
    import onnxruntime as ort


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """A speaker-embedding model: raw mono float32 16kHz audio -> a fixed-size vector."""

    name: str  # persisted alongside each voiceprint for cross-model namespacing

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray: ...


class ResemblyzerEmbedder:
    """SpeakerEmbedder backed by Resemblyzer's GE2E VoiceEncoder.

    256-dim, ~4.5% EER. Superseded as the default by ``CamPlusPlusEmbedder`` (its
    cosine scores separate too weakly on far-field mic audio), but kept selectable.
    """

    name = "resemblyzer"

    def __init__(self) -> None:
        self._encoder = VoiceEncoder()  # blocks on torch load — fail fast

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        return np.asarray(self._encoder.embed_utterance(audio_f32_16k))


# --- WeSpeaker CAM++ (ONNX) ---------------------------------------------------

# Pinned artifact: WeSpeaker CAM++_LM (VoxCeleb), 512-dim, ~0.71% EER. Code is
# Apache-2.0; weights CC-BY-4.0. The URL encodes the literal "+" as %2B.
_CAMPP_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/"
    "resolve/main/voxceleb_CAM%2B%2B_LM.onnx?download=true"
)
_CAMPP_SHA256 = "1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93"
_CAMPP_FILENAME = "voxceleb_CAM++_LM.onnx"
_DEFAULT_MODEL_DIR = Path("data") / "models"

_SAMPLE_RATE = 16000


def _ensure_model_cached(model_dir: Path | None = None) -> Path:
    """Download+verify the CAM++ onnx to *model_dir* (default ``data/models``) once.

    Split out as a module-level function so tests can monkeypatch it (as the
    Resemblyzer tests monkeypatch ``VoiceEncoder``) instead of hitting the network.
    """
    from larry.model_fetch import ensure_cached_model

    dest_dir = _DEFAULT_MODEL_DIR if model_dir is None else Path(model_dir)
    return ensure_cached_model(_CAMPP_URL, _CAMPP_SHA256, dest_dir / _CAMPP_FILENAME)


def _make_session(model_path: Path) -> ort.InferenceSession:
    """Construct the onnx runtime session (module-level for test monkeypatching)."""
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path))


def _compute_fbank(audio_f32_16k: np.ndarray) -> np.ndarray:
    """80-dim Kaldi fbank + CMN, matching WeSpeaker's reference featurization exactly.

    Every option is pinned to ``wespeaker/bin/infer_onnx.py``: the waveform is
    scaled from [-1, 1] to int16 range, a *Hamming* window is used (NOT
    kaldi-native-fbank's ``povey`` default — the difference degrades real matches
    while still passing a loose separation check), dither is 0, energy is unused,
    and per-utterance cepstral-mean normalization (no variance norm) is applied.
    Returns a ``(num_frames, 80)`` float32 matrix.
    """
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = _SAMPLE_RATE
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.use_energy = False

    fb = knf.OnlineFbank(opts)
    samples = np.asarray(audio_f32_16k, dtype=np.float32) * 32768.0
    # knf accepts a numpy float32 array at runtime; its stub types this as List[float].
    fb.accept_waveform(_SAMPLE_RATE, samples)  # type: ignore[arg-type]
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        raise ValueError("audio too short to produce a single fbank frame")
    mat = np.stack([np.asarray(fb.get_frame(i)) for i in range(n)], axis=0)
    return (mat - mat.mean(axis=0, keepdims=True)).astype(np.float32)


class CamPlusPlusEmbedder:
    """SpeakerEmbedder backed by WeSpeaker's CAM++ (``voxceleb_CAM++_LM.onnx``).

    512-dim, ~0.71% EER. Features are 80-dim Kaldi fbank (Hamming window, CMN) via
    ``kaldi-native-fbank``; the onnx graph runs on onnxruntime. Output embeddings
    are L2-normalized, so cosine similarity is a plain dot product and thresholds
    are stable across utterances.
    """

    name = "wespeaker_campplus"

    def __init__(self, model_dir: Path | None = None) -> None:
        model_path = _ensure_model_cached(model_dir)  # fail fast if unavailable
        self._session = _make_session(model_path)

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        feats = _compute_fbank(audio_f32_16k)[np.newaxis, :, :]
        raw = self._session.run(["embs"], {"feats": feats})[0]  # [1, 512]
        vec = cast("np.ndarray", raw)[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec


def get_speaker_embedder(name: str, model_dir: Path | None = None) -> SpeakerEmbedder:
    """Return the SpeakerEmbedder impl selected by *name* (the SPEAKER_EMBEDDER value).

    *model_dir* is the download cache for embedders that fetch a model file (CAM++);
    ignored by embedders that bundle their weights (Resemblyzer).
    """
    if name == "wespeaker_campplus":
        return CamPlusPlusEmbedder(model_dir)
    if name == "resemblyzer":
        return ResemblyzerEmbedder()
    raise ValueError(
        f"Unknown SPEAKER_EMBEDDER value: {name!r}. Expected 'wespeaker_campplus' or 'resemblyzer'."
    )

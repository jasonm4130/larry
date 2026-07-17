"""Model-agnostic speaker-embedding interface.

``SpeakerEmbedder`` abstracts the voice-embedding model behind a single
``embed()`` method so ``SpeakerIDProcessor`` and the voice-enroll capture path
never call a specific model's SDK directly. Swapping embedders — e.g. the
TitaNet/ONNX follow-up scoped in
docs/superpowers/plans/2026-07-17-identity-and-wake-fixes.md (Task 3) — means
adding an impl plus a branch in ``get_speaker_embedder``, with no change
anywhere else. Different embedders are different vector spaces (and different
dims — TitaNet-Small/ECAPA are 192-d vs Resemblyzer's 256-d), so voiceprints
are namespaced by embedder name in the speakers DB (see ``speaker_id.py``'s
schema) — a print from one embedder is never cosine-matched under another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from resemblyzer import VoiceEncoder


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """A speaker-embedding model: raw mono float32 16kHz audio -> a fixed-size vector."""

    name: str  # persisted alongside each voiceprint for cross-model namespacing

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray: ...


class ResemblyzerEmbedder:
    """SpeakerEmbedder backed by Resemblyzer's GE2E VoiceEncoder.

    256-dim, ~4.5% EER — today's default (SPEAKER_EMBEDDER=resemblyzer) until
    a more accurate model (TitaNet-Small/ECAPA, ~0.7-0.9% EER) is validated on
    real Pi hardware; see Task 3's deferred follow-up.
    """

    name = "resemblyzer"

    def __init__(self) -> None:
        self._encoder = VoiceEncoder()  # blocks on torch load — fail fast

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        return np.asarray(self._encoder.embed_utterance(audio_f32_16k))


def get_speaker_embedder(name: str) -> SpeakerEmbedder:
    """Return the SpeakerEmbedder impl selected by *name* (the SPEAKER_EMBEDDER config value).

    Only "resemblyzer" is implemented today; a future TitaNet/ONNX impl (Task 3
    follow-up) adds a branch here without touching any caller.
    """
    if name == "resemblyzer":
        return ResemblyzerEmbedder()
    raise ValueError(f"Unknown SPEAKER_EMBEDDER value: {name!r}. Expected 'resemblyzer'.")

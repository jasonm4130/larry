from __future__ import annotations

from typing import Protocol

import numpy as np


class JawDriver(Protocol):
    def set_open_fraction(self, fraction: float) -> None: ...
    def close(self) -> None: ...


class JawAmplitudeMapper:
    """Convert streaming int16 PCM bytes to a smoothed jaw-open fraction [0.0, 1.0].

    Phase 5 lip-sync. The Pipecat AudioBufferProcessor delivers chunks of
    ``bot_audio`` (the LLM-generated speech going to the speaker). We sample
    RMS in short windows, normalise against a running noise floor + peak,
    apply exponential smoothing, and emit a fraction ready for JawDriver.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        noise_floor: float = 0.01,
        peak: float = 0.3,
    ) -> None:
        self._alpha = smoothing_alpha
        self._noise_floor = noise_floor
        self._peak = peak
        self._smoothed: float = 0.0

    def feed(self, pcm_int16_bytes: bytes) -> float:
        """Consume a chunk of int16 PCM (mono); return current smoothed fraction in [0, 1]."""
        n = len(pcm_int16_bytes) // 2
        if n == 0:
            return self._smoothed

        samples = np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)) / 32768.0)

        span = self._peak - self._noise_floor
        raw = (rms - self._noise_floor) / span if span > 0 else 0.0
        clamped = max(0.0, min(1.0, raw))

        self._smoothed = self._alpha * clamped + (1.0 - self._alpha) * self._smoothed
        return self._smoothed

    def reset(self) -> None:
        """Reset smoothing state — call between utterances."""
        self._smoothed = 0.0

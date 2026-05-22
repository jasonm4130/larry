import numpy as np

from larry.jaw import JawAmplitudeMapper


def _silence(n_samples: int = 1600) -> bytes:
    return (np.zeros(n_samples, dtype=np.int16)).tobytes()


def _loud(n_samples: int = 1600, amp: int = 16000) -> bytes:
    t = np.linspace(0, 1, n_samples, endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()


def test_silence_close_to_zero():
    m = JawAmplitudeMapper()
    out = m.feed(_silence())
    assert out < 0.1


def test_loud_pushes_toward_one():
    m = JawAmplitudeMapper(smoothing_alpha=1.0)  # no smoothing for the test
    out = m.feed(_loud())
    assert out > 0.5


def test_reset_clears_state():
    m = JawAmplitudeMapper()
    m.feed(_loud())
    m.reset()
    out = m.feed(_silence())
    assert out < 0.1

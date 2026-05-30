import pytest

from larry.hardware import get_jaw_driver
from larry.hardware.jaw_mock import MockJawDriver
from larry.jaw import JawDriver


def test_mock_satisfies_jaw_driver_protocol():
    # JawDriver is a plain (non-runtime_checkable) Protocol, so verify the
    # required methods exist on the mock and are callable.
    jaw = MockJawDriver()
    for method in ("set_open_fraction", "close"):
        assert hasattr(jaw, method)
        assert callable(getattr(jaw, method))

    # Sanity: the Protocol declares exactly these methods.
    for method in ("set_open_fraction", "close"):
        assert hasattr(JawDriver, method)


def test_get_jaw_driver_returns_mock_for_mock_env(monkeypatch):
    monkeypatch.setenv("LARRY_HARDWARE", "mock")
    jaw = get_jaw_driver()
    assert isinstance(jaw, MockJawDriver)


def test_get_jaw_driver_raises_value_error_for_unknown(monkeypatch):
    monkeypatch.setenv("LARRY_HARDWARE", "bogus")
    with pytest.raises(ValueError, match="bogus"):
        get_jaw_driver()

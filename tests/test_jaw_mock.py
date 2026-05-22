from larry.hardware.jaw_mock import MockJawDriver


def test_mock_jaw_accepts_fraction():
    jaw = MockJawDriver()
    jaw.set_open_fraction(0.0)
    jaw.set_open_fraction(0.5)
    jaw.set_open_fraction(1.0)
    jaw.close()

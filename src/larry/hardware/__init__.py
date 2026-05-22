import sys
import os

from larry.jaw import JawDriver


def get_jaw_driver() -> JawDriver:
    """Return the appropriate JawDriver for the current platform/config.

    Selects MockJawDriver when LARRY_HARDWARE=mock (or on macOS by default).
    Selects PCA9685JawDriver when LARRY_HARDWARE=pca9685 (or on Linux by default).
    Imports are lazy so macOS never attempts to import adafruit_servokit.
    """
    hardware = os.environ.get(
        "LARRY_HARDWARE",
        "mock" if sys.platform == "darwin" else "pca9685",
    )

    if hardware == "mock":
        from larry.hardware.jaw_mock import MockJawDriver
        return MockJawDriver()
    elif hardware == "pca9685":
        from larry.hardware.jaw_pca9685 import PCA9685JawDriver
        from larry.config import load_config
        cfg = load_config()
        return PCA9685JawDriver(
            channel=cfg.jaw_servo_channel,
            open_angle=cfg.jaw_open_angle,
            closed_angle=cfg.jaw_closed_angle,
        )
    else:
        raise ValueError(
            f"Unknown LARRY_HARDWARE value: {hardware!r}. Expected 'mock' or 'pca9685'."
        )

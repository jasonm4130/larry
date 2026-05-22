# Pi-only: this file must only be imported on Linux/aarch64.
# If you see an ImportError for adafruit_servokit on macOS,
# the hardware factory in larry/hardware/__init__.py has a bug.
from adafruit_servokit import ServoKit


class PCA9685JawDriver:
    """JawDriver for the PCA9685 servo controller on Pi 5 via adafruit_servokit.

    The lgpio pin factory is set via GPIOZERO_PIN_FACTORY=lgpio in the systemd
    unit (or your shell env) — this class does not set it directly.
    """

    def __init__(self, channel: int, open_angle: int, closed_angle: int) -> None:
        self._kit = ServoKit(channels=16)
        self._channel = channel
        self._open_angle = open_angle
        self._closed_angle = closed_angle
        # Start closed
        self._kit.servo[self._channel].angle = self._closed_angle

    def set_open_fraction(self, fraction: float) -> None:
        """Move the jaw to a position interpolated between closed and open angles.

        Args:
            fraction: 0.0 = fully closed, 1.0 = fully open.
        """
        fraction = max(0.0, min(1.0, fraction))
        angle = self._closed_angle + fraction * (self._open_angle - self._closed_angle)
        self._kit.servo[self._channel].angle = angle

    def close(self) -> None:
        self._kit.servo[self._channel].angle = self._closed_angle

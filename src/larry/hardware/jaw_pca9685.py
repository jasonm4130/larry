"""PCA9685-backed JawDriver — drives a hobby servo for Larry's jaw on the Pi.

Importing this module requires adafruit_servokit, which is only installed on
the Pi (`uv sync --extra pi`). The hardware factory in `larry.hardware.__init__`
must never import this module on macOS.
"""

from adafruit_servokit import ServoKit


class PCA9685JawDriver:
    """JawDriver for the PCA9685 servo controller on Pi 5 via adafruit_servokit.

    The lgpio pin factory is set via GPIOZERO_PIN_FACTORY=lgpio in the systemd
    unit (or your shell env) — this class does not set it directly.
    """

    def __init__(
        self,
        channel: int = 0,
        open_angle: int = 60,
        closed_angle: int = 0,
        i2c_address: int = 0x40,
    ) -> None:
        self._kit = ServoKit(channels=16, address=i2c_address)
        self._channel = channel
        self._open_angle = open_angle
        self._closed_angle = closed_angle
        self.close()

    def set_open_fraction(self, fraction: float) -> None:
        f = max(0.0, min(1.0, float(fraction)))
        angle = self._closed_angle + f * (self._open_angle - self._closed_angle)
        self._kit.servo[self._channel].angle = angle

    def close(self) -> None:
        self._kit.servo[self._channel].angle = self._closed_angle

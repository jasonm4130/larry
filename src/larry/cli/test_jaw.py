import time

from larry.hardware import get_jaw_driver


def main() -> None:
    jaw = get_jaw_driver()
    for f in [0.0, 0.3, 0.6, 1.0, 0.6, 0.3, 0.0]:
        jaw.set_open_fraction(f)
        time.sleep(0.3)
    jaw.close()

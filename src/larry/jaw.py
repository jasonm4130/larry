from typing import Protocol


class JawDriver(Protocol):
    def set_open_fraction(self, fraction: float) -> None: ...
    def close(self) -> None: ...


# Phase 5: AudioBufferProcessor.on_track_audio_data handler computes RMS on bot_audio
# in 20ms windows and calls jaw.set_open_fraction(rms_normalised).

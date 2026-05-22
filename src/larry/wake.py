"""Wake-word gate: holds the pipeline asleep until 'Hey Larry' is detected."""

import struct
from time import monotonic

import pvporcupine
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_PORCUPINE_SAMPLE_RATE = 16000


class WakeWordGate(FrameProcessor):
    """Gates downstream pipeline activity until the configured wake word is heard.

    System frames (including InputAudioRawFrame, StartFrame, EndFrame) always
    pass through so pipeline control is never blocked.  Non-system data frames
    are dropped while asleep and forwarded while awake.
    """

    def __init__(
        self,
        porcupine: pvporcupine.Porcupine,
        sleep_timeout_s: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._porcupine = porcupine
        self._sleep_timeout_s = sleep_timeout_s
        self._awake: bool = False
        self._last_voice_activity: float = 0.0
        self._pcm_buffer: list[int] = []

    async def cleanup(self) -> None:
        await super().cleanup()
        self._porcupine.delete()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame, direction)
            return

        # Always forward system frames (StartFrame, EndFrame, CancelFrame, etc.).
        # Drop non-system frames while asleep.
        if isinstance(frame, SystemFrame) or self._awake:
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_audio(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        """Route audio based on wake state; feed asleep audio to Porcupine."""
        if self._awake:
            # Check sleep timeout on every audio frame — cheap monotonic compare.
            if monotonic() - self._last_voice_activity > self._sleep_timeout_s:
                self._awake = False
                logger.info("Larry going back to sleep (timeout).")
            else:
                if any(frame.audio):
                    self._last_voice_activity = monotonic()
                await self.push_frame(frame, direction)
            return

        # While asleep: feed audio to Porcupine; wake on detection.
        if frame.sample_rate != _PORCUPINE_SAMPLE_RATE or frame.num_channels != 1:
            logger.warning(
                "WakeWordGate expects 16kHz mono audio, got {}Hz {}ch"
                " — wake-word detection skipped.",
                frame.sample_rate,
                frame.num_channels,
            )
            return

        # Unpack int16 samples and accumulate into the ring buffer.
        n_samples = len(frame.audio) // 2
        samples = list(struct.unpack_from(f"{n_samples}h", frame.audio))
        self._pcm_buffer.extend(samples)

        fl = self._porcupine.frame_length
        while len(self._pcm_buffer) >= fl:
            chunk = self._pcm_buffer[:fl]
            self._pcm_buffer = self._pcm_buffer[fl:]
            keyword_index = self._porcupine.process(chunk)
            if keyword_index >= 0:
                self._awake = True
                self._last_voice_activity = monotonic()
                logger.info("Larry awoken by wake word (index={}).", keyword_index)
                # Forward this frame so the immediately following speech isn't lost.
                await self.push_frame(frame, direction)
                return


def make_wake_word_gate(
    access_key: str,
    keyword_path: str | None = None,
    sleep_timeout_s: float = 30.0,
) -> WakeWordGate:
    """Build a WakeWordGate.

    If keyword_path is None, falls back to the built-in 'computer' keyword so
    development works before a custom 'Hey Larry' model is trained in the
    Picovoice Console.
    """
    if keyword_path is not None:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path],
        )
    else:
        logger.warning("No wake-word .ppn path provided — using built-in 'computer' keyword.")
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=["computer"],
        )
    return WakeWordGate(porcupine=porcupine, sleep_timeout_s=sleep_timeout_s)

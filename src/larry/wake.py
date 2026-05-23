"""Wake-word gate: holds the pipeline asleep until the configured wake word is detected."""

import struct
from time import monotonic

import numpy as np
from loguru import logger
from openwakeword.model import Model
from pipecat.frames.frames import Frame, InputAudioRawFrame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — OpenWakeWord's required frame size


class WakeWordGate(FrameProcessor):
    """Gates downstream pipeline activity until the configured wake word is heard.

    System frames (including InputAudioRawFrame, StartFrame, EndFrame) always
    pass through so pipeline control is never blocked.  Non-system data frames
    are dropped while asleep and forwarded while awake.
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        threshold: float = 0.5,
        sleep_timeout_s: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._model_name = model_name
        self._threshold = threshold
        self._sleep_timeout_s = sleep_timeout_s
        self._awake: bool = False
        self._last_voice_activity: float = 0.0
        self._pcm_buffer: list[int] = []

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
        """Route audio based on wake state; feed asleep audio to OpenWakeWord."""
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

        # While asleep: feed audio to OpenWakeWord; wake on detection.
        if frame.sample_rate != _SAMPLE_RATE or frame.num_channels != 1:
            logger.warning(
                "WakeWordGate expects 16kHz mono audio, got {}Hz {}ch"
                " — wake-word detection skipped.",
                frame.sample_rate,
                frame.num_channels,
            )
            return

        # Unpack int16 samples and accumulate into the buffer.
        n_samples = len(frame.audio) // 2
        samples = list(struct.unpack_from(f"{n_samples}h", frame.audio))
        self._pcm_buffer.extend(samples)

        while len(self._pcm_buffer) >= _CHUNK_SAMPLES:
            chunk = self._pcm_buffer[:_CHUNK_SAMPLES]
            self._pcm_buffer = self._pcm_buffer[_CHUNK_SAMPLES:]
            chunk_np = np.array(chunk, dtype=np.int16)
            scores = self._model.predict(chunk_np)
            score = scores.get(self._model_name, 0.0)
            if score >= self._threshold:
                self._awake = True
                self._last_voice_activity = monotonic()
                logger.info(
                    "Larry awoken by wake word '{}' (score={:.3f}).",
                    self._model_name,
                    score,
                )
                # Forward this frame so the immediately following speech isn't lost.
                await self.push_frame(frame, direction)
                return


def make_wake_word_gate(
    model_name: str = "hey_jarvis",
    custom_model_path: str | None = None,
    sleep_timeout_s: float = 30.0,
    threshold: float = 0.5,
) -> WakeWordGate:
    """Build a WakeWordGate backed by OpenWakeWord (Apache-2.0, no API key needed).

    If custom_model_path is provided, loads that .onnx file instead of the
    named pretrained model.  Models auto-download to the openwakeword cache
    directory on first use.
    """
    if custom_model_path is not None:
        model = Model(wakeword_models=[custom_model_path], inference_framework="onnx")
        effective_name = custom_model_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    else:
        logger.info(
            "Loading OpenWakeWord pretrained model '{}'. "
            "Train a custom 'Hey Larry' via the OpenWakeWord Colab and set "
            "WAKE_WORD_CUSTOM_PATH to use it instead.",
            model_name,
        )
        model = Model(wakeword_models=[model_name], inference_framework="onnx")
        effective_name = model_name

    return WakeWordGate(
        model=model,
        model_name=effective_name,
        threshold=threshold,
        sleep_timeout_s=sleep_timeout_s,
    )

"""Wake-word gate: holds the pipeline asleep until the configured wake word is detected."""

import struct
from pathlib import Path
from time import monotonic

import numpy as np
import openwakeword
from loguru import logger
from openwakeword.model import Model
from pipecat.frames.frames import Frame, InputAudioRawFrame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — OpenWakeWord's required frame size


def _resolve_pretrained_path(name: str) -> str:
    """Map a friendly name like 'hey_jarvis' to its bundled .onnx file path."""
    for path in openwakeword.get_pretrained_model_paths():
        if Path(path).stem.startswith(name):
            return path
    available = [Path(p).stem for p in openwakeword.get_pretrained_model_paths()]
    raise ValueError(
        f"Unknown OpenWakeWord model {name!r}. Bundled options: {available}. "
        "For a custom 'Hey Larry' model, set WAKE_WORD_CUSTOM_PATH to a .onnx file."
    )


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

    If custom_model_path is provided, loads that .onnx file. Otherwise resolves
    model_name (e.g. 'hey_jarvis') to one of the .onnx files bundled with the
    openwakeword package (no download).
    """
    if custom_model_path is not None:
        model_path = custom_model_path
    else:
        model_path = _resolve_pretrained_path(model_name)
        logger.info(
            "Loading OpenWakeWord pretrained model '{}' from {}. "
            "Train a custom 'Hey Larry' via the OpenWakeWord Colab and set "
            "WAKE_WORD_CUSTOM_PATH to use it instead.",
            model_name,
            model_path,
        )

    model = Model(wakeword_model_paths=[model_path])
    # predict() returns a dict keyed by the .onnx filename stem (e.g. "hey_jarvis_v0.1").
    score_key = Path(model_path).stem

    return WakeWordGate(
        model=model,
        model_name=score_key,
        threshold=threshold,
        sleep_timeout_s=sleep_timeout_s,
    )

"""Wake-word gate: holds the pipeline asleep until the configured wake word is detected."""

import importlib.util
import struct
from collections.abc import Callable
from pathlib import Path
from time import monotonic

import numpy as np
import openwakeword
from loguru import logger
from openwakeword.model import Model
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SystemFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
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
        sleep_timeout_s: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._model_name = model_name
        self._threshold = threshold
        self._sleep_timeout_s = sleep_timeout_s
        self._awake: bool = False
        # _speaking is True between VADUserStartedSpeakingFrame and
        # VADUserStoppedSpeakingFrame.  We do NOT time out while speaking, no
        # matter how long the user talks; the 10s clock only runs during
        # post-speech silence (or immediately after wake-word with no speech yet).
        self._speaking: bool = False
        self._last_voice_activity: float = 0.0
        self._pcm_buffer: list[int] = []
        # Optional callbacks fired on state transitions so the rest of the
        # pipeline can surface an audio "Yes?" / "Hmph." listening cue.
        self.on_wake: Callable[[], None] | None = None
        self.on_sleep: Callable[[], None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # VAD speaking-state tracking — VAD frames are SystemFrames so they
        # reach us upstream even though VADProcessor sits downstream of us.
        # We only care while awake; pre-wake speaking is irrelevant.
        if self._awake and isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._last_voice_activity = monotonic()
        elif self._awake and isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            self._last_voice_activity = monotonic()

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
            # Timeout only fires during post-speech silence — never mid-sentence,
            # never on raw-audio energy (HVAC, room tone never make audio all-zero
            # so the old `any(frame.audio)` reset effectively disabled the
            # timeout entirely).  Speaking state is tracked via VAD frames.
            if (
                not self._speaking
                and monotonic() - self._last_voice_activity > self._sleep_timeout_s
            ):
                self._awake = False
                self._speaking = False
                logger.info("Larry going back to sleep (timeout).")
                if self.on_sleep is not None:
                    self.on_sleep()
            else:
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
            # patience=3 requires three consecutive 80 ms chunks (240 ms total)
            # above threshold before predict() reports a hit.  Without it, a
            # single spurious frame can wake — observed as 8-132 ms sleep→wake
            # bounces when the model resumes scoring after a long awake period
            # (we feed no audio while awake, so the preprocessor's mel buffer
            # and the model's internal state are stale on the first post-sleep
            # chunk and routinely cross 0.5 once).
            scores = self._model.predict(
                chunk_np,
                patience={self._model_name: 3},
                threshold={self._model_name: self._threshold},
            )
            # predict() returns dict[str, float] when timing=False (default);
            # openwakeword's untyped signature unions that with the timing=True
            # tuple shape, which pyright can't narrow without an overload.
            score = scores.get(self._model_name, 0.0)  # type: ignore[reportAttributeAccessIssue]
            if score >= self._threshold:
                self._awake = True
                self._speaking = False
                self._last_voice_activity = monotonic()
                logger.info(
                    "Larry awoken by wake word '{}' (score={:.3f}).",
                    self._model_name,
                    score,
                )
                if self.on_wake is not None:
                    self.on_wake()
                # Forward this frame so the immediately following speech isn't lost.
                await self.push_frame(frame, direction)
                return


def make_wake_word_gate(
    model_name: str = "hey_jarvis",
    custom_model_path: str | None = None,
    sleep_timeout_s: float = 10.0,
    threshold: float = 0.5,
) -> WakeWordGate:
    """Build a WakeWordGate backed by OpenWakeWord (Apache-2.0, no API key needed).

    If custom_model_path is provided, loads that .onnx file. Otherwise resolves
    model_name (e.g. 'hey_jarvis') to one of the .onnx files bundled with the
    openwakeword package (no download).
    """
    if custom_model_path is not None:
        model_path = custom_model_path
        logger.info("Loading custom OpenWakeWord model from {}.", model_path)
    else:
        model_path = _resolve_pretrained_path(model_name)
        logger.info(
            "Loading OpenWakeWord pretrained model '{}' from {}. "
            "Train a custom 'Hey Larry' via the OpenWakeWord Colab and set "
            "WAKE_WORD_CUSTOM_PATH to use it instead.",
            model_name,
            model_path,
        )

    # Speex DSP noise suppression is a Pi-only optional dep — the wheel
    # ships from openWakeWord's GitHub releases, not PyPI, and is Linux/
    # aarch64 only.  Feature-detect so macOS dev keeps working: if the
    # wheel isn't installed, silently fall back to no NS at this layer
    # (Pipecat's WebRTC NS in audio_filter.py is still running upstream).
    if importlib.util.find_spec("speexdsp_ns") is not None:
        enable_speex = True
        logger.info("Wake gate: Speex noise suppression enabled.")
    else:
        enable_speex = False
        logger.info(
            "Wake gate: Speex NS disabled (speexdsp_ns not installed). "
            "On Pi: `sudo apt install libspeexdsp-dev` then install the "
            "speexdsp-ns wheel from openwakeword's release assets."
        )

    model = Model(
        wakeword_model_paths=[model_path],
        # Silero VAD gate — zeros wake-word predictions whose 400-560ms
        # surrounding window has VAD score < 0.5.  Kills the false-positive
        # class we actually see in kitchen audio: HVAC drone, appliance
        # clicks, distant non-speech.  ~2ms/frame Pi 5 cost.
        vad_threshold=0.5,
        enable_speex_noise_suppression=enable_speex,
    )
    # predict() returns a dict keyed by the .onnx filename stem (e.g. "hey_jarvis_v0.1").
    score_key = Path(model_path).stem

    return WakeWordGate(
        model=model,
        model_name=score_key,
        threshold=threshold,
        sleep_timeout_s=sleep_timeout_s,
    )

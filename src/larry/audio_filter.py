"""Acoustic echo cancellation (AEC) for Larry's mic input.

On a single device — laptop on Mac, skull on the Pi — the mic and speaker
are co-located, so Larry's own TTS output bleeds back into the mic and
Whisper happily transcribes it as a phantom user turn.  AEC subtracts the
played audio (the "reference" / "far end") from the captured mic audio
(the "near end") before VAD / STT see it.

We use WebRTC AEC3 via `pywebrtc-audio`, wrapped in Pipecat's
``BaseAudioFilter`` contract.  The filter sits in ``audio_in_filter`` on
``LocalAudioTransportParams`` — Pipecat calls ``.filter(bytes) -> bytes``
on every 20 ms mic chunk before pushing it downstream.

The reference signal is a problem Pipecat doesn't solve in the filter API:
``filter()`` only sees mic bytes.  We get the reference out-of-band by
having ``pipeline.py`` push bot-audio chunks into ``feed_reference()`` from
its existing ``AudioBufferProcessor.on_track_audio_data`` event handler.
Bot audio arrives at 24 kHz (ElevenLabs output); mic audio at 16 kHz, so
we resample on the way in.

If reference frames never arrive (e.g. nobody calls ``feed_reference``),
the filter degrades gracefully: WebRTC AEC will be fed silent reference
audio, which means it acts mostly as a high-pass + (lightly) NS-only path.
That's still better than nothing for the desk-mic dev case.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame
from pywebrtc_audio import AudioProcessor
from scipy import signal

logger = logging.getLogger(__name__)


# WebRTC AEC3 always operates on 10 ms frames internally; the binding
# accepts any multiple but we feed it native 10 ms chunks for tightest
# alignment between near and far.  At 16 kHz that's 160 samples / 320
# bytes (int16).
_FRAME_MS = 10
_BYTES_PER_SAMPLE = 2  # int16


class WebRTCEchoCancellationFilter(BaseAudioFilter):
    """Mic-input filter that cancels Larry's own voice out of the mic.

    Wires into ``LocalAudioTransportParams(audio_in_filter=...)``.  Caller
    must push bot audio into ``feed_reference()`` (see ``pipeline.py`` —
    the ``AudioBufferProcessor.on_track_audio_data`` hook does this).

    Reference signal handling:

    * The far-end (bot) audio comes in at 24 kHz mono int16 (Larry's TTS
      output rate).  We resample to the mic sample rate inside
      ``feed_reference`` so the AEC sees aligned rates.
    * We buffer reference audio in a deque of 10 ms frames.  On every
      ``filter()`` call we pop one reference frame per near-end frame.
      If the reference buffer is empty (no bot audio playing) we feed
      silence — AEC handles this fine, it just has nothing to subtract.
    * We cap the reference buffer at ~500 ms to bound memory and to keep
      the AEC filter's delay estimate from going stale.
    """

    def __init__(
        self,
        *,
        reference_sample_rate: int = 24000,
        max_reference_lag_ms: int = 500,
        ns_level: int = 2,
        stream_delay_ms: int = 0,
    ) -> None:
        """Configure the filter.

        Args:
            reference_sample_rate: Sample rate of the bot/TTS audio that
                will be passed to ``feed_reference()``.  ElevenLabs in
                Larry's pipeline outputs 24 kHz.
            max_reference_lag_ms: Max amount of reference audio to keep
                queued.  Old reference frames past this are dropped — if
                we're behind by more than half a second the AEC's delay
                estimate is wrong anyway.
            ns_level: WebRTC noise-suppression aggressiveness (0..3).
                2 = 18 dB suppression, a sensible default.
            stream_delay_ms: Hint to AEC for the round-trip delay
                between far reference and near-end echo.  Leave 0 — AEC3
                estimates it adaptively.
        """
        self._reference_sample_rate = reference_sample_rate
        self._max_reference_lag_ms = max_reference_lag_ms
        self._ns_level = ns_level
        self._stream_delay_ms = stream_delay_ms

        # Set in start() once the input transport tells us the mic rate.
        self._sample_rate: int = 0
        self._frame_samples: int = 0  # samples per 10 ms at near-end rate
        self._frame_bytes: int = 0
        self._processor: AudioProcessor | None = None

        # Accumulator for partial near-end frames coming in from PyAudio.
        # Pipecat hands us 20 ms chunks, AEC eats 10 ms — we split.
        self._near_buf = bytearray()

        # Reference (far-end) ring of 10 ms int16 numpy frames at the
        # near-end sample rate.  feed_reference() resamples + chunks.
        self._ref_frames: deque[np.ndarray] = deque()
        self._max_ref_frames: int = 0  # set in start()

        # Resample state for the bot-audio stream.  We keep a leftover
        # tail so 24k→16k resampling doesn't drift on chunk boundaries.
        self._ref_resample_tail = np.zeros(0, dtype=np.int16)

        self._enabled = True

    # --------------------------------------------------------------- helpers

    def _silent_ref(self) -> np.ndarray:
        """A 10 ms silence buffer at the near-end rate."""
        return np.zeros(self._frame_samples, dtype=np.int16)

    def _push_reference_frames(self, audio: np.ndarray) -> None:
        """Split a contiguous mono int16 array into 10 ms ref frames."""
        # Slice into whole 10 ms frames; stash any tail for next call.
        usable = (audio.size // self._frame_samples) * self._frame_samples
        if usable:
            for i in range(0, usable, self._frame_samples):
                self._ref_frames.append(audio[i : i + self._frame_samples])
        leftover = audio[usable:]
        # Keep the tail so the next reference chunk concatenates cleanly.
        # (Tail is at the *near-end* rate already — short, typically <10 ms.)
        self._ref_resample_tail = leftover

        # Bound memory / staleness.
        while len(self._ref_frames) > self._max_ref_frames:
            self._ref_frames.popleft()

    # ------------------------------------------------------- public-ish API

    def feed_reference(
        self,
        audio: bytes,
        sample_rate: int,
        num_channels: int = 1,
    ) -> None:
        """Push a chunk of bot/TTS audio into the AEC's reference buffer.

        Called from ``AudioBufferProcessor.on_track_audio_data`` in
        ``pipeline.py``.  No-ops if the filter hasn't started yet or has
        been disabled — both safe states.

        Args:
            audio: Raw int16 PCM bytes of the audio Larry is currently
                playing through the speaker.
            sample_rate: Sample rate of ``audio``.  Typically 24000 for
                ElevenLabs output.  We resample to the mic rate.
            num_channels: Channel count of ``audio``.  Only mono is
                supported; stereo gets averaged down.
        """
        if not self._enabled or self._processor is None or not audio:
            return
        if self._frame_samples == 0:
            # start() not called yet — drop, the buffer is empty anyway.
            return

        # Interpret bytes as int16 mono.
        samples = np.frombuffer(audio, dtype=np.int16)
        if num_channels > 1:
            # Average down to mono — cheap and fine for AEC reference.
            samples = samples.reshape(-1, num_channels).mean(axis=1).astype(np.int16)

        # Prepend any leftover tail from last resample so we don't lose
        # samples across chunk boundaries.
        if self._ref_resample_tail.size:
            samples = np.concatenate([self._ref_resample_tail, samples])
            self._ref_resample_tail = np.zeros(0, dtype=np.int16)

        # Resample bot rate (e.g. 24k) → near-end rate (e.g. 16k).  We
        # use polyphase, which is fast and clean for integer ratios.
        if sample_rate != self._sample_rate:
            # scipy.signal.resample_poly uses np.gcd internally to pick
            # up/down factors; works fine for 24k→16k (up=2, down=3).
            up = self._sample_rate
            down = sample_rate
            gcd = int(np.gcd(up, down))
            up //= gcd
            down //= gcd
            resampled = signal.resample_poly(samples, up, down).astype(np.int16)
        else:
            resampled = samples.astype(np.int16, copy=False)

        self._push_reference_frames(resampled)

    # ----------------------------------------------------- BaseAudioFilter

    async def start(self, sample_rate: int) -> None:
        """Allocate the WebRTC AEC for the transport's mic sample rate.

        WebRTC AEC3 supports 16000 / 32000 / 48000.  If the input
        transport runs at something exotic (e.g. 8 kHz) we fall back to
        no-op — the filter passes audio through unchanged so the
        pipeline still works.
        """
        if sample_rate not in (16000, 32000, 48000):
            logger.warning(
                "WebRTC AEC does not support sample_rate=%s; running pass-through.",
                sample_rate,
            )
            self._sample_rate = sample_rate
            self._frame_samples = 0
            self._processor = None
            return

        self._sample_rate = sample_rate
        self._frame_samples = (sample_rate * _FRAME_MS) // 1000
        self._frame_bytes = self._frame_samples * _BYTES_PER_SAMPLE
        self._max_ref_frames = self._max_reference_lag_ms // _FRAME_MS

        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,  # mic AGC fights TTS leveling — skip
            ns_level=self._ns_level,
            stream_delay_ms=self._stream_delay_ms,
        )
        self._near_buf.clear()
        self._ref_frames.clear()
        self._ref_resample_tail = np.zeros(0, dtype=np.int16)
        logger.info(
            "WebRTC AEC started: sample_rate=%d frame_samples=%d ref_rate=%d",
            sample_rate,
            self._frame_samples,
            self._reference_sample_rate,
        )

    async def stop(self) -> None:
        """Drop the processor and any buffered audio."""
        self._processor = None
        self._near_buf.clear()
        self._ref_frames.clear()
        self._ref_resample_tail = np.zeros(0, dtype=np.int16)

    async def process_frame(self, frame: FilterControlFrame) -> None:
        """Honour runtime enable/disable control frames."""
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable
            logger.info("WebRTC AEC %s", "enabled" if frame.enable else "disabled")

    async def filter(self, audio: bytes) -> bytes:
        """Run AEC on a mic chunk and return the cleaned audio.

        Pipecat hands us 20 ms (or whatever) mic chunks; we split into
        10 ms frames, run each through ``AudioProcessor.process``, and
        return the concatenated output.  Partial frames are buffered for
        the next call so we don't lose tail samples.
        """
        if not self._enabled or self._processor is None or self._frame_samples == 0:
            return audio
        if not audio:
            return audio

        self._near_buf.extend(audio)

        available_frames = len(self._near_buf) // self._frame_bytes
        if available_frames == 0:
            # Need more data — let the input transport's "no audio = drop"
            # branch handle the empty return (see base_input.py:254).
            return b""

        total_bytes = available_frames * self._frame_bytes
        block = bytes(self._near_buf[:total_bytes])
        # Trim consumed data; bytearray slicing copies but the buffer is small.
        del self._near_buf[:total_bytes]

        near = np.frombuffer(block, dtype=np.int16).reshape(available_frames, -1)
        out_chunks: list[bytes] = []

        for i in range(available_frames):
            near_frame = near[i]
            far_frame = self._ref_frames.popleft() if self._ref_frames else self._silent_ref()
            try:
                processed = self._processor.process(near_frame, far_frame)
            except Exception as e:  # noqa: BLE001 — surface SDK errors, don't crash audio
                logger.warning("AEC process failed; passing near-end through: %s", e)
                processed = near_frame
            out_chunks.append(processed.tobytes())

        return b"".join(out_chunks)

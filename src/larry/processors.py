"""Module-level FrameProcessors extracted from the pipeline for unit testing.

Two small, self-contained ``FrameProcessor`` subclasses that don't depend on
the rest of ``pipeline.run()``'s wiring:

- ``WhisperHallucinationFilter`` — drops Whisper silence-hallucination
  transcripts (per-segment confidence + static denylist).
- ``STTMuteOnBotSpeech`` — mutes STT while the bot speaks, with a cool-down
  trail after it stops.
"""

import asyncio
import logging
import re

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    STTMuteFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


# Common Whisper-large-v3 silence-hallucinations.  When the audio buffer
# fed to STT is mostly silence (between phrases, or at the tail of a turn),
# Whisper sometimes invents these phrases because they're over-represented
# in its YouTube training data.  Dropping them prevents Larry from
# responding to phantom "Thank yous" he never received.
_WHISPER_HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thanks for watching.",
        "thanks for watching!",
        "thank you for watching.",
        "thank you for watching!",
        "subscribe.",
        "like and subscribe.",
        "bye.",
        "bye!",
        "you",
        ".",
        "",
    }
)


class WhisperHallucinationFilter(FrameProcessor):
    """Drop TranscriptionFrames that look like Whisper silence-hallucinations.

    Two layers, belt-and-braces:

    1. Per-segment confidence: when the STT is configured with
       ``include_prob_metrics=True`` (Groq returns ``verbose_json``), the
       transcription response carries a ``segments`` list with
       ``no_speech_prob``, ``avg_logprob`` and ``compression_ratio`` per
       segment.  Any segment that trips a threshold is dropped; if every
       segment is dropped, the whole frame is dropped.  Thresholds default
       to the OpenAI-documented "consider this silent / failed" values
       (no_speech_prob > 0.6, avg_logprob < -1.0, compression_ratio > 2.4)
       and are tunable from real logs.

    2. Static denylist: case-insensitive match against
       ``_WHISPER_HALLUCINATIONS`` for low-probability silence phrases that
       still slip through the confidence gate.  Matching ignores any
       ``[speaker: name]`` prefix added upstream by SpeakerIDProcessor.

    Contract assumption: ``frame.result`` is either ``None`` or an
    ``openai.types.audio.TranscriptionVerbose``-shaped object exposing
    ``.segments`` (a list of items with ``no_speech_prob``, ``avg_logprob``,
    ``compression_ratio``, ``text``).  We duck-type via ``getattr`` so a
    plain-json response (or a future Pipecat schema change) degrades to the
    static denylist instead of crashing.
    """

    def __init__(
        self,
        *,
        no_speech_prob_max: float = 0.6,
        avg_logprob_min: float = -1.0,
        compression_ratio_max: float = 2.4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._no_speech_prob_max = no_speech_prob_max
        self._avg_logprob_min = avg_logprob_min
        self._compression_ratio_max = compression_ratio_max

    def _segment_reject_reason(self, segment: object) -> str | None:
        """Return a short reason string if this segment should be dropped, else None."""
        no_speech = getattr(segment, "no_speech_prob", None)
        avg_logprob = getattr(segment, "avg_logprob", None)
        compression = getattr(segment, "compression_ratio", None)

        if isinstance(no_speech, (int, float)) and no_speech > self._no_speech_prob_max:
            return f"no_speech_prob={no_speech:.3f}>{self._no_speech_prob_max}"
        if isinstance(avg_logprob, (int, float)) and avg_logprob < self._avg_logprob_min:
            return f"avg_logprob={avg_logprob:.3f}<{self._avg_logprob_min}"
        if isinstance(compression, (int, float)) and compression > self._compression_ratio_max:
            return f"compression_ratio={compression:.3f}>{self._compression_ratio_max}"
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            # Layer 1: per-segment confidence (only when verbose_json result is attached).
            segments = getattr(frame.result, "segments", None)
            # Diagnostic: one-time log per process tells us why the verbose
            # confidence layer isn't engaging — wrong attr path? missing
            # response_format? thresholds tuned wrong? — without spamming.
            if segments is None and not getattr(self, "_logged_missing_segments", False):
                logger.info(
                    "verbose_json segments not found on TranscriptionFrame.result "
                    "(type=%s, attrs=%s, text=%r) — falling back to static denylist only.",
                    type(frame.result).__name__,
                    sorted(a for a in dir(frame.result) if not a.startswith("_"))[:20],
                    frame.text,
                )
                self._logged_missing_segments = True
            if segments:
                kept_any = False
                for seg in segments:
                    seg_text = getattr(seg, "text", "")
                    reason = self._segment_reject_reason(seg)
                    if reason is not None:
                        logger.info(
                            "Dropping low-confidence Whisper segment: text=%r "
                            "no_speech_prob=%r avg_logprob=%r compression_ratio=%r reason=%s",
                            seg_text,
                            getattr(seg, "no_speech_prob", None),
                            getattr(seg, "avg_logprob", None),
                            getattr(seg, "compression_ratio", None),
                            reason,
                        )
                    else:
                        kept_any = True
                if not kept_any:
                    logger.info(
                        "Dropping Whisper transcription (all segments low-confidence): %r",
                        frame.text,
                    )
                    return

            # Layer 2: static denylist on the final text (case-insensitive,
            # ignoring any [speaker: name] prefix).
            stripped = re.sub(r"^\[speaker:[^\]]*\]\s*", "", frame.text).strip().lower()
            if stripped in _WHISPER_HALLUCINATIONS:
                logger.info("Dropping Whisper hallucination: %r", frame.text)
                return
        await self.push_frame(frame, direction)


class STTMuteOnBotSpeech(FrameProcessor):
    """Mute the STT while the bot is speaking, plus a cool-down trail.

    AlwaysUserMuteStrategy only gates the user aggregator — VAD + Whisper
    still run on echo audio, producing phantom transcripts that have shown
    up in logs.  This filter sends STTMuteFrame(mute=True) when the bot
    starts and STTMuteFrame(mute=False) after `cool_down_s` of bot silence,
    so the mic-echo trail after Larry finishes a sentence doesn't get
    transcribed.
    """

    def __init__(self, cool_down_s: float = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cool_down_s = cool_down_s
        self._unmute_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            if self._unmute_task is not None and not self._unmute_task.done():
                self._unmute_task.cancel()
            await self.push_frame(STTMuteFrame(mute=True), FrameDirection.DOWNSTREAM)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._unmute_task is not None and not self._unmute_task.done():
                self._unmute_task.cancel()
            self._unmute_task = asyncio.create_task(self._delayed_unmute())

        await self.push_frame(frame, direction)

    async def _delayed_unmute(self) -> None:
        try:
            await asyncio.sleep(self._cool_down_s)
            await self.push_frame(STTMuteFrame(mute=False), FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

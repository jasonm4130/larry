"""Unit tests for the extracted FrameProcessors in larry.processors.

Self-contained: no heavy ML models, no audio devices, no network, and no
pytest-asyncio dependency. Each test body is an async coroutine driven through
``asyncio.run`` by a tiny ``_run`` helper.

The FrameProcessor base class needs a running event loop and a StartFrame to
set up its internal queues, so each test drives the processor through a small
in-process harness that captures pushed frames.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from larry.processors import STTMuteOnBotSpeech, WhisperHallucinationFilter


def _run(coro: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run an async test body on a fresh event loop."""
    asyncio.run(coro())


class _Sink(FrameProcessor):
    """Downstream collector: records every frame it receives.

    Runs in direct mode so frames are processed synchronously — no
    task_manager / event-loop plumbing needed for a unit test.
    """

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.received: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.received.append(frame)


async def _link(*processors: FrameProcessor) -> None:
    """Link processors into a chain and start them with a StartFrame.

    All processors run in direct mode (see helpers / fixtures), so queuing the
    StartFrame processes it synchronously and flips each processor's internal
    ``__started`` flag without needing the full pipeline runtime.
    """
    for upstream, downstream in zip(processors, processors[1:], strict=False):
        upstream.link(downstream)
    start = StartFrame(
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
    )
    await processors[0].queue_frame(start)


def _transcriptions(sink: _Sink) -> list[str]:
    return [f.text for f in sink.received if isinstance(f, TranscriptionFrame)]


def _mute_states(sink: _Sink) -> list[bool]:
    return [f.mute for f in sink.received if isinstance(f, STTMuteFrame)]


class _Seg:
    """Minimal verbose_json segment stand-in (duck-typed attrs)."""

    def __init__(self, text="hi", no_speech_prob=0.0, avg_logprob=0.0, compression_ratio=1.0):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


class _Result:
    def __init__(self, segments):
        self.segments = segments


def _transcription(text: str, result=None) -> TranscriptionFrame:
    frame = TranscriptionFrame(text=text, user_id="u", timestamp="2026-05-31T00:00:00Z")
    frame.result = result
    return frame


# --------------------------------------------------------------------------
# WhisperHallucinationFilter — _segment_reject_reason predicate (no harness)
# --------------------------------------------------------------------------


def test_segment_reject_reason_passes_good_segment():
    filt = WhisperHallucinationFilter()
    assert filt._segment_reject_reason(_Seg()) is None


def test_segment_reject_reason_no_speech_prob():
    filt = WhisperHallucinationFilter()
    reason = filt._segment_reject_reason(_Seg(no_speech_prob=0.95))
    assert reason is not None and "no_speech_prob" in reason


def test_segment_reject_reason_avg_logprob():
    filt = WhisperHallucinationFilter()
    reason = filt._segment_reject_reason(_Seg(avg_logprob=-2.0))
    assert reason is not None and "avg_logprob" in reason


def test_segment_reject_reason_compression_ratio():
    filt = WhisperHallucinationFilter()
    reason = filt._segment_reject_reason(_Seg(compression_ratio=3.0))
    assert reason is not None and "compression_ratio" in reason


def test_segment_reject_reason_boundary_is_inclusive_pass():
    # Predicate uses strict comparisons, so a segment exactly at a threshold
    # is kept (only values strictly past the threshold are rejected).
    filt = WhisperHallucinationFilter()
    assert (
        filt._segment_reject_reason(
            _Seg(no_speech_prob=0.6, avg_logprob=-1.0, compression_ratio=2.4)
        )
        is None
    )


def test_segment_reject_reason_ignores_missing_metrics():
    # A duck-typed object lacking the metric attrs must not crash or reject;
    # the filter degrades to the static denylist instead.
    class _Bare:
        text = "hi"

    filt = WhisperHallucinationFilter()
    assert filt._segment_reject_reason(_Bare()) is None


def test_segment_reject_reason_honors_custom_thresholds():
    filt = WhisperHallucinationFilter(no_speech_prob_max=0.9)
    # 0.7 trips the default 0.6 gate but not the loosened 0.9 one.
    assert filt._segment_reject_reason(_Seg(no_speech_prob=0.7)) is None
    assert filt._segment_reject_reason(_Seg(no_speech_prob=0.95)) is not None


# --------------------------------------------------------------------------
# WhisperHallucinationFilter — process_frame end-to-end
# --------------------------------------------------------------------------


def test_denylist_drops_hallucination():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        await filt.process_frame(_transcription("Thank you."), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == []

    _run(body)


def test_denylist_passes_real_speech():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        await filt.process_frame(_transcription("What time is it?"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["What time is it?"]

    _run(body)


def test_denylist_ignores_speaker_prefix():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        await filt.process_frame(
            _transcription("[speaker: Jason] thanks for watching!"), FrameDirection.DOWNSTREAM
        )
        await asyncio.sleep(0)

        assert _transcriptions(sink) == []

    _run(body)


def test_segment_confidence_drops_when_all_low():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        # Single segment trips the no_speech_prob gate (> 0.6 default).
        result = _Result([_Seg(text="ghost", no_speech_prob=0.95)])
        await filt.process_frame(_transcription("ghost", result), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == []

    _run(body)


def test_segment_confidence_keeps_when_one_good():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        result = _Result(
            [
                _Seg(text="ghost", no_speech_prob=0.95),
                _Seg(text="real words", no_speech_prob=0.01),
            ]
        )
        await filt.process_frame(
            _transcription("real words here", result), FrameDirection.DOWNSTREAM
        )
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["real words here"]

    _run(body)


def test_denylist_catches_text_when_segments_pass_confidence():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        # Segment confidence is fine (kept), but the final text is a denylisted
        # silence phrase — Layer 2 still drops it.
        result = _Result([_Seg(text="thank you", no_speech_prob=0.01)])
        await filt.process_frame(_transcription("Thank you.", result), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == []

    _run(body)


def test_missing_segments_falls_back_to_denylist_and_passes_real():
    async def body():
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(filt, sink)

        # result is None (no verbose_json): Layer 1 is skipped, real speech
        # passes via the denylist fallback without crashing.
        await filt.process_frame(_transcription("hello there"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["hello there"]

    _run(body)


# --------------------------------------------------------------------------
# STTMuteOnBotSpeech
# --------------------------------------------------------------------------


def test_mute_on_bot_started():
    async def body():
        muter = STTMuteOnBotSpeech(cool_down_s=0.01, enable_direct_mode=True)
        sink = _Sink()
        await _link(muter, sink)

        await muter.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _mute_states(sink) == [True]

    _run(body)


def test_speaking_frames_pass_through():
    async def body():
        muter = STTMuteOnBotSpeech(cool_down_s=0.01, enable_direct_mode=True)
        sink = _Sink()
        await _link(muter, sink)

        await muter.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await muter.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # The original speaking frames are forwarded, not consumed by the muter.
        started = [f for f in sink.received if isinstance(f, BotStartedSpeakingFrame)]
        stopped = [f for f in sink.received if isinstance(f, BotStoppedSpeakingFrame)]
        assert len(started) == 1 and len(stopped) == 1

    _run(body)


def test_unmute_after_cooldown():
    async def body():
        muter = STTMuteOnBotSpeech(cool_down_s=0.01, enable_direct_mode=True)
        sink = _Sink()
        await _link(muter, sink)

        await muter.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await muter.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        # Wait past the (tiny, real) cool-down so the delayed unmute fires.
        await asyncio.sleep(0.05)

        assert _mute_states(sink) == [True, False]

    _run(body)


def test_restart_cancels_pending_unmute():
    async def body():
        muter = STTMuteOnBotSpeech(cool_down_s=0.05, enable_direct_mode=True)
        sink = _Sink()
        await _link(muter, sink)

        # Bot stops, then starts again before the cool-down elapses: the
        # pending unmute must be cancelled so STT isn't re-enabled mid-speech.
        await muter.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await muter.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.01)
        await muter.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        # Two mute=True (two starts), and crucially no mute=False slipped in.
        assert _mute_states(sink) == [True, True]

    _run(body)

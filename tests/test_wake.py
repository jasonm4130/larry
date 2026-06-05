"""Unit tests for ``larry.wake.WakeWordGate``.

Self-contained: no heavy ML models, no ONNX, no audio devices, no network.
The OpenWakeWord model is replaced with a hand-controlled fake (its
``predict()`` / ``reset()`` are driven by the test), and ``larry.wake.monotonic``
is monkeypatched so the sleep clock is deterministic.

Mirrors the harness style of ``test_processors.py``: each test body is an async
coroutine driven through ``asyncio.run``, processors run in direct mode, and a
small ``_Sink`` collects pushed frames.  ``process_frame`` is called directly so
the wake/sleep state machine is exercised with no real pipeline runtime.
"""

import asyncio
import struct
from collections.abc import Callable, Coroutine
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TextFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import larry.wake as wake_module
from larry.wake import _CHUNK_SAMPLES, _SAMPLE_RATE, WakeWordGate

_MODEL_NAME = "hey_jarvis_v0.1"


def _run(coro: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run an async test body on a fresh event loop."""
    asyncio.run(coro())


class _Sink(FrameProcessor):
    """Downstream collector: records every frame it receives (direct mode)."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.received: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.received.append(frame)


async def _link(*processors: FrameProcessor) -> None:
    """Link processors into a chain and start them with a StartFrame."""
    for upstream, downstream in zip(processors, processors[1:], strict=False):
        upstream.link(downstream)
    start = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    await processors[0].queue_frame(start)


class _FakeModel:
    """Hand-controlled stand-in for openwakeword.model.Model.

    ``scores`` is a list of per-call return values consumed in order by
    ``predict``; once exhausted it yields ``0.0``.  ``predict`` records the
    keyword args it was called with (so tests can assert patience/threshold are
    forwarded), and ``reset`` counts its invocations.
    """

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = list(scores or [])
        self.predict_calls: list[dict] = []
        self.reset_calls = 0

    def predict(self, _audio, **kwargs):  # noqa: ANN001
        self.predict_calls.append(kwargs)
        score = self._scores.pop(0) if self._scores else 0.0
        return {_MODEL_NAME: score}

    def reset(self) -> None:
        self.reset_calls += 1


class _Clock:
    """Mutable monotonic clock; ``set``/``advance`` drive deterministic time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _audio_frame(n_chunks: int = 1) -> InputAudioRawFrame:
    """One InputAudioRawFrame carrying ``n_chunks`` * 1280 int16 zero samples."""
    n = _CHUNK_SAMPLES * n_chunks
    pcm = struct.pack(f"{n}h", *([0] * n))
    return InputAudioRawFrame(audio=pcm, sample_rate=_SAMPLE_RATE, num_channels=1)


async def _make_gate(
    monkeypatch,
    clock: _Clock,
    model: _FakeModel,
    *,
    threshold: float = 0.5,
    sleep_timeout_s: float = 10.0,
) -> tuple[WakeWordGate, _Sink]:
    """Build a linked, started WakeWordGate + sink with the clock patched."""
    monkeypatch.setattr(wake_module, "monotonic", clock)
    gate = WakeWordGate(
        model=model,  # pyright: ignore[reportArgumentType]  # _FakeModel duck-types Model
        model_name=_MODEL_NAME,
        threshold=threshold,
        sleep_timeout_s=sleep_timeout_s,
        enable_direct_mode=True,
    )
    sink = _Sink()
    await _link(gate, sink)
    return gate, sink


def _audio_received(sink: _Sink) -> list[InputAudioRawFrame]:
    return [f for f in sink.received if isinstance(f, InputAudioRawFrame)]


def _text_received(sink: _Sink) -> list[str]:
    return [f.text for f in sink.received if isinstance(f, TextFrame)]


# --------------------------------------------------------------------------
# (a) Wake requires N consecutive frames above threshold before opening.
#
# The N-consecutive-frame requirement is enforced by OpenWakeWord's
# patience parameter, which the gate forwards on every predict() call.  The
# fake model emulates that policy: below-threshold for the first two chunks,
# above on the third.  The gate must stay asleep until predict() reports the
# hit, and must request patience=3.
# --------------------------------------------------------------------------


def test_wake_requires_consecutive_frames_above_threshold(monkeypatch):
    async def body():
        clock = _Clock()
        # Two sub-threshold chunks, then one over threshold (patience satisfied).
        model = _FakeModel(scores=[0.4, 0.49, 0.92])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        # Feed three 80ms chunks in a single frame; predict() runs once/chunk.
        await gate.process_frame(_audio_frame(n_chunks=3), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is True
        # patience and threshold forwarded to OpenWakeWord on every call.
        assert model.predict_calls[0]["patience"] == {_MODEL_NAME: 3}
        assert model.predict_calls[0]["threshold"] == {_MODEL_NAME: 0.5}
        # Exactly three predict() calls — one per 80ms chunk.
        assert len(model.predict_calls) == 3

    _run(body)


def test_no_wake_when_all_frames_below_threshold(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.4, 0.49, 0.1])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        await gate.process_frame(_audio_frame(n_chunks=3), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is False
        # All three chunks consumed, none crossed threshold → stays asleep.
        assert len(model.predict_calls) == 3
        assert gate.on_wake is None  # nothing fired

    _run(body)


def test_wake_fires_on_wake_callback_and_forwards_frame(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.8])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        woke: list[bool] = []
        gate.on_wake = lambda: woke.append(True)

        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is True
        assert woke == [True]
        # The triggering frame is forwarded so the following speech isn't lost.
        assert len(_audio_received(sink)) == 1
        # Wake stamps the clock so the sleep timer measures from wake time.
        assert gate._last_voice_activity == clock.now

    _run(body)


# --------------------------------------------------------------------------
# (d) Frames are dropped while asleep and passed while awake.
# --------------------------------------------------------------------------


def test_non_system_frame_dropped_while_asleep(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel()
        gate, sink = await _make_gate(monkeypatch, clock, model)

        assert gate._awake is False
        await gate.process_frame(TextFrame("ignored while asleep"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _text_received(sink) == []

    _run(body)


def test_non_system_frame_passed_while_awake(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        # Wake first.
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        await gate.process_frame(TextFrame("now I'm heard"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _text_received(sink) == ["now I'm heard"]

    _run(body)


def test_audio_forwarded_while_awake(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True
        n_after_wake = len(_audio_received(sink))

        # A subsequent audio frame (not enough silence to time out) is forwarded,
        # and is NOT fed to predict() while awake.
        calls_before = len(model.predict_calls)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert len(_audio_received(sink)) == n_after_wake + 1
        assert len(model.predict_calls) == calls_before  # no scoring while awake

    _run(body)


# --------------------------------------------------------------------------
# (b) The gate sleeps after WAKE_SLEEP_TIMEOUT_S of post-speech silence,
# and (c) the OpenWakeWord prediction buffer is reset on the sleep transition.
# --------------------------------------------------------------------------


def test_sleeps_after_timeout_and_resets_model(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)

        slept: list[bool] = []
        gate.on_sleep = lambda: slept.append(True)

        # Wake.
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True
        assert model.reset_calls == 0

        # Advance past the timeout, then push an audio frame to trigger the check.
        clock.advance(10.5)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is False
        # (c) prediction buffer reset exactly once on the sleep transition.
        assert model.reset_calls == 1
        assert slept == [True]

    _run(body)


def test_no_sleep_before_timeout(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        # Just under the timeout — must stay awake, no reset.
        clock.advance(9.9)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is True
        assert model.reset_calls == 0

    _run(body)


def test_no_sleep_while_user_speaking(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # User starts speaking; even long past the timeout the gate must not sleep.
        await gate.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        clock.advance(100.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is True
        assert gate._speaking is True
        assert model.reset_calls == 0

    _run(body)


def test_no_sleep_while_bot_speaking(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # Larry talking longer than the timeout must not put him to sleep.
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        clock.advance(100.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert gate._awake is True
        assert gate._bot_speaking is True
        assert model.reset_calls == 0

    _run(body)


def test_sleep_clock_restarts_when_bot_stops(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # Bot speaks for a long while, then stops — the clock restarts from stop.
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        clock.advance(100.0)
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert gate._last_voice_activity == clock.now

        # 9s of post-speech silence: still awake (full window from bot-stop).
        clock.advance(9.0)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        # Cross the timeout: now sleep.
        clock.advance(1.5)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is False
        assert model.reset_calls == 1

    _run(body)


# --------------------------------------------------------------------------
# Re-wake after sleep: a fresh wake word should reopen the gate.
# --------------------------------------------------------------------------


def test_rewakes_after_sleeping(monkeypatch):
    async def body():
        clock = _Clock()
        # First chunk wakes; after sleep, a later chunk wakes again.
        model = _FakeModel(scores=[0.9, 0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model, sleep_timeout_s=10.0)

        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        # Time out → sleep (the 0.9 here is for the timeout-check frame; it's
        # not scored because the gate is awake, so the model score list is
        # untouched until the next asleep chunk).
        clock.advance(10.5)
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is False

        # New wake word arrives while asleep.
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

    _run(body)


def test_sleep_now_transitions_to_asleep(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel(scores=[0.9])
        gate, sink = await _make_gate(monkeypatch, clock, model)

        # Wake first.
        await gate.process_frame(_audio_frame(n_chunks=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._awake is True

        slept: list[bool] = []
        gate.on_sleep = lambda: slept.append(True)

        gate.sleep_now()

        assert gate._awake is False
        assert slept == [True]
        # Model reset — mirrors the timeout path.
        assert model.reset_calls == 1

    _run(body)


def test_sleep_now_idempotent_when_already_asleep(monkeypatch):
    async def body():
        clock = _Clock()
        model = _FakeModel()
        gate, sink = await _make_gate(monkeypatch, clock, model)

        assert gate._awake is False

        slept: list[bool] = []
        gate.on_sleep = lambda: slept.append(True)

        gate.sleep_now()
        gate.sleep_now()

        # Already asleep: on_sleep must NOT fire (no double cue), model NOT reset.
        assert slept == []
        assert model.reset_calls == 0

    _run(body)

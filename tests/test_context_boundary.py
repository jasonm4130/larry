"""Task 6 — cross-speaker context boundary.

`make_boundary_snapshot_provider` wraps a per-turn speaker snapshot provider
(in production, `SpeakerIDProcessor.take_turn_snapshot`) so that whenever
continuity with the standing speaker breaks — a confirmed speaker change, or
an unconfirmed ('unknown') turn following a different standing speaker — every
raw user/assistant turn is dropped from the live `LLMContext` before the new
turn is appended. No retained tail: a retained turn from the previous speaker
*is* the leak.

Driven end-to-end through the real `SpeakerTagProcessor` + `LLMUserAggregator`
(nested inside a `Pipeline` so `pipecat.tests.utils.run_test` can drive it) so
the test exercises the same commit path production uses, not a hand-rolled
substitute.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.tests.utils import run_test

from larry.context_boundary import make_boundary_snapshot_provider
from larry.speaker_id import SpeakerTagProcessor


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="2026-07-17T00:00:00Z")


def _fixed_provider(value: str) -> Callable[[], Coroutine[Any, Any, str]]:
    async def provider() -> str:
        return value

    return provider


def _user_contents(context: LLMContext) -> list[str]:
    return [
        str(m.get("content"))
        for m in context.get_messages()
        if isinstance(m, dict) and m.get("role") == "user"
    ]


async def _send_turn(tagger_and_agg: Pipeline, text: str) -> None:
    await run_test(
        tagger_and_agg,
        frames_to_send=[_transcription(text)],
        expected_down_frames=None,
        send_end_frame=True,
    )


def _build(snapshot_value: dict[str, str]) -> tuple[Pipeline, LLMContext]:
    context = LLMContext(messages=[{"role": "system", "content": "sys"}])
    user_agg = LLMContextAggregatorPair(context).user()

    async def raw_provider() -> str:
        return snapshot_value["value"]

    boundary_provider = make_boundary_snapshot_provider(raw_provider, context)
    tagger = SpeakerTagProcessor(boundary_provider)
    pipeline = Pipeline([tagger, user_agg])
    return pipeline, context


def test_same_speaker_continues_without_reset():
    async def body():
        snap = {"value": "alice"}
        pipeline, context = _build(snap)

        await _send_turn(pipeline, "hi, alice here")
        await _send_turn(pipeline, "still alice")

        contents = _user_contents(context)
        assert any("hi, alice here" in c for c in contents)
        assert any("still alice" in c for c in contents)

    asyncio.run(body())


def test_confirmed_speaker_change_drops_prior_speakers_turns():
    """A turn following a confirmed speaker change carries zero raw content
    from the previous speaker (brief verify (a))."""

    async def body():
        snap = {"value": "alice"}
        pipeline, context = _build(snap)

        await _send_turn(pipeline, "hi, alice here")
        assert any("alice here" in c for c in _user_contents(context))

        snap["value"] = "bob"  # confirmed switch (Task 4 hysteresis already resolved this)
        await _send_turn(pipeline, "hi, bob here")

        contents = _user_contents(context)
        assert not any("alice" in c for c in contents)
        assert any("bob here" in c for c in contents)

    asyncio.run(body())


def test_unknown_turn_after_a_different_standing_speaker_drops_context():
    """A turn that reads 'unknown' after a different confirmed speaker was
    standing also carries zero raw content from that speaker (brief verify
    (b)) — reset on unproven continuity, not only on a confirmed switch."""

    async def body():
        snap = {"value": "alice"}
        pipeline, context = _build(snap)

        await _send_turn(pipeline, "hi, alice here")
        assert any("alice here" in c for c in _user_contents(context))

        snap["value"] = "unknown"  # new voice, not yet confirmed
        await _send_turn(pipeline, "someone new")

        contents = _user_contents(context)
        assert not any("alice" in c for c in contents)
        assert any("someone new" in c for c in contents)

    asyncio.run(body())


def test_consecutive_unknown_turns_do_not_repeatedly_reset():
    """Two 'unknown' turns in a row (nobody yet identified) is not a change in
    the standing speaker, so the second turn's own content is not wiped out
    by a redundant reset."""

    async def body():
        snap = {"value": "unknown"}
        pipeline, context = _build(snap)

        await _send_turn(pipeline, "first unknown turn")
        await _send_turn(pipeline, "second unknown turn")

        contents = _user_contents(context)
        assert any("first unknown turn" in c for c in contents)
        assert any("second unknown turn" in c for c in contents)

    asyncio.run(body())


def test_system_message_always_survives_a_reset():
    async def body():
        snap = {"value": "alice"}
        pipeline, context = _build(snap)

        await _send_turn(pipeline, "hi, alice here")
        snap["value"] = "bob"
        await _send_turn(pipeline, "hi, bob here")

        roles = [m.get("role") for m in context.get_messages() if isinstance(m, dict)]
        assert roles.count("system") == 1

    asyncio.run(body())

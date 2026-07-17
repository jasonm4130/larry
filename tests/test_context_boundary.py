"""Task 6 — cross-speaker context boundary.

`make_context_boundary` returns a function applied by `SpeakerTagProcessor` to
each turn's resolved speaker snapshot so that whenever continuity with the
standing speaker breaks — a confirmed speaker change, or an unconfirmed
('unknown') turn following a different standing speaker — every raw
user/assistant turn is dropped from the live `LLMContext` before the new turn
is appended. No retained tail: a retained turn from the previous speaker *is*
the leak.

Driven end-to-end through the real `SpeakerTagProcessor` + `LLMUserAggregator`
(nested inside a `Pipeline` so `pipecat.tests.utils.run_test` can drive it) so
the test exercises the same commit path production uses, not a hand-rolled
substitute. Each turn's identity arrives in-band as an `IdentitySnapshotFrame`
ahead of its transcript, exactly as `SpeakerIDProcessor` emits it.
"""

import asyncio

from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.tests.utils import run_test

from larry.context_boundary import make_context_boundary
from larry.speaker_id import IdentitySnapshotFrame, SpeakerTagProcessor


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="2026-07-17T00:00:00Z")


def _identity(value: str) -> IdentitySnapshotFrame:
    fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    fut.set_result(value)
    return IdentitySnapshotFrame(snapshot=fut)


def _user_contents(context: LLMContext) -> list[str]:
    return [
        str(m.get("content"))
        for m in context.get_messages()
        if isinstance(m, dict) and m.get("role") == "user"
    ]


async def _send_turn(pipeline: Pipeline, snapshot: str, text: str) -> None:
    """One turn: its identity marker, then its transcript."""
    await run_test(
        pipeline,
        frames_to_send=[_identity(snapshot), _transcription(text)],
        expected_down_frames=None,
        send_end_frame=True,
    )


def _build() -> tuple[Pipeline, LLMContext]:
    context = LLMContext(messages=[{"role": "system", "content": "sys"}])
    user_agg = LLMContextAggregatorPair(context).user()
    tagger = SpeakerTagProcessor(make_context_boundary(context))
    pipeline = Pipeline([tagger, user_agg])
    return pipeline, context


def test_same_speaker_continues_without_reset():
    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "alice", "hi, alice here")
        await _send_turn(pipeline, "alice", "still alice")

        contents = _user_contents(context)
        assert any("hi, alice here" in c for c in contents)
        assert any("still alice" in c for c in contents)

    asyncio.run(body())


def test_confirmed_speaker_change_drops_prior_speakers_turns():
    """A turn following a confirmed speaker change carries zero raw content
    from the previous speaker (brief verify (a))."""

    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "alice", "hi, alice here")
        assert any("alice here" in c for c in _user_contents(context))

        await _send_turn(pipeline, "bob", "hi, bob here")  # confirmed switch

        contents = _user_contents(context)
        assert not any("alice" in c for c in contents)
        assert any("bob here" in c for c in contents)

    asyncio.run(body())


def test_unknown_turn_after_a_different_standing_speaker_drops_context():
    """A turn that reads 'unknown' after a different confirmed speaker was
    standing also carries zero raw content from that speaker (brief verify
    (b)) — reset on unproven continuity, not only on a confirmed switch."""

    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "alice", "hi, alice here")
        assert any("alice here" in c for c in _user_contents(context))

        await _send_turn(pipeline, "unknown", "someone new")  # new voice, not yet confirmed

        contents = _user_contents(context)
        assert not any("alice" in c for c in contents)
        assert any("someone new" in c for c in contents)

    asyncio.run(body())


def test_consecutive_unknown_turns_do_not_repeatedly_reset():
    """Two 'unknown' turns in a row (nobody yet identified) is not a change in
    the standing speaker, so the second turn's own content is not wiped out
    by a redundant reset."""

    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "unknown", "first unknown turn")
        await _send_turn(pipeline, "unknown", "second unknown turn")

        contents = _user_contents(context)
        assert any("first unknown turn" in c for c in contents)
        assert any("second unknown turn" in c for c in contents)

    asyncio.run(body())


def test_system_message_always_survives_a_reset():
    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "alice", "hi, alice here")
        await _send_turn(pipeline, "bob", "hi, bob here")

        roles = [m.get("role") for m in context.get_messages() if isinstance(m, dict)]
        assert roles.count("system") == 1

    asyncio.run(body())


def test_injected_memory_system_messages_are_dropped_on_reset():
    """Regression: `ScopedMem0MemoryService` injects each turn's retrieved
    Mem0 memories as an *additional* system message (position 1, not the
    base prompt at position 0). A reset that kept every role=='system'
    message would retain alice's injected memories across the boundary into
    bob's turn — the exact cross-speaker leak this module exists to close,
    resurfacing through the memory-injection channel instead of raw turns.
    """

    async def body():
        pipeline, context = _build()

        await _send_turn(pipeline, "alice", "hi, alice here")
        # Simulate ScopedMem0MemoryService inserting alice's retrieved
        # memories as a system message alongside the base prompt.
        messages = context.get_messages()
        messages.insert(1, {"role": "system", "content": "alice's private facts"})
        context.set_messages(messages)

        await _send_turn(pipeline, "bob", "hi, bob here")  # confirmed switch

        remaining = context.get_messages()
        system_contents = [
            m.get("content") for m in remaining if isinstance(m, dict) and m.get("role") == "system"
        ]
        assert system_contents == ["sys"]  # only the base prompt survives
        assert not any(
            "alice" in str(m.get("content", "")) for m in remaining if isinstance(m, dict)
        )

    asyncio.run(body())

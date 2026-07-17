"""Task 5 — one immutable per-turn speaker identity, threaded to every consumer.

Covers the snapshot side of Task 5:
  * the ``[speaker: …]`` tag format round-trip (speaker_tag.py),
  * ``SpeakerTagProcessor`` prefixing the tag onto a downstream STT
    ``TranscriptionFrame`` and it surviving ``WhisperHallucinationFilter``,
  * the tag landing in the real LLM context user message (regression guard for
    finding #2: an upstream tagger's tag never reached the LLM),
  * ``SpeakerIDProcessor.take_turn_snapshot`` returning this turn's *frozen*
    identity, not a live ``current_speaker`` read (Codex P2 / test c),
  * an unconfirmed new-speaker turn snapshotting ``unknown`` (test e).

Self-contained: the SpeakerEmbedder is monkeypatched so no torch model loads.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from pipecat.frames.frames import Frame, StartFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test

import larry.speaker_id as speaker_id
from larry.processors import WhisperHallucinationFilter
from larry.speaker_id import SpeakerIDProcessor, SpeakerTagProcessor
from larry.speaker_tag import format_speaker_tag, parse_speaker_tag

# --------------------------------------------------------------------------
# Tag format round-trip
# --------------------------------------------------------------------------


def test_tag_round_trips():
    tagged = format_speaker_tag("jason", "hey larry")
    assert tagged == "[speaker: jason] hey larry"
    assert parse_speaker_tag(tagged) == ("jason", "hey larry")


def test_parse_untagged_returns_none_speaker():
    # No tag (the streaming-STT path) must be distinguishable from [speaker: unknown].
    assert parse_speaker_tag("just words") == (None, "just words")


def test_parse_unknown_tag():
    assert parse_speaker_tag("[speaker: unknown] hi") == ("unknown", "hi")


# --------------------------------------------------------------------------
# Test harness (mirrors tests/test_processors.py) for the tagger + filter
# --------------------------------------------------------------------------


def _run(coro: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(coro())


class _Sink(FrameProcessor):
    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.received: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.received.append(frame)


async def _link(*processors: FrameProcessor) -> None:
    for upstream, downstream in zip(processors, processors[1:], strict=False):
        upstream.link(downstream)
    await processors[0].queue_frame(
        StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    )


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="2026-07-17T00:00:00Z")


def _transcriptions(sink: _Sink) -> list[str]:
    return [f.text for f in sink.received if isinstance(f, TranscriptionFrame)]


def _fixed_provider(value: str) -> Callable[[], Coroutine[Any, Any, str]]:
    async def provider() -> str:
        return value

    return provider


# --------------------------------------------------------------------------
# (a) SpeakerTagProcessor tags the transcript and it survives the filter
# --------------------------------------------------------------------------


def test_tagger_prefixes_snapshot_onto_downstream_transcript():
    async def body():
        tagger = SpeakerTagProcessor(_fixed_provider("alice"), enable_direct_mode=True)
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, filt, sink)

        # STT emits the transcript *downstream* of the tagger (the bug in
        # finding #2 was the tagger sitting upstream of STT and never seeing it).
        await tagger.process_frame(_transcription("hello larry"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # Tag is applied AND survives WhisperHallucinationFilter's non-mutating strip.
        assert _transcriptions(sink) == ["[speaker: alice] hello larry"]

    _run(body)


def test_tagger_tags_unknown_when_snapshot_unknown():
    async def body():
        tagger = SpeakerTagProcessor(_fixed_provider("unknown"), enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, sink)

        await tagger.process_frame(_transcription("who are you"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["[speaker: unknown] who are you"]

    _run(body)


def test_tagged_transcription_lands_in_llm_context_user_message():
    """The tag must reach the LLM: a tagged TranscriptionFrame fed to the real
    user aggregator produces a context user message that still carries the tag."""

    async def body():
        context = LLMContext(messages=[{"role": "system", "content": "sys"}])
        user_agg = LLMContextAggregatorPair(context).user()

        await run_test(
            user_agg,
            frames_to_send=[_transcription("[speaker: alice] hello larry")],
            expected_down_frames=None,
            send_end_frame=True,
        )

        user_msgs = [
            m for m in context.get_messages() if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert any(m.get("content") == "[speaker: alice] hello larry" for m in user_msgs)

    asyncio.run(body())


# --------------------------------------------------------------------------
# (c) take_turn_snapshot returns the frozen per-turn identity, not a live read
# --------------------------------------------------------------------------


class _FakeEmbedder:
    name = "resemblyzer"
    next_embedding: np.ndarray = np.zeros(4, dtype=np.float32)

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        return self.next_embedding


def _make_proc(monkeypatch, tmp_path, **kwargs) -> SpeakerIDProcessor:
    monkeypatch.setattr(speaker_id, "get_speaker_embedder", lambda name: _FakeEmbedder())
    return SpeakerIDProcessor(speakers_db_path=tmp_path / "speakers.db", **kwargs)


def test_take_turn_snapshot_uses_frozen_task_not_live_current_speaker(monkeypatch, tmp_path):
    async def body():
        proc = _make_proc(monkeypatch, tmp_path)

        async def _resolve(value: str) -> str:
            return value

        # This turn's own identification resolved to alice.
        proc._turn_identify_task = asyncio.create_task(_resolve("alice"))
        # ...but a *later* turn has since flipped the confirmed speaker to bob.
        proc._current_speaker = "bob"

        # The snapshot for this (already-closed) turn must be alice — read from
        # the frozen per-turn task, NOT from the now-flipped current_speaker.
        assert await proc.take_turn_snapshot() == "alice"

    asyncio.run(body())


def test_take_turn_snapshot_no_task_fails_closed_to_unknown(monkeypatch, tmp_path):
    async def body():
        proc = _make_proc(monkeypatch, tmp_path)
        proc._current_speaker = "alice"  # a confirmed speaker is standing...
        proc._turn_identify_task = None  # ...but this transcript had no turn task.
        assert await proc.take_turn_snapshot() == "unknown"  # never inherit

    asyncio.run(body())


# --------------------------------------------------------------------------
# (e) an unconfirmed new-speaker turn snapshots unknown, never the prior speaker
# --------------------------------------------------------------------------


def test_unconfirmed_new_speaker_turn_snapshots_unknown(monkeypatch, tmp_path):
    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)

        def embed_by_content(audio_f32_16k):
            return alice if audio_f32_16k[0] > 0 else bob

        fake = _FakeEmbedder()
        fake.embed = embed_by_content
        monkeypatch.setattr(speaker_id, "get_speaker_embedder", lambda name: fake)
        proc = SpeakerIDProcessor(speakers_db_path=tmp_path / "speakers.db")  # change_turns=2
        proc._enrolled = {"alice": alice, "bob": bob}

        a_bytes = np.array([10000, 0], dtype=np.int16).tobytes()  # → alice
        b_bytes = np.array([-10000, 0], dtype=np.int16).tobytes()  # → bob

        # Confirm alice over two turns.
        await proc._identify_turn(a_bytes)
        assert await proc._identify_turn(a_bytes) == "alice"
        assert proc.current_speaker == "alice"

        # A new voice (bob) speaks: the turn is unconfirmed, so its snapshot must
        # be 'unknown' — never the previously-confirmed alice — and the confirmed
        # speaker must not flip on this single turn.
        assert await proc._identify_turn(b_bytes) == "unknown"
        assert proc.current_speaker == "alice"

    asyncio.run(body())


def test_tagger_tags_each_frame_independently_after_flip():
    """A late flip cannot re-attribute an already-closed turn: frame A stays
    tagged with its own snapshot even after the provider starts returning bob."""

    async def body():
        current = {"value": "alice"}

        async def provider() -> str:
            return current["value"]

        tagger = SpeakerTagProcessor(provider, enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, sink)

        frame_a = _transcription("first")
        await tagger.process_frame(frame_a, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        current["value"] = "bob"  # a later turn flips the identity
        frame_b = _transcription("second")
        await tagger.process_frame(frame_b, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # Frame A keeps alice (already closed); only frame B gets bob.
        assert _transcriptions(sink) == ["[speaker: alice] first", "[speaker: bob] second"]

    _run(body)

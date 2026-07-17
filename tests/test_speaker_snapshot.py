"""Task 5 — one immutable per-turn speaker identity, threaded to every consumer.

Covers the snapshot side of Task 5, as re-worked to carry each turn's identity
in-band on an ``IdentitySnapshotFrame`` (emitted by ``SpeakerIDProcessor`` at
the VAD boundary, consumed by the downstream ``SpeakerTagProcessor``):

  * the ``[speaker: …]`` tag format round-trip (speaker_tag.py),
  * ``SpeakerTagProcessor`` prefixing the tag onto a downstream STT
    ``TranscriptionFrame`` and it surviving ``WhisperHallucinationFilter``,
  * the tag landing in the real LLM context user message (regression guard for
    finding #2: an upstream tagger's tag never reached the LLM),
  * the tagger's pending FIFO binding each transcript to its *own* turn's
    identification, not a live ``current_speaker`` a later turn flipped
    (Codex P2), and surviving two turns closing before either transcript,
  * a fully-muted bot-echo turn (VAD-stop with no transcript) emitting NO
    marker, so it can never desync attribution for later turns — the Critical
    the in-band redesign closes,
  * an unconfirmed new-speaker turn snapshotting ``unknown`` (never inherited).

Self-contained: the SpeakerEmbedder is monkeypatched so no torch model loads.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test

import larry.speaker_id as speaker_id
from larry.processors import WhisperHallucinationFilter
from larry.speaker_id import IdentitySnapshotFrame, SpeakerIDProcessor, SpeakerTagProcessor
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


def _resolved(value: str) -> asyncio.Future[str]:
    """A pre-resolved future — the snapshot an IdentitySnapshotFrame carries for
    a turn we already know the identity of (must be built inside a running loop)."""
    fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    fut.set_result(value)
    return fut


def _identity(value: str) -> IdentitySnapshotFrame:
    return IdentitySnapshotFrame(snapshot=_resolved(value))


def _transcriptions(sink: _Sink) -> list[str]:
    return [f.text for f in sink.received if isinstance(f, TranscriptionFrame)]


def _passthrough(snapshot: str) -> str:
    return snapshot


# --------------------------------------------------------------------------
# (a) SpeakerTagProcessor tags the transcript and it survives the filter
# --------------------------------------------------------------------------


def test_tagger_prefixes_snapshot_onto_downstream_transcript():
    async def body():
        tagger = SpeakerTagProcessor(_passthrough, enable_direct_mode=True)
        filt = WhisperHallucinationFilter(enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, filt, sink)

        # The turn's identity arrives in-band, ahead of its transcript...
        await tagger.process_frame(_identity("alice"), FrameDirection.DOWNSTREAM)
        # ...then STT emits the transcript *downstream* of the tagger (the bug in
        # finding #2 was the tagger sitting upstream of STT and never seeing it).
        await tagger.process_frame(_transcription("hello larry"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        # Tag is applied AND survives WhisperHallucinationFilter's non-mutating strip.
        assert _transcriptions(sink) == ["[speaker: alice] hello larry"]

    _run(body)


def test_tagger_tags_unknown_when_no_pending_snapshot_fails_closed():
    async def body():
        tagger = SpeakerTagProcessor(_passthrough, enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, sink)

        # No IdentitySnapshotFrame arrived for this transcript (e.g. its turn was
        # never identified) — the tagger must fail closed to 'unknown', never
        # inherit a prior turn's name.
        await tagger.process_frame(_transcription("who are you"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["[speaker: unknown] who are you"]

    _run(body)


def test_tagger_pops_pending_snapshots_in_fifo_order():
    """Two turns close (two IdentitySnapshotFrames arrive) before either
    transcript does — the tagger must bind each transcript to its OWN turn's
    identity in order, not to whichever marker arrived last (Codex P2)."""

    async def body():
        tagger = SpeakerTagProcessor(_passthrough, enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, sink)

        await tagger.process_frame(_identity("alice"), FrameDirection.DOWNSTREAM)
        await tagger.process_frame(_identity("bob"), FrameDirection.DOWNSTREAM)
        await tagger.process_frame(_transcription("first"), FrameDirection.DOWNSTREAM)
        await tagger.process_frame(_transcription("second"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["[speaker: alice] first", "[speaker: bob] second"]

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
# (c) end-to-end: SpeakerIDProcessor -> SpeakerTagProcessor via the in-band marker
# --------------------------------------------------------------------------


class _FakeEmbedder:
    name = "resemblyzer"
    next_embedding: np.ndarray = np.zeros(4, dtype=np.float32)

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        return self.next_embedding


def _by_content_embedder(monkeypatch, alice: np.ndarray, bob: np.ndarray) -> None:
    fake = _FakeEmbedder()
    # First PCM sample sign selects the speaker, so a turn's audio bytes decide
    # its identity — lets a test thread distinct speakers through the real embed.
    fake.embed = lambda audio_f32_16k: alice if audio_f32_16k[0] > 0 else bob
    monkeypatch.setattr(speaker_id, "get_speaker_embedder", lambda name: fake)


def _audio(sample: int) -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=np.array([sample, 0], dtype=np.int16).tobytes(), sample_rate=16000, num_channels=1
    )


_ALICE_SAMPLE = 10000  # -> alice
_BOB_SAMPLE = -10000  # -> bob


def _real_turn(sample: int) -> list[Frame]:
    """The frame sequence for a normal (voiced, un-muted) user turn."""
    return [VADUserStartedSpeakingFrame(), _audio(sample), VADUserStoppedSpeakingFrame()]


def _muted_echo_turn(sample: int) -> list[Frame]:
    """A bot-echo turn: STT muted for its whole duration, so it yields no
    transcript and must yield no IdentitySnapshotFrame."""
    return [
        STTMuteFrame(mute=True),
        VADUserStartedSpeakingFrame(),
        _audio(sample),
        VADUserStoppedSpeakingFrame(),
        STTMuteFrame(mute=False),
    ]


async def _drive(proc: SpeakerIDProcessor, frames: list[Frame]) -> list[str]:
    """Run *frames* through proc -> tagger -> hallucination filter (the real
    post-STT order) and return the surviving tagged transcripts. The filter
    drops empty transcripts, exactly as it does in production, so the empty
    transcript push_empty_transcripts emits for a silent/echo turn never shows
    up here."""
    tagger = SpeakerTagProcessor(_passthrough)
    pipeline = Pipeline([proc, tagger, WhisperHallucinationFilter()])
    down, _ = await run_test(
        pipeline,
        frames_to_send=frames,
        expected_down_frames=None,
        send_end_frame=True,
    )
    return [f.text for f in down if isinstance(f, TranscriptionFrame)]


async def _markers(proc: SpeakerIDProcessor, frames: list[Frame]) -> int:
    """Count the IdentitySnapshotFrames proc emits for *frames* (no tagger, so
    the markers flow through to the collected downstream frames)."""
    down, _ = await run_test(
        Pipeline([proc]),
        frames_to_send=frames,
        expected_down_frames=None,
        send_end_frame=True,
    )
    return sum(isinstance(f, IdentitySnapshotFrame) for f in down)


def _make_id_proc(monkeypatch, tmp_path, alice, bob, **kwargs) -> SpeakerIDProcessor:
    _by_content_embedder(monkeypatch, alice, bob)
    proc = SpeakerIDProcessor(speakers_db_path=tmp_path / "speakers.db", change_turns=1, **kwargs)
    proc._enrolled = {"alice": alice, "bob": bob}
    return proc


def test_each_transcript_bound_to_its_own_turn(monkeypatch, tmp_path):
    """Two real turns (alice then bob): each transcript is tagged with its own
    turn's identity, threaded end-to-end through the in-band marker."""

    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        proc = _make_id_proc(monkeypatch, tmp_path, alice, bob)

        tagged = await _drive(
            proc,
            [
                *_real_turn(_ALICE_SAMPLE),
                _transcription("hi it's alice"),
                *_real_turn(_BOB_SAMPLE),
                _transcription("bob here"),
            ],
        )
        assert tagged == ["[speaker: alice] hi it's alice", "[speaker: bob] bob here"]

    asyncio.run(body())


def test_racing_vad_stops_preserve_turn_order(monkeypatch, tmp_path):
    """Both turns close (two markers emitted) before either transcript arrives —
    order, not recency, must bind each transcript to its turn (Codex P2)."""

    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        proc = _make_id_proc(monkeypatch, tmp_path, alice, bob)

        tagged = await _drive(
            proc,
            [
                *_real_turn(_ALICE_SAMPLE),
                *_real_turn(_BOB_SAMPLE),
                _transcription("first"),
                _transcription("second"),
            ],
        )
        assert tagged == ["[speaker: alice] first", "[speaker: bob] second"]

    asyncio.run(body())


def test_marker_emitted_once_per_vad_stop(monkeypatch, tmp_path):
    """Regression (Critical / P1-a): exactly one IdentitySnapshotFrame per
    VAD-stop — a muted bot-echo turn included. Paired with push_empty_transcripts
    (one transcript per VAD turn), that keeps markers and transcripts 1:1 so the
    tagger's pending FIFO can never drift. A design that skipped the marker for
    muted (or empty-transcript) turns would emit fewer markers than transcripts
    and permanently mis-align attribution."""

    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        proc = _make_id_proc(monkeypatch, tmp_path, alice, bob)

        count = await _markers(
            proc,
            [
                *_real_turn(_ALICE_SAMPLE),
                *_muted_echo_turn(_ALICE_SAMPLE),  # muted echo STILL emits a marker
                *_real_turn(_BOB_SAMPLE),
            ],
        )
        assert count == 3  # one per VAD-stop: alice, muted echo, bob

    asyncio.run(body())


def test_muted_echo_turn_does_not_desync_attribution(monkeypatch, tmp_path):
    """Regression (Critical): a muted bot-echo turn emits a marker (resolving to
    'unknown', no embed) AND — via push_empty_transcripts — an empty transcript
    that pops it. Markers and transcripts stay 1:1, so bob's turn after the echo
    is still tagged bob, not shifted onto the echo's (or alice's) identity. If
    the echo emitted a transcript but no marker (or vice versa), the FIFO would
    drift and mis-attribute every later turn — the cross-speaker bug this branch
    exists to prevent."""

    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        proc = _make_id_proc(monkeypatch, tmp_path, alice, bob)

        tagged = await _drive(
            proc,
            [
                *_real_turn(_ALICE_SAMPLE),
                _transcription("alice speaking"),
                # Bot echo while muted: marker (unknown) + the empty transcript
                # push_empty_transcripts emits, which pops that marker and is then
                # dropped by the hallucination filter.
                *_muted_echo_turn(_ALICE_SAMPLE),
                _transcription(""),
                *_real_turn(_BOB_SAMPLE),
                _transcription("now bob"),
            ],
        )
        assert tagged == ["[speaker: alice] alice speaking", "[speaker: bob] now bob"]

    asyncio.run(body())


# --------------------------------------------------------------------------
# (e) an unconfirmed new-speaker turn snapshots unknown, never the prior speaker
# --------------------------------------------------------------------------


def test_unconfirmed_new_speaker_turn_snapshots_unknown(monkeypatch, tmp_path):
    async def body():
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        _by_content_embedder(monkeypatch, alice, bob)
        proc = SpeakerIDProcessor(speakers_db_path=tmp_path / "speakers.db")  # change_turns=2
        proc._enrolled = {"alice": alice, "bob": bob}

        a_bytes = np.array([10000, 0], dtype=np.int16).tobytes()  # -> alice
        b_bytes = np.array([-10000, 0], dtype=np.int16).tobytes()  # -> bob

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


def test_tagger_tags_each_frame_independently():
    """A late flip cannot re-attribute an already-closed turn: frame A stays
    tagged with its own snapshot even after a later turn's marker arrives."""

    async def body():
        tagger = SpeakerTagProcessor(_passthrough, enable_direct_mode=True)
        sink = _Sink()
        await _link(tagger, sink)

        await tagger.process_frame(_identity("alice"), FrameDirection.DOWNSTREAM)
        await tagger.process_frame(_transcription("first"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        await tagger.process_frame(_identity("bob"), FrameDirection.DOWNSTREAM)
        await tagger.process_frame(_transcription("second"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        assert _transcriptions(sink) == ["[speaker: alice] first", "[speaker: bob] second"]

    _run(body)

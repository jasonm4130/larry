"""Task 5 — the scoped Mem0 store: bind retrieval + store to the turn's frozen
speaker snapshot, and store ONLY the current turn's user message.

Guards the two Codex P1 findings on ``Mem0MemoryService``:
  1. the deferred store reads ``self.user_id`` *late* (when the background task
     runs), so a speaker change after queueing binds the store to the wrong
     person → we freeze the snapshot at queue time and pass it explicitly;
  2. the payload is built from the *whole shared context* (every user/assistant
     message), so one person's prior turns get stored under the next speaker →
     we build the payload from only the current turn's user message.

Self-contained: never constructs a real Mem0 client (no FastEmbed/Qdrant/torch).
The scoped I/O methods are exercised against a fake client on an instance built
with ``__new__`` so no heavy ``Memory.from_config`` runs.
"""

import asyncio
from typing import Any, cast

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.mem0.memory import Mem0MemoryService

from larry.memory import ScopedMem0MemoryService, scoped_turn


def test_scoped_service_overrides_base_process_frame():
    """Pin (plan risk note): the racy base handler — which reads self.user_id
    late and stores the whole shared context — must NOT be what runs. If a future
    pipecat-ai bump makes the subclass fall back to it, this trips."""
    assert ScopedMem0MemoryService.process_frame is not Mem0MemoryService.process_frame


class _FakeMemory:
    """Records Mem0 add()/search() calls so tests can assert the payload + id."""

    def __init__(self, search_result=None) -> None:
        self.add_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self._search_result = search_result or {"results": []}

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self._search_result


def _make_service(
    user_id="alice", search_result=None
) -> tuple[ScopedMem0MemoryService, _FakeMemory]:
    """A ScopedMem0MemoryService with a fake client, bypassing the heavy __init__.

    Attributes are configured through an ``Any`` alias so injecting the fake
    client / test doubles doesn't fight the checker's declared attribute types.
    """
    svc = ScopedMem0MemoryService.__new__(ScopedMem0MemoryService)
    fake = _FakeMemory(search_result=search_result)
    cfg: Any = svc
    cfg.memory_client = fake
    cfg.user_id = user_id
    cfg.agent_id = None
    cfg.run_id = None
    cfg.search_limit = 10
    cfg.search_threshold = 0.1
    cfg.api_version = "v2"
    cfg.system_prompt = "recall:\n"
    cfg.add_as_system_message = True
    cfg.position = 1
    cfg.last_query = None
    return svc, fake


# --------------------------------------------------------------------------
# scoped_turn: parse the frozen snapshot + isolate the current turn's message
# --------------------------------------------------------------------------


def test_scoped_turn_reads_latest_tagged_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[speaker: alice] earlier alice thing"},
        {"role": "assistant", "content": "boo"},
        {"role": "user", "content": "[speaker: bob] current bob thing"},
    ]
    assert scoped_turn(messages) == ("bob", "current bob thing")


def test_scoped_turn_unknown_when_untagged():
    messages = [{"role": "user", "content": "no tag here"}]
    assert scoped_turn(messages) == ("unknown", "no tag here")


def test_scoped_turn_none_when_no_user_message():
    messages = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "hi"}]
    assert scoped_turn(messages) is None


# --------------------------------------------------------------------------
# (d) payload contains ONLY the current turn's user message, no earlier speaker
# --------------------------------------------------------------------------


def test_store_payload_is_only_current_turn_user_message():
    svc, fake = _make_service(user_id="bob")
    binding = scoped_turn(
        [
            {"role": "user", "content": "[speaker: alice] alice's secret"},
            {"role": "assistant", "content": "mm"},
            {"role": "user", "content": "[speaker: bob] bob's words"},
        ]
    )
    assert binding is not None
    speaker, text = binding
    asyncio.run(svc._store_scoped([{"role": "user", "content": text}], speaker))

    assert len(fake.add_calls) == 1
    call = fake.add_calls[0]
    assert call["messages"] == [{"role": "user", "content": "bob's words"}]
    # No earlier speaker's message leaked into the payload.
    assert "alice's secret" not in str(call["messages"])
    assert call["user_id"] == "bob"


# --------------------------------------------------------------------------
# (b) the deferred store binds to the frozen snapshot, not a late self.user_id
# --------------------------------------------------------------------------


def test_store_uses_frozen_snapshot_even_when_user_id_changes_after():
    async def body():
        svc, fake = _make_service(user_id="alice")
        # Snapshot frozen for bob's turn (parsed at "queue" time).
        binding = scoped_turn([{"role": "user", "content": "[speaker: bob] hi"}])
        assert binding is not None and binding == ("bob", "hi")
        speaker, text = binding

        # A speaker change interleaves BEFORE the deferred store runs.
        svc.user_id = "carol"

        # The store still lands under bob (the turn's own snapshot), not carol.
        await svc._store_scoped([{"role": "user", "content": text}], speaker)

        assert fake.add_calls[0]["user_id"] == "bob"

    asyncio.run(body())


def test_retrieve_uses_explicit_snapshot_not_self_user_id():
    async def body():
        svc, fake = _make_service(user_id="carol")  # live user_id is stale/other
        await svc._retrieve_scoped("what did I say", "bob")
        assert fake.search_calls[0]["user_id"] == "bob"

    asyncio.run(body())


# --------------------------------------------------------------------------
# _scope_context_frame branch: unknown turns do NO Mem0 I/O; known turns scope
# both retrieval and store to the frozen snapshot.
# --------------------------------------------------------------------------


class _FakeContext:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self):
        return self._messages


def _instrument(svc: Any) -> tuple[list, list]:
    """Replace the two I/O collaborators with recorders; return their logs."""
    enhanced: list[tuple[str, str]] = []
    scheduled: list[str | None] = []

    async def fake_enhance(context, query, user_id):
        enhanced.append((query, user_id))

    def fake_create_task(coro, name=None):
        scheduled.append(name)
        coro.close()  # never actually run the store; avoid un-awaited warning

    svc._enhance_scoped = fake_enhance
    svc.create_task = fake_create_task
    return enhanced, scheduled


def test_scope_context_frame_skips_all_io_for_unknown_turn():
    svc, _ = _make_service()
    enhanced, scheduled = _instrument(svc)
    ctx = _FakeContext([{"role": "user", "content": "[speaker: unknown] hi larry"}])
    asyncio.run(svc._scope_context_frame(cast(LLMContext, ctx)))
    assert enhanced == [] and scheduled == []  # no retrieval, no store


def test_scope_context_frame_skips_when_no_user_message():
    svc, _ = _make_service()
    enhanced, scheduled = _instrument(svc)
    ctx = _FakeContext([{"role": "system", "content": "sys"}])
    asyncio.run(svc._scope_context_frame(cast(LLMContext, ctx)))
    assert enhanced == [] and scheduled == []


def test_scope_context_frame_scopes_known_speaker():
    svc, _ = _make_service()
    enhanced, scheduled = _instrument(svc)
    ctx = _FakeContext(
        [
            {"role": "user", "content": "[speaker: alice] earlier"},
            {"role": "user", "content": "[speaker: bob] current words"},
        ]
    )
    asyncio.run(svc._scope_context_frame(cast(LLMContext, ctx)))
    # Retrieval + store both scoped to bob, on bob's own current text only.
    assert enhanced == [("current words", "bob")]
    assert scheduled == ["mem0_store"]

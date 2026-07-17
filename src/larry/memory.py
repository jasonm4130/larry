"""Per-person memory via Mem0 (self-hosted) and a SQLite conversation log.

Embedding backend: FastEmbed (BAAI/bge-small-en-v1.5, ONNX, in-process).
  - Fully local, zero API spend, no key required.
  - ~80-120ms per query on a Pi 5; ~5-20ms on Apple Silicon.
  - 384-dim vectors. MTEB ~60% (vs ~62% for OpenAI text-embedding-3-small) -
    negligible for ~15-coworker fact retrieval.
Fact-extraction LLM: anthropic/claude-haiku-4-5 via OpenRouter.
  - Mem0's OpenAI provider auto-routes to OpenRouter when OPENROUTER_API_KEY is set.
  - Haiku is the cheapest capable model for short fact-extraction prompts.
Vector store: Qdrant with a local path under cfg.mem0_dir (no server needed).
"""

import asyncio
import contextlib
import sqlite3
from pathlib import Path
from typing import Any, cast

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.mem0.memory import Mem0MemoryService

from larry.config import Config
from larry.speaker_tag import parse_speaker_tag


def scoped_turn(context_messages: list[Any]) -> tuple[str, str] | None:
    """Resolve the current turn's ``(speaker, clean_user_text)`` from context.

    Reads the *latest* user message and parses the ``[speaker: …]`` tag frozen
    onto it at the turn boundary (by ``SpeakerTagProcessor``). Returns ``None``
    when the context has no string user message. ``speaker`` is ``'unknown'``
    when the message carries no tag (the streaming-STT path) or an explicit
    ``[speaker: unknown]`` — callers must skip all Mem0 I/O for it so every
    unidentified voice does not pile into one shared 'unknown' namespace.
    """
    for message in reversed(context_messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            name, text = parse_speaker_tag(message["content"])
            return (name or "unknown", text)
    return None


class ScopedMem0MemoryService(Mem0MemoryService):
    """Mem0 service that binds each turn to its *frozen* speaker snapshot.

    Fixes two Codex-P1 races in the base ``Mem0MemoryService`` for a
    multi-speaker deployment:

    1. **Deferred-store user_id race.** The base fires ``_store_messages`` as a
       background task that reads ``self.user_id`` *when it runs*; a speaker
       change between queueing and running binds the store to the wrong person.
       Here the snapshot is parsed from the turn's ``[speaker: …]`` tag at
       *queue* time and passed explicitly into the store.
    2. **Whole-context payload.** The base builds the store payload from *every*
       user/assistant message in the shared context, so one person's prior turns
       get stored under the next speaker. Here the payload is only the current
       turn's own user message.

    Retrieval uses the same frozen snapshot, and an ``unknown`` turn does no
    Mem0 I/O at all. Assumes a local ``mem0.Memory`` client (what
    ``make_memory_service`` always constructs).
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Scope memory I/O to the turn's frozen snapshot.

        Deliberately bypasses ``Mem0MemoryService.process_frame`` (its store is
        the racy, whole-context one) and runs only the ``FrameProcessor`` base
        bookkeeping, then reimplements the context-frame handling.
        """
        await FrameProcessor.process_frame(self, frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        try:
            await self._scope_context_frame(frame.context)
            await self.push_frame(frame)
        except Exception as e:
            await self.push_error(error_msg=f"Error processing with scoped Mem0: {e}", exception=e)
            await self.push_frame(frame)

    async def _scope_context_frame(self, context: LLMContext) -> None:
        """Retrieve + queue-store this turn, both bound to its frozen snapshot.

        Skips ALL Mem0 I/O for an ``unknown`` (or empty) turn so unidentified
        voices never share one persistent namespace (Codex audit P1). Only the
        current turn's own user message is stored, under the frozen snapshot.
        """
        binding = scoped_turn(context.get_messages())
        if binding is None:
            return
        speaker, text = binding
        if speaker == "unknown" or not text.strip():
            return
        await self._enhance_scoped(context, text, speaker)
        self.create_task(
            self._store_scoped([{"role": "user", "content": text}], speaker),
            name="mem0_store",
        )

    async def _retrieve_scoped(self, query: str, user_id: str) -> Any:
        """Search Mem0 under an explicit *user_id* (the turn's frozen snapshot)."""
        try:
            params: dict[str, Any] = {
                "query": query,
                "user_id": user_id,
                "limit": self.search_limit,
            }
            if self.agent_id:
                params["agent_id"] = self.agent_id
            if self.run_id:
                params["run_id"] = self.run_id
            return await asyncio.to_thread(lambda: self.memory_client.search(**params))
        except Exception as e:
            logger.error(f"Error retrieving memories from Mem0: {e}")
            return {"results": []}

    async def _store_scoped(self, messages: list[dict[str, Any]], user_id: str) -> None:
        """Store *messages* under an explicit *user_id* — never the late self.user_id."""
        try:
            params: dict[str, Any] = {
                "messages": messages,
                "metadata": {"platform": "pipecat"},
                "user_id": user_id,
            }
            if self.agent_id:
                params["agent_id"] = self.agent_id
            if self.run_id:
                params["run_id"] = self.run_id
            await asyncio.to_thread(lambda: self.memory_client.add(**params))
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")

    # Last (user_id, query) actually retrieved. The dedup key MUST include the
    # speaker: keyed on the query alone, a second speaker asking the identical
    # thing right after the first would be silently skipped and get no memories
    # (Codex P2). Class-level default so the __new__-built instances in tests and
    # the base-__init__ production path both start clean without an override.
    _last_scoped: tuple[str, str] | None = None

    async def _enhance_scoped(self, context: LLMContext, query: str, user_id: str) -> None:
        """Insert the *user_id*-scoped memories into the context before the LLM."""
        if self._last_scoped == (user_id, query):
            return
        self._last_scoped = (user_id, query)

        memories = await self._retrieve_scoped(query, user_id)
        results = memories.get("results", []) if isinstance(memories, dict) else memories
        if not results:
            return

        memory_text = self.system_prompt
        for i, memory in enumerate(results, 1):
            memory_text += f"{i}. {memory.get('memory', '')}\n\n"

        role = "system" if self.add_as_system_message else "user"
        memory_message = cast(LLMContextMessage, {"role": role, "content": memory_text})
        messages = context.get_messages()
        position = max(0, min(self.position, len(messages)))
        messages.insert(position, memory_message)
        context.set_messages(messages)


def make_memory_service(
    cfg: Config, *, user_id: str = "unknown", scoped: bool = True
) -> Mem0MemoryService:
    """Return a self-hosted Mem0MemoryService wired into the Pipecat pipeline.

    The service uses:
    - Qdrant (local path) for vector storage — persists across restarts.
    - FastEmbed (BAAI/bge-small-en-v1.5, 384-dim) for embeddings — fully local,
      no API key, ONNX runtime in-process.
    - anthropic/claude-haiku-4.5 via OpenRouter for fact extraction — requires
      OPENROUTER_API_KEY (Mem0's OpenAI provider auto-detects it from env).

    ``scoped`` selects ``ScopedMem0MemoryService`` (per-turn speaker binding —
    the default for the segmented-STT identity path). Pass ``scoped=False`` for
    the streaming-STT (xAI) path, which has no per-turn ``[speaker: …]`` tag to
    bind to and so falls back to the base service's single-namespace behaviour.

    Blocking Mem0 calls are already wrapped in asyncio.to_thread by the
    Pipecat plugin (pipecat.services.mem0.memory, resolved upstream issue #1741),
    so no additional wrapping is needed here.
    """
    cfg.mem0_dir.mkdir(parents=True, exist_ok=True)

    local_config: dict[str, Any] = {
        "llm": {
            # Mem0's OpenAI provider auto-routes to OpenRouter when
            # OPENROUTER_API_KEY is set in the environment — no explicit
            # api_key needed here.
            "provider": "openai",
            "config": {
                # anthropic/claude-haiku-4.5: $1/M in, $5/M out via OpenRouter.
                # Fact-extraction prompts are short (~200 tokens in, ~50 out),
                # so cost is well under a cent per conversation.
                "model": "anthropic/claude-haiku-4.5",
                "max_tokens": 2000,
            },
        },
        "embedder": {
            # FastEmbed: in-process ONNX, no API key, no recurring cost.
            # bge-small-en-v1.5: 384-dim, ~300MB on disk, ~80-120ms on Pi 5.
            "provider": "fastembed",
            "config": {
                "model": "BAAI/bge-small-en-v1.5",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "larry_mem0",
                "path": str(cfg.mem0_dir / "qdrant"),
                "embedding_model_dims": 384,
                "on_disk": True,
            },
        },
    }

    service_cls = ScopedMem0MemoryService if scoped else Mem0MemoryService
    return service_cls(
        local_config=local_config,
        user_id=user_id,
    )


class ConversationLog:
    """Append-only SQLite log of every conversation turn for debugging and replay."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            speaker TEXT NOT NULL,
            user_text TEXT NOT NULL,
            larry_text TEXT NOT NULL
        );
    """
    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_turns_speaker_ts ON turns(speaker, ts DESC);
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(self._CREATE_TABLE)
            self._conn.execute(self._CREATE_INDEX)

    def log_turn(self, speaker: str, user_text: str, larry_text: str) -> None:
        """Append one turn to the log."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO turns (speaker, user_text, larry_text) VALUES (?, ?, ?)",
                (speaker, user_text, larry_text),
            )

    def recent_turns(self, speaker: str, limit: int = 20) -> list[dict]:
        """Return the most recent turns for a given speaker, newest first."""
        cur = self._conn.execute(
            "SELECT id, ts, speaker, user_text, larry_text"
            " FROM turns WHERE speaker = ?"
            " ORDER BY ts DESC, rowid DESC LIMIT ?",
            (speaker, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

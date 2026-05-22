"""Per-person memory via Mem0 (self-hosted) and a SQLite conversation log.

Embedding backend: OpenAI text-embedding-3-small (cheap, no local inference).
Fact-extraction LLM: Anthropic claude-3-haiku-20240307 (fast, low-cost).
Vector store: Qdrant with a local path under cfg.mem0_dir (no server needed).
"""

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

from pipecat.services.mem0.memory import Mem0MemoryService


def make_memory_service(cfg: Any, *, user_id: str = "unknown") -> Mem0MemoryService:
    """Return a self-hosted Mem0MemoryService wired into the Pipecat pipeline.

    The service uses:
    - Qdrant (local path) for vector storage — persists across restarts.
    - OpenAI text-embedding-3-small for embeddings — no local model required.
    - Anthropic claude-3-haiku for fact extraction — uses cfg.anthropic_api_key.

    Blocking Mem0 calls are already wrapped in asyncio.to_thread by the
    Pipecat plugin (pipecat.services.mem0.memory, resolved upstream issue #1741),
    so no additional wrapping is needed here.
    """
    cfg.mem0_dir.mkdir(parents=True, exist_ok=True)

    local_config: dict[str, Any] = {
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": "claude-3-haiku-20240307",
                "api_key": cfg.anthropic_api_key,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                # text-embedding-3-small: ~$0.02 / 1M tokens — negligible.
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "larry_mem0",
                "path": str(cfg.mem0_dir / "qdrant"),
                "embedding_model_dims": 1536,
                "on_disk": True,
            },
        },
    }

    return Mem0MemoryService(
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
            " ORDER BY ts DESC LIMIT ?",
            (speaker, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

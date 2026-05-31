"""Speaker identification layer for Larry using Resemblyzer voice embeddings."""

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from resemblyzer import VoiceEncoder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors in [-1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to a float32 mono array in [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the speakers table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            name TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def load_enrolled(db_path: Path) -> dict[str, np.ndarray]:
    """Load all enrolled speakers from SQLite; return name → float32 embedding dict."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute("SELECT name, embedding FROM speakers").fetchall()
    return {name: np.frombuffer(blob, dtype=np.float32) for name, blob in rows}


class SpeakerIDProcessor(FrameProcessor):
    """Identify the active speaker from audio and tag TranscriptionFrames with their name."""

    def __init__(
        self,
        speakers_db_path: Path,
        on_speaker_change: Callable[[str], None] | None = None,
        match_threshold: float = 0.75,
        window_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self._db_path = speakers_db_path
        self._on_speaker_change = on_speaker_change
        self._match_threshold = match_threshold
        self._window_bytes = int(16000 * 2 * window_seconds)  # 16kHz int16 bytes

        self._enrolled: dict[str, np.ndarray] = load_enrolled(speakers_db_path)
        self._encoder: VoiceEncoder = VoiceEncoder()  # blocks on torch load — fail fast
        self._audio_buffer: bytearray = bytearray()
        self._current_speaker: str = "unknown"
        self._identify_task: asyncio.Task | None = None

        logger.info(
            f"SpeakerIDProcessor ready: {len(self._enrolled)} enrolled speaker(s), "
            f"threshold={match_threshold}, window={window_seconds}s"
        )

    @property
    def current_speaker(self) -> str:
        return self._current_speaker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._audio_buffer.extend(frame.audio)
            if len(self._audio_buffer) >= self._window_bytes:
                window = bytes(self._audio_buffer[: self._window_bytes])
                del self._audio_buffer[: self._window_bytes]
                # Fire-and-forget: Resemblyzer's torch embed is a 100-200ms
                # CPU stall.  Running it inline blocks the whole pipeline
                # (STT/LLM/TTS frames queue up behind us).  Skip new windows
                # while a prior embed is still running — torch state isn't
                # safe for concurrent calls, and "who is speaking right now"
                # is fine to sample every couple seconds rather than every
                # 1s window.
                if self._identify_task is None or self._identify_task.done():
                    self._identify_task = asyncio.create_task(
                        asyncio.to_thread(self._identify_speaker, window)
                    )
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            frame.text = f"[speaker: {self._current_speaker}] {frame.text}"
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    def _identify_speaker(self, pcm_window: bytes) -> None:
        """Embed one window of audio and update _current_speaker."""
        audio = pcm16_to_float32(pcm_window)
        embedding = np.asarray(self._encoder.embed_utterance(audio))

        if not self._enrolled:
            return

        best_name, best_score = max(
            ((name, cosine_similarity(embedding, emb)) for name, emb in self._enrolled.items()),
            key=lambda x: x[1],
        )

        new_speaker = best_name if best_score >= self._match_threshold else "unknown"
        if new_speaker != self._current_speaker:
            logger.info(
                f"Speaker change: {self._current_speaker!r} → {new_speaker!r} "
                f"(score={best_score:.3f})"
            )
            self._current_speaker = new_speaker
            if self._on_speaker_change:
                self._on_speaker_change(new_speaker)

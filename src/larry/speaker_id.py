"""Speaker identification layer for Larry using Resemblyzer voice embeddings."""

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from time import monotonic as _monotonic

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


def store_speaker(db_path: Path, name: str, embedding: np.ndarray) -> None:
    """Persist a speaker voiceprint (INSERT OR REPLACE on primary key name).

    Creates the database file and parent directories if absent. Both the CLI
    ``enroll`` command and the in-conversation voice-enroll path call this so
    there is exactly one storage code path.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)",
            (name, embedding.astype(np.float32).tobytes()),
        )
        conn.commit()


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

        # Capture state machine — arms on enroll_speaker tool call, accumulates
        # voiced audio post-BotStoppedSpeakingFrame, embeds on threshold.
        self._capture_state: str = "idle"   # "idle" | "armed" | "capturing"
        self._capture_name: str = ""
        self._capture_bytes: bytearray = bytearray()
        self._capture_voiced_bytes: int = 0
        self._capture_start: float = 0.0
        self._capture_target_voiced_s: float = 10.0
        self._capture_floor_voiced_s: float = 6.0
        self._capture_cap_wall_s: float = 20.0
        self._capture_embed_fn: Callable[[np.ndarray], np.ndarray] | None = None
        self._capture_db_path: Path | None = None

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

    # ------------------------------------------------------------------
    # Capture state machine
    # ------------------------------------------------------------------

    def arm_capture(
        self,
        name: str,
        *,
        embed_fn: Callable[[np.ndarray], np.ndarray],
        db_path: Path | None = None,
        target_voiced_s: float = 10.0,
        floor_voiced_s: float = 6.0,
        cap_wall_s: float = 20.0,
    ) -> None:
        """Arm a pending voiceprint capture for ``name``.

        Ignored if a capture is already in progress (one capture at a time).
        Accumulation only starts after ``bot_stopped_speaking()`` is called —
        so Larry's own "say this back to me" prompt can never pollute the print.

        ``embed_fn`` replaces direct Resemblyzer calls so the state machine is
        unit-testable without torch. In production, pipeline.py passes a lambda
        that calls ``self._encoder.embed_utterance``.
        """
        if self._capture_state != "idle":
            logger.info(
                "arm_capture(%r) ignored — capture already in state %r",
                name,
                self._capture_state,
            )
            return
        self._capture_name = name.strip()
        self._capture_state = "armed"
        self._capture_bytes = bytearray()
        self._capture_voiced_bytes = 0
        self._capture_start = _monotonic()
        self._capture_target_voiced_s = target_voiced_s
        self._capture_floor_voiced_s = floor_voiced_s
        self._capture_cap_wall_s = cap_wall_s
        self._capture_embed_fn = embed_fn
        self._capture_db_path = db_path if db_path is not None else self._db_path
        logger.info(
            "Capture armed for %r (target=%.0fs, cap=%.0fs)",
            name,
            target_voiced_s,
            cap_wall_s,
        )

    def bot_stopped_speaking(self) -> None:
        """Signal that the bot's TTS has finished — starts accumulation if armed."""
        if self._capture_state == "armed":
            self._capture_state = "capturing"
            self._capture_start = _monotonic()  # wall-clock cap measured from here
            logger.info("Capture accumulation started for %r", self._capture_name)

    def add_capture_audio(
        self,
        pcm_bytes: bytes,
        *,
        vad_voiced: bool,
        bot_speaking: bool,
    ) -> dict | None:
        """Feed a chunk of PCM audio into the capture accumulator.

        Returns a result dict ``{"status": "enrolled", "name": ...}`` on
        success, ``{"status": "failed", "reason": ...}`` on abort, or ``None``
        if still accumulating.  Only counts frames that are VAD-voiced AND bot-
        silent. Called from ``process_frame`` (or directly in tests).
        """
        if self._capture_state != "capturing":
            return None

        elapsed = _monotonic() - self._capture_start
        voiced_s = self._capture_voiced_bytes / (16000 * 2)  # int16 bytes → seconds

        # Cap exceeded with insufficient voiced audio → abort.
        if elapsed >= self._capture_cap_wall_s and voiced_s < self._capture_floor_voiced_s:
            logger.warning(
                "Capture aborted for %r: %.1fs voiced in %.1fs (floor=%.1fs, cap=%.1fs)",
                self._capture_name,
                voiced_s,
                elapsed,
                self._capture_floor_voiced_s,
                self._capture_cap_wall_s,
            )
            reason = (
                f"insufficient voiced audio ({voiced_s:.1f}s < "
                f"{self._capture_floor_voiced_s:.1f}s floor)"
            )
            self._capture_state = "idle"
            return {"status": "failed", "name": self._capture_name, "reason": reason}

        # Only accumulate bot-silent, VAD-voiced audio.
        if vad_voiced and not bot_speaking:
            self._capture_bytes.extend(pcm_bytes)
            self._capture_voiced_bytes += len(pcm_bytes)

        voiced_s = self._capture_voiced_bytes / (16000 * 2)

        # Target reached → embed + persist.
        if voiced_s >= self._capture_target_voiced_s:
            name = self._capture_name
            embed_fn = self._capture_embed_fn
            db_path = self._capture_db_path
            audio = pcm16_to_float32(bytes(self._capture_bytes))
            self._capture_state = "idle"

            if embed_fn is None or db_path is None:
                return {"status": "failed", "name": name, "reason": "embed_fn or db_path missing"}

            embedding = np.asarray(embed_fn(audio))
            store_speaker(db_path, name, embedding)
            # Reload in-memory enrolled set so new speaker is immediately identifiable.
            self._enrolled = load_enrolled(db_path)
            logger.info("Enrolled %r — %d speaker(s) now enrolled", name, len(self._enrolled))
            if self._on_speaker_change:
                self._on_speaker_change(name)
            return {"status": "enrolled", "name": name}

        return None

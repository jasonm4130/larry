"""Speaker identification layer for Larry, built on the SpeakerEmbedder interface.

The embedding model (Resemblyzer today; see speaker_embedder.py) is swappable
via SPEAKER_EMBEDDER — this module never calls a specific model's SDK.
"""

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from time import monotonic as _monotonic

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from larry.speaker_embedder import SpeakerEmbedder, get_speaker_embedder
from larry.voice_enroll import SAMPLE_RATE


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors in [-1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to a float32 mono array in [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the speakers table if needed; idempotently add later-added columns.

    ``embedder`` + ``dim`` namespace each voiceprint to the model that produced
    it — a different embedder is a different vector space (and often a
    different dimensionality), so a print must never be cosine-matched across
    embedders (see ``load_enrolled``'s ``embedder`` filter). Rows written
    before this migration default to 'resemblyzer'/256 — the only embedder
    ever shipped before it — so existing installs keep matching unmigrated.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            name TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TEXT,
            embedder TEXT NOT NULL DEFAULT 'resemblyzer',
            dim INTEGER NOT NULL DEFAULT 256
        )
    """)
    # Idempotent migrations for DBs created by an older schema.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(speakers)")}
    if "last_seen" not in existing_cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN last_seen TEXT")
    if "embedder" not in existing_cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN embedder TEXT NOT NULL DEFAULT 'resemblyzer'")
    if "dim" not in existing_cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN dim INTEGER NOT NULL DEFAULT 256")
    conn.commit()


def load_enrolled(db_path: Path, embedder: str | None = None) -> dict[str, np.ndarray]:
    """Load enrolled speakers from SQLite; return name → float32 embedding dict.

    When *embedder* is given, only voiceprints stored under that embedder name
    are returned — a print from a different embedder is a different vector
    space and must never be cosine-matched against it (fail closed to
    'unknown', not a garbage score). Pass None to load every print regardless
    of embedder (e.g. for inspection tooling).
    """
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        if embedder is None:
            rows = conn.execute("SELECT name, embedding FROM speakers").fetchall()
        else:
            rows = conn.execute(
                "SELECT name, embedding FROM speakers WHERE embedder = ?", (embedder,)
            ).fetchall()
    return {name: np.frombuffer(blob, dtype=np.float32) for name, blob in rows}


def touch_last_seen(db_path: Path, name: str, *, now: str | None = None) -> None:
    """Update the last_seen timestamp for *name* to *now* (ISO-8601).

    No-op if *name* is not in the DB (unknown speaker). Does not insert new rows.
    """
    import datetime as _dt

    stamp = now or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute("UPDATE speakers SET last_seen = ? WHERE name = ?", (stamp, name))
        conn.commit()


def load_last_seen(db_path: Path, name: str) -> str | None:
    """Return the stored last_seen ISO stamp for *name*, or None if absent/unknown."""
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT last_seen FROM speakers WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        return None
    return row[0]  # may be None (NULL) for rows enrolled before this migration


def store_speaker(
    db_path: Path, name: str, embedding: np.ndarray, *, embedder: str = "resemblyzer"
) -> None:
    """Persist a speaker voiceprint (INSERT OR REPLACE on primary key name).

    Creates the database file and parent directories if absent. Both the CLI
    ``enroll`` command and the in-conversation voice-enroll path call this so
    there is exactly one storage code path. ``embedder`` names the model that
    produced *embedding* (its dim is derived from the vector itself) — a
    future embedder swap means every speaker must re-enroll, since a print
    from one embedder is never matched under another (see ``load_enrolled``).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    emb = embedding.astype(np.float32)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        already = conn.execute("SELECT 1 FROM speakers WHERE name = ?", (name,)).fetchone()
        if already is not None:
            logger.warning(
                f"Re-enrolling {name!r} — overwriting the existing voiceprint for that name"
            )
        conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding, embedder, dim) VALUES (?, ?, ?, ?)",
            (name, emb.tobytes(), embedder, int(emb.shape[-1])),
        )
        conn.commit()


class SpeakerIDProcessor(FrameProcessor):
    """Identify the active speaker from audio and tag TranscriptionFrames with their name."""

    def __init__(
        self,
        speakers_db_path: Path,
        on_speaker_change: Callable[[str], None] | None = None,
        match_threshold: float = 0.75,
        change_turns: int = 2,
        margin: float = 0.06,
        embedder_name: str = "resemblyzer",
    ) -> None:
        super().__init__()
        self._db_path = speakers_db_path
        self._on_speaker_change = on_speaker_change
        self._match_threshold = match_threshold
        self._change_turns = change_turns
        self._margin = margin

        # Construct the embedder first — its .name selects which voiceprints
        # are even loadable, so a print from a different embedder is never
        # cosine-matched (fail closed to 'unknown', not a garbage score).
        self._encoder: SpeakerEmbedder = get_speaker_embedder(embedder_name)
        self._enrolled: dict[str, np.ndarray] = load_enrolled(
            speakers_db_path, embedder=self._encoder.name
        )

        # Turn-scoped identification: buffer this turn's voiced, bot-silent
        # audio (VAD-start -> VAD-stop) and embed it ONCE at VAD-stop, so the
        # embedding is inherently tied to its own turn — no rolling window that
        # straddles turns, no post-hoc result landing on the wrong turn.
        self._turn_audio: bytearray = bytearray()
        self._turn_identify_task: asyncio.Task | None = None

        # _current_speaker is the *confirmed* speaker: it switches only after
        # hysteresis commits a new one, never on a single stray turn.
        self._current_speaker: str = "unknown"
        # Hysteresis: how many consecutive turns the same named candidate has
        # won.  A switch is committed once the streak reaches _change_turns.
        self._pending_candidate: str = "unknown"
        self._pending_streak: int = 0

        # Shared across BOTH encoder call sites (the per-turn identify embed and
        # finalize_capture's enroll embed).  Cancelling an asyncio task that
        # wraps asyncio.to_thread() does NOT stop the torch call already running
        # in the worker thread, so a lock — not cancellation — is what keeps the
        # encoder from being entered concurrently (torch state isn't reentrant).
        self._encoder_lock = asyncio.Lock()

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
        # VAD / bot-speaking state tracked for capture + turn gating.
        self._vad_voiced: bool = False
        self._bot_speaking_for_capture: bool = False

        logger.info(
            f"SpeakerIDProcessor ready: embedder={self._encoder.name!r}, "
            f"{len(self._enrolled)} enrolled speaker(s), "
            f"threshold={match_threshold}, change_turns={change_turns}, margin={margin}"
        )

    @property
    def current_speaker(self) -> str:
        return self._current_speaker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            # Turn-scoped identification: accumulate only this turn's voiced,
            # bot-silent audio (mirrors the capture gate).  Bot-speaking audio
            # is TTS echo, not the speaker — never embed it.
            if self._vad_voiced and not self._bot_speaking_for_capture:
                self._turn_audio.extend(frame.audio)
            # Feed capture accumulator (if capturing) for every raw frame.
            if self._capture_state == "capturing":
                result = self.add_capture_audio(
                    frame.audio,
                    vad_voiced=self._vad_voiced,
                    bot_speaking=self._bot_speaking_for_capture,
                )
                if result is not None:
                    from larry.voice_enroll import ENROLL_CONFIRM, ENROLL_FAIL

                    if result["status"] == "ready":
                        await self.finalize_capture(result["name"], result["audio"])
                        await self.push_frame(TTSSpeakFrame(ENROLL_CONFIRM), direction)
                    else:
                        # "abort"
                        await self.push_frame(TTSSpeakFrame(ENROLL_FAIL), direction)
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking_for_capture = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking_for_capture = False
            self.bot_stopped_speaking()
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_voiced = True
            self._turn_audio = bytearray()  # start a fresh per-turn buffer
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_voiced = False
            # Turn boundary: embed THIS turn's own voiced audio once, off the
            # event loop.  The bytes are snapshotted here (passed by value into
            # the task), so a late-completing embed still attributes to this
            # turn, never a later one.  Task 5 awaits this task for the snapshot.
            turn_bytes = bytes(self._turn_audio)
            self._turn_audio = bytearray()
            # Skip identification while enrollment is in progress: the encoder is
            # busy with the enroll embed, and identifying the enrollment phrase is
            # pointless.  (_encoder_lock would serialize them safely anyway — this
            # just avoids the wasted embed and keeps enrollment turns out of the
            # hysteresis streak.)
            if self._capture_state == "idle":
                self._turn_identify_task = asyncio.create_task(self._identify_turn(turn_bytes))
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            frame.text = f"[speaker: {self._current_speaker}] {frame.text}"
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _identify_turn(self, pcm_bytes: bytes) -> str:
        """Embed this turn's own voiced audio once and resolve its speaker.

        Returns the turn's identity: the confirmed speaker when this turn
        confirms or continues them, else 'unknown' (fail closed — an
        unconfirmed turn never inherits the previously-confirmed speaker).

        ``pcm_bytes`` is this turn's audio, snapshotted at VAD-stop, so the
        result attributes to this turn even if the embed finishes late.  The
        encoder call is serialized with enrollment via ``_encoder_lock``.
        """
        if not pcm_bytes:
            # Silence turn — no voiced audio to embed.  Fail closed, no encoder.
            return self._resolve_identity(None)
        audio = pcm16_to_float32(pcm_bytes)
        async with self._encoder_lock:
            embedding = np.asarray(await asyncio.to_thread(self._encoder.embed, audio))
        return self._resolve_identity(embedding)

    def _candidate(self, embedding: np.ndarray) -> str:
        """The per-turn match decision: best enrolled name, or 'unknown'.

        Accepts the top match only when its cosine score clears
        ``_match_threshold`` AND (with >= 2 enrolled speakers) its top1-top2
        margin clears ``_margin``.  With < 2 enrolled speakers there is no
        runner-up, so the margin is undefined and waived — a fresh install with
        one voiceprint must still identify that speaker, not reject everyone.
        """
        if not self._enrolled:
            return "unknown"
        ranked = sorted(
            ((name, cosine_similarity(embedding, emb)) for name, emb in self._enrolled.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        best_name, best_score = ranked[0]
        if len(ranked) >= 2:
            margin: float | None = best_score - ranked[1][1]
            margin_ok = margin >= self._margin
        else:
            margin = None
            margin_ok = True  # single-candidate waiver
        candidate = best_name if (best_score >= self._match_threshold and margin_ok) else "unknown"
        # Diagnostic: log every turn's best candidate + score + margin, so
        # threshold/margin tuning is data-driven (we see near-misses, not just hits).
        margin_str = "n/a" if margin is None else f"{margin:.3f}"
        logger.info(
            f"Turn match: best={best_name!r} score={best_score:.3f} "
            f"thr={self._match_threshold:.2f} margin={margin_str} "
            f"(need {self._margin:.3f}) -> {candidate!r}"
        )
        return candidate

    def _resolve_identity(self, embedding: np.ndarray | None) -> str:
        """Run this turn's candidate through hysteresis; return the turn identity."""
        candidate = "unknown" if embedding is None else self._candidate(embedding)
        return self._apply_hysteresis(candidate)

    def _apply_hysteresis(self, candidate: str) -> str:
        """Update the consecutive-turn streak and, on confirmation, the speaker.

        The confirmed speaker (``_current_speaker``) switches only after
        ``_change_turns`` consecutive turns name the same speaker; until then an
        off-speaker turn snapshots 'unknown' rather than the prior confirmed
        name.  Returns this turn's identity snapshot.
        """
        # Track consecutive identical *named* candidates; an 'unknown' turn (or a
        # different name) breaks the run so the count is truly "consecutive".
        if candidate != "unknown" and candidate == self._pending_candidate:
            self._pending_streak += 1
        else:
            self._pending_candidate = candidate
            self._pending_streak = 1 if candidate != "unknown" else 0

        # Commit a switch once a named candidate has held for enough turns.
        if (
            candidate != "unknown"
            and self._pending_streak >= self._change_turns
            and candidate != self._current_speaker
        ):
            logger.info(
                f"Speaker confirmed: {self._current_speaker!r} -> {candidate!r} "
                f"(after {self._pending_streak} consecutive turn(s))"
            )
            self._current_speaker = candidate
            if self._on_speaker_change:
                self._on_speaker_change(candidate)

        # Fail closed: snapshot the confirmed speaker only when this turn's own
        # identification agrees with it; otherwise 'unknown' — never inherit.
        return self._current_speaker if candidate == self._current_speaker else "unknown"

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

        ``embed_fn`` replaces direct calls into the configured SpeakerEmbedder
        so the state machine is unit-testable without torch. In production,
        pipeline.py passes a lambda that calls ``self._encoder.embed``.
        """
        if self._capture_state != "idle":
            logger.info(
                f"arm_capture({name!r}) ignored — capture already in state {self._capture_state!r}"
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
            f"Capture armed for {name!r} (target={target_voiced_s:.0f}s, cap={cap_wall_s:.0f}s)"
        )

    def bot_stopped_speaking(self) -> None:
        """Signal that the bot's TTS has finished — starts accumulation if armed."""
        if self._capture_state == "armed":
            self._capture_state = "capturing"
            self._capture_start = _monotonic()  # wall-clock cap measured from here
            logger.info(f"Capture accumulation started for {self._capture_name!r}")

    def add_capture_audio(
        self,
        pcm_bytes: bytes,
        *,
        vad_voiced: bool,
        bot_speaking: bool,
    ) -> dict | None:
        """Feed a chunk of PCM audio into the capture accumulator.

        Returns one of:
          - ``None`` — still capturing
          - ``{"status": "ready",  "name": ..., "audio": <float32 ndarray>}``
            — enough voiced audio collected; caller must embed+store via
            ``finalize_capture``.
          - ``{"status": "abort",  "name": ..., "reason": ...}``
            — wall-clock cap expired with insufficient voiced audio; give up.

        On returning "ready" or "abort" the capture state transitions out of
        "capturing" so further frames are ignored.  Only VAD-voiced AND bot-
        silent audio is accumulated.  Called from ``process_frame`` (or
        directly in tests).
        """
        if self._capture_state != "capturing":
            return None

        elapsed = _monotonic() - self._capture_start
        voiced_s = self._capture_voiced_bytes / (SAMPLE_RATE * 2)  # int16 bytes → seconds

        # Wall-clock cap: always terminate when elapsed >= cap.
        if elapsed >= self._capture_cap_wall_s:
            name = self._capture_name
            if voiced_s >= self._capture_floor_voiced_s:
                # Floor met — encode what we have.
                audio = pcm16_to_float32(bytes(self._capture_bytes))
                self._capture_state = "embedding"
                logger.info(
                    f"Capture cap reached for {name!r}: {voiced_s:.1f}s voiced — "
                    f"ready (>= floor {self._capture_floor_voiced_s:.1f}s)"
                )
                return {"status": "ready", "name": name, "audio": audio}
            else:
                # Insufficient audio — abort.
                self._capture_state = "idle"
                reason = (
                    f"insufficient voiced audio ({voiced_s:.1f}s < "
                    f"{self._capture_floor_voiced_s:.1f}s floor)"
                )
                logger.warning(
                    f"Capture aborted for {name!r}: {voiced_s:.1f}s voiced in "
                    f"{elapsed:.1f}s (floor={self._capture_floor_voiced_s:.1f}s, "
                    f"cap={self._capture_cap_wall_s:.1f}s)"
                )
                return {"status": "abort", "name": name, "reason": reason}

        # Only accumulate bot-silent, VAD-voiced audio.
        if vad_voiced and not bot_speaking:
            self._capture_bytes.extend(pcm_bytes)
            self._capture_voiced_bytes += len(pcm_bytes)

        voiced_s = self._capture_voiced_bytes / (SAMPLE_RATE * 2)

        # Target voiced duration reached → ready to embed.
        if voiced_s >= self._capture_target_voiced_s:
            name = self._capture_name
            audio = pcm16_to_float32(bytes(self._capture_bytes))
            self._capture_state = "embedding"
            logger.info(f"Capture target reached for {name!r} ({voiced_s:.1f}s voiced)")
            return {"status": "ready", "name": name, "audio": audio}

        return None

    async def finalize_capture(self, name: str, audio: np.ndarray) -> dict:
        """Embed and persist a completed capture, off the event loop.

        Runs the configured SpeakerEmbedder's encode in a thread via
        ``asyncio.to_thread`` so the (typically torch-backed) encoder never
        blocks the pipeline event loop.  Reloads the in-memory enrolled set so
        the new speaker is immediately identifiable, fires
        ``_on_speaker_change`` if set, and returns
        ``{"status": "enrolled", "name": name}``.

        Must only be called after ``add_capture_audio`` returns "ready".
        """
        embed_fn = self._capture_embed_fn
        db_path = self._capture_db_path

        if embed_fn is None or db_path is None:
            self._capture_state = "idle"
            return {"status": "failed", "name": name, "reason": "embed_fn or db_path missing"}

        # Serialize with the per-turn identify embed via the shared encoder lock
        # (held across the to_thread await) — the encoder is not concurrency-safe.
        async with self._encoder_lock:
            embedding = np.asarray(await asyncio.to_thread(embed_fn, audio))
        store_speaker(db_path, name, embedding, embedder=self._encoder.name)
        # Reload in-memory enrolled set so new speaker is immediately identifiable.
        self._enrolled = load_enrolled(db_path, embedder=self._encoder.name)
        logger.info(f"Enrolled {name!r} — {len(self._enrolled)} speaker(s) now enrolled")
        if self._on_speaker_change:
            self._on_speaker_change(name)
        self._capture_state = "idle"
        return {"status": "enrolled", "name": name}

"""Speaker identification layer for Larry, built on the SpeakerEmbedder interface.

The embedding model (Resemblyzer today; see speaker_embedder.py) is swappable
via SPEAKER_EMBEDDER — this module never calls a specific model's SDK.
"""

import asyncio
import sqlite3
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic as _monotonic

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    STTMuteFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from larry.speaker_embedder import SpeakerEmbedder, get_speaker_embedder
from larry.speaker_tag import format_speaker_tag
from larry.voice_enroll import SAMPLE_RATE

# Upper bound on how long the SpeakerTagProcessor waits for a turn's own embed
# to resolve before failing closed to 'unknown'. The turn-scoped embed almost
# always finishes before the STT network round-trip returns, so this is a
# safety ceiling, not the expected wait. ponytail: fixed 3s ceiling; if real
# embeds ever approach it, tie it to the encoder latency budget instead.
_SNAPSHOT_AWAIT_TIMEOUT_S = 3.0


@dataclass
class IdentitySnapshotFrame(SystemFrame):
    """In-band carrier of one turn's frozen speaker identity.

    ``SpeakerIDProcessor`` emits this at the VAD turn boundary and the
    downstream ``SpeakerTagProcessor`` consumes it, so a turn's identity travels
    *with* the turn rather than being read off a shared mutable attribute a
    later turn could have already flipped (Codex P2).

    It is a ``SystemFrame`` — like ``VADUserStoppedSpeakingFrame`` — so it is
    processed inline in push order and reaches the tagger *ahead* of the turn's
    own ``TranscriptionFrame`` (a queued ``DataFrame``). A plain data frame
    would sit in the queue and arrive only after STT had already emitted the
    transcript, too late to tag it.

    One marker is emitted per VAD-stop, paired 1:1 with the single transcript
    the STT emits per turn (``push_empty_transcripts=True``), so the tagger's
    pending FIFO stays aligned regardless of transcript content.

    ``snapshot`` is an awaitable resolving to this turn's identity: an
    ``asyncio.Task`` from the turn-scoped embed, or a pre-resolved future for a
    turn we deliberately do not identify (a muted bot-echo turn, or an
    enrollment-capture turn).
    """

    snapshot: "Awaitable[str] | None" = None


async def _resolve_turn_snapshot(pending: "Awaitable[str] | None") -> str:
    """Await a pending per-turn identity, bounded and fail closed to 'unknown'.

    Fails closed if there is nothing pending, if the embed overruns
    ``_SNAPSHOT_AWAIT_TIMEOUT_S``, or if it errored — an unattributed turn is
    safe (Larry has a new-voice register); a wrong name is not. The awaitable is
    ``shield``ed so a wait-timeout here never cancels it: it keeps running to
    update the hysteresis streak for subsequent turns.
    """
    if pending is None:
        return "unknown"
    try:
        return await asyncio.wait_for(asyncio.shield(pending), _SNAPSHOT_AWAIT_TIMEOUT_S)
    except (TimeoutError, asyncio.CancelledError):
        logger.warning("Turn snapshot timed out waiting on embed — failing closed to unknown")
        return "unknown"
    except Exception:
        logger.warning("Turn snapshot embed failed — failing closed to unknown", exc_info=True)
        return "unknown"


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
        identify_enabled: bool = True,
    ) -> None:
        super().__init__()
        self._db_path = speakers_db_path
        self._on_speaker_change = on_speaker_change
        self._match_threshold = match_threshold
        self._change_turns = change_turns
        self._margin = margin
        # Per-turn identification + IdentitySnapshotFrame emission are only
        # meaningful on the segmented-STT path, where each transcript is bounded
        # to its own turn. On the xAI streaming path there is no downstream
        # tagger to consume the marker, so emitting one (and running the embed)
        # would burn CPU and grow no-one's queue — disable it there.
        self._identify_enabled = identify_enabled

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
        # Whether STT is currently muted (bot speaking + cool-down trail),
        # tracked from the STTMuteFrame that STTMuteOnBotSpeech emits and that
        # flows through here on its way to the STT service.
        self._stt_muted: bool = False
        # True once this turn accumulates voiced audio while un-muted — i.e.
        # real speech the STT heard, worth embedding. A turn that never sees an
        # un-muted voiced frame (pure echo during bot speech/cool-down) still
        # emits a marker, but resolves to 'unknown' without an embed, so echo
        # neither burns the encoder nor pollutes the hysteresis streak. Reset at
        # each VAD-start.
        self._turn_saw_unmuted: bool = False

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

        if isinstance(frame, STTMuteFrame):
            # STTMuteOnBotSpeech emits this upstream of us; track it (and forward
            # it, since the STT downstream is the real consumer) so we can gate
            # marker emission on whether this turn's audio will be transcribed.
            self._stt_muted = frame.mute
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            # Turn-scoped identification: accumulate only this turn's voiced,
            # bot-silent audio (mirrors the capture gate).  Bot-speaking audio
            # is TTS echo, not the speaker — never embed it.
            if self._vad_voiced and not self._bot_speaking_for_capture:
                self._turn_audio.extend(frame.audio)
            # This turn saw audio the STT will transcribe iff it was voiced while
            # un-muted — the same condition MutedGroqSTTService buffers under, so
            # marker emission stays 1:1 with transcript emission (no orphans).
            if self._vad_voiced and not self._stt_muted:
                self._turn_saw_unmuted = True
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
            self._turn_saw_unmuted = False  # ...and a fresh transcribable flag
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_voiced = False
            turn_bytes = bytes(self._turn_audio)
            self._turn_audio = bytearray()
            saw_unmuted = self._turn_saw_unmuted
            self._turn_saw_unmuted = False
            # Emit this turn's identity in-band, ahead of its transcript, on
            # EVERY VAD-stop (segmented-STT path only). The Groq STT is
            # configured with push_empty_transcripts=True, so every VAD turn
            # yields exactly one TranscriptionFrame — including a muted bot-echo
            # turn (empty, later dropped by WhisperHallucinationFilter) or a
            # silent turn Whisper transcribes to "". One marker per VAD-stop
            # therefore keeps markers and transcripts 1:1, so the tagger's FIFO
            # never drifts regardless of transcript content. (The one residual:
            # a raw Groq API error yields an ErrorFrame, not a transcript — a
            # rare, conversation-disrupting event that can orphan a single
            # marker; not worth an unsafe inline reconciliation to catch.) The
            # marker is a SystemFrame pushed BEFORE the VAD-stop so it reaches
            # the tagger ahead of the transcript.
            #
            # Run the (costly, torch) embed only for turns with voiced, un-muted
            # audio the STT actually heard. A muted echo turn or an enrollment
            # turn resolves to 'unknown' by value — never inheriting the last
            # confirmed speaker, and never letting echo pollute the hysteresis
            # streak. The embedded bytes are snapshotted here (by value), so a
            # late-completing embed still attributes to this turn, never a later
            # one.
            if self._identify_enabled:
                snapshot: Awaitable[str]
                if saw_unmuted and self._capture_state == "idle":
                    snapshot = asyncio.create_task(self._identify_turn(turn_bytes))
                else:
                    fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                    fut.set_result("unknown")
                    snapshot = fut
                await self.push_frame(IdentitySnapshotFrame(snapshot=snapshot), direction)
            await self.push_frame(frame, direction)

        else:
            # SpeakerIDProcessor is single-responsibility (audio → identity): it
            # sits upstream of STT and never sees TranscriptionFrames, so tagging
            # lives in the downstream SpeakerTagProcessor, not here.
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


class SpeakerTagProcessor(FrameProcessor):
    """Prefix each STT transcript with its turn's frozen ``[speaker: …]`` tag.

    Placed *after* STT (and before ``WhisperHallucinationFilter``) — unlike
    ``SpeakerIDProcessor``, which sits upstream of STT and so never sees the
    ``TranscriptionFrame`` STT emits (finding #2: that is exactly why the old
    upstream tag never reached the LLM).

    Each turn's identity arrives in-band as an ``IdentitySnapshotFrame``
    (emitted by ``SpeakerIDProcessor`` at the VAD boundary, ahead of the turn's
    transcript). We queue those snapshots FIFO and pop one per transcript, so a
    transcript is tagged with *its own* turn's identification — never a live
    ``current_speaker`` a later turn could have flipped. A FIFO (not a single
    "latest" slot) is required because ``IdentitySnapshotFrame`` is a
    ``SystemFrame`` processed inline: a second turn's marker can be handled
    before the first turn's queued ``TranscriptionFrame`` is dequeued, so order
    — not recency — is what binds each transcript to its turn. Markers and
    transcripts stay 1:1 *by construction*: ``SpeakerIDProcessor`` emits one
    marker per VAD-stop, and ``SegmentedSTTService`` runs STT once per VAD-stop
    yielding exactly one frame — a ``TranscriptionFrame`` (empty ones included,
    via ``push_empty_transcripts=True``) or, on a failed round-trip, a *downstream*
    ``ErrorFrame`` in place of the transcript. That downstream ErrorFrame drops
    its turn's marker here. Direction matters: ``push_error()`` sends unrelated
    LLM/TTS/Mem0 errors *upstream* through this processor, and those must NOT
    consume a marker — only a downstream STT error does. No BotStarted backstop:
    the 1:1 invariant needs none, and clearing on BotStarted would drop a marker
    left legitimately pending by a barge-in and re-introduce the desync.

    ``apply_boundary`` (Task 6) is applied to each resolved snapshot to drop the
    prior speaker's turns from the shared context on a continuity break. The tag
    then rides ``frame.text`` unchanged through the (non-mutating) hallucination
    filter into the LLM context, so the LLM, Mem0, and the conversation log all
    see the same identity that was frozen at the turn boundary.
    """

    def __init__(self, apply_boundary: Callable[[str], str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._apply_boundary = apply_boundary
        self._pending: deque[Awaitable[str]] = deque()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, IdentitySnapshotFrame):
            # This turn's identity, arriving ahead of its transcript. Queue it;
            # consume the frame (it is an internal handoff, never forwarded).
            if frame.snapshot is not None:
                self._pending.append(frame.snapshot)
            return
        if isinstance(frame, ErrorFrame):
            # A failed STT round-trip yields a *downstream* ErrorFrame in place of
            # this turn's transcript (push_empty_transcripts only covers an *empty*
            # result). That downstream error is this turn's one STT frame, so drop
            # its marker — else the NEXT turn's transcript pops it and shifts
            # attribution. But push_error() sends unrelated LLM/TTS/Mem0 errors
            # UPSTREAM through here; those are not STT turn errors and must leave
            # `_pending` untouched, or a downstream failure would silently
            # mis-attribute a live upstream turn. Gate strictly on direction.
            if direction == FrameDirection.DOWNSTREAM and self._pending:
                self._pending.popleft()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TranscriptionFrame):
            pending = self._pending.popleft() if self._pending else None
            snapshot = self._apply_boundary(await _resolve_turn_snapshot(pending))
            frame.text = format_speaker_tag(snapshot, frame.text)
        await self.push_frame(frame, direction)

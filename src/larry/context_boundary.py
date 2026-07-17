"""Task 6 — cross-speaker context boundary.

`pipeline.py` builds one shared `LLMContext` and the user aggregator appends
every turn to it, so without this module Bob's next turn would be sent to the
LLM carrying Alice's raw prior messages — even once Mem0 and `ConversationLog`
are correctly scoped to the frozen per-turn snapshot (Task 5).

`make_boundary_snapshot_provider` wraps a per-turn snapshot provider (in
production, `SpeakerIDProcessor.take_turn_snapshot`) with the fail-closed
rule: whenever continuity with the standing speaker breaks, drop every raw
user/assistant turn from the live context, keeping only the system prompt, so
the *next* turn is appended onto a clean slate. Continuity beyond that point
is carried by Mem0 facts + the recency line, never by replaying another
person's transcript — no retained tail, since a retained turn *is* the leak.
"""

from collections.abc import Awaitable, Callable
from typing import cast

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


def make_boundary_snapshot_provider(
    snapshot_provider: Callable[[], Awaitable[str]],
    context: LLMContext,
) -> Callable[[], Awaitable[str]]:
    """Wrap *snapshot_provider* to reset *context* on a continuity break.

    Compares each turn's snapshot to the previous turn's — the "standing"
    speaker whose raw turns currently live in context. Given
    `SpeakerIDProcessor`'s hysteresis (a snapshot is the confirmed speaker
    only when this turn's own identification agrees with it, else
    'unknown'), a change in that value is exactly the plan's two trigger
    conditions:
      - a confirmed switch to a new named speaker, or
      - an unconfirmed ('unknown') turn following a different (named)
        standing speaker.
    Two identical 'unknown' turns in a row (nobody yet identified) is not a
    change, so it does not force a redundant reset.

    Must be wired in *before* the turn's tagged transcript reaches the user
    aggregator (i.e. as the `snapshot_provider` passed into
    `SpeakerTagProcessor`, which sits upstream of it) so the reset lands
    before this turn's own message is appended.
    """
    standing: dict[str, str] = {"value": "unknown"}

    async def provider() -> str:
        snapshot = await snapshot_provider()
        if snapshot != standing["value"]:
            before = context.get_messages()
            kept = [m for m in before if isinstance(m, dict) and m.get("role") == "system"]
            dropped = len(before) - len(kept)
            if dropped:
                context.set_messages(cast(list[LLMContextMessage], kept))
                logger.info(
                    f"Context boundary: {standing['value']!r} -> {snapshot!r} — "
                    f"dropped {dropped} prior turn message(s)"
                )
            standing["value"] = snapshot
        return snapshot

    return provider

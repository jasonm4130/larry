"""Task 6 — cross-speaker context boundary.

`pipeline.py` builds one shared `LLMContext` and the user aggregator appends
every turn to it, so without this module Bob's next turn would be sent to the
LLM carrying Alice's raw prior messages — even once Mem0 and `ConversationLog`
are correctly scoped to the frozen per-turn snapshot (Task 5).

`make_context_boundary` returns a small function applied to each turn's
resolved speaker snapshot (by `SpeakerTagProcessor`, before the tagged
transcript reaches the user aggregator). Whenever continuity with the standing
speaker breaks, it drops every raw user/assistant turn from the live context,
keeping only the base system prompt, so the *next* turn is appended onto a
clean slate. Continuity beyond that point is carried by Mem0 facts + the
recency line, never by replaying another person's transcript — no retained
tail, since a retained turn *is* the leak.

Accepted tradeoff (intentional, not a bug): hysteresis (`SPEAKER_CHANGE_TURNS`,
default 2) means a new speaker's turn 1 always snapshots 'unknown', so it gets
no Mem0 store (an unknown turn does no Mem0 I/O — see `memory.py`). When turn 2
then confirms the speaker, the snapshot changes ('unknown' -> name) and this
module resets the boundary, dropping turn 1's raw content from the live
context too. So the opening line of every new speaker's conversation is
neither stored in Mem0 nor retained in the LLM context by the time Larry
replies to turn 2 — only the SQLite `ConversationLog` keeps it. This is the
fail-closed cost of never mis-attributing an unconfirmed turn; it is not fixed
by this module.

Second accepted tradeoff (P1-b): every 'unknown' turn resets, so a lone
stranger speaking across several still-unconfirmed turns also loses continuity
between their own turns — the price of guaranteeing two different guests can
never share context. Their turns still land in `ConversationLog`; only the live
LLM context is reset.
"""

from collections.abc import Callable
from typing import cast

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


def make_context_boundary(context: LLMContext) -> Callable[[str], str]:
    """Return a function that resets *context* on a continuity break.

    The returned `apply(snapshot)` compares each turn's snapshot to the
    previous turn's — the "standing" speaker whose raw turns currently live in
    context — and returns the snapshot unchanged. Given `SpeakerIDProcessor`'s
    hysteresis (a snapshot is the confirmed speaker only when this turn's own
    identification agrees with it, else 'unknown'), it resets on:
      - a confirmed switch to a new named speaker, or
      - any 'unknown' turn — including one following another 'unknown'. Because
        'unknown' carries no identity, two consecutive unknowns may be two
        DIFFERENT unproven voices, so neither shares context with the other
        (P1-b: isolate strangers even from each other).
    Only a named turn that matches the standing named speaker continues without
    a reset.

    Must be applied *before* the turn's tagged transcript reaches the user
    aggregator (i.e. inside `SpeakerTagProcessor`, which sits upstream of it) so
    the reset lands before this turn's own message is appended.

    Keeps only the *base* system prompt — the first system-role message, the
    same one `_refresh_system_prompt` (pipeline.py) treats as canonical — not
    every system-role message. `ScopedMem0MemoryService` (Task 5) injects each
    turn's retrieved Mem0 memories as an additional system message
    (`add_as_system_message` defaults True), and those accumulate across turns
    since only the first system message is ever refreshed in place. Retaining
    them on a reset would leak the previous speaker's memories into the next
    speaker's context through that channel, so they are dropped along with the
    raw user/assistant turns.
    """
    standing: dict[str, str] = {"value": "unknown"}

    def apply(snapshot: str) -> str:
        # Reset on any continuity break, AND on every 'unknown' turn: 'unknown'
        # carries no identity, so two consecutive unknowns may be two DIFFERENT
        # unproven voices — we cannot assume the second is the first continuing,
        # so neither shares context (P1-b). A named turn matching the standing
        # named speaker is the only case that continues without a reset.
        if snapshot == "unknown" or snapshot != standing["value"]:
            before = context.get_messages()
            base_system = next(
                (m for m in before if isinstance(m, dict) and m.get("role") == "system"), None
            )
            kept = [base_system] if base_system is not None else []
            dropped = len(before) - len(kept)
            if dropped:
                context.set_messages(cast(list[LLMContextMessage], kept))
                logger.info(
                    f"Context boundary: {standing['value']!r} -> {snapshot!r} — "
                    f"dropped {dropped} prior turn message(s)"
                )
            standing["value"] = snapshot
        return snapshot

    return apply

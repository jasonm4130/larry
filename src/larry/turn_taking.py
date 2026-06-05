"""User-turn aggregator wiring, isolated so it can be unit-tested.

The front-end ``VADProcessor`` in the pipeline is the SINGLE source of VAD
segmentation. The user aggregator must NOT be given its own ``vad_analyzer``:
a second Silero analyzer on the same audio stream wraps its own
``VADController`` that injects a duplicate ``VADUserStoppedSpeakingFrame`` per
silence, so every user turn gets aggregated twice — the "Larry thinks you're
repeating yourself" bug. The duplication is code-level, so it survives the
Jabra Speak 510's hardware AEC. See tests/test_turn_taking.py.
"""

from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import BaseUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies


def make_user_aggregator_params(
    *,
    user_idle_timeout: float,
    user_mute_strategies: list[BaseUserMuteStrategy],
    user_turn_strategies: UserTurnStrategies,
) -> LLMUserAggregatorParams:
    """Build the user aggregator params with NO VAD analyzer of its own.

    Deliberately omits ``vad_analyzer``: the upstream ``VADProcessor`` already
    emits the ``VADUserStarted/StoppedSpeakingFrame``s the aggregator's turn
    strategies consume. Re-adding an analyzer here re-introduces the 2x bug.
    """
    return LLMUserAggregatorParams(
        user_idle_timeout=user_idle_timeout,
        user_mute_strategies=user_mute_strategies,
        user_turn_strategies=user_turn_strategies,
    )

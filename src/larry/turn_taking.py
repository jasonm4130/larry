"""User-turn aggregator wiring, isolated so it can be unit-tested.

Two independent guards keep one spoken utterance == one user turn, both
defending against the "Larry thinks you're repeating yourself" bug:

1. Start strategy (the confirmed hardware cause). Pipecat's default user-turn
   *start* set is ``[VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]``.
   With streaming STT, a transcription frame can land AFTER the VAD-segmented
   turn has already closed and inference fired; the transcription-start strategy
   then opens a SECOND turn for the same utterance, so the LLM runs twice. We pin
   start to VAD-only (``make_user_turn_strategies``) — the front-end
   ``VADProcessor`` is the single segmentation source.

2. No second VAD analyzer (``make_user_aggregator_params``). The user aggregator
   must not carry its own ``vad_analyzer``: a second Silero analyzer wraps its
   own ``VADController`` that injects duplicate ``VADUserStoppedSpeakingFrame``s.

Both duplications are code-level, so they survive the Jabra Speak 510's hardware
AEC. See tests/test_turn_taking.py.
"""

from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import BaseUserMuteStrategy
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
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


def make_user_turn_strategies(
    stop: list[BaseUserTurnStopStrategy],
) -> UserTurnStrategies:
    """Build user-turn strategies with a SINGLE, VAD-only *start* strategy.

    Pipecat defaults ``start`` to ``[VADUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy]`` when it is left unset. The
    transcription-start fallback opens a duplicate turn whenever a streaming-STT
    transcription frame arrives after the VAD turn has already closed — the
    2x-repeat bug. Passing ``start`` explicitly (a non-empty list) suppresses the
    default, leaving the front-end ``VADProcessor`` as the only turn-start
    signal. The caller supplies the ``stop`` strategy (Smart Turn or
    speech-timeout); we never touch it.
    """
    return UserTurnStrategies(start=[VADUserTurnStartStrategy()], stop=stop)

"""Regression test for the "Larry repeats you 2x" bug (#double-vad).

Root cause (confirmed): two independent Silero VAD analyzers ran on the same
audio stream — one in the front-end ``VADProcessor`` and a second inside the
user aggregator (``LLMUserAggregatorParams.vad_analyzer``). The aggregator's
analyzer wraps its own ``VADController`` that injects a *second*
``VADUserStoppedSpeakingFrame`` per silence, so every user turn was aggregated
twice and Larry reacted as though the speaker had repeated themselves. Because
the duplication is purely code-level it survived the Jabra Speak 510's hardware
AEC, which is why swapping to the Jabra never fixed it.

The fix: the user aggregator must run NO VAD of its own — the upstream
``VADProcessor`` is the single source of turn segmentation. These tests pin
that: the params we build for the user aggregator carry no analyzer, and the
constructed aggregator therefore builds no internal ``VADController``.

Self-contained: no audio devices, no model load (the fixed path constructs no
SileroVADAnalyzer), no network.
"""

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from larry.turn_taking import make_user_aggregator_params, make_user_turn_strategies


def _params():
    return make_user_aggregator_params(
        user_idle_timeout=10.0,
        user_mute_strategies=[],
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
    )


def test_user_aggregator_params_carry_no_vad_analyzer():
    # A second analyzer here is the bug. The front-end VADProcessor is the only VAD.
    assert _params().vad_analyzer is None


def test_user_aggregator_builds_no_internal_vad_controller():
    pair = LLMContextAggregatorPair(
        LLMContext(messages=[{"role": "system", "content": "x"}]),
        user_params=_params(),
    )
    user = pair.user()
    # No analyzer -> Pipecat leaves _vad_controller None (llm_response_universal.py:647-648),
    # so no duplicate VADUserStoppedSpeakingFrame is ever injected.
    assert user._vad_controller is None


def test_user_turn_strategies_start_is_vad_only():
    # The ACTUAL 2x-repeat cause (confirmed on hardware): Pipecat's default start
    # set is [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]. With
    # streaming STT, a transcription frame landing AFTER the VAD-segmented turn has
    # already closed opens a SECOND user turn for the same utterance -> the LLM runs
    # twice and Larry reacts as if you repeated yourself. Pin start to VAD-only so
    # the front-end VADProcessor is the single turn-segmentation source.
    strategies = make_user_turn_strategies(stop=[SpeechTimeoutUserTurnStopStrategy()])
    assert len(strategies.start or []) == 1
    assert isinstance((strategies.start or [])[0], VADUserTurnStartStrategy)
    # The transcription-start fallback (which fires on the late streaming frames)
    # must NOT be present — its absence is the fix.
    assert not any(
        isinstance(s, TranscriptionUserTurnStartStrategy) for s in (strategies.start or [])
    )


def test_user_turn_strategies_passes_stop_through_unchanged():
    stop = [SpeechTimeoutUserTurnStopStrategy()]
    strategies = make_user_turn_strategies(stop=stop)
    assert strategies.stop == stop

"""Voice-triggered speaker enrollment and dismiss tools for Larry.

Two LLM function tools, registered alongside keep_about_self:

  enroll_speaker(name)  — arms the SpeakerIDProcessor capture state machine;
                          Larry prompts the user to speak a phrase so ~10s of
                          voiced audio can be captured, embedded, and persisted.
  dismiss()             — makes Larry go to sleep immediately (same dormant
                          state as the wake-gate silence timeout).

The Resemblyzer embed and the sleep_now() call are injected so this module is
unit-testable without audio, torch, or the pipeline runtime.

See docs/superpowers/specs/2026-06-05-voice-enrollment-and-dismiss-design.md.
"""

import random
from collections.abc import Awaitable, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties

# ---------------------------------------------------------------------------
# Module constants — tunable here, not env vars (not worth the noise).
# ---------------------------------------------------------------------------
# Tuned so a single repeat of one ENROLL_PHRASE (~4-5s spoken) clears the target;
# floor is the quality minimum; cap is the hard stop (embeds if >= floor, else aborts).
CAPTURE_TARGET_VOICED_S: float = 4.0
CAPTURE_FLOOR_VOICED_S: float = 2.5
CAPTURE_CAP_WALL_S: float = 12.0
SAMPLE_RATE: int = 16000

# Larry asks the speaker to repeat ONE of these. Resemblyzer is text-independent
# (content is irrelevant, duration is what matters), so these are chosen for length
# (~4-5s spoken) and in-character voice.
ENROLL_PHRASES: list[str] = [
    "the skull keeps what it's given, and it keeps me too",
    "I leave my voice with the one who lives in the box",
    "what the bone remembers, it never lets go of",
    "say it once and the hollow dark will know my name",
    "I hand the quiet my voice, and the skull holds it",
]
ENROLL_CONFIRM: str = "Kept. I'll know you now."
ENROLL_FAIL: str = "the quiet swallowed it — try again, my kept one"

# run_llm=False suppresses the LLM's own spoken turn after a tool call, so the
# deterministic line we queue is the only thing Larry says (no double-talk).
_SUPPRESS_LLM = FunctionCallResultProperties(run_llm=False)


def enrollment_instruction(phrase: str) -> str:
    """The exact line Larry speaks to prompt a repeat — deterministic, not LLM-relayed."""
    return f"Repeat the phrase, so I can recognise you: {phrase}."


# ---------------------------------------------------------------------------
# Tool schema factory
# ---------------------------------------------------------------------------

def build_voice_tools() -> ToolsSchema:
    """Build ToolsSchema for both voice tools (enroll_speaker + dismiss)."""
    enroll_fn = FunctionSchema(
        name="enroll_speaker",
        description=(
            "When someone introduces themselves by name ('it's Jason', 'I'm Dan', "
            "'this is Sarah'), call this to begin learning their voice. The system "
            "speaks the enrollment prompt itself — you do not need to say anything "
            "after calling it."
        ),
        properties={
            "name": {
                "type": "string",
                "description": "The speaker's first name as they gave it.",
            }
        },
        required=["name"],
    )
    dismiss_fn = FunctionSchema(
        name="dismiss",
        description=(
            "When someone says goodbye, 'that's all', 'go to sleep', or otherwise "
            "dismisses you, call this to go dormant immediately. Do not reply further "
            "after calling it — the cue line in the result is the last thing to speak."
        ),
        properties={},
        required=[],
    )
    return ToolsSchema(standard_tools=[enroll_fn, dismiss_fn])


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def make_enroll_speaker_handler(
    *,
    arm_capture_fn: Callable[..., None],
    speak_fn: Callable[[str], Awaitable[None]],
    rng: random.Random | None = None,
) -> Callable:
    """Build an async Pipecat function handler for enroll_speaker.

    ``arm_capture_fn(name, **kwargs)`` arms the capture state machine.
    ``speak_fn(line)`` queues a deterministic spoken line (a TTSSpeakFrame in
    production) — this is how Larry reliably ASKS the speaker to repeat a phrase,
    instead of relying on the LLM to relay the instruction (which a non-reasoning
    model does unreliably, speaking the in-character phrase as a statement). The
    LLM's own response is suppressed (run_llm=False) so the queued line is the
    only thing spoken. ``rng`` is injectable for deterministic tests.
    """
    chooser = rng or random.Random()

    async def handler(params) -> None:
        raw_name = str(params.arguments.get("name", "")).strip()
        name = raw_name.lower()
        if not name:
            await params.result_callback(
                {"status": "error", "reason": "name was empty"}, properties=_SUPPRESS_LLM
            )
            return
        arm_capture_fn(
            name,
            target_voiced_s=CAPTURE_TARGET_VOICED_S,
            floor_voiced_s=CAPTURE_FLOOR_VOICED_S,
            cap_wall_s=CAPTURE_CAP_WALL_S,
        )
        phrase = chooser.choice(ENROLL_PHRASES)
        await speak_fn(enrollment_instruction(phrase))
        await params.result_callback(
            {"status": "enrolling", "name": name, "phrase": phrase},
            properties=_SUPPRESS_LLM,
        )

    return handler


def make_dismiss_handler(
    *,
    sleep_now_fn: Callable[[], Awaitable[None]],
) -> Callable:
    """Build an async Pipecat function handler for dismiss.

    ``sleep_now_fn()`` is awaited to transition the wake gate to sleep.  In
    production this wraps ``wake_gate.sleep_now()``; in tests it is a coroutine
    fake.
    """

    async def handler(params) -> None:
        await sleep_now_fn()
        await params.result_callback({"status": "dismissed"}, properties=_SUPPRESS_LLM)

    return handler

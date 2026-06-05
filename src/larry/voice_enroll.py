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

# ---------------------------------------------------------------------------
# Module constants — tunable here, not env vars (not worth the noise).
# ---------------------------------------------------------------------------
CAPTURE_TARGET_VOICED_S: float = 10.0
CAPTURE_FLOOR_VOICED_S: float = 6.0
CAPTURE_CAP_WALL_S: float = 20.0
REPEAT_PHRASE: str = "the skull keeps what it's given"
ENROLL_CONFIRM: str = "Kept. I'll know you now."
ENROLL_FAIL: str = "the quiet swallowed it — try again, my kept one"

_DISMISS_CUES: list[str] = [
    "I'll keep listening.",
    "Until you come back.",
    "I never really sleep.",
    "Go on. I'll wait.",
]


# ---------------------------------------------------------------------------
# Tool schema factory
# ---------------------------------------------------------------------------

def build_voice_tools() -> ToolsSchema:
    """Build ToolsSchema for both voice tools (enroll_speaker + dismiss)."""
    enroll_fn = FunctionSchema(
        name="enroll_speaker",
        description=(
            "When someone introduces themselves by name ('it's Jason', 'I'm Dan', "
            "'this is Sarah'), call this to capture their voice and remember them. "
            "After calling it, speak the repeat phrase back to them so they know "
            "to say it aloud."
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
) -> Callable:
    """Build an async Pipecat function handler for enroll_speaker.

    ``arm_capture_fn(name, **kwargs)`` is called with the normalised name and
    the capture parameters from this module's constants.  In production this is
    ``speaker_id_proc.arm_capture``; in tests it is a fake.
    """

    async def handler(params) -> None:
        raw_name = str(params.arguments.get("name", "")).strip()
        name = raw_name.lower()
        if not name:
            await params.result_callback({"status": "error", "reason": "name was empty"})
            return
        arm_capture_fn(
            name,
            target_voiced_s=CAPTURE_TARGET_VOICED_S,
            floor_voiced_s=CAPTURE_FLOOR_VOICED_S,
            cap_wall_s=CAPTURE_CAP_WALL_S,
        )
        await params.result_callback(
            {
                "status": "pending",
                "name": name,
                "prompt": (
                    f"Say this back to me — '{REPEAT_PHRASE}.' "
                    f"That's how I'll keep you."
                ),
            }
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
        cue = random.choice(_DISMISS_CUES)
        await sleep_now_fn()
        await params.result_callback({"status": "dismissed", "cue": cue})

    return handler

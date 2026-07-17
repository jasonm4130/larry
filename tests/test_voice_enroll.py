"""Unit tests for larry.voice_enroll — tool schemas and handler factories.

Self-contained: no audio, no torch, no network. Pipecat FunctionSchema and
ToolsSchema are imported (pure Python dataclasses — no pipeline runtime needed).
Handlers are exercised via duck-typed _Params just like test_self_layer.py.
"""

import asyncio
import random

from larry import voice_enroll as ve

# ── Schema shape tests ──────────────────────────────────────────────────────


def test_build_voice_tools_contains_both_schemas():
    tools = ve.build_voice_tools()
    names = {fn.name for fn in tools.standard_tools}
    assert "enroll_speaker" in names
    assert "dismiss" in names


def test_enroll_speaker_schema_requires_name():
    tools = ve.build_voice_tools()
    enroll_fn = next(fn for fn in tools.standard_tools if fn.name == "enroll_speaker")
    schema = enroll_fn.to_default_dict()
    assert "name" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["name"]


def test_dismiss_schema_has_no_required_params():
    tools = ve.build_voice_tools()
    dismiss_fn = next(fn for fn in tools.standard_tools if fn.name == "dismiss")
    schema = dismiss_fn.to_default_dict()
    assert schema["parameters"].get("required", []) == []


# ── Handler: enroll_speaker ─────────────────────────────────────────────────


class _Params:
    """Duck-type for Pipecat FunctionCallParams."""

    def __init__(self, arguments: dict):
        self.arguments = arguments
        self._result = None
        self._properties = None

    async def result_callback(self, result, *, properties=None):
        self._result = result
        self._properties = properties


def test_enroll_handler_arms_capture_and_speaks_repeat_prompt():
    armed: list[str] = []
    spoken: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def fake_speak(line):
        spoken.append(line)

    async def body():
        params = _Params({"name": "Jason"})
        expected_phrase = random.Random(0).choice(ve.ENROLL_PHRASES)
        handler = ve.make_enroll_speaker_handler(
            arm_capture_fn=fake_arm, speak_fn=fake_speak, rng=random.Random(0)
        )
        await handler(params)
        assert armed == ["jason"], "arm should receive the normalised name"
        # Larry deterministically SPEAKS a 'repeat the phrase' instruction — not
        # relayed through the LLM — and it carries the chosen phrase.
        assert spoken == [ve.enrollment_instruction(expected_phrase)]
        assert spoken[0].lower().startswith("repeat the phrase")
        assert params._result["status"] == "enrolling"
        # The LLM's own spoken turn is suppressed so there is no double-talk.
        assert params._properties is not None and params._properties.run_llm is False

    asyncio.run(body())


def test_enroll_handler_normalises_name():
    """Name is stripped and lower-cased for the DB key."""
    armed_names: list[str] = []

    async def fake_speak(_line):
        pass

    async def body():
        params = _Params({"name": "  ALICE  "})
        handler = ve.make_enroll_speaker_handler(
            arm_capture_fn=lambda name, **_k: armed_names.append(name), speak_fn=fake_speak
        )
        await handler(params)
        assert armed_names[0] == "alice"  # normalised

    asyncio.run(body())


def test_enroll_handler_ignores_empty_name():
    armed: list[str] = []
    spoken: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def fake_speak(line):
        spoken.append(line)

    async def body():
        params = _Params({"name": "   "})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm, speak_fn=fake_speak)
        await handler(params)
        assert armed == []
        assert spoken == []  # nothing spoken on a bad name
        assert params._result["status"] == "error"
        assert params._properties is not None and params._properties.run_llm is False

    asyncio.run(body())


# ── Handler: dismiss ────────────────────────────────────────────────────────


def test_dismiss_handler_calls_sleep_now_and_returns_dismissed():
    slept: list[bool] = []

    async def fake_sleep_now():
        slept.append(True)

    async def body():
        params = _Params({})
        handler = ve.make_dismiss_handler(sleep_now_fn=fake_sleep_now)
        await handler(params)
        assert slept == [True], "sleep_now should be called"
        assert params._result is not None
        assert params._result["status"] == "dismissed"
        # Cue is played via _on_sleep; handler only returns status.
        assert "cue" not in params._result
        # LLM's own turn suppressed — the sleep cue is the last word.
        assert params._properties is not None and params._properties.run_llm is False

    asyncio.run(body())

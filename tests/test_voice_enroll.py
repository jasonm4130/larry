"""Unit tests for larry.voice_enroll — tool schemas and handler factories.

Self-contained: no audio, no torch, no network. Pipecat FunctionSchema and
ToolsSchema are imported (pure Python dataclasses — no pipeline runtime needed).
Handlers are exercised via duck-typed _Params just like test_self_layer.py.
"""

import asyncio

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

    async def result_callback(self, result):
        self._result = result


def test_enroll_handler_arms_capture_and_returns_prompt():
    armed: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def body():
        params = _Params({"name": "Jason"})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert "jason" in armed[0].lower() or "Jason" in armed[0], "arm should receive the name"
        assert params._result is not None
        assert params._result["status"] == "pending"
        # Result must include the repeat phrase so Larry speaks it.
        assert ve.REPEAT_PHRASE in params._result["prompt"]

    asyncio.run(body())


def test_enroll_handler_normalises_name():
    """Name is stripped and stored; casing kept for display but lowered for key."""
    armed_names: list[str] = []

    def fake_arm(name, **_kwargs):
        armed_names.append(name)

    async def body():
        params = _Params({"name": "  ALICE  "})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert armed_names[0] == "alice"   # normalised

    asyncio.run(body())


def test_enroll_handler_ignores_empty_name():
    armed: list[str] = []

    def fake_arm(name, **_kwargs):
        armed.append(name)

    async def body():
        params = _Params({"name": "   "})
        handler = ve.make_enroll_speaker_handler(arm_capture_fn=fake_arm)
        await handler(params)
        assert armed == []
        assert params._result["status"] == "error"

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

    asyncio.run(body())

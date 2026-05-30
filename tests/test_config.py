"""Unit tests for larry.config.load_config / Config.

Self-contained: no heavy ML models, no audio devices, no network. The config
module reads ``os.environ`` directly at call time, so monkeypatch.setenv /
delenv fully control the inputs. ``LARRY_HARDWARE`` is set explicitly in tests
that care about provider/hardware-dependent defaults so results don't depend on
the host platform (``_default_hardware`` keys off ``sys.platform``).
"""

import pytest

from larry.config import load_config


@pytest.fixture
def required_keys(monkeypatch):
    """Set the three required API keys so a Config can be constructed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")


# --- Required keys ---------------------------------------------------------


@pytest.mark.parametrize("missing", ["OPENROUTER_API_KEY", "GROQ_API_KEY", "ELEVENLABS_API_KEY"])
def test_missing_required_key_raises(monkeypatch, required_keys, missing):
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_config()
    assert missing in str(exc.value)


def test_empty_required_key_raises(monkeypatch, required_keys):
    # _require treats empty string as missing too.
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(RuntimeError):
        load_config()


def test_required_keys_present_constructs(required_keys):
    cfg = load_config()
    assert cfg.openrouter_api_key == "or-key"
    assert cfg.groq_api_key == "groq-key"
    assert cfg.elevenlabs_api_key == "el-key"


# --- Provider-dependent default llm_model ----------------------------------


def test_default_llm_without_xai(monkeypatch, required_keys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = load_config()
    assert cfg.xai_api_key is None
    assert cfg.llm_model == "anthropic/claude-sonnet-4-6"


def test_default_llm_with_xai(monkeypatch, required_keys):
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = load_config()
    assert cfg.xai_api_key == "xai-key"
    assert cfg.llm_model == "grok-4.20-non-reasoning"


def test_llm_model_overrides_without_xai(monkeypatch, required_keys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "google/gemini-2.5-pro")
    cfg = load_config()
    assert cfg.llm_model == "google/gemini-2.5-pro"


def test_llm_model_overrides_with_xai(monkeypatch, required_keys):
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    monkeypatch.setenv("LLM_MODEL", "x-ai/grok-custom")
    cfg = load_config()
    assert cfg.llm_model == "x-ai/grok-custom"


# --- _bool env parsing (via ENABLE_SMART_TURN) -----------------------------


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "Yes", "on", "ON", " true "])
def test_bool_truthy_values(monkeypatch, required_keys, truthy):
    # Force the default to False so a True result must come from the value.
    monkeypatch.setenv("LARRY_HARDWARE", "mock")
    monkeypatch.setenv("ENABLE_SMART_TURN", truthy)
    cfg = load_config()
    assert cfg.enable_smart_turn is True


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "", "maybe", "2"])
def test_bool_falsy_values(monkeypatch, required_keys, falsy):
    # Force the default to True so a False result must come from the value.
    monkeypatch.setenv("LARRY_HARDWARE", "pca9685")
    monkeypatch.setenv("ENABLE_SMART_TURN", falsy)
    cfg = load_config()
    assert cfg.enable_smart_turn is False


def test_bool_honors_default_when_unset(monkeypatch, required_keys):
    monkeypatch.delenv("ENABLE_SMART_TURN", raising=False)
    monkeypatch.setenv("LARRY_HARDWARE", "mock")
    assert load_config().enable_smart_turn is False
    monkeypatch.setenv("LARRY_HARDWARE", "pca9685")
    assert load_config().enable_smart_turn is True


# --- smart_turn default keyed on LARRY_HARDWARE ----------------------------


def test_smart_turn_default_true_on_pca9685(monkeypatch, required_keys):
    monkeypatch.delenv("ENABLE_SMART_TURN", raising=False)
    monkeypatch.setenv("LARRY_HARDWARE", "pca9685")
    assert load_config().enable_smart_turn is True


def test_smart_turn_default_false_otherwise(monkeypatch, required_keys):
    monkeypatch.delenv("ENABLE_SMART_TURN", raising=False)
    monkeypatch.setenv("LARRY_HARDWARE", "mock")
    assert load_config().enable_smart_turn is False


# --- ElevenLabs voice / model defaults -------------------------------------


def test_elevenlabs_defaults(monkeypatch, required_keys):
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)
    cfg = load_config()
    assert cfg.elevenlabs_voice_id == "cPoqAvGWCPfCfyPMwe4z"
    assert cfg.elevenlabs_model == "eleven_turbo_v2_5"


def test_elevenlabs_overrides(monkeypatch, required_keys):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "my-voice")
    monkeypatch.setenv("ELEVENLABS_MODEL", "eleven_v3")
    cfg = load_config()
    assert cfg.elevenlabs_voice_id == "my-voice"
    assert cfg.elevenlabs_model == "eleven_v3"

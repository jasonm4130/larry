"""Unit tests for larry.config.load_config / Config.

Self-contained: no heavy ML models, no audio devices, no network. The config
module reads ``os.environ`` directly at call time, so monkeypatch.setenv /
delenv fully control the inputs. ``LARRY_HARDWARE`` is set explicitly in tests
that care about provider/hardware-dependent defaults so results don't depend on
the host platform (``_default_hardware`` keys off ``sys.platform``).
"""

import dataclasses

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


# --- STT provider selection ------------------------------------------------


def test_stt_provider_defaults_to_groq(monkeypatch, required_keys):
    # xAI streaming STT carries transcripts across VAD turns (the loop bug), so
    # Groq segmented STT is the safe default even when XAI_API_KEY is set.
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    assert load_config().stt_provider == "groq"


def test_stt_provider_override_xai_normalized(monkeypatch, required_keys):
    monkeypatch.setenv("STT_PROVIDER", "  XAI  ")
    assert load_config().stt_provider == "xai"


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


# --- __post_init__ fail-fast validation (env-driven fields) ----------------


@pytest.mark.parametrize(
    ("env_var", "bad_value", "field_name"),
    [
        ("PROACTIVE_PROBABILITY", "1.5", "proactive_probability"),
        ("PROACTIVE_PROBABILITY", "-0.1", "proactive_probability"),
        ("IDLE_TIMEOUT_S", "0", "idle_timeout_s"),
        ("IDLE_TIMEOUT_S", "-5", "idle_timeout_s"),
        ("WAKE_SLEEP_TIMEOUT_S", "0", "wake_sleep_timeout_s"),
        ("WAKE_SLEEP_TIMEOUT_S", "-1", "wake_sleep_timeout_s"),
        ("STT_MUTE_COOLDOWN_S", "-0.1", "stt_mute_cooldown_s"),
        ("VAD_START_SECS", "-0.1", "vad_start_secs"),
        ("SMART_TURN_CPU_COUNT", "0", "smart_turn_cpu_count"),
        ("SMART_TURN_CPU_COUNT", "-1", "smart_turn_cpu_count"),
        ("JAW_NOISE_FLOOR", "-0.01", "jaw_noise_floor"),
    ],
)
def test_invalid_env_value_raises_value_error(
    monkeypatch, required_keys, env_var, bad_value, field_name
):
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError) as exc:
        load_config()
    assert field_name in str(exc.value)
    assert bad_value.lstrip("-") in str(exc.value) or bad_value in str(exc.value)


def test_jaw_peak_not_above_noise_floor_raises(monkeypatch, required_keys):
    # jaw_peak must be strictly greater than jaw_noise_floor.
    monkeypatch.setenv("JAW_NOISE_FLOOR", "0.3")
    monkeypatch.setenv("JAW_PEAK", "0.3")
    with pytest.raises(ValueError) as exc:
        load_config()
    assert "jaw_peak" in str(exc.value)


# --- __post_init__ validation (hardcoded fields, via direct construction) ---


def _valid_config_kwargs(required_keys):
    """A fully-valid kwargs dict mirroring load_config() defaults."""
    cfg = load_config()
    return dataclasses.asdict(cfg)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("jaw_open_angle", 181),
        ("jaw_open_angle", -1),
        ("jaw_closed_angle", 200),
        ("jaw_closed_angle", -5),
        ("jaw_servo_channel", 16),
        ("jaw_servo_channel", -1),
    ],
)
def test_invalid_hardcoded_field_raises_value_error(required_keys, field_name, bad_value):
    from larry.config import Config

    kwargs = _valid_config_kwargs(required_keys)
    kwargs[field_name] = bad_value
    with pytest.raises(ValueError) as exc:
        Config(**kwargs)
    assert field_name in str(exc.value)
    assert str(bad_value) in str(exc.value)


def test_valid_config_constructs_without_error(required_keys):
    # Sanity: the default config passes validation.
    from larry.config import Config

    kwargs = _valid_config_kwargs(required_keys)
    Config(**kwargs)


def test_self_evolution_defaults(monkeypatch, required_keys):
    cfg = load_config()
    assert cfg.self_layer_path.name == "larry_self.md"
    assert cfg.self_layer_cap_chars == 5000
    assert cfg.self_evolution_enabled is True


def test_self_evolution_overrides(monkeypatch, required_keys):
    monkeypatch.setenv("SELF_LAYER_CAP_CHARS", "8000")
    monkeypatch.setenv("SELF_EVOLUTION_ENABLED", "false")
    cfg = load_config()
    assert cfg.self_layer_cap_chars == 8000
    assert cfg.self_evolution_enabled is False


def test_voice_tools_enabled_defaults_true(monkeypatch, required_keys):
    cfg = load_config()
    assert cfg.voice_tools_enabled is True


def test_voice_tools_enabled_overridable(monkeypatch, required_keys):
    monkeypatch.setenv("VOICE_TOOLS_ENABLED", "false")
    cfg = load_config()
    assert cfg.voice_tools_enabled is False

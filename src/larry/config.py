import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_hardware() -> str:
    return "mock" if sys.platform == "darwin" else "pca9685"


@dataclass(frozen=True)
class Config:
    # API keys
    openrouter_api_key: str
    openrouter_base_url: str  # default: https://openrouter.ai/api/v1
    # If XAI_API_KEY is set, the main chat LLM routes directly to xAI's API
    # (faster + cheaper than Claude via OpenRouter, per May 2026 research).
    # Mem0's fact-extraction LLM still uses OpenRouter regardless.
    xai_api_key: str | None
    # Default depends on which provider is active: grok-4.20-non-reasoning for
    # xAI direct, anthropic/claude-sonnet-4-6 for OpenRouter.  Override via LLM_MODEL.
    llm_model: str
    groq_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model: str  # default eleven_turbo_v2_5; v3 needs alpha access

    # Hardware
    larry_hardware: str
    wake_word_model: str  # OpenWakeWord pretrained model name (default: hey_jarvis)
    wake_word_custom_path: str | None  # path to a custom .onnx model, if any

    # Turn-taking / VAD tuning (see docs/plan: turn-taking robustness)
    stt_mute_cooldown_s: float  # STT stays muted this long after bot stops speaking
    vad_start_secs: float  # speech must persist this long before VAD declares a turn start
    wake_sleep_timeout_s: float  # post-speech silence before the wake gate sleeps
    enable_smart_turn: bool  # layer Smart Turn v3 neural end-of-turn on top of VAD
    smart_turn_cpu_count: int  # threads for the local Smart Turn ONNX model

    # Paths
    data_dir: Path
    speakers_db: Path
    conversations_db: Path
    mem0_dir: Path
    logs_dir: Path
    personality_path: Path

    # Servo calibration
    jaw_open_angle: int
    jaw_closed_angle: int
    jaw_servo_channel: int

    # Jaw lip-sync
    jaw_noise_floor: float  # RMS level treated as silence (normalisation floor)
    jaw_peak: float  # RMS level treated as full-open (normalisation ceiling)

    # Idle / proactive behaviour
    idle_timeout_s: float  # seconds of user silence before proactive utterance fires
    proactive_probability: float  # probability [0, 1] of speaking on each idle trigger


def load_config() -> Config:
    def _require(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(
                f"Required environment variable {key!r} is not set. "
                "Check your .env file or environment."
            )
        return val

    data_dir = Path("data")

    def _bool(key: str, default: bool) -> bool:
        val = os.environ.get(key)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "on"}

    larry_hardware = os.environ.get("LARRY_HARDWARE", _default_hardware())
    # Smart Turn defaults on for the Pi (pca9685 + Jabra hardware AEC), off for
    # Mac dev where it adds latency to a feedback-prone single-device loop.
    smart_turn_default = larry_hardware == "pca9685"

    xai_api_key = os.environ.get("XAI_API_KEY")
    # Default model depends on active provider — xAI's grok-4.20 non-reasoning
    # variant is the fast/cheap pick when routing direct, Claude Sonnet 4.6
    # remains the OpenRouter fallback.
    default_llm = "grok-4.20-non-reasoning" if xai_api_key else "anthropic/claude-sonnet-4-6"

    return Config(
        openrouter_api_key=_require("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        xai_api_key=xai_api_key,
        llm_model=os.environ.get("LLM_MODEL", default_llm),
        groq_api_key=_require("GROQ_API_KEY"),
        elevenlabs_api_key=_require("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "cPoqAvGWCPfCfyPMwe4z"),
        elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        larry_hardware=larry_hardware,
        wake_word_model=os.environ.get("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_custom_path=os.environ.get("WAKE_WORD_CUSTOM_PATH"),
        stt_mute_cooldown_s=float(os.environ.get("STT_MUTE_COOLDOWN_S", "0.2")),
        vad_start_secs=float(os.environ.get("VAD_START_SECS", "0.1")),
        wake_sleep_timeout_s=float(os.environ.get("WAKE_SLEEP_TIMEOUT_S", "20")),
        enable_smart_turn=_bool("ENABLE_SMART_TURN", smart_turn_default),
        smart_turn_cpu_count=int(os.environ.get("SMART_TURN_CPU_COUNT", "2")),
        data_dir=data_dir,
        speakers_db=data_dir / "speakers.db",
        conversations_db=data_dir / "conversations.db",
        mem0_dir=data_dir / "mem0",
        logs_dir=data_dir / "logs",
        personality_path=Path(__file__).parent / "personality" / "larry.md",
        jaw_open_angle=60,
        jaw_closed_angle=0,
        jaw_servo_channel=0,
        jaw_noise_floor=float(os.environ.get("JAW_NOISE_FLOOR", "0.01")),
        jaw_peak=float(os.environ.get("JAW_PEAK", "0.3")),
        idle_timeout_s=float(os.environ.get("IDLE_TIMEOUT_S", "600")),
        proactive_probability=float(os.environ.get("PROACTIVE_PROBABILITY", "1.0")),
    )

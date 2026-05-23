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
    llm_model: str  # default: anthropic/claude-sonnet-4-6; override via LLM_MODEL
    groq_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str

    # Hardware
    larry_hardware: str
    wake_word_model: str  # OpenWakeWord pretrained model name (default: hey_jarvis)
    wake_word_custom_path: str | None  # path to a custom .onnx model, if any

    # Paths
    data_dir: Path
    speakers_db: Path
    conversations_db: Path
    mem0_dir: Path
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

    return Config(
        openrouter_api_key=_require("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_model=os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-6"),
        groq_api_key=_require("GROQ_API_KEY"),
        elevenlabs_api_key=_require("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "cPoqAvGWCPfCfyPMwe4z"),
        larry_hardware=os.environ.get("LARRY_HARDWARE", _default_hardware()),
        wake_word_model=os.environ.get("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_custom_path=os.environ.get("WAKE_WORD_CUSTOM_PATH"),
        data_dir=data_dir,
        speakers_db=data_dir / "speakers.db",
        conversations_db=data_dir / "conversations.db",
        mem0_dir=data_dir / "mem0",
        personality_path=Path(__file__).parent / "personality" / "larry.md",
        jaw_open_angle=60,
        jaw_closed_angle=0,
        jaw_servo_channel=0,
        jaw_noise_floor=float(os.environ.get("JAW_NOISE_FLOOR", "0.01")),
        jaw_peak=float(os.environ.get("JAW_PEAK", "0.3")),
        idle_timeout_s=float(os.environ.get("IDLE_TIMEOUT_S", "600")),
        proactive_probability=float(os.environ.get("PROACTIVE_PROBABILITY", "1.0")),
    )

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_hardware() -> str:
    return "mock" if sys.platform == "darwin" else "pca9685"


@dataclass(frozen=True)
class Config:
    # API keys
    anthropic_api_key: str
    groq_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    picovoice_access_key: str | None  # required from Phase 3 onwards

    # Hardware
    larry_hardware: str
    wake_word_keyword_path: str | None

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
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        groq_api_key=_require("GROQ_API_KEY"),
        elevenlabs_api_key=_require("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "cPoqAvGWCPfCfyPMwe4z"),
        picovoice_access_key=os.environ.get("PICOVOICE_ACCESS_KEY"),
        larry_hardware=os.environ.get("LARRY_HARDWARE", _default_hardware()),
        wake_word_keyword_path=os.environ.get("WAKE_WORD_KEYWORD_PATH"),
        data_dir=data_dir,
        speakers_db=data_dir / "speakers.db",
        conversations_db=data_dir / "conversations.db",
        mem0_dir=data_dir / "mem0",
        personality_path=Path(__file__).parent / "personality" / "larry.md",
        jaw_open_angle=60,
        jaw_closed_angle=0,
        jaw_servo_channel=0,
    )

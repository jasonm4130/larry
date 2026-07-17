import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_hardware() -> str:
    return "mock" if sys.platform == "darwin" else "pca9685"


def _default_wake_word_custom_path() -> str | None:
    """Package-relative path to the committed "Hey Larry" model, mirroring
    personality_path's resolution.  Returns None (falls back to the
    hey_jarvis pretrained model in wake.py) if the .onnx file isn't present —
    e.g. a checkout predating training, or one that intentionally omits it."""
    candidate = Path(__file__).parent / "wake_models" / "hey_larry.onnx"
    return str(candidate) if candidate.exists() else None


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
    # xAI direct, anthropic/claude-sonnet-5 for OpenRouter.  Override via LLM_MODEL.
    llm_model: str
    # STT provider: "groq" (segmented, per-VAD-turn — default) or "xai" (streaming).
    # xAI streaming STT shares one WebSocket session across turns and carries the
    # previous utterance's transcript into the next (the "looping" bug), so Groq
    # is the safe default even when XAI_API_KEY is set for the chat LLM.
    stt_provider: str
    groq_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model: str  # default eleven_flash_v2_5; v3 needs alpha access

    # Hardware
    larry_hardware: str
    wake_word_model: str  # OpenWakeWord pretrained model name (default: hey_jarvis)
    # Path to a custom .onnx model. Defaults to the committed wake_models/hey_larry.onnx
    # when present; override with WAKE_WORD_CUSTOM_PATH, or unset to fall back to
    # wake_word_model (hey_jarvis) if the committed model is absent.
    wake_word_custom_path: str | None

    # Turn-taking / VAD tuning (see docs/plan: turn-taking robustness)
    stt_mute_cooldown_s: float  # STT stays muted this long after bot stops speaking
    vad_start_secs: float  # speech must persist this long before VAD declares a turn start
    wake_sleep_timeout_s: float  # post-speech silence before the wake gate sleeps
    enable_smart_turn: bool  # layer Smart Turn v3 neural end-of-turn on top of VAD
    smart_turn_cpu_count: int  # threads for the local Smart Turn ONNX model

    # Speaker identification
    speaker_match_threshold: float  # cosine-sim cutoff to accept an enrolled match
    # Hysteresis: switch the confirmed speaker only after this many consecutive
    # turns identify the same best match (>= 1; default 2 — confirms a switch
    # quickly while rejecting a single stray identification).
    speaker_change_turns: int
    # Minimum top1-top2 cosine margin required to accept a match when >= 2
    # speakers are enrolled (in [0, 1]; default 0.06 — rejects near-ties without
    # starving normal matches). Waived when < 2 speakers are enrolled (no runner-up).
    speaker_margin: float
    # SpeakerEmbedder impl name (src/larry/speaker_embedder.py). Only "resemblyzer"
    # is implemented; a future TitaNet/ONNX impl is a deferred follow-up (see Task 3
    # in docs/superpowers/plans/2026-07-17-identity-and-wake-fixes.md). Voiceprints
    # are namespaced by embedder name, so changing this requires every speaker to
    # re-enroll.
    speaker_embedder: str

    # Paths
    data_dir: Path
    speakers_db: Path
    conversations_db: Path
    mem0_dir: Path
    logs_dir: Path
    personality_path: Path
    self_layer_path: Path
    self_layer_cap_chars: int
    self_evolution_enabled: bool
    voice_tools_enabled: bool

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

    def __post_init__(self) -> None:
        def _check(cond: bool, name: str, val: object, msg: str) -> None:
            if not cond:
                raise ValueError(f"{name} {msg}, got {val!r}")

        _check(
            0.0 <= self.proactive_probability <= 1.0,
            "proactive_probability",
            self.proactive_probability,
            "must be in [0, 1]",
        )
        _check(
            0 <= self.jaw_open_angle <= 180,
            "jaw_open_angle",
            self.jaw_open_angle,
            "must be in [0, 180]",
        )
        _check(
            0 <= self.jaw_closed_angle <= 180,
            "jaw_closed_angle",
            self.jaw_closed_angle,
            "must be in [0, 180]",
        )
        _check(
            0 <= self.jaw_servo_channel <= 15,
            "jaw_servo_channel",
            self.jaw_servo_channel,
            "must be in [0, 15]",
        )
        _check(self.idle_timeout_s > 0, "idle_timeout_s", self.idle_timeout_s, "must be > 0")
        _check(
            self.wake_sleep_timeout_s > 0,
            "wake_sleep_timeout_s",
            self.wake_sleep_timeout_s,
            "must be > 0",
        )
        _check(
            self.stt_mute_cooldown_s >= 0,
            "stt_mute_cooldown_s",
            self.stt_mute_cooldown_s,
            "must be >= 0",
        )
        _check(self.vad_start_secs >= 0, "vad_start_secs", self.vad_start_secs, "must be >= 0")
        _check(
            0.0 <= self.speaker_match_threshold <= 1.0,
            "speaker_match_threshold",
            self.speaker_match_threshold,
            "must be in [0, 1]",
        )
        _check(
            self.speaker_change_turns >= 1,
            "speaker_change_turns",
            self.speaker_change_turns,
            "must be >= 1",
        )
        _check(
            0.0 <= self.speaker_margin <= 1.0,
            "speaker_margin",
            self.speaker_margin,
            "must be in [0, 1]",
        )
        _check(
            self.speaker_embedder in {"resemblyzer"},
            "speaker_embedder",
            self.speaker_embedder,
            "must be one of {'resemblyzer'}",
        )
        _check(
            self.smart_turn_cpu_count >= 1,
            "smart_turn_cpu_count",
            self.smart_turn_cpu_count,
            "must be >= 1",
        )
        _check(self.jaw_noise_floor >= 0, "jaw_noise_floor", self.jaw_noise_floor, "must be >= 0")
        _check(
            self.jaw_peak > self.jaw_noise_floor,
            "jaw_peak",
            self.jaw_peak,
            f"must be > jaw_noise_floor ({self.jaw_noise_floor!r})",
        )


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
    # variant is the fast/cheap pick when routing direct, Claude Sonnet 5
    # remains the OpenRouter fallback.
    default_llm = "grok-4.20-non-reasoning" if xai_api_key else "anthropic/claude-sonnet-5"

    return Config(
        openrouter_api_key=_require("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        xai_api_key=xai_api_key,
        llm_model=os.environ.get("LLM_MODEL", default_llm),
        stt_provider=os.environ.get("STT_PROVIDER", "groq").strip().lower(),
        groq_api_key=_require("GROQ_API_KEY"),
        elevenlabs_api_key=_require("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "cPoqAvGWCPfCfyPMwe4z"),
        elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
        larry_hardware=larry_hardware,
        wake_word_model=os.environ.get("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_custom_path=os.environ.get("WAKE_WORD_CUSTOM_PATH")
        or _default_wake_word_custom_path(),
        stt_mute_cooldown_s=float(os.environ.get("STT_MUTE_COOLDOWN_S", "0.2")),
        vad_start_secs=float(os.environ.get("VAD_START_SECS", "0.1")),
        wake_sleep_timeout_s=float(os.environ.get("WAKE_SLEEP_TIMEOUT_S", "20")),
        enable_smart_turn=_bool("ENABLE_SMART_TURN", smart_turn_default),
        smart_turn_cpu_count=int(os.environ.get("SMART_TURN_CPU_COUNT", "2")),
        speaker_match_threshold=float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.75")),
        speaker_change_turns=int(os.environ.get("SPEAKER_CHANGE_TURNS", "2")),
        speaker_margin=float(os.environ.get("SPEAKER_MARGIN", "0.06")),
        speaker_embedder=os.environ.get("SPEAKER_EMBEDDER", "resemblyzer").strip().lower(),
        data_dir=data_dir,
        speakers_db=data_dir / "speakers.db",
        conversations_db=data_dir / "conversations.db",
        mem0_dir=data_dir / "mem0",
        logs_dir=data_dir / "logs",
        personality_path=Path(__file__).parent / "personality" / "larry.md",
        self_layer_path=data_dir / "larry_self.md",
        self_layer_cap_chars=int(os.environ.get("SELF_LAYER_CAP_CHARS", "5000")),
        self_evolution_enabled=os.environ.get("SELF_EVOLUTION_ENABLED", "true").lower()
        not in ("false", "0", "no"),
        voice_tools_enabled=os.environ.get("VOICE_TOOLS_ENABLED", "true").lower()
        not in ("false", "0", "no"),
        jaw_open_angle=60,
        jaw_closed_angle=0,
        jaw_servo_channel=0,
        jaw_noise_floor=float(os.environ.get("JAW_NOISE_FLOOR", "0.01")),
        jaw_peak=float(os.environ.get("JAW_PEAK", "0.3")),
        idle_timeout_s=float(os.environ.get("IDLE_TIMEOUT_S", "600")),
        proactive_probability=float(os.environ.get("PROACTIVE_PROBABILITY", "1.0")),
    )

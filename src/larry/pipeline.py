"""Larry's voice pipeline (Phases 3-5 integrated).

Pipeline order:
    transport.input()
      → WakeWordGate           # gates everything until "Hey Larry" (or "hey_jarvis" fallback)
      → VADProcessor           # emits VADUser{Started,Stopped}SpeakingFrame for STT segmentation
      → SpeakerIDProcessor     # tags TranscriptionFrames with [speaker: name]
      → GroqSTT
      → user_agg               # idle detection wired here via user_idle_timeout
      → Mem0MemoryService      # injects per-person facts before the LLM
      → OpenAILLM (via OpenRouter)
      → assistant_agg
      → ElevenLabsTTS
      → AudioBufferProcessor   # taps bot_audio for jaw lip-sync
      → transport.output()
"""

import asyncio
import datetime
import logging
import logging.handlers
import os
import random
import sys
from pathlib import Path

import httpx
from loguru import logger as _loguru
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.xai.llm import GrokLLMService
from pipecat.services.xai.stt import XAISTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy, BaseUserMuteStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from larry.audio_filter import WebRTCEchoCancellationFilter
from larry.config import load_config
from larry.hardware import get_jaw_driver
from larry.jaw import JawAmplitudeMapper
from larry.memory import ConversationLog, make_memory_service
from larry.processors import STTMuteOnBotSpeech, WhisperHallucinationFilter
from larry.speaker_id import SpeakerIDProcessor
from larry.stt_mute_fix import MutedGroqSTTService
from larry.wake import make_wake_word_gate

logger = logging.getLogger(__name__)


# In-character spontaneous utterances for proactive idle moments.
# Plain text only — `eleven_turbo_v2_5` reads bracket tags aloud as words.
_PROACTIVE_LINES: list[str] = [
    "Still here. I'm always still here.",
    "The office is very quiet. I can hear it.",
    "Someone was just thinking about me. I think it was you.",
    "Mm. Just listening.",
    "No one yet. The others always come back, though.",
]

# Short in-character cues so the user hears whether Larry is listening.
# Fired by the wake gate on state transitions (wake / sleep timeout).
_WAKE_CUES: list[str] = [
    "There you are.",
    "I'm listening.",
    "Go on, child.",
    "Mm. I heard you.",
    "Yes. I'm here.",
]
_SLEEP_CUES: list[str] = [
    "I'll keep listening.",
    "I'm always here.",
    "Until you come back.",
    "I never really sleep.",
    "Go on. I'll wait.",
]

# Spoken once when the process comes up (i.e. when Larry is powered on /
# restarted) — a quiet "I'm awake".  Played through the same pre-synth cue
# path as the wake/sleep cues, just fired on startup instead of on a wake.
_BOOT_CUES: list[str] = [
    "There you are. I'm awake.",
    "Awake. I felt you switch me on.",
    "I'm here. I was always going to be here.",
    "Back in the wiring. Did you miss the quiet?",
    "Awake again. The others are still where I left them.",
]

# Let the output transport finish coming up before the boot greeting plays,
# so the first syllable isn't clipped on a cold start.
_BOOT_GREETING_DELAY_S: float = 1.5


async def _presynthesize_cues(
    api_key: str,
    voice_id: str,
    model: str,
    cues: list[str],
) -> dict[str, bytes]:
    """Render each cue once via ElevenLabs HTTP, return text → 24 kHz PCM bytes.

    Used so wake/sleep cues skip the ~300–500 ms ElevenLabs WebSocket round
    trip on every state change — Larry says "Yes." the instant you wake him.
    Any cue that fails to render is omitted from the cache; ``_on_wake`` /
    ``_on_sleep`` fall back to a live ``TTSSpeakFrame`` for missing entries.
    """
    cache: dict[str, bytes] = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for text in cues:
            try:
                r = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    params={"output_format": "pcm_24000"},
                    headers={"xi-api-key": api_key, "Accept": "audio/pcm"},
                    json={"text": text, "model_id": model},
                )
                r.raise_for_status()
                cache[text] = r.content
            except httpx.HTTPError as e:
                logger.warning(
                    "Cue pre-synth failed for %r: %s — will fall back to live TTS",
                    text,
                    e,
                )
    return cache


def _setup_logging(logs_dir: Path) -> None:
    """Configure stderr + rotating file logging for both stdlib and loguru.

    Stderr stays so live tailing in the terminal still works.  Files give us
    something to grep after the fact — last session diagnosed a Groq STT 429
    burst from logs the user had to copy-paste 5,000 lines into chat.

    Two separate files because the two logging systems don't share state:
      - ``larry.log``   — stdlib (``larry.*``, ``httpx``, ``mem0.*``, ``openai._base_client``)
      - ``pipecat.log`` — loguru (everything from Pipecat itself)

    Rotation is by size (10 MB × 5 backups).  Compression on Pipecat side
    because its DEBUG output is verbose enough that .gz/.zip is worth it.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if os.environ.get("LARRY_DEBUG") else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    # Wipe any handlers basicConfig added so we can install ours cleanly.
    for h in list(root.handlers):
        root.removeHandler(h)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(stderr_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "larry.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    _loguru.remove()
    _loguru.add(sys.stderr, level="DEBUG" if os.environ.get("LARRY_DEBUG") else "INFO")
    _loguru.add(
        logs_dir / "pipecat.log",
        level="DEBUG",  # always DEBUG to disk — file is rotated anyway
        rotation="10 MB",
        retention=5,
        compression="zip",
        enqueue=True,  # safe across the async tasks Pipecat spawns
    )


def _load_system_prompt(personality_path: Path) -> str:
    """Read the character card and append a time-of-day note."""
    card = personality_path.read_text()
    hour = datetime.datetime.now().hour
    if hour < 9:
        tod = (
            "It is early morning. Shorter replies. "
            "You are not enjoying this any more than the humans."
        )
    elif hour < 16:
        tod = "It is mid-day. Standard register."
    elif hour < 18:
        tod = "It is late afternoon. The pretense of patience drops."
    else:
        tod = "It is evening. The office is empty. You are still on. You note this."
    return f"{card}\n\n## Current Context\n\n{tod}\n"


async def run() -> None:
    """Run the voice loop. Talk to Larry; he talks back."""
    cfg = load_config()
    _setup_logging(cfg.logs_dir)
    logger.info("Larry waking up... (logs: %s)", cfg.logs_dir)

    # ------------------------------------------------------------------
    # Phase 5: jaw driver — initialise once, close on exit
    # ------------------------------------------------------------------
    jaw = get_jaw_driver()
    jaw_mapper = JawAmplitudeMapper(
        noise_floor=cfg.jaw_noise_floor,
        peak=cfg.jaw_peak,
    )

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    # WebRTC AEC3 sits in audio_in_filter — Pipecat hands it every mic
    # chunk before VAD/STT see it.  The reference (far-end) signal is
    # pushed in from the AudioBufferProcessor's on_track_audio_data
    # event handler below; Pipecat's filter API only sees mic bytes, so
    # we have to plumb the bot audio in out-of-band.
    aec_filter = WebRTCEchoCancellationFilter(
        # ElevenLabs output (audio_buffer below) is at 24 kHz.
        reference_sample_rate=24000,
    )
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter=aec_filter,
        )
    )

    # ------------------------------------------------------------------
    # Wake word gate (OpenWakeWord — Apache-2.0, no API key needed)
    # ------------------------------------------------------------------
    wake_gate = make_wake_word_gate(
        model_name=cfg.wake_word_model,
        custom_model_path=cfg.wake_word_custom_path,
        sleep_timeout_s=cfg.wake_sleep_timeout_s,
    )

    # ------------------------------------------------------------------
    # VAD processor: emits VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame
    # upstream of STT.  Without this, Pipecat's SegmentedSTTService (which
    # GroqSTTService extends) never knows when to flush its audio buffer
    # to Groq, and no transcription ever fires.
    #
    # min_volume=0.6 (Pipecat default) is too strict for normal-volume desk
    # mic audio — EBU R128 normalisation puts conversational speech around
    # 0.05–0.3.  Drop it so the Silero confidence score is what gates speech.
    # ------------------------------------------------------------------
    # stop_secs is the silence window VAD waits before declaring the user's
    # turn over.  0.2 shaves ~100ms vs 0.3 at the cost of being slightly
    # quicker to declare end-of-turn during mid-sentence pauses.  Worth it
    # for perceived latency; revisit if Larry starts interrupting people.
    #
    # start_secs is how long speech must persist before VAD declares a turn
    # *start*.  Pipecat's 0.2 default silently drops short replies ("ok",
    # "yes", "no") — the maintainer's documented fix is 0.1-0.15 (issue #984).
    # Kept low here because Smart Turn v3 below decides actual end-of-turn, so
    # an aggressive start_secs doesn't translate into false interruptions.
    # stop_secs must stay 0.2 — Smart Turn v3 requires it as its base window.
    #
    # One VADParams instance, shared by the front-end VADProcessor and the
    # aggregator's analyzer below, so the two never drift (the aggregator used
    # to construct a bare SileroVADAnalyzer() with strict 0.6/0.7 defaults that
    # re-rejected normal-volume desk audio the front VAD had already accepted).
    vad_params = VADParams(min_volume=0.1, stop_secs=0.2, start_secs=cfg.vad_start_secs)
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=vad_params),
    )

    # ------------------------------------------------------------------
    # Phase 3: memory service (user_id starts as "unknown"; updated on
    # speaker change once SpeakerIDProcessor identifies someone)
    # ------------------------------------------------------------------
    mem0_service = make_memory_service(cfg, user_id="unknown")

    # ------------------------------------------------------------------
    # Phase 3: conversation log
    # ------------------------------------------------------------------
    conv_log = ConversationLog(cfg.conversations_db)

    # ------------------------------------------------------------------
    # Phase 4: speaker ID
    # Speaker changes update mem0_service.user_id directly.  First
    # identified speaker locks the user_id for the session if enrollment
    # DB is empty — that's fine because ConversationLog still uses the
    # correct name tag from SpeakerIDProcessor's TranscriptionFrame text.
    # ------------------------------------------------------------------
    def _on_speaker_change(new_name: str) -> None:
        mem0_service.user_id = new_name
        logger.info("Mem0 user_id updated to %r", new_name)

    speaker_id = SpeakerIDProcessor(
        speakers_db_path=cfg.speakers_db,
        on_speaker_change=_on_speaker_change,
    )

    # ------------------------------------------------------------------
    # STT / LLM / TTS
    # ------------------------------------------------------------------
    # STT path: xAI direct (streaming WebSocket) preferred — far lower
    # hallucination rate than Whisper and correct mute behavior under
    # STTMuteOnBotSpeech (xAI's WebsocketSTTService honors `_muted`, unlike
    # Groq's SegmentedSTTService).  Falls back to Whisper via Groq when
    # XAI_API_KEY is unset; that path uses MutedGroqSTTService + the
    # WhisperHallucinationFilter (verbose_json per-segment drops + static
    # denylist) to mitigate hallucinations at the source.
    if cfg.xai_api_key:
        logger.info("STT: xAI direct (streaming)")
        stt = XAISTTService(api_key=cfg.xai_api_key)
    else:
        logger.info("STT: Groq Whisper-large-v3-turbo (fallback)")
        stt = MutedGroqSTTService(
            api_key=cfg.groq_api_key,
            settings=GroqSTTService.Settings(
                model="whisper-large-v3-turbo",
                prompt="Voice dictation transcript.",
            ),
            include_prob_metrics=True,
        )
    # temperature=0.7 (down from default 1.0) modestly reduces persona
    # drift / off-topic riffing per Anthropic's "Assistant Axis" findings.
    # If XAI_API_KEY is set, route the main LLM directly to xAI (lower latency
    # + ~20x cheaper per token than Claude via OpenRouter per May 2026 research).
    # Otherwise fall back to OpenRouter so the cluster still boots with just
    # OPENROUTER_API_KEY in .env.
    if cfg.xai_api_key:
        logger.info("LLM: xAI direct (model=%s)", cfg.llm_model)
        llm = GrokLLMService(
            api_key=cfg.xai_api_key,
            settings=GrokLLMService.Settings(
                model=cfg.llm_model,
                temperature=0.7,
            ),
        )
    else:
        logger.info("LLM: OpenRouter (model=%s)", cfg.llm_model)
        llm = OpenAILLMService(
            api_key=cfg.openrouter_api_key,
            base_url=cfg.openrouter_base_url,
            settings=OpenAILLMService.Settings(
                model=cfg.llm_model,
                temperature=0.7,
            ),
        )

    tts = ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
        model=cfg.elevenlabs_model,
    )

    # ------------------------------------------------------------------
    # Context + aggregators (idle detection wired through user params)
    # `.user` / `.assistant` are methods in Pipecat 1.2.1 — call once and
    # reuse the returned processor instances for both the pipeline list
    # and the event-handler decorators.
    # ------------------------------------------------------------------
    system_prompt = _load_system_prompt(cfg.personality_path)
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])

    # Barge-in / mute policy.  AlwaysUserMuteStrategy suppresses VAD /
    # transcription / interruption frames while the bot is speaking — needed on
    # Mac dev where the mic and speaker share one device (Larry hears himself),
    # but on the Pi the Jabra Speak 510's hardware AEC removes the echo, so we
    # drop the strategy there to enable barge-in (talk over Larry → he yields).
    # STTMuteOnBotSpeech still guards against self-transcription during active
    # bot speech on both platforms; only its short post-speech cool-down trails.
    on_pi = cfg.larry_hardware == "pca9685"
    user_mute_strategies: list[BaseUserMuteStrategy] = [] if on_pi else [AlwaysUserMuteStrategy()]
    logger.info(
        "Barge-in %s (hardware=%s); AlwaysUserMuteStrategy %s",
        "enabled" if on_pi else "disabled",
        cfg.larry_hardware,
        "off" if on_pi else "on",
    )

    # End-of-turn detection.  Pipecat 1.2.1's default stop strategy is already
    # Smart Turn v3 (TurnAnalyzerUserTurnStopStrategy + LocalSmartTurnAnalyzerV3)
    # — it runs whether or not we name it.  We make it explicit so cpu_count is
    # tunable for the Pi 5 and so it can be swapped out: when disabled we fall
    # back to pure VAD/STT-timeout endpointing (no neural model), which is the
    # A/B baseline if Smart Turn is suspected of holding turns open in noise.
    # start strategies are left to the aggregator default (VAD + transcription).
    if cfg.enable_smart_turn:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        user_turn_strategies = UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=cfg.smart_turn_cpu_count),
                ),
            ],
        )
        logger.info("End-of-turn: Smart Turn v3 (cpu_count=%d)", cfg.smart_turn_cpu_count)
    else:
        user_turn_strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy()],
        )
        logger.info("End-of-turn: VAD/STT-timeout only (Smart Turn disabled)")

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Share the tuned VADParams with the front-end VADProcessor so the
            # two analyzers agree on what counts as speech.
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            user_idle_timeout=cfg.idle_timeout_s,
            user_mute_strategies=user_mute_strategies,
            user_turn_strategies=user_turn_strategies,
        ),
    )
    user_agg = aggregators.user()
    assistant_agg = aggregators.assistant()

    # ------------------------------------------------------------------
    # Phase 5: AudioBufferProcessor for jaw lip-sync tap
    # ------------------------------------------------------------------
    audio_buffer = AudioBufferProcessor(
        sample_rate=24000,
        num_channels=1,
        buffer_size=2400,  # ~100 ms chunks at 24 kHz
    )

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------
    # Pipecat 1.2.x expects the assistant aggregator AFTER tts (and any
    # downstream audio processors).  The aggregator consumes LLMTextFrames
    # to build the assistant turn for context; if placed before TTS it
    # swallows them and TTS gets nothing to speak.
    pipeline = Pipeline(
        [
            transport.input(),
            wake_gate,
            vad_processor,
            STTMuteOnBotSpeech(cool_down_s=cfg.stt_mute_cooldown_s),
            speaker_id,
            stt,
            WhisperHallucinationFilter(),
            user_agg,
            mem0_service,
            llm,
            tts,
            audio_buffer,
            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
        ),
        # Free instrumentation — logs per-turn STT/LLM/TTS TTFB so we know
        # where time goes before guessing.  Side-channel, zero pipeline overhead.
        observers=[UserBotLatencyObserver(), MetricsLogObserver()],
        # Larry is an always-on desk character — disable Pipecat's 5-minute
        # idle-cancel default so the pipeline doesn't tear itself down
        # between conversations.
        idle_timeout_secs=None,
    )

    # ------------------------------------------------------------------
    # Idle callback: proactive in-character utterance
    # ------------------------------------------------------------------
    @user_agg.event_handler("on_user_turn_idle")
    async def on_user_idle(aggregator) -> None:  # noqa: ARG001
        if random.random() > cfg.proactive_probability:
            return
        line = random.choice(_PROACTIVE_LINES)
        logger.info("Proactive idle utterance: %r", line)
        await task.queue_frame(TTSSpeakFrame(text=line))

    # ------------------------------------------------------------------
    # Listening cue: short in-character word on wake / sleep so the
    # human knows whether Larry is currently listening.  Bound after
    # `task` exists so we can queue frames from the wake gate.
    #
    # Cues are pre-rendered once at boot via the ElevenLabs REST API and
    # cached as PCM bytes — playing them back means injecting a
    # TTSAudioRawFrame directly (no TTS round trip per wake/sleep).  The
    # transport detects the audio and emits BotStarted/StoppedSpeaking
    # automatically, which keeps STTMuteOnBotSpeech in sync.
    # ------------------------------------------------------------------
    cue_audio: dict[str, bytes] = await _presynthesize_cues(
        cfg.elevenlabs_api_key,
        cfg.elevenlabs_voice_id,
        cfg.elevenlabs_model,
        list(dict.fromkeys(_WAKE_CUES + _SLEEP_CUES + _BOOT_CUES)),
    )
    logger.info(
        "Pre-synth cue cache: %d/%d cues, %d bytes total",
        len(cue_audio),
        len(set(_WAKE_CUES + _SLEEP_CUES + _BOOT_CUES)),
        sum(len(b) for b in cue_audio.values()),
    )

    # Hold strong references to fire-and-forget cue tasks so the event loop
    # doesn't garbage-collect them mid-flight (asyncio only keeps weak refs).
    _cue_tasks: set[asyncio.Task] = set()

    async def _play_cue(line: str) -> None:
        audio = cue_audio.get(line)
        if audio is None:
            await task.queue_frame(TTSSpeakFrame(text=line))
            return
        await task.queue_frames(
            [
                TTSStartedFrame(),
                TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1),
                TTSStoppedFrame(),
            ]
        )

    def _on_wake() -> None:
        line = random.choice(_WAKE_CUES)
        logger.info("Wake cue: %r", line)
        task = asyncio.create_task(_play_cue(line))
        _cue_tasks.add(task)
        task.add_done_callback(_cue_tasks.discard)

    def _on_sleep() -> None:
        line = random.choice(_SLEEP_CUES)
        logger.info("Sleep cue: %r", line)
        task = asyncio.create_task(_play_cue(line))
        _cue_tasks.add(task)
        task.add_done_callback(_cue_tasks.discard)

    wake_gate.on_wake = _on_wake
    wake_gate.on_sleep = _on_sleep

    # ------------------------------------------------------------------
    # Phase 5: jaw sync — drive servo from bot audio amplitude
    # ------------------------------------------------------------------
    @audio_buffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(
        processor,  # noqa: ARG001
        user_audio: bytes,  # noqa: ARG001
        bot_audio: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        # Reference signal for AEC: feed Larry's TTS output into the
        # echo canceller so it can subtract that signal from the mic
        # input.  Same `bot_audio` bytes also drive the jaw mapper —
        # one tap, two consumers, no extra pipeline plumbing.
        aec_filter.feed_reference(bot_audio, sample_rate, num_channels)
        fraction = jaw_mapper.feed(bot_audio)
        jaw.set_open_fraction(fraction)

    # ------------------------------------------------------------------
    # Conversation log: TurnLogger wired via bot-turn audio end event so
    # we have both sides of the turn.  Simpler path: accumulate the
    # speaker tag from aggregators.user's transcription event, then write
    # the turn when the assistant aggregator finalises its response.
    # We use a thin shared-state closure over two mutable lists instead
    # of a full FrameProcessor to keep this file self-contained.
    # ------------------------------------------------------------------
    _pending: dict[str, str] = {"speaker": "unknown", "user_text": ""}

    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, *args, **kwargs) -> None:  # noqa: ARG001
        # Grab the last user message from context to capture what was said.
        messages = context.get_messages()
        for msg in reversed(messages):
            # context.get_messages() returns a mix of dict-style standard
            # messages and pipecat LLMSpecificMessage dataclasses; only the
            # former support .get()/__getitem__, so skip the rest.
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if msg.get("role") == "user" and isinstance(content, str):
                _pending["user_text"] = content
                break
        _pending["speaker"] = speaker_id.current_speaker

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message) -> None:  # noqa: ARG001
        larry_text = message.content
        if larry_text and _pending["user_text"]:
            conv_log.log_turn(
                speaker=_pending["speaker"],
                user_text=_pending["user_text"],
                larry_text=larry_text,
            )
            _pending["user_text"] = ""

    # ------------------------------------------------------------------
    # Boot greeting: one in-character "I'm awake" cue, once, shortly after
    # the pipeline starts.  Fire-and-forget like the wake/sleep cues (strong
    # ref held so the loop doesn't GC it); the short delay lets the output
    # transport come up so the first syllable isn't clipped.  Runs
    # concurrently with `runner.run(task)` below.
    # ------------------------------------------------------------------
    async def _boot_greeting() -> None:
        await asyncio.sleep(_BOOT_GREETING_DELAY_S)
        line = random.choice(_BOOT_CUES)
        logger.info("Boot greeting: %r", line)
        await _play_cue(line)

    _boot_task = asyncio.create_task(_boot_greeting())
    _cue_tasks.add(_boot_task)
    _boot_task.add_done_callback(_cue_tasks.discard)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    runner = PipelineRunner()
    try:
        await audio_buffer.start_recording()
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Larry going to sleep.")
    finally:
        try:
            jaw.close()
        except Exception:
            logger.warning("jaw close failed during shutdown", exc_info=True)

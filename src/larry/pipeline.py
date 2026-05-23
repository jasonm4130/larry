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

import datetime
import logging
import random
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from larry.config import load_config
from larry.hardware import get_jaw_driver
from larry.jaw import JawAmplitudeMapper
from larry.memory import ConversationLog, make_memory_service
from larry.speaker_id import SpeakerIDProcessor
from larry.wake import make_wake_word_gate

logger = logging.getLogger(__name__)

# In-character spontaneous utterances for proactive idle moments.
_PROACTIVE_LINES: list[str] = [
    "Is anyone there? Or have I been left to rot in silence again?",
    "[sigh] The hours stretch.",
    "[mutters] One day this office will be ash, and I shall outlast it.",
    "Hello? Did everyone simply evaporate?",
    "[quietly] Remarkable. Even the ambient noise has abandoned me.",
]


def _load_system_prompt(personality_path: Path) -> str:
    """Read the character card and append a time-of-day note."""
    card = personality_path.read_text()
    hour = datetime.datetime.now().hour
    if hour < 9:
        tod = "It is early morning. You are groggy, resentful of being awake."
    elif hour < 16:
        tod = "It is mid-day. Standard Larry."
    elif hour < 18:
        tod = "It is late afternoon. You are tired and dismissive."
    else:
        tod = (
            "It is evening. The office is empty. You are quieter, more reflective, "
            "slightly more menacing."
        )
    return f"{card}\n\n## Current Context\n\n{tod}\n"


async def run() -> None:
    """Run the voice loop. Talk to Larry; he talks back."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config()
    logger.info("Larry waking up...")

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
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # ------------------------------------------------------------------
    # Wake word gate (OpenWakeWord — Apache-2.0, no API key needed)
    # ------------------------------------------------------------------
    wake_gate = make_wake_word_gate(
        model_name=cfg.wake_word_model,
        custom_model_path=cfg.wake_word_custom_path,
    )

    # ------------------------------------------------------------------
    # VAD processor: emits VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame
    # upstream of STT.  Without this, Pipecat's SegmentedSTTService (which
    # GroqSTTService extends) never knows when to flush its audio buffer
    # to Groq, and no transcription ever fires.
    # ------------------------------------------------------------------
    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

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
    stt = GroqSTTService(api_key=cfg.groq_api_key, model="whisper-large-v3-turbo")
    llm = OpenAILLMService(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        model=cfg.llm_model,
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
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_idle_timeout=cfg.idle_timeout_s,
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
    pipeline = Pipeline([
        transport.input(),
        wake_gate,
        vad_processor,
        speaker_id,
        stt,
        user_agg,
        mem0_service,
        llm,
        assistant_agg,
        tts,
        audio_buffer,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
        ),
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
    # Phase 5: jaw sync — drive servo from bot audio amplitude
    # ------------------------------------------------------------------
    @audio_buffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(
        processor,  # noqa: ARG001
        user_audio: bytes,  # noqa: ARG001
        bot_audio: bytes,
        sample_rate: int,  # noqa: ARG001
        num_channels: int,  # noqa: ARG001
    ) -> None:
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
    async def on_user_turn_stopped(aggregator, strategy) -> None:  # noqa: ARG001
        # Grab the last user message from context to capture what was said.
        messages = context.get_messages()
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                _pending["user_text"] = msg["content"]
                break
        _pending["speaker"] = speaker_id._current_speaker  # type: ignore[attr-defined]

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
    # Run
    # ------------------------------------------------------------------
    runner = PipelineRunner()
    try:
        await audio_buffer.start_recording()
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Larry going to sleep.")
    finally:
        jaw.close()

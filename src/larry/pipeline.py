"""Larry's voice pipeline (Phase 2 — basic loop, no wake word/memory/speaker ID yet)."""

import datetime
import logging
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from larry.config import load_config

logger = logging.getLogger(__name__)


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

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = GroqSTTService(api_key=cfg.groq_api_key, model="whisper-large-v3-turbo")

    # TODO Phase 3: insert wake word gate (WakeWordProcessor from wake.py) before STT
    # TODO Phase 3: insert Mem0MemoryService between aggregators.user and llm
    # TODO Phase 4: insert SpeakerIDProcessor (Resemblyzer) before STT
    # TODO Phase 5: insert AudioBufferProcessor after TTS for jaw sync tap

    llm = AnthropicLLMService(api_key=cfg.anthropic_api_key)  # default model: claude-sonnet-4-6

    tts = ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
        model="eleven_v3",
    )

    system_prompt = _load_system_prompt(cfg.personality_path)
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        aggregators.user,
        llm,
        aggregators.assistant,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
        ),
    )

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Larry going to sleep.")

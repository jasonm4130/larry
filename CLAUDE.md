# Larry Project Guide

## What Is Larry

Cursed AI character in a motorized Halloween skull on a Raspberry Pi 5; runs on cloud APIs (Anthropic, ElevenLabs, Groq, Mem0) with voice in/out and a servo-controlled jaw.

## Architecture at a Glance

**Orchestration:** Pipecat (single Python process, self-hosted).

**Pipeline order:**
```
LocalAudioTransport
  → SpeakerID (Resemblyzer)
  → GroqSTT
  → Mem0 (short-term memory)
  → OpenAILLM (via OpenRouter → Claude Sonnet 4.6)
  → ElevenLabsTTS
  → AudioBufferProcessor (tap for jaw sync)
  → transport.output
```

**Wake word** ("Hey Larry") gates the pipeline via custom `FrameProcessor` in `wake.py` — Pipecat has no first-party Porcupine plugin. All other services use first-party Pipecat implementations.

## Where to Make Changes

- **Personality / vibe**: `src/larry/personality/larry.md` (character card). Never edit prompts in `pipeline.py`.
- **Audio tags** (`[cackle]`, `[whispers]`, etc.): also `personality/larry.md` — there's an explicit allow-list.
- **Pipeline plumbing** (services, model swaps): `src/larry/pipeline.py`. For LLM model swaps, set `LLM_MODEL` env var (e.g. `LLM_MODEL=google/gemini-2.5-pro`) — no code change needed.
- **Hardware**: `src/larry/hardware/`. `JawDriver` Protocol in `src/larry/jaw.py`. Mock impl in `hardware/jaw_mock.py`, real impl in `hardware/jaw_pca9685.py`. Select via `LARRY_HARDWARE=mock|pca9685`; defaults to mock on macOS, pca9685 on Linux/aarch64.

## Cross-Platform Development

macOS = dev, Pi 5 = production.

- **macOS prerequisite**: `brew install portaudio` (Pipecat's `[local]` extra builds PyAudio against the system portaudio headers). Once is enough.
- **Pi prerequisite**: `sudo apt install portaudio19-dev` (same reason, apt side).
- **macOS**: `GPIOZERO_PIN_FACTORY=mock` makes all GPIO code no-ops. Run `uv sync` (no extras).
- **Pi 5**: Uses `lgpio` pin factory (NOT `RPi.GPIO` — broken on Pi 5; NOT `pigpio` — no Pi 5 support). Run `uv sync --extra pi`.
- **Pi-only deps** (`lgpio`, `adafruit-circuitpython-pca9685`, `adafruit-circuitpython-servokit`) live in `[project.optional-dependencies.pi]`.
- **Dev tools** (`pytest`, `ruff`, `pyright`) live in `[dependency-groups.dev]` and install automatically with `uv sync`.

## Standard Commands

- `uv run larry` — start main loop
- `uv run larry enroll <name>` — record 10s voice for speaker ID
- `uv run larry test-jaw` — sweep servo (Pi) or print mock angles (Mac)
- `uv run pytest` — tests (Mac only; test code avoids hardware imports)
- `uv run ruff check && uv run pyright` — lint and typecheck

## API Keys

- **OPENROUTER_API_KEY**: chat LLM (Claude Sonnet 4.6 by default) + Mem0 fact extraction (Claude Haiku 4.5). Single key for both via OpenRouter.
- **GROQ_API_KEY**: Groq Whisper-large-v3-turbo (STT).
- **ELEVENLABS_API_KEY**: ElevenLabs v3 (TTS).
- **PICOVOICE_ACCESS_KEY**: Porcupine wake word.

Embeddings (Mem0 vector layer) run locally via FastEmbed (BAAI/bge-small-en-v1.5, ONNX, in-process) — no API key, no recurring cost. The `[local]` extra of Pipecat already pulls the audio stack; `fastembed` adds the embedding runtime.

## Pipecat-Specific Gotchas

- Default LLM model is `anthropic/claude-sonnet-4-6` routed via OpenRouter through `OpenAILLMService`. Override with `LLM_MODEL` env var.
- Proactive utterances (speak outside pipeline flow): `await task.queue_frame(TTSSpeakFrame("..."))`
- User idle detection: `UserIdleProcessor` from `pipecat.processors.user_idle_processor` (NOT a transport hook).
- Jaw sync: `AudioBufferProcessor` after TTS, register `@event_handler("on_track_audio_data")`, consume `bot_audio` (not `user_audio`).
- Mem0 blocking: `_store_messages` can lag replies; wrap in `asyncio.create_task` if Larry feels slow.

## Personality Safety Boundaries

**Hard line:** no slurs, no harassment on protected characteristics, no real-coworker impersonation, no self-harm encouragement, no threats.

**Soft line:** stay in character under pushback; deflect rather than refuse for in-bounds prods.

Both encoded in `personality/larry.md` as strength-5 (triple-nested) negative constraints. Don't weaken without explicit reason.

## Working Preferences for AI Agents

When spanning multiple independent files, dispatch parallel sub-agents (one per file or module). Orchestrator stays focused on integration and verification; subagents do file-level writing. Preserves context window across long sessions.

Don't read `RESEARCH_larry_stack.md` by default — reference only. Consult when making stack-level decisions.

## Out of Scope

- Llama Guard or external content safety (Claude's instruction-following is the layer).
- Voice cloning subscription (stock ElevenLabs voice `cPoqAvGWCPfCfyPMwe4z`).
- Zep memory (Mem0 sufficient at our scale).
- Pyannote diarization (Resemblyzer lighter, sufficient for ~15 known speakers).

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

**Wake word** ("Hey Larry") gates the pipeline via custom `FrameProcessor` in `wake.py` — OpenWakeWord (Apache-2.0, no API key). Default model `hey_jarvis`. Train a custom "Hey Larry" via the [OpenWakeWord Colab](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb) and point `WAKE_WORD_CUSTOM_PATH` at the resulting .onnx file.

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

## Audio Hardware

- **macOS dev**: built-in laptop mic + speaker. Acoustic feedback is real (Whisper transcribes Larry's own TTS) — **use headphones for any serious testing**. The software AEC (`src/larry/audio_filter.py`, WebRTC AEC3 via `pywebrtc-audio`) is loaded but marginal on a single-device acoustic loop. See [issue #1](https://github.com/jasonm4130/larry/issues/1).
- **Pi 5 production**: **Jabra Speak 510** (USB-A conference speakerphone — mic + speaker + hardware AEC in one device). Solves the single-enclosure feedback problem in hardware. UAC-compliant, no vendor drivers, `snd-usb-audio` auto-detects.
  - **Known PipeWire bug**: on PipeWire ≥ 23.10 the *speaker* side of the Jabra goes silent (mic still works). Fix: add `default.clock.quantum = 2048` to `/etc/pipewire/pipewire.conf`. Ref: [Ubuntu #2059401](https://bugs.launchpad.net/ubuntu/+source/pipewire/+bug/2059401).
  - **Stay on Bookworm kernel 6.6**, not Trixie 6.12 — USB audio dropouts reported on 6.12.
  - **Don't use HDMI on the Pi for the skull build** — there's a PipeWire/wireplumber restart bug on HDMI hotplug that kills USB audio mid-conversation.
  - **Use the official Pi 5 PSU** (5V/5A). Undervoltage causes Jabra USB transfer failures.
  - **Design implication**: with Jabra handling AEC in hardware, the software AEC + `STTMuteOnBotSpeech` cool-down become belt-and-braces, not primary defense. The skull is a visual prop; voice comes from the Jabra on the desk (jaw still syncs via `bot_audio` tap).

## Standard Commands

- `uv run larry` — start main loop
- `uv run larry enroll <name>` — record 10s voice for speaker ID
- `uv run larry test-jaw` — sweep servo (Pi) or print mock angles (Mac)
- `uv run pytest` — tests (Mac only; test code avoids hardware imports)
- `uv run ruff check && uv run pyright` — lint and typecheck

## API Keys

- **XAI_API_KEY** *(optional, preferred)*: when set, the main chat LLM routes direct to xAI (`grok-4.20-non-reasoning` default — ~600ms TTFT, ~20× cheaper than Claude per May 2026 research). Falls back to OpenRouter if unset.
- **OPENROUTER_API_KEY**: always required for Mem0 fact extraction (Claude Haiku 4.5). Also serves the main chat LLM if XAI_API_KEY is unset.
- **GROQ_API_KEY**: Groq Whisper-large-v3-turbo (STT).
- **ELEVENLABS_API_KEY**: ElevenLabs `eleven_turbo_v2_5` (TTS, voice `cPoqAvGWCPfCfyPMwe4z`).

Wake word runs locally via OpenWakeWord (Apache-2.0) — no API key required.

Embeddings (Mem0 vector layer) run locally via FastEmbed (BAAI/bge-small-en-v1.5, ONNX, in-process) — no API key, no recurring cost. The `[local]` extra of Pipecat already pulls the audio stack; `fastembed` adds the embedding runtime.

## Pipecat-Specific Gotchas

- Default LLM model depends on which provider is active: `grok-4.20-non-reasoning` via `GrokLLMService` when `XAI_API_KEY` is set (preferred path), else `anthropic/claude-sonnet-4-6` via `OpenAILLMService` + OpenRouter. Override with `LLM_MODEL` env var; model name semantics differ by provider (prefixed `x-ai/grok-...` for OpenRouter, plain `grok-4.20-non-reasoning` for direct xAI).
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

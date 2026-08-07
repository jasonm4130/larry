# Larry Project Guide

Voice-driven Halloween skull on a Pi 5. `README.md` covers the stack and quick start, `docs/ARCHITECTURE.md` the pipeline and module map, and `src/larry/pipeline.py`'s module docstring the authoritative pipeline order. Read those rather than a copy here.

## Invariants

**One immutable per-turn identity.** The speaker is snapshotted once at `VADUserStoppedSpeakingFrame` and that frozen value — never a live `current_speaker` — is threaded to the `[speaker: name]` tag, Mem0 retrieval, the deferred Mem0 store, and the conversation log. Any new consumer must take the same snapshot or it will cross-attribute a later turn. `unknown` turns are ephemeral: no Mem0 read, no Mem0 store.

**Identity requires segmented STT.** Keep `STT_PROVIDER=groq` (the default). `STT_PROVIDER=xai` is a single streaming WebSocket shared across turns: it carries the previous utterance's transcript into the next (the "looping" bug, confirmed on hardware) and disables per-speaker attribution entirely. Don't reach for it to cut latency.

**The Strength-5 guardrails in `personality/larry.md` are immutable.** They are re-asserted as a system-prompt footer *after* the self-layer, so nothing Larry appends to `data/larry_self.md` can override them. Don't weaken them and don't move them ahead of the self-layer.

**Never write prompts into `pipeline.py`.** Personality, tone and constraints live in `src/larry/personality/larry.md`, loaded at startup.

**No bracketed stage directions in Larry's output.** The card bans `[cackle]`, `[whispers]`, `[sigh]` and friends outright — the shipping TTS model reads brackets aloud as words. (`eleven_v3` would perform them; it needs alpha access and is not what ships.)

**CAM++ speaker matching fails closed by default.** `SPEAKER_MATCH_THRESHOLD` defaults to `1.0` for `wespeaker_campplus`, which matches nobody until an operator installs a threshold from on-device genuine+impostor calibration. "Identity always says unknown" is expected on a fresh install, not a bug.

## Two subsystems nothing else points at

Neither `README.md` nor `docs/ARCHITECTURE.md` mentions either of these, and neither toggle is in `.env.example` — this is their only live pointer.

- **Self-evolution** (`src/larry/self_layer.py`, `SELF_EVOLUTION_ENABLED`): Larry keeps an append-only self-concept at `data/larry_self.md`, distinct from Mem0's per-person facts. He appends via the `keep_about_self` LLM tool; the layer compacts on sleep past `SELF_LAYER_CAP_CHARS`.
- **Voice tools** (`src/larry/voice_enroll.py`, `VOICE_TOOLS_ENABLED`, default true): in-conversation enrollment and dismiss, driven by a capture state machine in `speaker_id.py`. `WakeWordGate.sleep_now()` is the dismiss entry point; the CLI `enroll` shares the same `store_speaker` helper.

## Cross-platform

macOS = dev, Pi 5 = production. Both need portaudio for Pipecat's `[local]` extra — `brew install portaudio` on Mac, `sudo apt install portaudio19-dev` on the Pi. Once each.

- macOS: `GPIOZERO_PIN_FACTORY=mock`, `uv sync`.
- Pi 5: `uv sync --extra pi`. The pin factory must be `lgpio` — `RPi.GPIO` is broken on Pi 5 and `pigpio` has no Pi 5 support.

Mac dev has a real acoustic feedback loop (Whisper transcribes Larry's own TTS back through the laptop mic). **Use headphones for any serious local testing** — the software AEC in `audio_filter.py` is marginal on a single-device loop. See [issue #1](https://github.com/jasonm4130/larry/issues/1). Pi production uses a Jabra Speak 510 whose hardware AEC solves this properly.

## Commands

- `uv run larry` — main loop
- `uv run larry enroll <name>` — record 10s of voice for speaker ID
- `uv run larry fetch-models` — pre-download the CAM++ embedder; optional (auto-fetched on first run), handy for Pi provisioning
- `uv run larry test-jaw` — sweep the servo (Pi) or print mock angles (Mac)
- `uv run pytest` — Mac only; test code avoids hardware imports
- `uv run ruff check && uv run pyright`

## Gotchas

- `OPENROUTER_API_KEY` is required even when `XAI_API_KEY` is set — Mem0's fact-extraction LLM always routes through OpenRouter. Every other key and tunable is documented in `.env.example`.
- Proactive speech outside pipeline flow: `await task.queue_frame(TTSSpeakFrame(text=...))`.
- Don't read `docs/RESEARCH_larry_stack.md` by default — it's reference material, for stack-level decisions only.

## Settled out of scope

Llama Guard / external content safety, a voice-cloning subscription, Zep, pyannote diarization. Rationale is in `docs/RESEARCH_larry_stack.md`; don't re-litigate without new evidence.

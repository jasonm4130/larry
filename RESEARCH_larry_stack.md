# Larry: Technical Stack Research & Rationale

**Larry** is a cursed AI office companion living in a Halloween skull with a motorized jaw on a Raspberry Pi 5. This document synthesizes the research that informed the final technical stack.

## Table of Contents

- [Project Goal](#project-goal)
- [Research Methodology](#research-methodology)
- [Orchestration: Pipecat](#orchestration-pipecat)
- [Wake Word: Picovoice Porcupine](#wake-word-picovoice-porcupine)
- [Speech-to-Text: Groq Whisper-large-v3-turbo](#speech-to-text-groq-whisper-large-v3-turbo)
- [LLM: Claude Sonnet 4.6 — The Persona Flip](#llm-claude-sonnet-46--the-persona-flip)
- [Text-to-Speech: ElevenLabs v3](#text-to-speech-elevenlabs-v3)
- [Memory: Mem0 Self-Hosted](#memory-mem0-self-hosted)
- [Speaker ID: Resemblyzer](#speaker-id-resemblyzer)
- [Hardware](#hardware)
- [Costs](#costs)
- [Deliberate Non-Features](#deliberate-non-features)
- [Sources](#sources)
- [Research Scope](#research-scope)

---

## Project Goal

Build a cursed AI office friend on a Pi 5 + Halloween skull with motorized jaw. Cloud APIs only. Multi-user (~15 coworkers), multi-month operation (May 2026 onward). Personality consistency matters more than IQ. Budget ~$10/mo recurring, ~$170 one-time hardware.

---

## Research Methodology

Research happened in three phases:

1. **Initial scan** (Apr 2026): Voice/LLM APIs and Pi hardware tooling landscape
2. **Targeted dive** (user request): Grok, Whisper, OpenRouter evaluation
3. **Five-angle deep research** (May 2026): Orchestration frameworks, TTS for creepy voices, LLM persona consistency + safety, memory + speaker ID architectures, real-world precedents

Each layer that follows lists: **what we picked**, **what we rejected**, **why**, with key sources.

---

## Orchestration: Pipecat

**Pick:** Pipecat

Pipecat is the only orchestration candidate that is:
- Fully self-hostable on a Pi 5
- Ships first-party native plugins for Deepgram, Groq, Anthropic, ElevenLabs, Mem0
- Includes interruption by default (Silero VAD + SemanticTurnDetection)
- Has built-in idle / proactive utterance hooks for reactive characters
- Explicitly positioned for "AI companions / characters" (not generic call centers)

For a character that needs sub-second interruption ("Larry, wait—"), stable personality routing, and per-user memory integration, Pipecat's plugin-first architecture sidesteps orchestration glue code entirely.

**Rejected:**
- **Vapi**: Closed-source SaaS only; no self-hosting
- **Vocode**: Last commit Nov 2024; effectively abandoned
- **LiveKit Agents**: Requires running a heavyweight WebRTC SFU server; overscaled for one Pi device
- **Cloudflare Agents SDK**: TypeScript only, runs on Workers; not the right shape for hardware with GPIO + audio
- **Daily Bots**: Pipecat's managed cloud tier, not a separate architectural choice

**Sources:** docs.pipecat.ai; github.com/pipecat-ai/pipecat (BSD-2, 11K stars)

---

## Wake Word: Picovoice Porcupine

**Pick:** Porcupine ("Hey Larry")

Porcupine offers:
- Free tier (1 active keyword per device)
- CPU footprint: sub-4% of a Pi 3 core, ~1MB RAM
- Custom keyword training in Picovoice Console (~30 seconds)
- Reliable in noisy office environments
- Polished cloud console UX for enrollment

Pipecat has no first-party Porcupine plugin (open issue #1985); we implement a custom FrameProcessor that taps raw audio frames and signals to the main orchestration loop.

**Rejected:**
- **OpenWakeWord**: FOSS alternative; slightly more setup overhead. Porcupine wins on console UX and proven noise robustness in office settings
- **Push-to-talk button**: Zero-magic fallback only; kills the "cursed character always listening" personality

**Sources:** picovoice.ai/docs; github.com/Picovoice/porcupine

---

## Speech-to-Text: Groq Whisper-large-v3-turbo

**Pick:** Groq Whisper-large-v3-turbo

Cost + latency win:
- ~$0.04/hr = ~$0.0007/min
- ~10× cheaper than Deepgram Nova-3 ($0.0077/min)
- Fast batch transcription on Groq's LPUs
- Pipecat has a first-party `GroqSTTService`

For a wake-word-gated flow, we don't need streaming STT; Silero VAD (built into Pipecat) handles end-of-turn detection. Batch transcription after silence is sufficient.

**Rejected:**
- **Deepgram Nova-3**: Excellent streaming latency; overkill for our use case (wake-word gating already serializes the input). 10× cost premium not justified
- **OpenAI Whisper API**: No true streaming as of May 2026; batch-only, no cost advantage
- **Local whisper.cpp on Pi 5**: `tiny.en` achieves RTF ~0.3 but accuracy degrades sharply on short utterances (<2 seconds). User constraint: no local inference

**Sources:** console.groq.com pricing; github.com/openai/whisper; docs.deepgram.com

---

## LLM: Claude Sonnet 4.6 — The Persona Flip

**Initial recommendation:** Grok 4.3 (for "edgy" baseline personality)

**Deep research flipped that decision.**

Recent persona stability research converges on a clear finding: Claude Opus / Sonnet 4.6 are the **most persona-robust under multi-turn adversarial pressure**. Grok is the most permissive in tone but the **least robust to being pushed off-persona** — a critical liability when 15 coworkers spend months trying to break Larry.

Key research:
- **ContinuityBench** (Ning, Mar 2026): Persona adherence under adversarial interruptions
- **arXiv 2511.08565**: Multi-turn persona drift
- **arXiv 2601.22812 & 2602.00016**: Instruction-following robustness vs. persona permissiveness (not the same axis)
- **arXiv 2602.00016**: GPT-5.3 shows steepest persona decline at high intensity

**The right move:** Pick the most instruction-following model (Claude), then engineer edginess via **prompting + ElevenLabs audio tags**, rather than relying on an unstable persona ceiling.

**Prompt pattern:** FIVE-style (dev.to/kiro0x, May 2026):
- Input gates over output filters
- Strength-graded negative constraints (1–5 severity)
- Concrete if-then branches over vague adjectives
- Frequency caps on verbal tics

Example:
```
[Input Gate: If user asks for financial advice, decline with a cackle]
[Constraint Level 3: Avoid specific names of real demons; use fictional or archaic references]
[If-then: If interrupted mid-sentence, resume immediately with no acknowledgment]
```

**Rejected:**
- **Grok 4.3**: Permissive but unstable; best-in-tone, worst-in-robustness
- **Gemini 3.1 Pro**: Top-tier instruction-following, no compelling reason over Claude
- **GPT-5.3**: Steepest persona decline at high intensity per arXiv 2602.00016
- **OpenRouter as primary brain**: Text-only, no streaming audio, plus 100% markup on Claude
- **Character.AI / Inworld / Convai**: Less direct control, more vendor lock-in, no meaningful persona advantage

**Sources:** 
- github.com/ning-coeva/continuity-bench
- arXiv 2511.08565, 2601.22812, 2602.00016
- dev.to/kiro0x FIVE article (May 2026)
- Anthropic prompt caching docs (pipecat's default model is `claude-sonnet-4-6`)

---

## Text-to-Speech: ElevenLabs v3

**Pick:** ElevenLabs v3 with stock villain voice ID `cPoqAvGWCPfCfyPMwe4z`

Why ElevenLabs:
- Dedicated villain/horror/demon voice library (Malyx, Matthew Schmitz, Lilith, etc.) — no competitors offer character-archetype presets out of the box
- v3 audio tags (`[cackle]`, `[whispers]`, `[demonic laugh]`) pass through Pipecat untouched; the TTS performs them as character direction
- Starter tier $5/mo covers ~30k characters (~375 replies/month) — ample for office volume
- No voice cloning subscription needed; user selected a pre-trained voice

**Rejected:**
- **Cartesia Sonic**: Lower latency (40-90ms vs ElevenLabs' 75-300ms), but no villain library
- **Hume.ai**: Emotional modeling but no character archetypes
- **PlayHT, Resemble, Inworld**: No compelling tooling for "cursed Victorian demon" out of the box
- **Local TTS (TacotronX, Glow-TTS)**: User constraint: no local inference

**Sources:** 
- elevenlabs.io voice library
- texttolab.com Cartesia vs ElevenLabs comparison (May 2026)
- github.com/pipecat-ai/pipecat (ElevenLabs plugin docs)

---

## Memory: Mem0 Self-Hosted

**Pick:** Mem0 self-hosted

Mem0 provides:
- First-party Pipecat plugin (`Mem0MemoryService`)
- Sits between context aggregator and LLM
- Auto-summarizes facts per-user at no additional cost (self-hosted)
- `user_id` set per-turn from identified speaker → per-person facts surface automatically

Example: Larry remembers Dave's coffee preference, the inside joke about Tuesdays, that Priya always arrives early Friday.

**Known issue #1741:** `_store_messages` can block the conversation. Mitigation: wrap in `asyncio.create_task()` if response latency creeps above acceptable.

**Rejected:**
- **Zep**: $25/mo, better temporal reasoning, but overkill at 15-user, 6-month horizon
- **Letta**: Designed for long-running autonomy, not reactive character responses
- **Rolling context window**: Loses per-person facts after a week or two
- **pgvector DIY**: More boilerplate, no advantage over Mem0's turnkey integration

**Sources:**
- apiscout.dev Zep vs Mem0 vs Letta (2026)
- aicraftguide.com Mem0/Letta/Zep production (Apr 2026)
- github.com/mem0ai/mem0

---

## Speaker ID: Resemblyzer

**Pick:** Resemblyzer

Resemblyzer provides:
- 256-dimensional voice embeddings
- Free, lightweight
- Custom Pipecat FrameProcessor taps `InputAudioRawFrame`s, buffers ~1s of audio
- Cosine-matches against enrolled embeddings (threshold 0.75)
- Enrollment via `uv run larry enroll <name>` (10s of voice per person)

Solves the core need: "Who is talking?" without full diarization machinery.

**Rejected:**
- **Pyannote 3.1**: Best-in-class diarization, but pulls a large torch wheel — heavy on Pi 5. We only need fingerprint matching, not temporal diarization
- **Picovoice Eagle**: <100ms identification latency, but proprietary SDK with tighter licensing
- **Deepgram diarization**: Separates speakers within one stream, doesn't identify by name
- **Google Cloud Speaker ID**: Expensive; overkill for 15 known speakers

**Sources:**
- github.com/resemble-ai/Resemblyzer
- pyannote.ai (comparison reference)

---

## Hardware

**Compute:**
- **Pi 5 4GB**: Not Pi 4 (same idle power, 2-3× faster CPU, cleaner GPIO via RP1); not Zero 2 W (too constrained for orchestration + audio + servo + speaker ID)

**Jaw:**
- **MG90S servo**: Metal gears — SG90's plastic strips after a few hundred cycles
- **PCA9685 PWM driver** ($8): Pi 5 GPIO PWM has audible jitter under CPU bursts; PCA9685 has its own oscillator
- **Separate 5V/2A PSU for servo**: Not Pi's 5V rail (stall current would brown out the Pi; grounds tied)

**Audio:**
- **Fifine K669B USB mic**: Zero-driver on Linux, ~$30, good for desk use
- **Cheap powered USB speaker** (Creative Pebble / Logitech S150): Pi 5 has no onboard analog jack
- **`sounddevice`** for cross-platform audio (PortAudio wrapper; PyAudio unmaintained)

**Software:**
- **`gpiozero` with `lgpio` pin factory**: Pi 5 support (RPi.GPIO and pigpio don't support Pi 5)
- **`gpiozero.MockFactory`** on Mac for same code, no hardware

**Sources:**
- raspberry.tips Pi power comparison 2026
- gpiozero.readthedocs.io
- ben.akrin.com Pi servo jitter analysis

---

## Costs

**Recurring (~$7/mo):**
- Groq STT: ~$0.60/mo (at ~2 min/day average office usage)
- Claude API (with prompt caching): ~$1-2/mo
- ElevenLabs Starter: $5/mo

**One-time hardware (~$170):**
- Pi 5 4GB: $80
- Servo + PCA9685 + PSU: $25
- Mic + Speaker: $45
- Misc (GPIO cable, breadboard, housing): $20

Skull already owned.

---

## Deliberate Non-Features

The following were evaluated and rejected for v1 and beyond:

- **Llama Guard safety filter**: Adds ~150ms latency. Claude's instruction-following is the safety layer at this risk level (private office, not public deployment)
- **Voice cloning**: User picked a stock voice from ElevenLabs' library
- **Zep memory upgrade**: Overkill for 15-user, 6-month horizon
- **Pyannote diarization**: Too heavy for Pi 5; Resemblyzer covers the actual need
- **OpenAI Realtime / Gemini Live (monolithic speech-to-speech)**: Trades away ElevenLabs' character voices + audio tags + Claude's instruction-following for ~500ms latency saving — wrong tradeoff for a character-first product

---

## Sources

**Orchestration & Plugins:**
- github.com/pipecat-ai/pipecat
- docs.pipecat.ai

**LLM Persona Research:**
- github.com/ning-coeva/continuity-bench (ContinuityBench, Ning, Mar 2026)
- arXiv 2511.08565 (Multi-turn persona drift)
- arXiv 2601.22812 (Instruction-following robustness)
- arXiv 2602.00016 (Model intensity vs. persona degradation)

**Prompt Engineering:**
- dev.to/kiro0x (FIVE-style prompting, May 2026)

**TTS & Voice Libraries:**
- elevenlabs.io
- texttolab.com (Cartesia vs ElevenLabs comparison, May 2026)

**STT & Speech APIs:**
- console.groq.com (pricing & models)
- docs.deepgram.com
- github.com/openai/whisper

**Memory & Context:**
- apiscout.dev (Zep vs Mem0 vs Letta, 2026)
- aicraftguide.com (Mem0/Letta/Zep production survey, Apr 2026)
- github.com/mem0ai/mem0

**Speaker ID:**
- github.com/resemble-ai/Resemblyzer
- pyannote.ai

**Hardware & GPIO:**
- raspberry.tips (Pi power comparison, 2026)
- gpiozero.readthedocs.io
- ben.akrin.com (Pi servo jitter analysis)

**Prior Art (Inspiration):**
- github.com/menemy/franky (AI companion in housing)
- github.com/Thokoop/billy-b-assistant (character voice implementation)
- github.com/opsnlops/creature-listener (Pi-based creature with servos)

---

## Research Scope

**Deliberately not researched:**
- Local TTS / STT (user constraint: cloud APIs only)
- Distributed multi-skull deployments (Larry is one device)
- Phone / SIP integration (Larry is a desk character)

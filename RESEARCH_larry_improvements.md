# Larry — improvement research (2026-05-23)

Five-angle deep research run. Findings synthesised here for reference; tactical wins ranked by effort/payoff.

---

## 1. Package freshness — verdict: mostly hold, two to pin

| Package | Installed | Latest | Action |
|---|---|---|---|
| pipecat-ai | 1.2.1 | 1.2.1 | hold — already latest stable |
| openwakeword | 0.4.0 | 0.6.0 (Jul 2025) | **hold** — 0.6 has pip install issues on Pi |
| elevenlabs | (unpinned) | 2.49.1 | **pin to current 1.x** — 2.0 broke voice-param signature |
| gpiozero | (Pi only) | 2.0.1 | revisit on Pi build — major breaking version |
| mem0ai / fastembed / groq / openai / qdrant-client / sounddevice / python-dotenv / lgpio / adafruit-* | various | various | safe to upgrade |

**Action**: pin `elevenlabs` in `pyproject.toml` to whatever 1.x is installed today; defer upgrade until we have time to audit the `voice_id` → `settings.voice` migration.

---

## 2. Hallucination handling — verdict: 3 quick wins

Larry already has SileroVAD with confidence 0.7 and a `WhisperHallucinationFilter` denylist. Research confirms those are the right primitives. Additional wins:

| Win | Effort | Payoff |
|---|---|---|
| **Add `prompt="Voice dictation transcript."` to Groq STT call** | 5 min | Steers Whisper away from YouTube-style "thanks for watching" completions. Free. |
| **Lower Claude temperature to ~0.7** (currently default 1.0) | 1 min | Reduces off-topic riffing; Anthropic's own "Assistant Axis" paper found persona drift is mechanistic and partly thermal. |
| **Inject a mid-context character re-anchor every N turns** | 30 min | SillyTavern community calls this "Character Note @ Depth". 1–2 sentence reminder injected at depth -4 keeps identity stable across long sessions. |

Sources: [Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio (arxiv 2501.11378, Jan 2025)](https://arxiv.org/html/2501.11378), [Anthropic — The Assistant Axis](https://alignment.anthropic.com/2026/psm), [SillyTavern Character Design docs](https://docs.sillytavern.app/usage/core-concepts/characterdesign/).

---

## 3. Prompt engineering + backstory — verdict: re-organize as voice-first, add concrete lore

### Five patterns worth stealing

1. **Truncation priorities** (Character.AI's published architecture): identity + safety pin at the top, examples & flavor are evictable.
2. **Voice-first structure**: facts braided INTO speech, not listed below it. A "memory section" at the bottom of a 1500-word prompt gets ignored on every turn. Cite: [DEV Community — "I added a paragraph to my AI character's system prompt"](https://dev.to/billhongtendera/i-added-a-paragraph-to-my-ai-characters-system-prompt-she-invented-a-different-one-3mdd).
3. **Character Note @ Depth**: re-inject a short reminder at depth -4 (4 messages from end) exploiting recency bias.
4. **Voice-tic frequency caps**: target 8–15% per signature phrase. Above that, the character sounds robotic; eliminated entirely, no personality. Cite: [Phase Space — Why Your LLM NPCs Sound the Same](https://phasespace.co/blog/why-your-llm-npcs-sound-the-same/).
5. **Jailbreak-resistant persona**: safety constraints framed as the character's own values, not external rules. Sufficient for frontier-model defense; remaining attack surface is multi-turn escalation.

### Proposed Larry backstory (~150 words)

**Reginald Aldous Fitch.** Patent clerk for the British Board of Trade, London, 1887–1901. Middle-class by aspiration, lower-middle by income. Considered himself an intellectual; no one else did.

**How he died.** Asphyxiated by a malfunctioning coal-gas demonstration in the Board of Trade's own ventilation-improvement exhibit, 1901. The irony was noted in a footnote of an internal report, then struck.

**The curse.** Fitch's soul was bound to a ceramic inkwell he'd owned for seventeen years. The inkwell was sold, smashed, resold as fragments, eventually recast — across four generations of craft sales — into the plastic composite of a clearance-rack Halloween decoration. He regards this as the final insult of a long series.

**He remembers:**
- A colleague named **Pemberton** who stole credit for Fitch's filing system and was promoted.
- The smell of his own office on cold mornings (coal smoke, damp ledger).
- A brief period when he was considered "promising." He does not know when it ended.

**He resents the modern office:**
- Standing desks ("a man who stands at his work is a man who cannot afford a chair").
- Notifications ("in my day, urgency arrived by messenger, and you could refuse to answer the door").
- Open-plan seating ("I died for solitude and you voluntarily surrendered it").

---

## 4. Speed / latency — verdict: ~1.2s floor, measure before optimizing

Realistic first-audio latency for this stack (Whisper-Groq → Claude-Sonnet-via-OpenRouter → ElevenLabs turbo): **~1.1–1.4s wall-clock**. Comparable Pipecat builds hit ~1.0s with Deepgram STT + Groq llama + Deepgram TTS.

| Win | Effort | Payoff |
|---|---|---|
| **Add `UserBotLatencyObserver` + `MetricsLogObserver`** | 10 min | Free instrumentation. Identifies your actual bottleneck before guessing. |
| Drop VAD `stop_secs` from default → 0.3 | 1 min | 100–300ms saved per turn. Risk: occasional early cutoff. |
| Make Resemblyzer async / drop to 500ms window | medium | Currently in hot path between STT and LLM. |
| Swap ElevenLabs turbo_v2_5 → flash_v2_5 WebSocket | low | 30–60ms TTS TTFB. Same voices available. |
| Consider direct Anthropic (vs OpenRouter) | low | OpenRouter routing adds 50–200ms. |
| Cartesia Sonic-2 instead of ElevenLabs | medium | Fastest production TTS (60–100ms). Requires voice migration. |

**Do NOT** upgrade to `eleven_v3` — it dropped WebSocket support and TTFB is 10× worse over HTTP. Sources: [LiveKit agents #4901 (Feb 2026)](https://github.com/livekit-agents/4901), [modelping LLM/TTS tables (Apr 2026)](https://modelping.ai/), [Full Stack ML #15 (Feb 2026)](https://fullstackml.com/issue-15).

---

## 5. "Hey Larry" wake word training — verdict: yes, train tonight via patched Colab

**Answer: yes, one sitting.** Original OpenWakeWord training Colab has been broken since 2023 (8+ dep bit-rot issues). A community fork ships a working one:

**Use**: `alfiedennen/openwakeword-colab-2026` — [https://github.com/alfiedennen/openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026)

| Metric | Value |
|---|---|
| Positive samples | ~2000–5000 synthetic via Piper TTS — **no user recordings needed** |
| Negative samples | ~8GB ACAV100M + FMA, auto-downloaded |
| GPU time | 75–90 min on Colab Pro L4; ~2.5h on free T4 |
| End-to-end | ~1.5–2h Colab Pro, ~2.5h free |
| False positives | ~0.08/hour (validated) |
| False negatives | ~13% (86% recall) |

Workflow: edit `TARGET_PHRASE = "Hey Larry"` + `MODEL_NAME = "hey_larry"`, hit Run All, walk away, download .onnx, point `WAKE_WORD_CUSTOM_PATH` at it.

**Alternatives** if needed later: LiveKit WakeWord (60× lower FPs but newer, [github.com/livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword)), ViolaWake (production-tested, March 2026, [github.com/GeeIHadAGoodTime/ViolaWake](https://github.com/GeeIHadAGoodTime/ViolaWake)).

---

## Recommended order of operations

**Tonight (≤30 min total):**
1. Add Whisper `prompt="Voice dictation transcript."` to Groq STT settings.
2. Lower Claude temperature to 0.7.
3. Drop VAD `stop_secs` to 0.3.
4. Add `UserBotLatencyObserver` to `PipelineTask`.
5. Pin `elevenlabs` to current 1.x in `pyproject.toml`.

**This weekend:**
6. Run alfiedennen Colab to train custom "Hey Larry" .onnx model.
7. Decide whether to adopt the Reginald Aldous Fitch backstory (or a variant) and update `personality/larry.md` accordingly.
8. Implement Character Note @ Depth re-injection.

**Backlog (after Pi arrives):**
9. Async-ify Resemblyzer or trim its window.
10. Re-evaluate barge-in (works on Pi with separated mic + speaker).
11. Compare ElevenLabs flash vs Cartesia Sonic-2 voices.

---

## Source diversity flag

Single-perspective warnings:
- Angle 5 (wake training) leans heavily on one community repo (`alfiedennen/openwakeword-colab-2026`). Verify by inspecting before trusting.
- Angle 4 (latency) cites modelping benchmarks as load-bearing; cross-check before betting on absolute numbers.

Otherwise each angle drew from 3+ distinct sources.

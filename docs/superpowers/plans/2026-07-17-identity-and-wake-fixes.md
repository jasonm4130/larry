# Larry — Identity & Wake-Word Fixes (P0)

Date: 2026-07-17
Status: draft for review

## Problem

Larry "doesn't quite work right," and the failures cluster in **identity**: he wakes
to the wrong word, doesn't reliably know who's talking, and can attach one person's
memories to another. Four verified root causes (findings below), decomposed — with one
embedder-accuracy swap and two config-freshness fixes — into six tasks numbered in
dependency order. Everything stays inside the current architecture (Pipecat cascade, Mem0).

## Verified findings

1. **Wake word is `hey_jarvis`, not "Hey Larry."** A trained
   `src/larry/wake_models/hey_larry.onnx` (790 KB, real ONNX) is committed but unused:
   config defaults are `wake_word_model=hey_jarvis`, `wake_word_custom_path=None`. The
   character card even concedes "your wake word is still Hey Jarvis."

2. **The `[speaker: name]` tag never reaches the LLM.** `SpeakerIDProcessor` sits
   *upstream* of STT (pipeline.py assembly order), but its tagging branch
   (`speaker_id.py:212-214`) only fires on `TranscriptionFrame`s — which STT emits
   *downstream*, away from it. Verified empirically with a 3-processor harness on the
   installed Pipecat 1.3.0: an upstream processor does **not** receive a
   downstream-emitted `TranscriptionFrame`; the sink got `'hello'`, not
   `'[speaker: test] hello'`. So the entire "Speaker Context" section and all 13
   few-shot examples in `larry.md` describe an input format the model never sees. Larry
   gets the current name only via the system-prompt recency line, and only when a
   known-speaker change fires.

3. **Mem0 facts can bind to the wrong speaker.** `mem0_service.user_id` is written from
   `_on_speaker_change`, which runs in a worker thread (`_identify_speaker` via
   `asyncio.to_thread`) at arbitrary times, and is read live on the event loop by
   `Mem0MemoryService`. A short first utterance from a new speaker can be processed
   under the *previous* speaker's `user_id` → that turn's facts are retrieved/stored
   against the wrong person. `ConversationLog`'s speaker tag has the same exposure.

4. **Speaker ID flickers.** The identify window (`speaker_id.py:160-164`) buffers audio
   unconditionally — no VAD gate — so it embeds silence, room tone, and TTS echo.
   `_identify_speaker` (`232-246`) then fires a change on a **single** window with a
   fixed 0.75 threshold, no hysteresis and no top1–top2 margin. One noisy window flips
   the speaker (and, via #3, the memory namespace).

## Tasks

*Numbered in dependency order. Tasks 1 and 2 are independent; 3 → 4 → 5 → 6 is a chain.*

### Task 1 — Activate the "Hey Larry" wake word  *(independent)*
- In `config.py`, default `wake_word_custom_path` to the committed model when present:
  resolve `src/larry/wake_models/hey_larry.onnx` package-relative (like
  `personality_path`) when the file exists and no env override is set; else keep the
  `hey_jarvis` pretrained path. Preserve `WAKE_WORD_MODEL` / `WAKE_WORD_CUSTOM_PATH`
  env overrides.
- Verify the ONNX output key: `make_wake_word_gate` uses `score_key =
  Path(model_path).stem` → `"hey_larry"`; confirm that matches the model's actual
  prediction key (OWW keys `predict()` output by it). Map correctly if it differs.
- Update the "wake word is still Hey Jarvis" line in `larry.md`.
- **Acceptance rule (not a one-off wake).** Before the custom model becomes the default,
  run a fixed corpus on the Pi in the deployment's acoustic environment:
  - **Recall:** ≥30 "Hey Larry" utterances across the enrolled speakers at desk and
    across-room distance; require **recall ≥ 90%**.
  - **False-accepts:** a **multi-hour soak (≥4 h, ideally an overnight/full workday)** of
    normal conversation, HVAC, and silence. Note the statistics honestly — 30 minutes
    cannot substantiate a ≤0.5/h rate (a true 1/h source has ~60% odds of zero events in
    30 min). Treat ≤0.5/h over the soak as a **promotion threshold, not a proven rate**.
  - **Any threshold change re-runs the whole suite.** A `threshold` bump to kill false
    accepts can push recall below 90%, so recall and the false-accept soak must both be
    re-measured after every change — never accept a threshold validated on only one half.
  - If it misses either, keep `hey_jarvis` as the shipped default and leave the custom
    model reachable via `WAKE_WORD_CUSTOM_PATH` — do not wire-and-ship on the ~13% FN
    training figure.
- **Verify:** unit test that config selects the custom path when the file exists and
  falls back when it doesn't; the on-device corpus above (manual, on-Pi).

### Task 2 — Cheap config freshness fixes  *(independent)*
Two stale model defaults surfaced by the mid-2026 stack research; both are config-string
changes, no code logic. ⚠️ Model IDs are past the research's verification horizon —
**confirm each string is live at implementation** (OpenRouter / ElevenLabs slugs drift).
- **TTS: `eleven_turbo_v2_5` → `eleven_flash_v2_5`.** ElevenLabs' own docs deprecate turbo
  for flash (same voices, lower TTFB). Change the `ELEVENLABS_MODEL` default in `config.py`
  + `.env.example` + the README/ARCHITECTURE mentions. Flash trades a little fidelity for
  latency, so **ear-check Larry's voice** on flash before committing (it's a character
  voice); keep the env override as the fallback. Confirm the voice-settings / streaming
  signature is unchanged (the SDK 2.0 voice-param break was a separate concern).
- **LLM fallback freshness: `claude-sonnet-4-6` → current Sonnet (`claude-sonnet-5`).**
  Only the OpenRouter *fallback* default (used when `XAI_API_KEY` is unset); xAI Grok stays
  the primary path. This is a like-for-like freshness bump — **not** a drop to Haiku: the
  main chat model carries Larry's character and needs the quality tier. (Haiku 4.5 already
  correctly serves Mem0 fact-extraction — the right place for the fast tier — and is
  current; leave it.) Update the default in `config.py` + `.env.example` + docs.
- **Verify:** config-default test asserts the new defaults; a boot smoke-test on the
  OpenRouter fallback path with the new model string; confirm both IDs resolve live before
  merge.

### Task 3 — Swap the speaker embedder behind an interface (Resemblyzer today; TitaNet later)
Resemblyzer (2018 GE2E, ~4.5% EER) is the biggest *verified* accuracy gap in the stack;
TitaNet-Small and ECAPA-TDNN sit at ~0.7–0.9% EER. Better embeddings attack the flicker at
its source, complementing Task 4's gating/hysteresis. **Recommended target (integration
research):** export **NVIDIA TitaNet-Small** (6.4M params, depthwise-separable convs built
for edge use, 192-dim) to **ONNX** and run it in-process via the **already-vendored
`onnxruntime` 1.24.4** (aarch64 wheel already resolved in `uv.lock`), paired with
**`kaldi-native-fbank`** (pip, aarch64 + macOS wheels, no torch) for the Kaldi-style fbank
frontend that NeMo's ONNX export strips out of the graph. Footprint stays ~flat vs today's
Resemblyzer+torch. The NeMo export runs **off-Pi** (dev Mac / cloud — NeMo is too heavy for
the Pi); the resulting `.onnx` is committed like `hey_larry.onnx`. **Fallback: SpeechBrain
ECAPA** — zero preprocessing (fbank bundled in the graph) but adds ~4 heavy deps
(torchaudio/hyperpyyaml/transformers/sentencepiece) and equally-unverified Pi latency; use
only if the ONNX/fbank path proves too fiddly.
- **SDD scope for this run = the model-agnostic half only.** Implement now (testable on the
  Mac): the `SpeakerEmbedder` interface, the Resemblyzer impl behind it, the DB
  schema/migration + cross-model namespacing, and the `SPEAKER_EMBEDDER` env (default
  `resemblyzer`). **Deferred to a manual follow-up** (needs an artifact + a real Pi — cannot
  run in SDD): the NeMo→ONNX TitaNet export, the `kaldi-native-fbank` frontend, the on-device
  latency gate, and the empirical threshold/margin re-tune (the three bullets so marked
  below). Task 4 builds on the interface, so the abstraction must land; the TitaNet impl does
  not gate the identity fixes.
- **Abstract the embedder behind an interface.** Introduce a `SpeakerEmbedder` protocol —
  `embed(audio_f32_16k: np.ndarray) -> np.ndarray` — with a `resemblyzer` impl (today's
  `VoiceEncoder`) and a slot for the new impl, selected by env (`SPEAKER_EMBEDDER`, default
  `resemblyzer` until a new model is validated, mirroring Task 1's A/B discipline). Task 4's
  per-turn embed and the shared encoder `asyncio.Lock` both route through this interface, so
  the concurrency guard is embedder-independent. `speaker_id.py`'s `_encoder` and the
  `voice_enroll` embed lambda become the interface, not a hard Resemblyzer reference.
- **Enrollment DB migration — a future swap invalidates existing voiceprints.** A different
  model is a different embedding space (and a different dim — TitaNet-Small/ECAPA are 192
  vs Resemblyzer's 256), so stored prints from model A are meaningless to model B.
  Persist the embedder name + dim alongside each voiceprint (schema bump on `speakers`);
  refuse to cosine-match a print against a different embedder (fail closed to `unknown`,
  not a garbage score). On a future model change **all ~15 speakers must re-enroll** via the
  CLI `enroll` and the in-conversation enroll path; document it.
- *(Deferred, manual)* **Re-tune threshold + margin.** ECAPA/TitaNet cosine distributions
  differ from Resemblyzer's, so `SPEAKER_MATCH_THRESHOLD` (0.75) and the Task 4
  `SPEAKER_MARGIN` (0.06) defaults must be re-derived from the per-window score logs
  (`f32d60b` already logs them) on real enrolled voices — do not carry Resemblyzer's numbers
  over.
- *(Deferred, manual)* **On-device latency gate — no trustworthy Pi numbers (conflicting
  evidence).** One source (Turku UAS thesis) reports TitaNet-Small at RTF 0.709 on a Pi 4; a
  second research pass could not independently locate *any* ARM CPU number for either model,
  and SpeechBrain users report ONNX CPU inference *slower* than native PyTorch until
  INT8-quantized. Pi-5 latency is therefore **unvalidated** — before the swap becomes the
  default, time the per-turn embed on the Pi for a 2–3 s clip and require it under budget
  (target <300 ms; apply INT8 dynamic quantization if it misses). If neither model clears it,
  keep Resemblyzer (`SPEAKER_EMBEDDER=resemblyzer`).
- *(Deferred, manual)* **Deps + attribution:** the TitaNet path adds `kaldi-native-fbank`
  (dev-Mac and Pi) and the committed `.onnx`; `onnxruntime` is already present. The fbank
  frontend config must **exactly match** the params NeMo baked into the export — a mismatch
  silently degrades embeddings; pin and document them. Weights are **CC-BY-4.0** (attribution
  only — MIT-compatible; add to `README`/`NOTICE`). Caveat: titanet-small's own license
  wasn't directly confirmed (inferred from the family — titanet-large's card is CC-BY-4.0);
  verify the model card before committing the artifact.
- **Verify (this run):** interface parity test (the Resemblyzer impl satisfies
  `SpeakerEmbedder`); a migration test that a print embedded/labelled as model A is never
  matched under model B (→ `unknown`); config-default test that `SPEAKER_EMBEDDER` defaults
  to `resemblyzer` and rejects an unknown value.

### Task 4 — Turn-scoped identification (gate + hysteresis)
- **Identify per turn, not per rolling window (Codex audit P1 — the design pivot).**
  Replace the fire-and-forget 1 s-window identification with a **turn-scoped** one: buffer
  the current turn's voiced audio (VAD-start → VAD-stop, `_vad_voiced and not
  _bot_speaking_for_capture`, mirroring the capture path at `speaker_id.py:353`) and embed
  it **once at VAD-stop** (through the Task 3 `SpeakerEmbedder` interface). That embedding is
  inherently associated with its own turn (no straddling, no post-hoc result arriving for the
  wrong turn) and is more accurate than a 1 s window (2–3 s of speech). The Task 5 snapshot
  awaits this turn's embed (bounded; it typically finishes before the STT network round-trip
  returns, so ~no added latency).
- **Hysteresis across turns.** Switch the *confirmed* speaker only after
  `SPEAKER_CHANGE_TURNS` consecutive turns identify the same best match **and** its
  top1–top2 margin ≥ `SPEAKER_MARGIN`. Until a new speaker is confirmed, their turns
  snapshot `unknown` (Task 5 fail-closed) rather than the previously confirmed name.
- **Single-candidate case (Codex P1).** With <2 enrolled speakers there is no second
  score, so the margin is undefined — waive it and fall back to threshold-only. A fresh
  install with exactly one voiceprint must still identify that speaker, not reject
  everyone. Cover this case explicitly in tests.
- **Validated config with stated defaults (Codex P2).** Add `SPEAKER_CHANGE_TURNS`
  (int ≥ 1, **default 2**) and `SPEAKER_MARGIN` (float in [0, 1], **default 0.06**) as
  `Config` fields with bounds checked in `__post_init__`, matching the existing `_check`
  pattern. Rationale: 2 turns confirms a switch quickly while rejecting a single stray
  identification (wrong-name greetings are the cost being avoided); 0.06 rejects near-ties
  without starving normal matches. `SPEAKER_CHANGE_TURNS=0` (silently disables hysteresis)
  and an out-of-range margin (everyone → unknown) must be rejected at load. Include the
  defaults in the config-default test plus boundary tests.
- **Encoder race — lock, not cancel (Codex P2).** Cancelling an `asyncio` task that wraps
  `asyncio.to_thread()` does *not* stop the torch call already running in the worker
  thread, so "cancel the in-flight identify" is insufficient. Wrap **both** encoder call
  sites (the per-turn identify embed and `finalize_capture`'s enroll embed) in a shared
  `asyncio.Lock` held across the `to_thread` await, or have enroll await actual identify
  completion. Test the specific interleaving (identify running when enroll starts).
- (Same area, cheap) warn on re-enrollment overwrite of an existing name.
- **Verify:** unit tests for gating (silence turn → no embed / `unknown`), turn-scoped
  correlation (a late-arriving embed cannot attribute to a different turn), hysteresis (a
  single deviating turn doesn't flip the confirmed speaker), margin, the single-candidate
  waiver, the config bounds, and a concurrency test that identify + enroll never run the
  encoder simultaneously.

### Task 5 — One immutable per-turn speaker identity, threaded to every consumer
The tag, Mem0 retrieval, the Mem0 **deferred store**, and `ConversationLog` must all use
the *same* identity, frozen at the turn it belongs to — never a live mutable value read
later. This is the core of the fix; findings below are why a naive version fails.
- **Scope: segmented STT only (Codex P1).** This design assumes one transcript per VAD
  turn, so a snapshot taken at the VAD boundary correlates to the transcript that follows.
  That holds for the default `STT_PROVIDER=groq` (segmented). `STT_PROVIDER=xai` is
  *streaming* — it emits `TranscriptionFrame`s asynchronously with no VAD-turn identifier
  and can emit Alice's final transcript after Bob's boundary is recorded, which a FIFO
  snapshot would mis-attribute. xAI STT is already opt-in and discouraged (it's the
  documented source of the cross-turn looping bug); this plan **formally scopes it out** —
  the identity path requires segmented STT, and selecting `xai` must log a loud warning
  that per-speaker attribution is disabled. A streaming-safe correlation mechanism is a
  separate effort, not P0.
- **Snapshot at the turn boundary — fail closed, never inherit (Codex P1 + audit P1).**
  The snapshot must be derived from *this turn's own audio*, which the (pre-Task-4) rolling
  fire-and-forget 1 s windows cannot guarantee. Task 4 makes identification **turn-scoped**
  (embed the turn's own voiced audio at VAD-stop); this snapshot uses that turn-scoped
  result, awaited (bounded) at VAD-stop. Fail-closed rule: if the turn's own identification
  does not positively confirm a named speaker (per Task 4 hysteresis), snapshot **`unknown`**
  — never inherit the previously confirmed speaker into an unconfirmed turn. "unknown" is
  safe (Larry has an in-character register for a new voice) and, per Task 6, carries no
  memory or prior context; a wrong name is not.
- **Tag from the snapshot, not a live getter.** A `SpeakerTagProcessor` placed **after
  `stt`** (before `WhisperHallucinationFilter`) prefixes the `TranscriptionFrame.text`
  with `[speaker: <snapshot>]`. It must read the snapshot for *that* utterance, not
  `get_speaker()` live: STT latency means a later identify task can flip
  `_current_speaker` to Bob before Alice's transcript arrives, mis-attributing her words
  (Codex P2). Carry the snapshot on/with the frame's turn, not off a shared attribute.
- **Fix the Mem0 deferred-store race AND its payload (Codex P1 ×2).** `Mem0MemoryService`
  fires `_store_messages()` as a background task (`create_task`, memory.py:289) that (a)
  reads `self.user_id` **when it runs** (memory.py:193), and (b) stores
  `messages_to_store` built from **every** user/assistant message in the *shared*
  `LLMContext` (memory.py:283-285) — not just the current turn. So even with a frozen
  `user_id`, Bob's store submits Alice's prior messages under Bob. Fix by subclassing
  `Mem0MemoryService` to, at queue time: (1) freeze the turn's snapshot and pass it
  explicitly into the `add` params (overriding the late `self.user_id` read), and (2)
  build the payload from **only the current turn's user message**, not the whole context.
  Use the same snapshot for retrieval.
- Remove the dead tagging branch from `SpeakerIDProcessor.process_frame`; keep it
  single-responsibility (audio → identity). Order the tagger before
  `WhisperHallucinationFilter`; that filter's `^\[speaker:…\]` strip (`processors.py:149`)
  is **non-mutating** — it builds a local stripped string only for the denylist compare
  and must stay that way, so the tag persists on `frame.text` all the way to the LLM.
- **`unknown` is ephemeral (Codex audit P1).** When the turn's snapshot is `unknown`, do
  **not** run Mem0 retrieval or store for it — otherwise every unidentified person shares
  one persistent `unknown` namespace and cross-retrieves. Unknown turns get the persona's
  new-voice register and no memory I/O; only confirmed, named speakers persist to Mem0.
- Source the recency / self-prompt refresh name from the snapshot too.
- **Verify:** (a) pipeline test that a stub-STT `TranscriptionFrame` reaches a downstream
  sink carrying the tag and the LLM-context user message includes it; (b) a test that
  interleaves a speaker change *between* a turn's queue and its deferred store, asserting
  the store still lands under the turn's own snapshot; (c) a test that a late identify
  flip does not re-attribute an already-closed turn's transcript; (d) a test asserting the
  fake Mem0 `add()` **payload** for a turn contains **only that turn's user message** and
  no earlier speaker's message (guards the payload fix, not just the `user_id`); (e) a
  test that an unconfirmed new-speaker turn snapshots `unknown`, never the prior speaker.

### Task 6 — Cross-speaker context boundary
`pipeline.py` builds **one shared `LLMContext`** (line 444) and the user aggregator
appends every turn to it, so after Alice speaks, Bob's next request is sent to the LLM
with Alice's raw prior messages in context — even once Mem0 and `ConversationLog` are
correctly scoped (Codex P1). For an office-desk character this leaks one person's words
to the next.
- **Enforced boundary (specified, not deferred):** drop **all** of the prior speaker's
  raw user/assistant turns from the live context, keeping only the system prompt, whenever
  continuity is broken — that is, on a *confirmed* speaker change **and** on any turn whose
  snapshot is `unknown` while a different speaker was the standing one (Codex audit P1).
  Reset on *unproven* continuity, not only on a *confirmed switch* — otherwise a new person
  reading as `unknown` keeps seeing the prior speaker's transcript. No retained tail — a
  retained turn *is* the leak. Continuity is preserved the way Larry is designed to
  remember — Mem0 facts + the recency line — not by replaying another person's transcript.
- Behavior note for Jason: this trades a little mid-thread "shared-mind" banter for
  correct isolation. If the shared-mind feel is later judged more important, the softer
  alternative (keep one context, tag every retained turn with `[speaker:]` so the model
  attributes) can replace this — but that is a follow-up, not an open question blocking
  this plan. The boundary above is the shipped spec.
- **Verify:** test that a turn following a confirmed speaker change (and a turn that reads
  `unknown` after a different confirmed speaker) contains **zero** raw message content from
  the previous speaker (consistent with the no-tail rule above).

## Sequencing
Tasks 1 and 2 are independent — land them first (fastest visible wins). Then
**3 → 4 → 5 → 6** (dependency order): Task 3 puts the embedder behind the `SpeakerEmbedder`
interface; Task 4 builds turn-scoped identification + hysteresis on that interface and emits
the per-turn embed result + the *confirmed* speaker-change signal; Task 5 consumes that
turn-scoped result for the immutable snapshot / tag / Mem0 binding; Task 6 consumes the
confirmed-change signal for the context boundary. **Task 5 must follow Task 4** — its
snapshot is derived from Task 4's turn-scoped identification, not the reverse.

## Out of scope (deferred to P1/P2 from the audit)
Jaw envelope + off-loop I2C write, self-layer length/rate caps + consolidation backup,
AEC resample-tail bug + tests, larger TTS/LLM/STT swaps (Cartesia, realtime speech-to-speech,
re-benchmarking Grok TTFT). The TitaNet embedder *implementation* (Task 3 ships only the
interface + Resemblyzer + migration this run) and the on-device wake/latency gates (Tasks 1
and 3) are manual follow-ups.

## Risks / notes
- Pipecat 1.3.0 deprecates `PipelineTask` / `PipelineRunner` (warnings observed) and the
  turn-taking comments still say "1.2.1"; not fixed here, but the API is drifting — add
  the regression test that pins the turn-strategy + `VADController` defaults (P2).
- Hysteresis adds a small latency-to-first-correct-name (a couple of turns).
  Acceptable vs. wrong-name greetings; keep `SPEAKER_CHANGE_TURNS` low.
- The custom wake model may underperform `hey_jarvis`; Task 1 must A/B on device, not
  wire-and-ship.
- Task 5's Mem0 subclass depends on `Mem0MemoryService` internals (the `create_task`
  store at memory.py:289 reading `self.user_id` at memory.py:193). Pin that behavior with
  a test so a future `pipecat-ai` bump can't silently reopen the cross-speaker store race.

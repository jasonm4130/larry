# Task 2 report — Cheap config freshness fixes

## What I built

Two stale model-string defaults bumped, pure config change, no logic touched:

1. **TTS**: `ELEVENLABS_MODEL` default `eleven_turbo_v2_5` → `eleven_flash_v2_5` in
   `src/larry/config.py`, `.env.example`, `README.md`, `docs/ARCHITECTURE.md`,
   `CLAUDE.md` (project), and a comment in `src/larry/pipeline.py` that names the
   model.
2. **LLM fallback**: OpenRouter-only default (used when `XAI_API_KEY` is unset)
   `anthropic/claude-sonnet-4-6` → `anthropic/claude-sonnet-5` in the same set of
   files. xAI Grok stays the primary chat path (`XAI_API_KEY` set → unchanged);
   Mem0's fact-extraction model (Haiku 4.5) untouched, as instructed.

No env-var names changed, no new `Config` fields, no `__post_init__` validation
changes — both fields already existed and already had `_check`-free simple string
defaults, so this was a value-only edit inside `load_config()`.

## Live verification of both slugs (brief required this before merge)

- **`eleven_flash_v2_5`** — WebSearch confirmed current, and that ElevenLabs
  "recommends using the Flash models over Turbo models in all use cases" (same
  voices, lower TTFB, no model-list deprecation found). Turbo v2.5 and Flash v2.5
  are documented as functionally equivalent aside from latency, so the streaming/
  voice-settings call signature into `ElevenLabsTTSService` in `pipeline.py` is
  unaffected — confirmed by reading `pipeline.py:431-434`, which passes
  `cfg.elevenlabs_model` straight through with no model-specific branching.
- **`anthropic/claude-sonnet-5`** — cross-checked against the bundled `claude-api`
  skill's current model table (Claude Sonnet 5, `claude-sonnet-5`, cached
  2026-06-24) and then WebSearch-confirmed live on OpenRouter at
  `openrouter.ai/anthropic/claude-sonnet-5` ($2/$10 per MTok intro pricing, 1M
  context, 128K max output) — the exact OpenRouter-prefixed slug the fallback
  path constructs.

Neither slug is hard-guessed; both resolve today per the brief's "verify at
implementation" instruction.

## Not verified — flagging per brief instruction

**"Ear-check Larry's voice on flash before committing"** — could not do this in
this environment: no `ELEVENLABS_API_KEY` / `.env` is present in the worktree
(gitignored, and this sandbox has no audio output device), so there is no way to
actually synthesize and listen to a sample. The env override (`ELEVENLABS_MODEL`)
remains the escape hatch if flash sounds worse in practice — flagging this as an
open human action item, not silently marking it done.

**Boot smoke-test on the OpenRouter fallback path** — same constraint: no
`OPENROUTER_API_KEY` in this sandbox, so an actual live chat-completion call
against `anthropic/claude-sonnet-5` via OpenRouter could not be executed. The
config-default unit test (below) verifies the string is correctly threaded
through `load_config()`; the live-resolution check above (OpenRouter model page)
is the closest substitute available in this environment for "confirm the ID
resolves."

## Tests — TDD evidence (RED → GREEN)

Two existing config-default assertions were updated to the new expected values
first, confirmed to fail for the right reason, then made to pass.

**RED** (`uv run pytest tests/test_config.py -k "test_default_llm_without_xai or test_elevenlabs_defaults" -v`):
```
>       assert cfg.elevenlabs_model == "eleven_flash_v2_5"
E       AssertionError: assert 'eleven_turbo_v2_5' == 'eleven_flash_v2_5'
...
FAILED tests/test_config.py::test_default_llm_without_xai - AssertionError: a...
FAILED tests/test_config.py::test_elevenlabs_defaults - AssertionError: asser...
======================= 2 failed, 56 deselected in 0.17s =======================
```

**GREEN** (same command, after editing `config.py`):
```
======================= 2 passed, 56 deselected in 0.14s =======================
```

**Full suite** (`uv run pytest`):
```
======================= 175 passed, 3 warnings in 54.43s =======================
```
(The 3 warnings are pre-existing `DeprecationWarning`/`UserWarning` from
`pipecat`/`resemblyzer`/`webrtcvad` third-party imports, unrelated to this
change — present on the base commit too.)

**Override tests still pass** — confirming `ELEVENLABS_MODEL` and `LLM_MODEL` env
overrides still take priority over the new defaults (`test_elevenlabs_overrides`,
`test_llm_model_overrides_without_xai`, `test_llm_model_overrides_with_xai`, all
green in the full run above).

**Lint / typecheck:**
```
$ uv run ruff check src/ tests/
All checks passed!

$ uv run pyright src/larry/config.py
0 errors, 0 warnings, 0 informations
```

## Files changed

- `src/larry/config.py` — two defaults + two comments
- `tests/test_config.py` — two assertions updated to new expected defaults
- `.env.example` — TTS and LLM comment blocks
- `README.md` — stack table row (TTS + LLM)
- `docs/ARCHITECTURE.md` — mermaid diagram label + prose (TTS + LLM)
- `CLAUDE.md` (project) — pipeline-order diagram, API-keys section, Pipecat
  gotchas section (not explicitly named in the brief's file list, but is the
  primary AI-agent-facing doc describing these exact defaults; left it stale
  would contradict the point of a "freshness fix" task — this is a doc-string
  edit only, no new decision)
- `src/larry/pipeline.py` — one comment naming the TTS model (no logic change)

Deliberately **not** touched: `docs/RESEARCH_larry_stack.md` (explicitly marked
"reference only, don't read by default" in project CLAUDE.md — it's a dated
historical decision record, not current-state documentation) and
`docs/portfolio-gap-analysis.md` (a dated gap-analysis snapshot referencing the
old default as a documented-at-the-time fact, not a currency claim to fix).

## Self-review

- Implemented exactly the two brief items, nothing extra — no code-logic
  changes, no new config fields, no touched hardware/self-layer/AEC code.
- Diff is small and mechanical: 7 files, 21 insertions / 20 deletions.
- Verified both model IDs live before merge, per the global constraint on this
  task ("verify-at-implementation, do not hard-fail the build on an unresolved
  slug but flag it") — both resolved cleanly, nothing to flag as unresolved.
- The two items I could *not* verify (ear-check, live boot smoke-test) are
  environment-blocked, not skipped — called out above and in `concerns`.

## Concerns

- Flash's audio quality/character-voice fit versus Turbo has not been ear-checked
  by a human — recommend a quick listen on the Pi (or any machine with
  `ELEVENLABS_API_KEY` set) before this merges to production, since Larry's voice
  is a character asset. `ELEVENLABS_MODEL=eleven_turbo_v2_5` in `.env` reverts
  instantly if flash sounds worse.
- No live OpenRouter chat-completion smoke test was run (no API key in this
  sandbox) — the OpenRouter model-catalog page confirms the slug exists and is
  billable, which is strong evidence but not identical to a successful
  `messages.create` round trip. Low risk: `anthropic/claude-sonnet-5` is a
  standard Anthropic-via-OpenRouter slug with no unusual routing requirements.

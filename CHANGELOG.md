# Changelog

All notable changes to Larry are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI that runs the full quality gate (ruff lint, ruff format
  check, pyright, pytest) on every push to `main` and every pull request, with
  hardware mocked and third-party actions SHA-pinned.
- `pre-commit` config mirroring CI: ruff (check + format) plus trailing-whitespace,
  end-of-file, large-file, YAML, and TOML hooks.
- Test suite grown from 4 to 98 tests, with coverage reporting (`pytest-cov`).
- `docs/ARCHITECTURE.md` — a full tour of the wake gate and voice pipeline.
- MIT `LICENSE`.
- Fail-fast config validation: `Config` now rejects out-of-range values
  (probabilities, servo angles/channel, non-positive timeouts) at startup with
  a clear, variable-named error instead of crashing mid-run.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog, and
  a Dependabot config for Python and GitHub Actions updates.

### Changed

- Extracted the custom Pipecat frame processors into `src/larry/processors.py`,
  separating pipeline plumbing from processor logic.
- Reconciled stack documentation with the real implementation and reorganized
  the `docs/` tree.
- Made the type tree pyright-green.

### Fixed

- Crash in `on_user_turn_stopped` caused by a latent `LLMSpecificMessage`
  handling bug.
- Dropped replies after bot speech; the end-of-turn timing is now tunable.
- Wake-gate responsiveness: lowered the Silero VAD threshold (0.5 → 0.3),
  reset the OpenWakeWord prediction buffer on the sleep transition, and required
  consecutive frames above threshold to eliminate sleep↔wake bounces.
- Stopped tracking the `.coverage` artifact and gitignored coverage outputs.
- `ConversationLog.recent_turns` now orders deterministically (`ts DESC, rowid
  DESC`) so turns logged within the same second still return newest-first.
- Hardened cue playback (strong task references prevent GC of in-flight cues)
  and shutdown (a failing `jaw.close()` is logged, not allowed to mask exit).

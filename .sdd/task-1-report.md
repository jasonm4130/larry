# Task 1 — Activate the "Hey Larry" wake word

## What I built

1. **`src/larry/config.py`** — added `_default_wake_word_custom_path()`, a
   package-relative resolver (mirrors `personality_path`'s pattern) that
   returns `str(src/larry/wake_models/hey_larry.onnx)` when that file exists,
   else `None`. Wired into `load_config()`:
   `wake_word_custom_path=os.environ.get("WAKE_WORD_CUSTOM_PATH") or
   _default_wake_word_custom_path()`. `WAKE_WORD_CUSTOM_PATH` /
   `WAKE_WORD_MODEL` env overrides are unchanged and take precedence.

2. **ONNX output-key verification** (brief item: "Verify the ONNX output key
   ... map correctly if it differs") — verified, no code change needed.
   Inspected `openwakeword/model.py`'s `predict()`: for a single-output model
   (`model_outputs[mdl] == 1`), predictions are keyed by `mdl`, which is
   `os.path.basename(path[0:-5])` — i.e. the filename stem, exactly what
   `wake.py`'s `score_key = Path(model_path).stem` already computes. Confirmed
   empirically against the actual committed model:
   - `onnxruntime` inspection of `hey_larry.onnx` shows a single output of
     shape `['batch', 1]` → single-output path.
   - `make_wake_word_gate(custom_model_path="src/larry/wake_models/hey_larry.onnx")`
     → `gate._model_name == "hey_larry"`, `gate._model.models.keys() ==
     ['hey_larry']`.
   - A real `predict()` call on a silent 1280-sample chunk returned
     `{"hey_larry": 0.0}` — the key matches `score_key` exactly.
   No mapping fix was required.

3. **`src/larry/personality/larry.md`** — updated the self-reference line
   from "Your wake word is still 'Hey Jarvis.'..." to reflect "Hey Larry" as
   the active wake word, keeping the character's voice/register.

## Acceptance rule (on-device validation) — explicitly NOT done here

The brief's acceptance rule (≥30-utterance recall ≥90%, ≥4h false-accept
soak, re-run both after any threshold change, on the Pi in the deployment
acoustic environment) is manual and on-Pi by definition — it cannot be
executed from this macOS dev worktree. I did not perform it and did not
claim it passed. The code change makes `hey_larry.onnx` the default
whenever it's present in the checkout (it already is, committed at
`11c9d35`), which activates it in this worktree's config resolution;
whether it should be the *shipped production default* on the actual Pi is
gated on that manual validation per the brief. Flagging this as the primary
concern for whoever deploys next — see Concerns.

## Files changed

- `src/larry/config.py` (+13/-1)
- `src/larry/personality/larry.md` (+1/-1)
- `tests/test_config.py` (+33, new tests only)

## TDD evidence

**RED** — new tests added first, run before implementation:

```
$ uv run pytest tests/test_config.py -k wake_word_custom_path -q
F.F                                                                      [100%]
=================================== FAILURES ===================================
____________ test_wake_word_custom_path_defaults_to_committed_model ____________
    assert cfg.wake_word_custom_path is not None
E   AssertionError: assert None is not None
...
______ test_wake_word_custom_path_falls_back_when_committed_model_missing ______
    monkeypatch.setattr(config_module, "_default_wake_word_custom_path", lambda: None)
E   AttributeError: <module 'larry.config' ...> has no attribute '_default_wake_word_custom_path'
...
2 failed, 1 passed, 58 deselected in 0.22s
```

(The third test, `test_wake_word_custom_path_env_override_wins`, passed
trivially since env-override behavior already existed — included for
completeness of the "unit test that config selects the custom path... and
falls back" requirement.)

**GREEN** — after implementing `_default_wake_word_custom_path()` and wiring
it into `load_config()`:

```
$ uv run pytest tests/test_config.py -k wake_word_custom_path -q
...                                                                      [100%]
3 passed, 58 deselected in 0.14s
```

**Full suite, lint, typecheck** (final, after commit):

```
$ uv run pytest -q
...
178 passed, 3 warnings in 7.29s

$ uv run ruff check src/larry/config.py tests/test_config.py src/larry/personality/larry.md
All checks passed!

$ uv run ruff format --check src/larry/config.py tests/test_config.py
2 files already formatted

$ uv run pyright src/larry/config.py
0 errors, 0 warnings, 0 informations
```

The 3 warnings in the full-suite run are pre-existing (`pkg_resources`
deprecation from `webrtcvad`, unrelated to this change) — same warnings
present in the baseline run before any edits.

## Self-review

- Implemented exactly the brief's three code/doc items; did not touch
  `wake.py` since the key-mapping verification found no bug to fix.
- Did not touch `pipeline.py` — it already passes
  `cfg.wake_word_custom_path` / `cfg.wake_word_model` through to
  `make_wake_word_gate`, so the new default flows through with no change.
- Did not touch root `CLAUDE.md` (the brief named `larry.md` only; the root
  doc's wording about training a custom model via Colab is still accurate
  guidance for anyone regenerating the model, not something this task's
  brief asked to change).
- No new dependency, no public API change, no schema change — nothing to
  escalate.
- Tests exercise real behavior (`load_config()` end to end against the real
  committed file, plus one `monkeypatch` of the resolver function to
  simulate the file's absence, which can't otherwise be produced since the
  real file is committed) rather than mocking `load_config()` itself.

## Concerns

- **Production default is a deployment decision, not just a code
  decision.** This change makes `hey_larry.onnx` the active default the
  moment it's present in a checkout. The brief's acceptance rule requires a
  manual on-Pi recall/false-accept validation before that default should be
  trusted in production; if it fails, the brief says to keep `hey_jarvis`
  shipped and reach the custom model only via `WAKE_WORD_CUSTOM_PATH`. That
  validation is out of scope for this task (explicitly "manual, on-Pi" per
  the brief) and I have not run it — someone must run it before relying on
  this default on the actual skull, and if it fails, `WAKE_WORD_MODEL` /
  unsetting the custom default would need to be revisited (not a code
  change since the env override already exists, just an operational
  choice).

# Contributing to Larry

Larry is a Halloween skull that talks back. The code underneath is taken
seriously even if the skull isn't — so contributions are welcome as long as
they clear the same quality gates the maintainer runs locally and CI runs on
every PR.

## Dev setup

Larry uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
# macOS needs the portaudio headers (PyAudio builds against them):
brew install portaudio          # Pi: sudo apt install portaudio19-dev

uv sync                          # install deps + dev tools (pytest, ruff, pyright)
uv run pre-commit install        # wire up the local hygiene hooks
```

No `--extra pi` needed for development — the hardware is mocked on macOS and on
any non-Linux/aarch64 host (`LARRY_HARDWARE=mock`). You do not need a Raspberry
Pi, a servo, or a skull to work on Larry.

## Quality gates

Run this before you push. It mirrors CI exactly:

```bash
uv run ruff check && uv run ruff format && uv run pyright && uv run pytest
```

- **ruff check** — lint
- **ruff format** — formatting (CI runs `--check`; locally just format)
- **pyright** — type checking; the tree is pyright-green, keep it that way
- **pytest** — the test suite

CI (`.github/workflows/ci.yml`) runs the same gates on every pull request, with
the hardware mocked. If it's green locally, it's green on CI. The `pre-commit`
hooks catch the cheap stuff (ruff, trailing whitespace, EOF, YAML/TOML) before
you ever commit.

## macOS-dev / Pi-prod split

macOS is the development target; the Raspberry Pi 5 is production. All GPIO and
servo code goes through the `JawDriver` Protocol (`src/larry/jaw.py`) with a mock
implementation, so the full pipeline runs on a laptop with no hardware attached.
Keep test code free of hardware imports — that's what lets `uv run pytest` pass
on a Mac.

## Where to make changes

- **Personality, vibe, audio tags** (`[cackle]`, `[whispers]`, …): edit
  `src/larry/personality/larry.md`. This is the character card. Never bury prompt
  text in `pipeline.py`.
- **Pipeline plumbing** (services, model swaps, processors): `src/larry/pipeline.py`
  and `src/larry/processors.py`. For an LLM swap you usually just set `LLM_MODEL`
  — no code change.
- **Hardware**: `src/larry/hardware/`, behind the `JawDriver` Protocol.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline tour.

## PR etiquette

- One logical change per PR. Keep the diff surgical — touch only what the change
  needs.
- Write a descriptive title and a short body explaining the *why*. Conventional,
  imperative subject lines (`Fix dropped replies after bot speech`) match the
  existing history.
- Make sure the quality gates pass and update `CHANGELOG.md` under `[Unreleased]`
  when your change is user-visible.
- Be excellent to each other — see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

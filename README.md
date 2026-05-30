# Larry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A cursed Halloween skull living on your office desk. Raspberry Pi 5 inside, motorized jaw, cloud AI doing all the thinking. People talk to him. He talks back—sardonic, theatrical, and completely unhinged. No local inference, no ethical guardrails, just vibes and servo angles.

## What Is This

Larry is a conversational skull. He has a local wake word detector (listens for "Hey Larry" via OpenWakeWord — no account needed), understands speech via cloud Whisper, generates responses via Claude, and runs them through text-to-speech with custom audio tags like `[cackle]` and `[whispers]`. A PCA9685 servo driver moves his jaw while he speaks. He remembers who you are after you enroll (10 seconds of voice), and his personality is entirely driven by a Markdown file you can edit to make him meaner, sillier, or more pretentious.

**TL;DR:** It's a toy for offices. Not practical. Extremely fun.

## Stack

| Component | Service/Library |
|-----------|-----------------|
| Orchestration | Pipecat |
| Wake word | OpenWakeWord (local, no key) |
| Speech-to-text | Groq Whisper-large-v3-turbo |
| LLM | Claude Sonnet 4.6 via OpenRouter |
| Text-to-speech | ElevenLabs v3 (custom audio tags) |
| Memory | Mem0 (self-hosted) |
| Speaker ID | Resemblyzer |
| Hardware | Raspberry Pi 5, MG90S servo, PCA9685, Fifine K669B mic, USB speaker |

## Quick Start (macOS)

```bash
brew install portaudio          # one-time system dep for audio I/O
cp .env.example .env            # fill in API keys (OpenRouter + Groq + ElevenLabs)

uv sync
uv run larry enroll yourname    # record 10 seconds of your voice
uv run larry                    # start the skull
```

On macOS, hardware is mocked—jaw angles print to stdout. The experience is otherwise identical.

## On a Raspberry Pi

```bash
uv sync --extra pi

# Install the templated systemd unit (replace `jason` with your username):
sudo cp systemd/larry@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now larry@jason   # journalctl -u larry@jason -f
```

You'll need to wire up the PCA9685 to your Pi's I2C bus and point the servo to a real MG90S. See `RESEARCH_larry_stack.md` for hardware detail.

## Personality

Larry's voice, tone, and character live in `src/larry/personality/larry.md`. Edit it to change how sardonic, verbose, or unhinged he is. The file is loaded at startup, so you can tweak vibe without redeploying.

## Cost

- **Recurring:** ~$7/month in cloud API calls (at typical office chatting volume)
- **One-time:** ~$170 in hardware (Pi, servo, driver board, mic, speaker)

## Status

Very early. Built for fun. Will probably eat your batteries and ask existential questions at 3 AM.

## More Reading

See `RESEARCH_larry_stack.md` for the full rationale behind every choice (why Groq over OpenAI's Whisper, why Resemblyzer for speaker ID, etc.).

## License

MIT — see [LICENSE](LICENSE).

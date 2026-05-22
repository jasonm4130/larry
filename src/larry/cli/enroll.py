"""CLI command to enroll a new speaker into Larry's voice-fingerprint database."""

import sqlite3
import time

import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder

from larry.config import load_config
from larry.speaker_id import _ensure_schema

_SAMPLE_RATE = 16000
_DURATION_SECONDS = 10


def main(name: str) -> None:
    """Record 10s of <name>'s voice; compute Resemblyzer embedding; store in SQLite."""
    print(
        f"Enrolling {name}. Speak naturally for {_DURATION_SECONDS} seconds. "
        "Recording starts in 3..."
    )
    for countdown in (3, 2, 1):
        print(countdown)
        time.sleep(1)

    print("Recording...")
    audio = sd.rec(
        frames=_DURATION_SECONDS * _SAMPLE_RATE,
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Done recording.")

    # Resemblyzer expects a 1-D float32 array
    wav = audio.squeeze()

    encoder = VoiceEncoder()
    embedding: np.ndarray = encoder.embed_utterance(wav)

    cfg = load_config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(cfg.speakers_db) as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)",
            (name, embedding.astype(np.float32).tobytes()),
        )
        conn.commit()

    print(f"Enrolled {name}. Embedding shape: {embedding.shape}.")

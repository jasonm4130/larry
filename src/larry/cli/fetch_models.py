"""CLI command to pre-download the speaker-embedder model file.

Run during Pi provisioning (``uv run larry fetch-models``) so the first
conversation doesn't block on the ~28 MB CAM++ download. Idempotent and
checksum-verified (see ``model_fetch.py``); needs no API keys, so it can run
before ``.env`` is fully populated.
"""

from larry.speaker_embedder import _ensure_model_cached


def main() -> None:
    print("Fetching CAM++ speaker-embedder model...")
    path = _ensure_model_cached()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Ready: {path} ({size_mb:.1f} MB)")

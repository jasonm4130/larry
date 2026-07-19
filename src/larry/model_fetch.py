"""Idempotent, checksum-verified cache for downloaded model files.

Fetches a pinned model artifact to a local cache exactly once and verifies its
SHA-256, so a truncated or tampered download is never silently used. Shared by any
component that needs a downloaded model (today: the CAM++ speaker embedder). No
third-party dependency — ``urllib`` from the stdlib.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from loguru import logger

_CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*, read in bounded chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_cached_model(url: str, sha256: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Return *dest*, downloading and verifying it from *url* if not already valid.

    Idempotent: if *dest* already exists and matches *sha256*, returns immediately
    with no network I/O. Otherwise downloads to a temp file in *dest*'s directory,
    verifies the digest, and atomically renames it into place. A checksum mismatch
    raises ``RuntimeError`` and never leaves a bad file at *dest* — callers get a
    valid file or an exception, never a corrupt cache.
    """
    dest = Path(dest)
    if dest.exists():
        actual = sha256_of(dest)
        if actual == sha256:
            return dest
        logger.warning(
            f"Cached model {dest} has sha256 {actual}, expected {sha256} — re-downloading"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model: {url} -> {dest}")
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        actual = sha256_of(tmp)
        if actual != sha256:
            raise RuntimeError(
                f"Downloaded model from {url} has sha256 {actual}, expected {sha256}"
            )
        tmp.replace(dest)
        logger.info(f"Model cached at {dest} ({dest.stat().st_size} bytes, sha256 verified)")
        return dest
    finally:
        tmp.unlink(missing_ok=True)

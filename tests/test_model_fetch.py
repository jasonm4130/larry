"""Unit tests for the checksum-verified model cache. Offline — uses file:// URLs
and a monkeypatched urlopen, never the network."""

import hashlib

import pytest

from larry import model_fetch
from larry.model_fetch import ensure_cached_model, sha256_of


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    data = b"larry" * 1000
    p.write_bytes(data)
    assert sha256_of(p) == _sha(data)


def test_ensure_cached_model_downloads_and_verifies(tmp_path):
    src = tmp_path / "src.onnx"
    data = b"model-bytes-0123456789" * 512
    src.write_bytes(data)
    dest = tmp_path / "cache" / "model.onnx"

    got = ensure_cached_model(src.as_uri(), _sha(data), dest)

    assert got == dest
    assert dest.read_bytes() == data


def test_ensure_cached_model_rejects_bad_checksum(tmp_path):
    src = tmp_path / "src.onnx"
    src.write_bytes(b"corrupt-download")
    dest = tmp_path / "cache" / "model.onnx"

    with pytest.raises(RuntimeError, match="sha256"):
        ensure_cached_model(src.as_uri(), _sha(b"a totally different artifact"), dest)

    # A failed-checksum download must never be left behind as a valid cache entry.
    assert not dest.exists()


def test_ensure_cached_model_idempotent_no_redownload(tmp_path, monkeypatch):
    data = b"already-cached" * 100
    dest = tmp_path / "model.onnx"
    dest.write_bytes(data)

    def _boom(*args, **kwargs):
        raise AssertionError("must not download when the cache is already valid")

    monkeypatch.setattr(model_fetch.urllib.request, "urlopen", _boom)

    got = ensure_cached_model("file:///nonexistent", _sha(data), dest)
    assert got == dest

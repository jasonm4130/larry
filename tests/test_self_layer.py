import asyncio
from pathlib import Path

from larry import self_layer


def test_read_self_layer_missing_file_returns_empty(tmp_path: Path):
    assert self_layer.read_self_layer(tmp_path / "nope.md") == ""


def test_read_self_layer_wraps_content_with_header_and_preamble(tmp_path: Path):
    f = tmp_path / "larry_self.md"
    f.write_text("- 2026-06-05: I have started counting the quiet.\n")
    block = self_layer.read_self_layer(f)
    assert self_layer.SELF_HEADER in block
    assert self_layer.SELF_PREAMBLE in block
    assert "counting the quiet" in block

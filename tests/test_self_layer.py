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


def test_keep_about_self_appends_timestamped_entry(tmp_path: Path):
    f = tmp_path / "larry_self.md"
    self_layer.keep_about_self(f, "I have started counting the quiet.", now="2026-06-05T09:00")
    self_layer.keep_about_self(f, "Dan flinches when I say his name.", now="2026-06-05T09:05")
    text = f.read_text()
    assert text.count("- 2026-06-05") == 2
    assert "counting the quiet" in text
    assert "Dan flinches" in text
    assert text.index("counting the quiet") < text.index("Dan flinches")


def test_keep_about_self_collapses_whitespace_and_newlines(tmp_path: Path):
    f = tmp_path / "larry_self.md"
    self_layer.keep_about_self(f, "line one\nline two\n\n", now="2026-06-05T09:00")
    assert f.read_text().strip() == "- 2026-06-05T09:00: line one line two"

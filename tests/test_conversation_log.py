"""Unit tests for the SQLite ConversationLog in larry.memory (#18).

Self-contained: touches only the local sqlite log, never Mem0 / the cloud
memory layer or any network. Uses pytest's built-in tmp_path / monkeypatch
fixtures so the test file can run in parallel with the rest of the suite
without a shared conftest.
"""

import sqlite3

from larry.memory import ConversationLog


def _log_turn_at(
    log: ConversationLog, ts: str, speaker: str, user_text: str, larry_text: str
) -> None:
    """Insert a turn with an explicit timestamp.

    ``log_turn`` relies on SQLite's ``CURRENT_TIMESTAMP``, which has
    second-resolution. Several inserts within the same wall-clock second tie on
    ``ts``, so the ``ORDER BY ts DESC`` query has no deterministic ordering
    among them. Ordering tests therefore pin distinct timestamps via the same
    connection / schema the class uses, exercising the real ``recent_turns``
    query without depending on the host clock.
    """
    with log._conn:
        log._conn.execute(
            "INSERT INTO turns (ts, speaker, user_text, larry_text) VALUES (?, ?, ?, ?)",
            (ts, speaker, user_text, larry_text),
        )


def test_init_creates_schema(tmp_path):
    db_path = tmp_path / "conv.db"
    ConversationLog(db_path)

    # The constructor should have created the db file and the turns table.
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "turns" in names

        cols = {row[1] for row in conn.execute("PRAGMA table_info(turns)")}
        assert {"id", "ts", "speaker", "user_text", "larry_text"} <= cols

        index_names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert "idx_turns_speaker_ts" in index_names
    finally:
        conn.close()


def test_init_creates_parent_dirs(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "conv.db"
    ConversationLog(db_path)
    assert db_path.exists()


def test_log_and_read_back_single_turn(tmp_path):
    log = ConversationLog(tmp_path / "conv.db")
    log.log_turn("alice", "hello larry", "boo, hello yourself")

    turns = log.recent_turns("alice")
    assert len(turns) == 1
    turn = turns[0]
    assert turn["speaker"] == "alice"
    assert turn["user_text"] == "hello larry"
    assert turn["larry_text"] == "boo, hello yourself"
    # Each row carries the autoincrement id and a timestamp.
    assert turn["id"] == 1
    assert turn["ts"] is not None


def test_recent_turns_newest_first(tmp_path):
    log = ConversationLog(tmp_path / "conv.db")
    _log_turn_at(log, "2026-05-31 10:00:00", "bob", "u1", "l1")
    _log_turn_at(log, "2026-05-31 10:00:01", "bob", "u2", "l2")
    _log_turn_at(log, "2026-05-31 10:00:02", "bob", "u3", "l3")

    turns = log.recent_turns("bob")
    assert [t["user_text"] for t in turns] == ["u3", "u2", "u1"]
    # ids stay monotonically increasing in insertion order, so newest-first
    # ordering means descending ids.
    assert [t["id"] for t in turns] == sorted((t["id"] for t in turns), reverse=True)


def test_recent_turns_respects_limit(tmp_path):
    log = ConversationLog(tmp_path / "conv.db")
    for i in range(5):
        _log_turn_at(log, f"2026-05-31 10:00:0{i}", "carol", f"u{i}", f"l{i}")

    turns = log.recent_turns("carol", limit=2)
    assert len(turns) == 2
    assert turns[0]["user_text"] == "u4"
    assert turns[1]["user_text"] == "u3"


def test_recent_turns_filters_by_speaker(tmp_path):
    log = ConversationLog(tmp_path / "conv.db")
    log.log_turn("dave", "dave-said", "to-dave")
    log.log_turn("erin", "erin-said", "to-erin")

    dave_turns = log.recent_turns("dave")
    assert len(dave_turns) == 1
    assert dave_turns[0]["user_text"] == "dave-said"

    erin_turns = log.recent_turns("erin")
    assert len(erin_turns) == 1
    assert erin_turns[0]["user_text"] == "erin-said"

    assert log.recent_turns("nobody") == []


def test_log_persists_across_instances(tmp_path):
    db_path = tmp_path / "conv.db"
    first = ConversationLog(db_path)
    first.log_turn("frank", "remember me", "always")
    first._conn.close()

    second = ConversationLog(db_path)
    turns = second.recent_turns("frank")
    assert len(turns) == 1
    assert turns[0]["user_text"] == "remember me"

"""Tests that a corrupt sessions index recovers by rebuilding from files
rather than silently hiding existing sessions."""

from datetime import datetime

from agentic_cli.config import BaseSettings
from agentic_cli.persistence.session import SessionPersistence, SessionSnapshot


def _persistence(tmp_path) -> SessionPersistence:
    return SessionPersistence(BaseSettings(workspace_dir=tmp_path))


def _snapshot(session_id: str) -> SessionSnapshot:
    now = datetime.now()
    return SessionSnapshot(
        session_id=session_id,
        created_at=now,
        saved_at=now,
        messages=[{"role": "user", "content": "hi"}],
    )


def test_corrupt_index_rebuilds_from_files(tmp_path):
    p = _persistence(tmp_path)
    for sid in ("a", "b", "c"):
        p.save_snapshot(_snapshot(sid))

    p._get_sessions_index_path().write_text("{ not valid json ", encoding="utf-8")

    listed = {s["session_id"] for s in p.list_sessions()}
    assert listed == {"a", "b", "c"}


def test_non_dict_index_rebuilds_from_files(tmp_path):
    p = _persistence(tmp_path)
    for sid in ("a", "b"):
        p.save_snapshot(_snapshot(sid))

    # A JSON array (not an object) used to raise AttributeError on data.get(...).
    p._get_sessions_index_path().write_text('["not", "a", "dict"]', encoding="utf-8")

    listed = {s["session_id"] for s in p.list_sessions()}
    assert listed == {"a", "b"}


def test_save_after_corrupt_index_does_not_hide_existing(tmp_path):
    """Regression: corrupt index, save a new session, list must still show all.

    Previously a corrupt index read as empty, so the next save reset it to a
    single entry and every prior session vanished from the listing.
    """
    p = _persistence(tmp_path)
    p.save_snapshot(_snapshot("a"))
    p.save_snapshot(_snapshot("b"))

    p._get_sessions_index_path().write_text("totally broken", encoding="utf-8")

    p.save_snapshot(_snapshot("c"))  # triggers index reload (corrupt) -> rebuild

    listed = {s["session_id"] for s in p.list_sessions()}
    assert listed == {"a", "b", "c"}, f"existing sessions were hidden: {listed}"

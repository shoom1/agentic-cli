"""Tests for atomic-write durability hardening in file_utils._atomic_write."""

import json
import threading

from agentic_cli.file_utils import atomic_write_json, atomic_write_text


def test_utf8_content_roundtrips(tmp_path):
    """Non-ASCII content must be written as UTF-8 regardless of locale."""
    path = tmp_path / "u.txt"
    content = "café — naïve — 日本語 — 🚀"
    atomic_write_text(path, content)
    assert path.read_text(encoding="utf-8") == content


def test_json_roundtrips_with_unicode(tmp_path):
    path = tmp_path / "d.json"
    data = {"name": "café", "emoji": "🚀", "nested": [1, 2, {"k": "日本語"}]}
    atomic_write_json(path, data)
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_no_temp_file_left_behind(tmp_path):
    path = tmp_path / "x.json"
    atomic_write_json(path, {"a": 1})
    # Only the target file should exist — no stray .tmp artifacts.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "x.json"]
    assert leftovers == [], f"unexpected leftover files: {leftovers}"


def test_temp_file_cleaned_up_on_serialization_error(tmp_path):
    path = tmp_path / "bad.json"

    class _Unserializable:
        pass

    try:
        atomic_write_json(path, {"obj": _Unserializable()})
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for unserializable data")

    # The failed write must not leave the target or a temp file behind.
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writes_produce_valid_file(tmp_path):
    """Many threads writing the same path: final file is valid, no leftovers.

    A fixed .tmp name would let writers truncate each other's temp file and
    leave torn JSON or a missing file for losers; unique temp names + atomic
    replace make every write self-consistent.
    """
    path = tmp_path / "shared.json"
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            for _ in range(20):
                atomic_write_json(path, {"writer": i, "payload": list(range(50))})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writers raised: {errors}"
    # Final file parses cleanly (some writer's complete payload).
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["payload"] == list(range(50))
    # No temp files left behind.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "shared.json"]
    assert leftovers == [], f"unexpected leftover files: {leftovers}"

"""Atomic writes for the memory package's own files."""

from __future__ import annotations

import json
import os
import stat

import pytest

from agentic_cli.memory._core.io import atomic_write_json, atomic_write_text
from agentic_cli.memory._core.log import get_logger


def test_json_round_trips_and_the_file_is_private(tmp_path):
    path = tmp_path / "nested" / "data.json"

    atomic_write_json(path, {"a": [1, 2]})

    assert json.loads(path.read_text()) == {"a": [1, 2]}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_text_replaces_the_file_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "notes.md"

    atomic_write_text(path, "old")
    atomic_write_text(path, "new — ünïcode")

    assert path.read_text(encoding="utf-8") == "new — ünïcode"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.md"]


def test_a_failed_write_keeps_the_old_content(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    atomic_write_text(path, "old")

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new")

    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.md"]


def test_get_logger_logs_without_host_configuration():
    get_logger("agentic_cli.memory.test").debug("memory_log_smoke", ok=True)

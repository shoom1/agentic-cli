"""Calls that write one file at once run one after the other.

One model step's calls run together in worker threads (both backends), and
``edit_file`` reads a file, then writes it back: two edits of one file both
read it before either wrote, and the second write undid the first, though
both reported success. ``edit_file`` and ``write_file`` hold a lock per file
(by resolved path) from the read to the write.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agentic_cli.tools import file_write


def slow_writes(monkeypatch, *, first_only: bool = False) -> threading.Event:
    """Make each write (or only the first) wait a moment, long enough for
    another call to reach the file, and return an event set as the first
    write begins."""
    real_write = file_write.atomic_write_text
    writing = threading.Event()

    def slow_write(*args, **kwargs):
        if not (first_only and writing.is_set()):
            writing.set()
            time.sleep(0.2)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(file_write, "atomic_write_text", slow_write)
    return writing


def test_two_edits_of_one_file_at_once_both_land(tmp_path, monkeypatch):
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")
    slow_writes(monkeypatch)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(
            lambda edit: file_write.edit_file(str(target), *edit),
            [("alpha", "ALPHA"), ("beta", "BETA")],
        ))

    assert [r["success"] for r in results] == [True, True]
    assert target.read_text() == "ALPHA\nBETA\n"


def test_a_write_during_an_edit_is_not_undone(tmp_path, monkeypatch):
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n")
    writing = slow_writes(monkeypatch, first_only=True)

    with ThreadPoolExecutor(2) as pool:
        edit = pool.submit(file_write.edit_file, str(target), "alpha", "ALPHA")
        assert writing.wait(5)  # the edit has read the file
        write = pool.submit(file_write.write_file, str(target), "gamma\n")
        assert edit.result()["success"] and write.result()["success"]

    assert target.read_text() == "gamma\n"

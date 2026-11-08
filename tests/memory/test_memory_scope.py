"""One scope's folder: reading files as they change, history, the lock (spec §5.3-5.5)."""

from __future__ import annotations

import os
import stat
import threading
from datetime import timedelta

import pytest

from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory.store import _scope
from agentic_cli.memory.store._scope import Entry, ScopeFolder, resolve_duplicates
from agentic_cli.memory.store.models import MemoryBusy, UnsafeMemoryStore
from tests.memory.memory_files import T0, memory_text, write_memory


def _folder(tmp_path) -> ScopeFolder:
    return ScopeFolder(tmp_path / "user", "user")


def test_a_missing_folder_reads_empty(tmp_path):
    assert _folder(tmp_path).scan() == ([], [])


def test_memory_files_are_read_and_others_are_not(tmp_path):
    folder = _folder(tmp_path)
    write_memory(folder.path, "prefers-pytest-4f2a9c1e.md", memory_text())
    write_memory(folder.path, ".prefers.md.abc.tmp", "half written")
    write_memory(folder.path, "notes.txt", "not a memory")
    (folder.path / ".inbox").mkdir()
    entries, problems = folder.scan()
    assert [e.memory.id for e in entries] == ["4f2a9c1e"]
    assert problems == []
    assert entries[0].memory.scope == "user"


def test_a_hand_edit_is_picked_up_at_the_next_scan(tmp_path):
    folder = _folder(tmp_path)
    path = write_memory(folder.path, "m-4f2a9c1e.md", memory_text(content="Old."))
    assert folder.scan()[0][0].memory.content == "Old."
    path.write_text(memory_text(content="New, and longer."), encoding="utf-8")
    assert folder.scan()[0][0].memory.content == "New, and longer."


def test_an_unchanged_file_is_not_read_again(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    write_memory(folder.path, "m-4f2a9c1e.md", memory_text())
    folder.scan()
    reads = []
    real = _scope.read_regular_file
    monkeypatch.setattr(_scope, "read_regular_file", lambda *a, **k: reads.append(a) or real(*a, **k))
    folder.scan()
    assert reads == []


@pytest.mark.parametrize("name, data, kind", [
    ("broken-a1.md", b"no frontmatter\n", "bad_frontmatter"),
    ("bad-type-a1.md", memory_text(type="preference").encode(), "bad_field"),
    ("latin1-a1.md", "---\nid: a1\ndescription: caf\xe9\n".encode("latin-1"), "unreadable"),
    ("huge-a1.md", memory_text(content="x" * (64 * 1024)).encode(), "too_large"),
])
def test_a_file_that_cannot_be_used_is_a_problem(tmp_path, name, data, kind):
    folder = _folder(tmp_path)
    folder.path.mkdir(parents=True)
    (folder.path / name).write_bytes(data)
    entries, problems = folder.scan()
    assert entries == []
    assert [(p.path.name, p.kind) for p in problems] == [(name, kind)]


def test_a_symlinked_memory_file_is_a_problem(tmp_path):
    folder = _folder(tmp_path)
    outside = write_memory(tmp_path / "elsewhere", "x.md", memory_text())
    folder.path.mkdir(parents=True)
    os.symlink(outside, folder.path / "linked-4f2a9c1e.md")
    entries, problems = folder.scan()
    assert entries == []
    assert [p.kind for p in problems] == ["unreadable"]


@pytest.mark.parametrize("inside", [None, ".inbox", ".history"])
def test_a_symlinked_folder_is_refused(tmp_path, inside):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    folder = _folder(tmp_path)
    if inside is None:
        os.symlink(elsewhere, folder.path)
    else:
        folder.path.mkdir()
        os.symlink(elsewhere, folder.path / inside)
    with pytest.raises(UnsafeMemoryStore):
        folder.check()


def test_duplicate_ids_keep_the_newer_and_report_the_other(tmp_path):
    folder = _folder(tmp_path)
    write_memory(folder.path, "old-a1.md", memory_text("a1", updated="2026-11-01T00:00:00Z"))
    write_memory(folder.path, "new-a1.md", memory_text("a1", updated="2026-11-02T00:00:00Z"))
    entries, _ = folder.scan()
    winners, problems = resolve_duplicates(entries)
    assert winners["a1"].memory.path.name == "new-a1.md"
    assert [(p.path.name, p.kind) for p in problems] == [("old-a1.md", "duplicate_id")]


def test_ids_that_differ_only_by_case_are_one_id(tmp_path):
    """On a case-insensitive file system ``.history/ABCD`` and ``.history/abcd`` are one folder."""
    folder = _folder(tmp_path)
    write_memory(folder.path, "upper-ABCD.md", memory_text("ABCD", "Upper one", updated="2026-11-01T00:00:00Z"))
    write_memory(folder.path, "lower-abcd.md", memory_text("abcd", "Lower one", updated="2026-11-02T00:00:00Z"))
    entries, _ = folder.scan()
    winners, problems = resolve_duplicates(entries)
    assert list(winners) == ["abcd"]
    assert [(p.path.name, p.kind) for p in problems] == [("upper-ABCD.md", "duplicate_id")]


def test_write_creates_private_folders_and_files_and_the_note(tmp_path):
    folder = ScopeFolder(tmp_path / "projects" / "app-0123" / "memory", "project",
                         note=(tmp_path / "projects" / "app-0123" / "project-path.txt", "/src/app"))
    entry = _entry(folder)
    folder.write(entry.memory, entry.extra)
    for path in (tmp_path / "projects", tmp_path / "projects" / "app-0123", folder.path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.memory.path.stat().st_mode) == 0o600
    assert (tmp_path / "projects" / "app-0123" / "project-path.txt").read_text() == "/src/app\n"


def _entry(folder: ScopeFolder, memory_id: str = "4f2a9c1e") -> Entry:
    from agentic_cli.memory.store.models import Memory

    memory = Memory(
        id=memory_id, description="Prefers pytest fixtures", content="Use fixtures.",
        type="feedback", scope=folder.scope, source="agent", conversation=None,
        created=T0, updated=T0, path=folder.target("Prefers pytest fixtures", memory_id),
    )
    return Entry(memory, {"tags": "a"}, 0)


def test_versions_are_kept_named_by_time_and_read_back(tmp_path):
    folder = _folder(tmp_path)
    entry = _entry(folder)
    path = folder.keep_version(entry, T0)
    assert path == folder.path / ".history" / "4f2a9c1e" / "20261108T140322.000000Z.md"
    [(memory_id, saved, found)] = folder.versions()
    assert (memory_id, saved, found) == ("4f2a9c1e", T0, path)
    assert folder.read_version(path).memory.content == "Use fixtures."
    assert folder.read_version(path).extra == {"tags": "a"}


def test_two_versions_at_one_instant_both_stay(tmp_path):
    folder = _folder(tmp_path)
    folder.keep_version(_entry(folder), T0)
    folder.keep_version(_entry(folder), T0)
    assert len(folder.versions("4f2a9c1e")) == 2


def test_history_keeps_30_days_and_20_versions(tmp_path):
    folder = _folder(tmp_path)
    entry = _entry(folder)
    folder.keep_version(entry, T0 - timedelta(days=31))
    for minute in range(25):
        folder.keep_version(entry, T0 + timedelta(minutes=minute))
    folder.prune_history(T0 + timedelta(hours=1))
    kept = sorted(saved for _, saved, _ in folder.versions())
    assert len(kept) == 20
    assert kept[0] == T0 + timedelta(minutes=5)


def test_the_lock_gives_up_after_its_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(_scope, "LOCK_TIMEOUT_S", 0.2)
    folder = _folder(tmp_path)
    folder.ensure()
    held, release = threading.Event(), threading.Event()

    def hold():
        with folder_lock(folder.path / ".lock"):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        with pytest.raises(MemoryBusy, match="memory is busy"):
            with folder.lock():
                pass
    finally:
        release.set()
        holder.join()

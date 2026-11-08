"""0.6's memories.json is imported once into the user scope (spec §10)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import pytest

from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory.store import _import_06, _inbox, _scope
from agentic_cli.memory.store.store import MemoryStore
from tests.memory.kb_storage import LogRecorder

SAVED = {"id": "0b6f3e2a-9c1d-4f7e-8a51-2d3c4b5a6f70", "content": "User prefers metric units\nAlways.",
         "tags": ["preference"], "created_at": "2026-10-01T09:30:00", "updated_at": "2026-10-02T09:30:00",
         "last_accessed_at": "", "access_count": 3, "importance": 7, "archived": False}
EXTRACTED = {"id": "1c7f4e3b-0000-4f7e-8a51-2d3c4b5a6f71", "content": "Project uses Python 3.12",
             "tags": ["auto-extracted", "session"], "archived": False}
EXTRACTED_AGAIN = {**EXTRACTED, "id": "2d8f5e4c-0000-4f7e-8a51-2d3c4b5a6f72"}
ARCHIVED = {"id": "3e9f6e5d-0000-4f7e-8a51-2d3c4b5a6f73", "content": "Old fact", "archived": True}


def _write(folder, data) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "memories.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "memories_embeddings.json").write_text("{}")
    (folder / "embedding_model.json").write_text("{}")


@pytest.fixture
def log(monkeypatch) -> LogRecorder:
    return LogRecorder().install(monkeypatch, _import_06)


@pytest.mark.parametrize("wrap", [lambda items: items, lambda items: {"items": items}])
def test_saved_items_become_memories_and_extracted_ones_proposals(tmp_path, log, wrap):
    user = tmp_path / "memory"
    _write(user, wrap([SAVED, EXTRACTED, EXTRACTED_AGAIN, ARCHIVED]))

    store = MemoryStore(user)

    [memory] = store.memories()
    assert memory.id == SAVED["id"]
    assert (memory.description, memory.type, memory.scope, memory.source) == (
        "User prefers metric units", "user", "user", "imported")
    assert memory.content == "User prefers metric units\nAlways.\n\nTags: preference"
    local = datetime(2026, 10, 1, 9, 30).astimezone()
    assert memory.created == local.astimezone(timezone.utc)
    [proposal] = store.proposals()
    assert (proposal.description, proposal.source, proposal.action) == (
        "Project uses Python 3.12", "imported", "new")
    assert sorted(p.name for p in (user / "pre-0.7.0").iterdir()) == [
        "embedding_model.json", "memories.json", "memories_embeddings.json"]
    assert not (user / "memories.json").exists()
    [counts] = log.infos("memory_imported")
    assert (counts["memories"], counts["proposals"]) == (1, 1)


def test_extracted_items_past_the_inbox_cap_are_counted_and_stay_in_the_old_file(tmp_path, log, monkeypatch):
    monkeypatch.setattr(_inbox, "INBOX_MAX", 2)
    extracted = [{"id": f"e{n}", "content": f"Extracted fact number {n}", "tags": ["auto-extracted"]}
                 for n in range(5)]
    user = tmp_path / "memory"
    _write(user, [SAVED, *extracted])

    store = MemoryStore(user)

    assert [m.id for m in store.memories()] == [SAVED["id"]]  # saved items are not capped
    assert len(store.proposals()) == 2
    [counts] = log.infos("memory_imported")
    assert (counts["memories"], counts["proposals"], counts["proposals_not_filed"]) == (1, 2, 3)
    assert "Extracted fact number 4" in (user / "pre-0.7.0" / "memories.json").read_text()


def test_an_extracted_item_that_repeats_one_is_not_counted_as_unfiled(tmp_path, log):
    _write(tmp_path / "memory", [EXTRACTED, EXTRACTED_AGAIN])
    MemoryStore(tmp_path / "memory")
    [counts] = log.infos("memory_imported")
    assert (counts["proposals"], counts["proposals_not_filed"]) == (1, 0)


def test_a_long_first_line_is_cut_at_a_word(tmp_path):
    words = " ".join(["otters"] * 40)
    _write(tmp_path / "memory", [{"id": "a1", "content": words}])
    [memory] = MemoryStore(tmp_path / "memory").memories()
    assert len(memory.description) <= 120
    assert memory.description.endswith("otters…")


def test_control_characters_in_the_first_line_become_spaces(tmp_path):
    _write(tmp_path / "memory", [{"id": "x1", "content": "Name:\tAndrey\x07 Shiryaev\nmore"}])
    [memory] = MemoryStore(tmp_path / "memory").memories()
    assert memory.description == "Name: Andrey  Shiryaev"


def test_items_with_the_same_first_line_each_get_a_description(tmp_path):
    items = [{"id": f"n{n}", "content": f"Note:\n{text}"} for n, text in enumerate("ABC")]
    store = MemoryStore(_import_of(tmp_path, items))
    assert sorted((m.description, m.content) for m in store.memories()) == [
        ("Note:", "Note:\nA"), ("Note: (2)", "Note:\nB"), ("Note: (3)", "Note:\nC")]


def test_a_derived_description_a_memory_already_has_is_numbered(tmp_path):
    user = tmp_path / "memory"
    MemoryStore(user).save("Note:", "saved by hand", "user", scope="user")
    _write(user, [{"id": "n1", "content": "Note:\nfrom 0.6"}])
    assert sorted(m.description for m in MemoryStore(user).memories()) == ["Note:", "Note: (2)"]


def test_a_number_never_takes_a_description_past_150_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(_import_06, "DESCRIPTION_CUT", 150)  # a first line cut as long as a description may be
    first = " ".join(["otters"] * 40)
    items = [{"id": f"n{n}", "content": f"{first} {n}"} for n in range(3)]
    descriptions = sorted(m.description for m in MemoryStore(_import_of(tmp_path, items)).memories())
    assert len(set(descriptions)) == 3
    assert max(len(d) for d in descriptions) == 150
    assert [d[-4:] for d in descriptions if d.endswith(")")] == [" (2)", " (3)"]


def _import_of(tmp_path, items):
    user = tmp_path / "memory"
    _write(user, items)
    return user


def test_a_second_run_adds_nothing(tmp_path):
    user = tmp_path / "memory"
    _write(user, [SAVED, EXTRACTED])
    MemoryStore(user)
    _write(user, [SAVED, EXTRACTED])
    store = MemoryStore(user)
    assert len(store.memories()) == 1 and len(store.proposals()) == 1
    assert sorted(p.name for p in (user / "pre-0.7.0").iterdir())[0].startswith("embedding_model.json")
    assert len(list((user / "pre-0.7.0").iterdir())) == 6


def test_what_0_6_writes_after_the_import_is_imported_at_the_next_open(tmp_path):
    """Review Focus 3: a 0.6 session still running writes memories.json again."""
    user = tmp_path / "memory"
    _write(user, [SAVED])
    MemoryStore(user)
    later = {"id": "4f0a7f6e-0000-4f7e-8a51-2d3c4b5a6f74", "content": "Prefers dark mode"}
    (user / "memories.json").write_text(json.dumps([SAVED, later]))
    store = MemoryStore(user)
    assert sorted(m.description for m in store.memories()) == ["Prefers dark mode", "User prefers metric units"]
    assert not (user / "memories.json").exists()


def test_a_0_6_rewrite_does_not_bring_back_what_was_deleted_in_0_7(tmp_path):
    """Released 0.6 takes no lock and rewrites memories.json on every change, a search included."""
    user = tmp_path / "memory"
    other = {"id": "5a1b8e7f-0000-4f7e-8a51-2d3c4b5a6f75", "content": "Prefers tabs"}
    _write(user, [SAVED, other])
    store = MemoryStore(user)
    store.delete(SAVED["id"])
    later = {"id": "4f0a7f6e-0000-4f7e-8a51-2d3c4b5a6f74", "content": "Prefers dark mode"}
    (user / "memories.json").write_text(json.dumps([SAVED, other, later]))

    store = MemoryStore(user)

    assert sorted(m.description for m in store.memories()) == ["Prefers dark mode", "Prefers tabs"]
    assert store.read(SAVED["id"]) is None
    assert not (user / "memories.json").exists()
    assert any(p.name.startswith("memories.json.2") for p in (user / "pre-0.7.0").iterdir())


def test_a_proposal_that_was_accepted_and_deleted_is_not_proposed_again(tmp_path):
    user = tmp_path / "memory"
    _write(user, [EXTRACTED])
    store = MemoryStore(user)
    store.delete(store.accept(store.proposals()[0].id).id)
    (user / "memories.json").write_text(json.dumps([EXTRACTED]))

    store = MemoryStore(user)

    assert store.proposals() == [] and store.memories() == []


def test_an_old_file_that_cannot_be_read_says_nothing_of_what_was_imported(tmp_path):
    user = tmp_path / "memory"
    (user / "pre-0.7.0").mkdir(parents=True)
    (user / "pre-0.7.0" / "memories.json.20260101T000000000000Z").write_text("not json {")
    (user / "pre-0.7.0" / "memories.json.unreadable-20260101T000000000000Z").write_text("[")
    _write(user, [SAVED])
    assert [m.id for m in MemoryStore(user).memories()] == [SAVED["id"]]


def test_a_running_0_6_session_holding_the_lock_postpones_the_import(tmp_path, monkeypatch, log):
    monkeypatch.setattr(_scope, "LOCK_TIMEOUT_S", 0.2)
    user = tmp_path / "memory"
    _write(user, [SAVED])
    held, release = threading.Event(), threading.Event()

    def hold():
        with folder_lock(user / ".lock"):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        store = MemoryStore(user)
        assert store.memories() == []
        assert log.infos("memory_import_waiting")
    finally:
        release.set()
        holder.join()
    assert len(MemoryStore(user).memories()) == 1


@pytest.mark.parametrize("content", ["not json {", '"just a string"', "\x00\xff"])
def test_an_unreadable_file_is_left_where_it_is(tmp_path, log, content):
    user = tmp_path / "memory"
    user.mkdir()
    (user / "memories.json").write_bytes(content.encode("latin-1"))
    store = MemoryStore(user)
    assert (user / "memories.json").exists()
    assert [p.kind for p in store.problems()] == ["import_unreadable"]
    assert log.warnings("memory_import_unreadable")


def test_a_symlinked_file_is_not_followed(tmp_path):
    user = tmp_path / "memory"
    user.mkdir()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps([SAVED]))
    os.symlink(elsewhere, user / "memories.json")
    store = MemoryStore(user)
    assert store.memories() == []
    assert [p.kind for p in store.problems()] == ["import_unreadable"]


def test_an_item_the_rules_refuse_stays_in_the_old_file(tmp_path, log):
    user = tmp_path / "memory"
    _write(user, [{"id": "a1", "content": "key AKIAIOSFODNN7EXAMPLE"}, {"id": "b2", "content": "x" * 5000}])
    store = MemoryStore(user)
    assert store.memories() == []
    assert len(log.infos("memory_import_item_skipped")) == 2
    assert "AKIAIOSFODNN7EXAMPLE" in (user / "pre-0.7.0" / "memories.json").read_text()


def test_unreadable_sets_aside_move_too(tmp_path):
    user = tmp_path / "memory"
    _write(user, [SAVED])
    (user / "memories.json.unreadable-20261001T000000000000Z").write_text("[")
    MemoryStore(user)
    assert (user / "pre-0.7.0" / "memories.json.unreadable-20261001T000000000000Z").exists()


def test_nothing_to_import_creates_nothing(tmp_path):
    MemoryStore(tmp_path / "memory")
    assert not (tmp_path / "memory").exists()

"""The memory store's memories: the save rules, changes, history (spec §6.3-6.4)."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest

from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory.store import _scope
from agentic_cli.memory.store import store as store_module
from agentic_cli.memory.store.models import (
    DuplicateMemory,
    InvalidMemory,
    MemoryBusy,
    MemoryStoreFull,
    NoProjectScope,
)
from agentic_cli.memory.store.store import MemoryStore
from tests.memory.memory_files import T0, Clock, memory_text, write_memory


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> MemoryStore:
    return MemoryStore(tmp_path / "user", tmp_path / "projects" / "app-0123" / "memory",
                       project_name="app", project_path=tmp_path / "src" / "app", now=clock)


class TestSave:
    def test_a_memory_is_a_file_named_by_its_description(self, store, tmp_path):
        memory = store.save("Prefers pytest fixtures", "Use fixtures.", "feedback",
                            source="user", conversation="c-1")
        assert memory.path == tmp_path / "projects" / "app-0123" / "memory" / f"prefers-pytest-fixtures-{memory.id}.md"
        assert (memory.scope, memory.source, memory.conversation) == ("project", "user", "c-1")
        assert memory.created == memory.updated == T0
        assert memory.path.read_text() == (
            f"---\nid: {memory.id}\ndescription: Prefers pytest fixtures\ntype: feedback\n"
            "source: user\nconversation: c-1\ncreated: 2026-11-08T14:03:22Z\n"
            "updated: 2026-11-08T14:03:22Z\n---\nUse fixtures.\n"
        )
        assert (tmp_path / "projects" / "app-0123" / "project-path.txt").read_text() == f"{tmp_path / 'src' / 'app'}\n"
        assert store.read(memory.id) == memory

    def test_description_and_content_are_stripped(self, store):
        memory = store.save("  Spaced  ", "  text \n", "user", scope="user")
        assert (memory.description, memory.content) == ("Spaced", "text")

    @pytest.mark.parametrize("args, message", [
        (("", "c", "user"), "description: empty"),
        (("d" * 151, "c", "user"), "description: longer than 150"),
        (("d", "c" * 4001, "user"), "content: longer than 4000"),
        (("d", "   ", "user"), "content: empty"),
        (("d", "c", "preference"), "type: 'preference' is not one of"),
    ])
    def test_a_field_that_fails_its_rule(self, store, args, message):
        with pytest.raises(InvalidMemory, match=message):
            store.save(*args)

    def test_an_unknown_scope(self, store):
        with pytest.raises(InvalidMemory, match="scope: 'team'"):
            store.save("d", "c", "user", scope="team")

    def test_no_project_scope(self, tmp_path):
        bare = MemoryStore(tmp_path / "user")
        assert not bare.has_project_scope
        with pytest.raises(NoProjectScope):
            bare.save("d", "c", "user")
        assert bare.save("d", "c", "user", scope="user").scope == "user"

    def test_a_secret_is_refused(self, store):
        with pytest.raises(InvalidMemory, match=r"looks like a secret \(an AWS access key\)"):
            store.save("Staging key", "AKIAIOSFODNN7EXAMPLE", "reference")

    def test_the_screen_can_be_the_app_s(self, tmp_path):
        relaxed = MemoryStore(tmp_path / "user", screen=None)
        assert relaxed.save("Staging key", "AKIAIOSFODNN7EXAMPLE", "reference", scope="user")

    def test_a_description_another_memory_has_in_either_scope(self, store):
        first = store.save("Prefers pytest", "a", "feedback", scope="user")
        with pytest.raises(DuplicateMemory) as raised:
            store.save("  PREFERS   pytest ", "b", "feedback")
        assert raised.value.existing_id == first.id

    def test_a_full_scope(self, store, monkeypatch):
        monkeypatch.setattr(store_module, "SCOPE_MAX", 2)
        store.save("one", "1", "user")
        store.save("two", "2", "user")
        with pytest.raises(MemoryStoreFull, match="delete or merge some first"):
            store.save("three", "3", "user")
        assert store.save("three", "3", "user", scope="user")

    def test_the_rules_are_checked_in_order(self, tmp_path):
        bare = MemoryStore(tmp_path / "user")
        with pytest.raises(InvalidMemory, match="description"):
            bare.save("", "AKIAIOSFODNN7EXAMPLE", "user")  # rule 1 before 4 and 5
        with pytest.raises(NoProjectScope):
            bare.save("d", "AKIAIOSFODNN7EXAMPLE", "user")  # rule 4 before 5

    def test_a_busy_scope(self, store, monkeypatch):
        monkeypatch.setattr(_scope, "LOCK_TIMEOUT_S", 0.2)
        folder = store._folders["user"]
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
            with pytest.raises(MemoryBusy):
                store.save("d", "c", "user", scope="user")
        finally:
            release.set()
            holder.join()


class TestChanges:
    def test_update_keeps_the_identity_and_the_previous_version(self, store, clock):
        old = store.save("Prefers spaces", "Spaces.", "feedback", source="user", conversation="c-1")
        clock.now = T0 + timedelta(hours=1)
        new = store.update(old.id, description="Prefers tabs", content="Tabs.")
        assert (new.id, new.created, new.source, new.conversation) == (old.id, T0, "user", "c-1")
        assert new.updated == T0 + timedelta(hours=1)
        assert not old.path.exists() and new.path.name == f"prefers-tabs-{old.id}.md"
        [version] = store.history(old.id)
        assert (version.memory.content, version.deleted) == ("Spaces.", False)

    def test_keys_a_person_added_survive_an_update(self, store):
        memory = store.save("Prefers tabs", "Tabs.", "feedback")
        text = memory.path.read_text().replace("---\nTabs.", "mood: calm\n---\nTabs.")
        memory.path.write_text(text)
        store.update(memory.id, content="Tabs, width 4.")
        assert "mood: calm\n---\nTabs, width 4.\n" in store.read(memory.id).path.read_text()

    def test_update_needs_a_field_and_a_known_id(self, store):
        memory = store.save("d", "c", "user")
        with pytest.raises(InvalidMemory, match="give a new description, content or type"):
            store.update(memory.id)
        assert store.update("deadbeef", content="x") is None
        assert store.update("../../etc", content="x") is None

    def test_update_into_another_memory_s_description(self, store):
        store.save("one", "1", "user")
        two = store.save("two", "2", "user")
        with pytest.raises(DuplicateMemory):
            store.update(two.id, description="ONE")
        assert store.update(two.id, description="two", content="2b").content == "2b"

    def test_delete_keeps_a_restorable_version(self, store, clock):
        memory = store.save("Prefers tabs", "Tabs.", "feedback")
        clock.now = T0 + timedelta(minutes=5)
        assert store.delete(memory.id) is True
        assert store.read(memory.id) is None and not memory.path.exists()
        [version] = store.history()
        assert version.deleted is True
        clock.now = T0 + timedelta(minutes=10)
        restored = store.restore(memory.id, version.saved)
        assert (restored.id, restored.content, restored.updated) == (memory.id, "Tabs.", clock.now)
        assert store.delete("deadbeef") is False

    def test_a_restore_can_be_restored(self, store, clock):
        memory = store.save("Prefers tabs", "Tabs.", "feedback")
        clock.now = T0 + timedelta(minutes=1)
        store.update(memory.id, content="Tabs, width 4.")
        [first] = store.history(memory.id)
        clock.now = T0 + timedelta(minutes=2)
        store.restore(memory.id, first.saved)
        assert store.read(memory.id).content == "Tabs."
        newest = store.history(memory.id)[0]
        assert newest.memory.content == "Tabs, width 4."
        clock.now = T0 + timedelta(minutes=3)
        assert store.restore(memory.id, newest.saved).content == "Tabs, width 4."

    def test_restore_of_an_unknown_version(self, store):
        with pytest.raises(InvalidMemory, match="no version"):
            store.restore("deadbeef", T0)

    def test_history_shows_the_last_days_asked_for(self, store, clock):
        memory = store.save("d", "c", "user")
        store.update(memory.id, content="c2")
        clock.now = T0 + timedelta(days=10)
        assert store.history(days=30) != []
        assert store.history(days=5) == []


class TestLists:
    def test_memories_newest_first_then_by_id(self, store, clock):
        a = store.save("a", "1", "user", scope="user")
        clock.now = T0 + timedelta(minutes=1)
        b = store.save("b", "2", "user")
        c = store.save("c", "3", "user", scope="user")
        expected = [m.id for m in sorted([b, c], key=lambda m: m.id)] + [a.id]
        assert [m.id for m in store.memories()] == expected
        assert [m.id for m in store.memories("project")] == [b.id]

    def test_problems_lists_files_the_store_cannot_use(self, store, tmp_path):
        write_memory(tmp_path / "user", "broken-a1.md", "nonsense\n")
        assert [(p.path.name, p.kind) for p in store.problems()] == [("broken-a1.md", "bad_frontmatter")]


def test_two_stores_saving_at_once_keep_both(tmp_path):
    """Review Focus 1: two sessions (two projects) save to the user scope together."""
    one = MemoryStore(tmp_path / "user", tmp_path / "p1" / "memory")
    two = MemoryStore(tmp_path / "user", tmp_path / "p2" / "memory")
    errors: list[BaseException] = []

    def save_many(store: MemoryStore, prefix: str) -> None:
        try:
            for n in range(20):
                store.save(f"{prefix} fact {n}", f"{prefix} {n}", "user", scope="user")
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=save_many, args=(s, p)) for s, p in ((one, "one"), (two, "two"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    memories = MemoryStore(tmp_path / "user").memories("user")
    assert len(memories) == 40
    assert len({m.id for m in memories}) == 40
    assert MemoryStore(tmp_path / "user").problems() == []


def test_a_broken_file_survives_every_change_to_the_others(store, tmp_path):
    """Review Focus 2: a person's broken edit is never rewritten, moved or deleted."""
    broken = write_memory(tmp_path / "user", "mine-a1.md", memory_text("a1", type="preference"))
    before = broken.read_bytes()
    kept = store.save("kept", "1", "user", scope="user")
    gone = store.save("gone", "2", "user", scope="user")
    store.update(kept.id, content="1b")
    store.delete(gone.id)
    store.restore(gone.id, store.history(gone.id)[0].saved)
    assert broken.read_bytes() == before
    assert [p.path for p in store.problems()] == [broken]
    assert {m.description for m in store.memories()} == {"kept", "gone"}

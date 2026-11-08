"""The memory store's memories: the save rules, changes, history (spec §6.3-6.4)."""

from __future__ import annotations

import os
import threading
from datetime import timedelta

import pytest

from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory.store import _files, _scope
from agentic_cli.memory.store import store as store_module
from agentic_cli.memory.store.models import (
    DuplicateMemory,
    FileInTheWay,
    InvalidMemory,
    MemoryBusy,
    MemoryStoreError,
    MemoryStoreFull,
    NoProjectScope,
    UnsafeMemoryStore,
)
from agentic_cli.memory.store.store import MemoryStore
from tests.memory.kb_storage import LogRecorder
from tests.memory.memory_files import T0, Clock, memory_text, write_memory


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> MemoryStore:
    return MemoryStore(tmp_path / "user", tmp_path / "projects" / "app-0123" / "memory",
                       project_name="app", project_path=tmp_path / "src" / "app", now=clock)


class TestSave:
    def test_a_memory_is_a_file_named_by_its_description(self, store, tmp_path, monkeypatch):
        # An id of digits only is written quoted ("84249433" would read as a number), so fix the draw.
        monkeypatch.setattr(_files.secrets, "token_hex", lambda n: "4f2a9c1e")
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

    @pytest.mark.parametrize("description", ["Ticket #42 blocks the release", "課題　#42 が残っている"])
    def test_a_hash_in_a_description_survives_a_save_and_an_update(self, store, description):
        memory = store.save(description, "text", "project")
        assert store.read(memory.id).description == description
        store.update(memory.id, content="more text")
        assert store.read(memory.id).description == description

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


def test_a_new_id_avoids_one_that_differs_only_by_case(tmp_path, monkeypatch):
    write_memory(tmp_path / "user", "mine-4F2A9C1E.md", memory_text("4F2A9C1E", "Written by hand"))
    draws = iter(["4f2a9c1e", "5a5a5a5a"])
    monkeypatch.setattr(_files.secrets, "token_hex", lambda n: next(draws))
    assert MemoryStore(tmp_path / "user").save("d", "c", "user", scope="user").id == "5a5a5a5a"


class TestFilesThatAreNotTheEntrys:
    """A file the store cannot use is never rewritten, moved or deleted (spec §5.3)."""

    def test_restore_does_not_overwrite_a_file_a_person_broke(self, store):
        memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
        store.update(memory.id, content="Tabs, width 4.")
        path = store.read(memory.id).path
        broken = path.read_text().replace("type: feedback", "type: preference").replace("Tabs, width 4.", "MY EDIT")
        path.write_text(broken)
        assert [(p.path, p.kind) for p in store.problems()] == [(path, "bad_field")]
        [version] = store.history(memory.id)
        assert version.deleted is False  # the id is still claimed by the file they edited
        with pytest.raises(FileInTheWay, match=path.name):
            store.restore(memory.id, version.saved)
        assert path.read_text() == broken

    def test_restore_does_not_overwrite_a_file_in_its_way(self, store):
        memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
        store.delete(memory.id)
        memory.path.write_text("nonsense, my notes\n")
        [version] = store.history(memory.id)
        with pytest.raises(FileInTheWay, match=memory.path.name):
            store.restore(memory.id, version.saved)
        assert memory.path.read_text() == "nonsense, my notes\n"

    def test_update_does_not_rename_onto_a_file_in_its_way(self, store, tmp_path):
        alpha = store.save("alpha", "a", "user", scope="user")
        mine = tmp_path / "user" / f"beta-{alpha.id}.md"
        mine.write_text("not frontmatter, my notes\n")
        with pytest.raises(FileInTheWay, match=mine.name):
            store.update(alpha.id, description="beta")
        assert mine.read_text() == "not frontmatter, my notes\n"
        assert store.read(alpha.id).description == "alpha"
        assert store.history(alpha.id) == []  # nothing was written before it refused


class TestUnsafeFolders:
    """Links and read-only folders: the store refuses them in words, and writes nothing through them."""

    def _linked_history(self, store, tmp_path):
        memory = store.save("note", "c", "user", scope="user")
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "20200101T000000.000000Z.md"  # old enough to be pruned, if it were followed
        victim.write_text("victim")
        (tmp_path / "user" / ".history").mkdir(exist_ok=True)
        os.symlink(outside, tmp_path / "user" / ".history" / memory.id)
        return memory, outside, victim

    def test_a_symlinked_history_folder_is_never_written_into(self, store, tmp_path):
        memory, outside, victim = self._linked_history(store, tmp_path)
        with pytest.raises(UnsafeMemoryStore, match=memory.id):
            store.update(memory.id, content="c2")
        with pytest.raises(UnsafeMemoryStore, match=memory.id):
            store.delete(memory.id)
        assert os.listdir(outside) == [victim.name]
        assert store.read(memory.id).content == "c"

    def test_a_symlinked_history_folder_is_not_pruned_and_is_logged(self, store, tmp_path, monkeypatch):
        memory, outside, victim = self._linked_history(store, tmp_path)
        log = LogRecorder().install(monkeypatch, _scope)
        MemoryStore(tmp_path / "user")
        assert victim.read_text() == "victim"
        assert [f["path"] for f in log.warnings("memory_history_folder_refused")] == [
            str(tmp_path / "user" / ".history" / memory.id)
        ]

    def test_a_symlinked_lock_is_refused_when_the_store_opens(self, tmp_path):
        user = tmp_path / "user"
        user.mkdir()
        os.symlink(tmp_path / "elsewhere.lock", user / ".lock")
        with pytest.raises(UnsafeMemoryStore, match=r"\.lock"):
            MemoryStore(user)
        assert not (tmp_path / "elsewhere.lock").exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
    def test_a_read_only_scope_folder_is_a_store_error_naming_it(self, tmp_path):
        user = tmp_path / "user"
        user.mkdir()
        user.chmod(0o500)
        try:
            with pytest.raises(MemoryStoreError, match=str(user)):
                MemoryStore(user)
        finally:
            user.chmod(0o700)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
    def test_a_folder_that_cannot_be_created_is_a_store_error_naming_it(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            store = MemoryStore(parent / "user")  # nothing exists yet, so nothing is opened
            with pytest.raises(MemoryStoreError, match=str(parent)):
                store.save("d", "c", "user", scope="user")
        finally:
            parent.chmod(0o700)


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

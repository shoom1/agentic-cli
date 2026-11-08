"""One memory the store cannot read no longer costs the others (review K1, memory-kb F2).

``_load`` parsed memories.json in one ``try``: a memory missing a key emptied
the store, and the next save rewrote the file with only the new memory. An
entry that does not parse is now kept as it was and written back by every
save; a file that does not parse at all is renamed aside before anything is
written over it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from agentic_cli.memory import MemoryStore, MockEmbeddingService

OTTERS = {"id": "a1", "content": "otters nest in burrows"}
NO_CONTENT = {"id": "b2", "tags": ["broken"]}
HERONS = {"id": "c3", "content": "herons wade at dawn"}


def _write(folder: Path, data) -> None:
    (folder / "memories.json").write_text(json.dumps(data))


def _saved(folder: Path) -> list:
    return json.loads((folder / "memories.json").read_text())


def test_one_bad_entry_does_not_lose_the_others(tmp_path):
    _write(tmp_path, [OTTERS, NO_CONTENT, HERONS])

    store = MemoryStore(tmp_path)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]
    assert [m.content for m in store.search("herons")] == ["herons wade at dawn"]
    store.store("kingfishers dive")

    saved = _saved(tmp_path)
    assert NO_CONTENT in saved
    assert {entry.get("content") for entry in saved} >= {
        "otters nest in burrows", "herons wade at dawn", "kingfishers dive",
    }


def test_an_entry_that_is_not_an_object_is_kept(tmp_path):
    _write(tmp_path, ["just a string", OTTERS])

    store = MemoryStore(tmp_path)
    store.store("kingfishers dive")

    assert "just a string" in _saved(tmp_path)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]


def test_a_file_that_is_not_json_is_set_aside(tmp_path):
    (tmp_path / "memories.json").write_text("{not json")

    store = MemoryStore(tmp_path)
    store.store("kingfishers dive")

    [aside] = tmp_path.glob("memories.json.unreadable-*")
    assert aside.read_text() == "{not json"
    assert [entry["content"] for entry in _saved(tmp_path)] == ["kingfishers dive"]


def test_a_file_of_another_shape_is_set_aside(tmp_path):
    _write(tmp_path, {"memories": [OTTERS]})

    MemoryStore(tmp_path).store("kingfishers dive")

    [aside] = tmp_path.glob("memories.json.unreadable-*")
    assert json.loads(aside.read_text()) == {"memories": [OTTERS]}


def test_a_store_that_reloads_after_another_wrote_keeps_the_bad_entry(tmp_path):
    _write(tmp_path, [OTTERS, NO_CONTENT])
    first, second = MemoryStore(tmp_path), MemoryStore(tmp_path)

    second.store("herons wade at dawn")
    first.store("kingfishers dive")

    # Each reload starts the list again: the entry is not added to itself.
    assert _saved(tmp_path).count(NO_CONTENT) == 1


def test_two_entries_with_one_id_are_both_kept_and_the_first_is_the_memory(tmp_path):
    later = {"id": "a1", "content": "herons wade at dawn"}
    _write(tmp_path, [OTTERS, later])

    MemoryStore(tmp_path).store("kingfishers dive")
    reopened = MemoryStore(tmp_path)

    assert {entry["content"] for entry in _saved(tmp_path)} >= {
        "otters nest in burrows", "herons wade at dawn", "kingfishers dive",
    }
    assert [m.content for m in reopened.search("otters")] == ["otters nest in burrows"]
    assert reopened.search("herons") == []
    # The first copy stays the memory, so an edit of it is not undone by a reload.
    assert reopened.update("a1", content="otters nest in holts")
    assert [m.content for m in MemoryStore(tmp_path).search("holts")] == ["otters nest in holts"]
    assert later in _saved(tmp_path)


def test_ids_that_are_not_strings_are_kept_apart(tmp_path):
    # 1, true and 1.0 are one key to a dict: only one of them survived a save.
    odd = [
        {"id": 1, "content": "an id of one"},
        {"id": True, "content": "an id of true"},
        {"id": 1.0, "content": "an id of one point zero"},
    ]
    _write(tmp_path, [OTTERS, *odd])

    store = MemoryStore(tmp_path)
    store.store("kingfishers dive")

    saved = _saved(tmp_path)
    assert all(entry in saved for entry in odd)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]


def test_an_entry_whose_id_cannot_be_a_key_is_kept(tmp_path):
    unhashable = {"id": ["a", "list"], "content": "an id that is a list"}
    _write(tmp_path, [unhashable, OTTERS])

    store = MemoryStore(tmp_path)
    store.store("kingfishers dive")

    assert unhashable in _saved(tmp_path)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]


WRONG_TYPES = [
    pytest.param({"id": "d1", "content": None}, id="content-null"),
    pytest.param({"id": "d2", "content": 5}, id="content-number"),
    pytest.param({"id": "d3", "content": "otters too", "tags": 5}, id="tags-number"),
    pytest.param({"id": "d4", "content": "otters too", "tags": ["a", 1]}, id="tags-not-strings"),
    pytest.param({"id": "d5", "content": "otters too", "access_count": "3"}, id="access-count-string"),
    pytest.param({"id": "d6", "content": "otters too", "access_count": True}, id="access-count-bool"),
    pytest.param({"id": "d7", "content": "otters too", "importance": "7"}, id="importance-string"),
    pytest.param({"id": "d8", "content": "otters too", "importance": False}, id="importance-bool"),
]


@pytest.mark.parametrize("bad", WRONG_TYPES)
def test_an_entry_with_a_field_of_the_wrong_type_is_kept_and_costs_only_itself(tmp_path, bad):
    _write(tmp_path, [OTTERS, bad])

    store = MemoryStore(tmp_path)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]
    assert "otters nest in burrows" in store.load_all()
    store.store("kingfishers dive")

    assert bad in _saved(tmp_path)


class _StrictEmbedder(MockEmbeddingService):
    """An embedding model that refuses what is not text, as the real ones do."""

    def embed_batch(self, texts):
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("text input must be of type str")
        return super().embed_batch(texts)


def test_an_entry_with_no_text_does_not_stop_the_store_opening_with_an_embedder(tmp_path):
    bad = {"id": "d1", "content": None}
    _write(tmp_path, [OTTERS, bad])

    store = MemoryStore(tmp_path, embedding_service=_StrictEmbedder())
    store.store("kingfishers dive")

    assert bad in _saved(tmp_path)


def test_a_float_importance_is_a_readable_memory(tmp_path):
    _write(tmp_path, [{"id": "f1", "content": "otters float", "importance": 7.5}])

    assert [m.importance for m in MemoryStore(tmp_path).search("otters")] == [7.5]


def test_a_file_nested_too_deep_to_parse_is_set_aside(tmp_path):
    (tmp_path / "memories.json").write_text("[" * 100_000)

    MemoryStore(tmp_path).store("kingfishers dive")

    [aside] = tmp_path.glob("memories.json.unreadable-*")
    assert aside.read_text() == "[" * 100_000


@pytest.mark.parametrize(
    "content",
    [b"[]", b"null", b'{"a1": [0.1, \xff]}'],
    ids=["a-list", "null", "not-utf-8"],
)
def test_an_unreadable_embeddings_file_is_ignored_and_the_vectors_are_made_again(tmp_path, content):
    _write(tmp_path, [OTTERS])
    (tmp_path / "memories_embeddings.json").write_bytes(content)

    store = MemoryStore(tmp_path, embedding_service=MockEmbeddingService())

    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]
    assert "a1" in json.loads((tmp_path / "memories_embeddings.json").read_text())


def test_a_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    (tmp_path / "memories.json").write_bytes(b"\xef\xbb\xbf" + json.dumps([OTTERS]).encode())

    store = MemoryStore(tmp_path)

    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]
    assert not list(tmp_path.glob("memories.json.unreadable-*"))


def test_an_unreadable_entry_is_reported_once_by_a_store(tmp_path):
    _write(tmp_path, [OTTERS, NO_CONTENT])
    other = MemoryStore(tmp_path)

    with capture_logs() as logs:
        store = MemoryStore(tmp_path)
        other.store("herons wade at dawn")
        store.search("otters")  # reads what the other wrote
        other.store("kingfishers dive")
        store.search("otters")  # and again

    reports = [entry for entry in logs if entry["event"] == "memory_entry_unreadable"]
    assert len(reports) == 1

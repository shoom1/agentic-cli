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

from agentic_cli.memory import MemoryStore

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

    assert NO_CONTENT in _saved(tmp_path)


def test_an_entry_whose_id_cannot_be_a_key_is_kept(tmp_path):
    unhashable = {"id": ["a", "list"], "content": "an id that is a list"}
    _write(tmp_path, [unhashable, OTTERS])

    store = MemoryStore(tmp_path)
    store.store("kingfishers dive")

    assert unhashable in _saved(tmp_path)
    assert [m.content for m in store.search("otters")] == ["otters nest in burrows"]

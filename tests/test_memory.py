"""Tests for memory module."""

import pytest


class TestWorkingMemory:
    """Tests for WorkingMemory class."""

    def test_set_and_get_value(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        memory.set("key1", "value1")
        assert memory.get("key1") == "value1"

    def test_get_missing_key_returns_default(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        assert memory.get("missing") is None
        assert memory.get("missing", "default") == "default"

    def test_set_with_tags(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        memory.set("paper1", {"title": "ML Paper"}, tags=["research", "ml"])
        memory.set("paper2", {"title": "NLP Paper"}, tags=["research", "nlp"])
        memory.set("note1", "Some note", tags=["personal"])
        all_keys = memory.list()
        assert set(all_keys) == {"paper1", "paper2", "note1"}
        research_keys = memory.list(tags=["research"])
        assert set(research_keys) == {"paper1", "paper2"}

    def test_delete_key(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        memory.set("key1", "value1")
        memory.delete("key1")
        assert memory.get("key1") is None

    def test_clear_all(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        memory.set("key1", "value1")
        memory.set("key2", "value2")
        memory.clear()
        assert memory.list() == []

    def test_to_snapshot_and_from_snapshot(self):
        from agentic_cli.memory import WorkingMemory
        memory = WorkingMemory()
        memory.set("key1", "value1", tags=["tag1"])
        memory.set("key2", {"nested": "data"}, tags=["tag2"])
        snapshot = memory.to_snapshot()
        restored = WorkingMemory.from_snapshot(snapshot)
        assert restored.get("key1") == "value1"
        assert restored.get("key2") == {"nested": "data"}
        assert set(restored.list(tags=["tag1"])) == {"key1"}

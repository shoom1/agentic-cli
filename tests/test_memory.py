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


class TestLongTermMemory:
    """Tests for LongTermMemory class."""

    def test_store_and_recall_fact(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        entry_id = memory.store(type=MemoryType.FACT, content="Test fact", source="session_1")
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.content == "Test fact"
        assert entry.type == MemoryType.FACT

    def test_recall_by_query(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        memory.store(type=MemoryType.FACT, content="Basel III requires 99% confidence", source="s1")
        memory.store(type=MemoryType.PREFERENCE, content="User prefers APA format", source="s2")
        results = memory.recall("Basel")
        assert len(results) > 0
        assert "Basel" in results[0].content

    def test_recall_by_type(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        memory.store(type=MemoryType.FACT, content="Fact 1", source="s1")
        memory.store(type=MemoryType.PREFERENCE, content="Pref 1", source="s2")
        prefs = memory.recall("", type=MemoryType.PREFERENCE)
        assert len(prefs) == 1
        assert prefs[0].type == MemoryType.PREFERENCE

    def test_store_with_kb_references(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        entry_id = memory.store(type=MemoryType.REFERENCE, content="Paper X", source="s1", kb_references=["doc_123"])
        entry = memory.get(entry_id)
        assert entry.kb_references == ["doc_123"]

    def test_update_entry(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        entry_id = memory.store(type=MemoryType.FACT, content="Original", source="s1")
        memory.update(entry_id, content="Updated")
        assert memory.get(entry_id).content == "Updated"

    def test_forget_entry(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        entry_id = memory.store(type=MemoryType.FACT, content="To forget", source="s1")
        memory.forget(entry_id)
        assert memory.get(entry_id) is None

    def test_get_preferences(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory = LongTermMemory(mock_context.settings)
        memory.store(type=MemoryType.PREFERENCE, content="Pref 1", source="s1")
        memory.store(type=MemoryType.PREFERENCE, content="Pref 2", source="s2")
        memory.store(type=MemoryType.FACT, content="Fact 1", source="s3")
        prefs = memory.get_preferences()
        assert len(prefs) == 2
        assert all(p.type == MemoryType.PREFERENCE for p in prefs)

    def test_persistence_across_instances(self, mock_context):
        from agentic_cli.memory import LongTermMemory, MemoryType
        memory1 = LongTermMemory(mock_context.settings)
        entry_id = memory1.store(type=MemoryType.FACT, content="Persistent", source="s1")
        memory2 = LongTermMemory(mock_context.settings)
        assert memory2.get(entry_id).content == "Persistent"

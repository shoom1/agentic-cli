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


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_access_working_and_longterm(self, mock_context):
        from agentic_cli.memory import MemoryManager, MemoryType

        manager = MemoryManager(mock_context.settings)
        manager.working.set("task", "analyzing papers")
        assert manager.working.get("task") == "analyzing papers"
        entry_id = manager.longterm.store(
            type=MemoryType.FACT, content="Test", source="test"
        )
        assert manager.longterm.get(entry_id) is not None

    def test_search_across_all_memory(self, mock_context):
        from agentic_cli.memory import MemoryManager, MemoryType

        manager = MemoryManager(mock_context.settings)
        manager.working.set("current_topic", "VaR calculation", tags=["risk"])
        manager.longterm.store(
            type=MemoryType.FACT, content="VaR requires 99% confidence", source="s1"
        )
        results = manager.search("VaR")
        assert results.working_results is not None
        assert results.longterm_results is not None
        assert len(results.longterm_results) > 0

    def test_search_with_tier_filtering(self, mock_context):
        from agentic_cli.memory import MemoryManager, MemoryType

        manager = MemoryManager(mock_context.settings)
        manager.working.set("note", "working note about VaR")
        manager.longterm.store(
            type=MemoryType.FACT, content="Long-term fact about VaR", source="s1"
        )
        results = manager.search("VaR", include_working=True, include_longterm=False)
        assert results.working_results is not None
        assert results.longterm_results == []

    def test_clear_working(self, mock_context):
        from agentic_cli.memory import MemoryManager

        manager = MemoryManager(mock_context.settings)
        manager.working.set("key1", "value1")
        manager.working.set("key2", "value2")
        manager.clear_working()
        assert manager.working.list() == []

    def test_get_working_snapshot(self, mock_context):
        from agentic_cli.memory import MemoryManager

        manager = MemoryManager(mock_context.settings)
        manager.working.set("key1", "value1", tags=["tag1"])
        manager.working.set("key2", {"nested": "data"})
        snapshot = manager.get_working_snapshot()
        assert "entries" in snapshot
        assert "key1" in snapshot["entries"]
        assert snapshot["entries"]["key1"]["value"] == "value1"

    def test_restore_working(self, mock_context):
        from agentic_cli.memory import MemoryManager

        manager = MemoryManager(mock_context.settings)
        manager.working.set("key1", "value1", tags=["tag1"])
        snapshot = manager.get_working_snapshot()

        # Clear and restore
        manager.clear_working()
        assert manager.working.list() == []

        manager.restore_working(snapshot)
        assert manager.working.get("key1") == "value1"
        assert "key1" in manager.working.list(tags=["tag1"])

    def test_search_working_by_key(self, mock_context):
        from agentic_cli.memory import MemoryManager

        manager = MemoryManager(mock_context.settings)
        manager.working.set("current_analysis", "some value")
        manager.working.set("other_key", "other value")
        results = manager.search("analysis", include_longterm=False)
        # Should find by key match
        assert len(results.working_results) >= 1
        assert any(key == "current_analysis" for key, _ in results.working_results)

    def test_search_working_by_string_value(self, mock_context):
        from agentic_cli.memory import MemoryManager

        manager = MemoryManager(mock_context.settings)
        manager.working.set("task", "analyzing papers about risk")
        manager.working.set("note", {"dict": "value"})  # Should not match string search
        results = manager.search("risk", include_longterm=False)
        # Should find by string value match
        assert len(results.working_results) >= 1
        assert any(key == "task" for key, _ in results.working_results)


class TestMemorySearchResult:
    """Tests for MemorySearchResult dataclass."""

    def test_default_values(self):
        from agentic_cli.memory import MemorySearchResult

        result = MemorySearchResult(query="test")
        assert result.query == "test"
        assert result.working_results == []
        assert result.longterm_results == []
        assert result.kb_results == []

    def test_with_values(self, mock_context):
        from agentic_cli.memory import (
            MemorySearchResult,
            LongTermMemoryEntry,
            MemoryType,
        )

        entry = LongTermMemoryEntry(
            id="123",
            type=MemoryType.FACT,
            content="Test",
            source="test",
        )
        result = MemorySearchResult(
            query="search",
            working_results=[("key1", "value1")],
            longterm_results=[entry],
            kb_results=[{"doc_id": "doc1"}],
        )
        assert result.query == "search"
        assert len(result.working_results) == 1
        assert len(result.longterm_results) == 1
        assert len(result.kb_results) == 1


class TestMemoryTools:
    """Tests for memory tools that agents can use."""

    def test_working_memory_tool_set_get(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        result = working_memory_tool(
            operation="set", key="task", value="analyzing", settings=mock_context.settings
        )
        assert result["success"] is True
        result = working_memory_tool(
            operation="get", key="task", settings=mock_context.settings
        )
        assert result["success"] is True
        assert result["value"] == "analyzing"

    def test_working_memory_tool_list(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        working_memory_tool(
            operation="set",
            key="item1",
            value="v1",
            tags=["tag1"],
            settings=mock_context.settings,
        )
        working_memory_tool(
            operation="set",
            key="item2",
            value="v2",
            tags=["tag2"],
            settings=mock_context.settings,
        )
        result = working_memory_tool(operation="list", settings=mock_context.settings)
        assert result["success"] is True
        assert set(result["keys"]) == {"item1", "item2"}

    def test_working_memory_tool_list_with_tags(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        working_memory_tool(
            operation="set",
            key="item1",
            value="v1",
            tags=["research"],
            settings=mock_context.settings,
        )
        working_memory_tool(
            operation="set",
            key="item2",
            value="v2",
            tags=["personal"],
            settings=mock_context.settings,
        )
        result = working_memory_tool(
            operation="list", tags=["research"], settings=mock_context.settings
        )
        assert result["success"] is True
        assert result["keys"] == ["item1"]

    def test_working_memory_tool_delete(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        working_memory_tool(
            operation="set", key="to_delete", value="temp", settings=mock_context.settings
        )
        result = working_memory_tool(
            operation="delete", key="to_delete", settings=mock_context.settings
        )
        assert result["success"] is True
        assert result["key"] == "to_delete"
        result = working_memory_tool(
            operation="get", key="to_delete", settings=mock_context.settings
        )
        assert result["value"] is None

    def test_working_memory_tool_clear(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        working_memory_tool(
            operation="set", key="item1", value="v1", settings=mock_context.settings
        )
        working_memory_tool(
            operation="set", key="item2", value="v2", settings=mock_context.settings
        )
        result = working_memory_tool(operation="clear", settings=mock_context.settings)
        assert result["success"] is True
        result = working_memory_tool(operation="list", settings=mock_context.settings)
        assert result["keys"] == []

    def test_working_memory_tool_invalid_operation(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        result = working_memory_tool(
            operation="invalid", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_working_memory_tool_set_missing_key(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        result = working_memory_tool(
            operation="set", value="no key", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_working_memory_tool_get_missing_key(self, mock_context):
        from agentic_cli.memory.tools import working_memory_tool, reset_working_memory

        reset_working_memory()
        result = working_memory_tool(
            operation="get", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_store_recall(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store",
            content="User prefers markdown",
            type="preference",
            settings=mock_context.settings,
        )
        assert result["success"] is True
        assert "entry_id" in result
        result = long_term_memory_tool(
            operation="recall", query="markdown", settings=mock_context.settings
        )
        assert result["success"] is True
        assert len(result["entries"]) > 0

    def test_long_term_memory_tool_store_with_tags(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store",
            content="Risk management fact",
            type="fact",
            tags=["risk", "finance"],
            settings=mock_context.settings,
        )
        assert result["success"] is True
        assert "entry_id" in result

    def test_long_term_memory_tool_store_with_kb_references(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store",
            content="Related to document",
            type="reference",
            kb_references=["doc_123", "doc_456"],
            settings=mock_context.settings,
        )
        assert result["success"] is True
        assert "entry_id" in result

    def test_long_term_memory_tool_update(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        # First store an entry
        store_result = long_term_memory_tool(
            operation="store",
            content="Original content",
            type="fact",
            settings=mock_context.settings,
        )
        entry_id = store_result["entry_id"]

        # Update it
        result = long_term_memory_tool(
            operation="update",
            entry_id=entry_id,
            content="Updated content",
            settings=mock_context.settings,
        )
        assert result["success"] is True
        assert result["entry_id"] == entry_id

        # Recall and verify
        recall_result = long_term_memory_tool(
            operation="recall", query="Updated", settings=mock_context.settings
        )
        assert len(recall_result["entries"]) > 0
        assert "Updated content" in recall_result["entries"][0]["content"]

    def test_long_term_memory_tool_forget(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        # Store an entry
        store_result = long_term_memory_tool(
            operation="store",
            content="To be forgotten",
            type="fact",
            settings=mock_context.settings,
        )
        entry_id = store_result["entry_id"]

        # Forget it
        result = long_term_memory_tool(
            operation="forget", entry_id=entry_id, settings=mock_context.settings
        )
        assert result["success"] is True
        assert result["entry_id"] == entry_id

        # Verify it's gone
        recall_result = long_term_memory_tool(
            operation="recall", query="forgotten", settings=mock_context.settings
        )
        assert len(recall_result["entries"]) == 0

    def test_long_term_memory_tool_recall_by_type(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        # Store entries of different types
        long_term_memory_tool(
            operation="store",
            content="A fact",
            type="fact",
            settings=mock_context.settings,
        )
        long_term_memory_tool(
            operation="store",
            content="A preference",
            type="preference",
            settings=mock_context.settings,
        )

        # Recall only preferences
        result = long_term_memory_tool(
            operation="recall", query="", type="preference", settings=mock_context.settings
        )
        assert result["success"] is True
        assert len(result["entries"]) == 1
        assert result["entries"][0]["type"] == "preference"

    def test_long_term_memory_tool_invalid_operation(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="invalid", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_store_missing_content(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store", type="fact", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_store_missing_type(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store", content="some content", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_invalid_type(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="store",
            content="some content",
            type="invalid_type",
            settings=mock_context.settings,
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_update_missing_entry_id(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="update", content="new content", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_forget_missing_entry_id(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        result = long_term_memory_tool(
            operation="forget", settings=mock_context.settings
        )
        assert result["success"] is False
        assert "error" in result

    def test_long_term_memory_tool_recall_entry_format(self, mock_context):
        from agentic_cli.memory.tools import long_term_memory_tool

        # Store an entry with all fields
        store_result = long_term_memory_tool(
            operation="store",
            content="Full entry content",
            type="learning",
            tags=["test", "full"],
            kb_references=["doc_1"],
            settings=mock_context.settings,
        )
        entry_id = store_result["entry_id"]

        # Recall and check format
        result = long_term_memory_tool(
            operation="recall", query="Full entry", settings=mock_context.settings
        )
        assert result["success"] is True
        assert len(result["entries"]) > 0
        entry = result["entries"][0]
        assert "id" in entry
        assert "type" in entry
        assert "content" in entry
        assert "tags" in entry
        assert "kb_references" in entry
        assert entry["id"] == entry_id
        assert entry["type"] == "learning"
        assert entry["content"] == "Full entry content"
        assert entry["tags"] == ["test", "full"]
        assert entry["kb_references"] == ["doc_1"]

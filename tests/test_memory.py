"""Tests for simplified memory module."""

import pytest


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_store_and_search(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        item_id = store.store("User prefers markdown output")
        assert item_id is not None

        results = store.search("markdown")
        assert len(results) == 1
        assert results[0].content == "User prefers markdown output"
        assert results[0].id == item_id

    def test_store_with_tags(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        item_id = store.store("Important fact", tags=["fact", "finance"])

        results = store.search("Important")
        assert len(results) == 1
        assert results[0].tags == ["fact", "finance"]

    def test_search_case_insensitive(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Basel III requires 99% confidence")

        results = store.search("BASEL")
        assert len(results) == 1
        assert "Basel" in results[0].content

    def test_search_empty_query_returns_all(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Item 1")
        store.store("Item 2")
        store.store("Item 3")

        results = store.search("")
        assert len(results) == 3

    def test_search_with_limit(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        for i in range(5):
            store.store(f"Memory item {i}")

        results = store.search("", limit=3)
        assert len(results) == 3

    def test_search_no_match(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Something about Python")

        results = store.search("JavaScript")
        assert len(results) == 0

    def test_load_all_empty(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        assert store.load_all() == ""

    def test_load_all_with_items(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Fact one", tags=["fact"])
        store.store("Fact two")

        output = store.load_all()
        assert "Fact one" in output
        assert "Fact two" in output
        assert "[fact]" in output

    def test_persistence_across_instances(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store1 = MemoryStore(mock_context.settings)
        store1.store("Persistent data")

        store2 = MemoryStore(mock_context.settings)
        results = store2.search("Persistent")
        assert len(results) == 1
        assert results[0].content == "Persistent data"

    def test_corrupted_file_starts_fresh(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        # Create a store and add data
        store = MemoryStore(mock_context.settings)
        store.store("Some data")

        # Corrupt the file
        storage_path = mock_context.settings.workspace_dir / "memory" / "memories.json"
        storage_path.write_text("{invalid json")

        # New instance should start fresh
        store2 = MemoryStore(mock_context.settings)
        assert store2.search("") == []

    def test_atomic_write(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Test data")

        # Verify no .tmp file left behind
        tmp_path = mock_context.settings.workspace_dir / "memory" / "memories.tmp"
        assert not tmp_path.exists()

    def test_created_at_is_set(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        store.store("Timestamped item")

        results = store.search("Timestamped")
        assert results[0].created_at != ""


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_to_dict_and_from_dict(self):
        from agentic_cli.tools.memory_tools import MemoryItem

        item = MemoryItem(
            id="test-id",
            content="Test content",
            tags=["tag1", "tag2"],
            created_at="2024-01-01T00:00:00",
        )
        data = item.to_dict()
        restored = MemoryItem.from_dict(data)
        assert restored.id == item.id
        assert restored.content == item.content
        assert restored.tags == item.tags
        assert restored.created_at == item.created_at

    def test_from_dict_defaults(self):
        from agentic_cli.tools.memory_tools import MemoryItem

        data = {"id": "abc", "content": "Hello"}
        item = MemoryItem.from_dict(data)
        assert item.tags == []
        assert item.created_at == ""


class TestMemoryToolFunctions:
    """Tests for save_memory and search_memory tool functions."""

    def test_save_memory_without_context(self):
        from agentic_cli.tools.memory_tools import save_memory

        result = save_memory(content="test")
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_search_memory_without_context(self):
        from agentic_cli.tools.memory_tools import search_memory

        result = search_memory(query="test")
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_save_and_search_with_context(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore
        from agentic_cli.tools.memory_tools import save_memory, search_memory
        from agentic_cli.workflow.context import (
            set_context_memory_manager,
        )

        store = MemoryStore(mock_context.settings)
        token = set_context_memory_manager(store)
        try:
            # Save
            result = save_memory(content="Important learning", tags=["test"])
            assert result["success"] is True
            assert "item_id" in result

            # Search
            result = search_memory(query="Important")
            assert result["success"] is True
            assert result["count"] == 1
            assert result["items"][0]["content"] == "Important learning"
            assert result["items"][0]["tags"] == ["test"]
        finally:
            token.var.reset(token)

    def test_save_memory_with_tags(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore
        from agentic_cli.tools.memory_tools import save_memory, search_memory
        from agentic_cli.workflow.context import set_context_memory_manager

        store = MemoryStore(mock_context.settings)
        token = set_context_memory_manager(store)
        try:
            result = save_memory(content="Tagged item", tags=["a", "b"])
            assert result["success"] is True

            search_result = search_memory(query="Tagged")
            assert search_result["items"][0]["tags"] == ["a", "b"]
        finally:
            token.var.reset(token)

    def test_search_memory_with_limit(self, mock_context):
        from agentic_cli.tools.memory_tools import MemoryStore
        from agentic_cli.tools.memory_tools import save_memory, search_memory
        from agentic_cli.workflow.context import set_context_memory_manager

        store = MemoryStore(mock_context.settings)
        token = set_context_memory_manager(store)
        try:
            for i in range(5):
                save_memory(content=f"Item {i}")

            result = search_memory(query="", limit=2)
            assert result["count"] == 2
        finally:
            token.var.reset(token)

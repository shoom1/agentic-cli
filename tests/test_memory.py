"""Tests for simplified memory module."""

import pytest

from agentic_cli.tools.memory_tools import MemoryStore


@pytest.fixture
def memory_store_ctx(mock_context):
    """Provide a MemoryStore with context set, auto-cleanup."""
    from agentic_cli.workflow.service_registry import set_service_registry

    store = MemoryStore(mock_context.settings)
    token = set_service_registry({"memory_store": store})
    yield store
    token.var.reset(token)


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

    def test_update_existing_memory(self, mock_context):
        store = MemoryStore(mock_context.settings)
        item_id = store.store("original content", tags=["tag1"])
        result = store.update(item_id, content="updated content", tags=["tag2"])
        assert result is True
        items = store.search("updated")
        assert len(items) == 1
        assert items[0].content == "updated content"
        assert items[0].tags == ["tag2"]
        assert items[0].updated_at >= items[0].created_at

    def test_update_nonexistent_memory(self, mock_context):
        store = MemoryStore(mock_context.settings)
        result = store.update("nonexistent-id", content="new content")
        assert result is False

    def test_update_partial_fields(self, mock_context):
        store = MemoryStore(mock_context.settings)
        item_id = store.store("content", tags=["original"])
        store.update(item_id, content="new content")
        item = store._items[item_id]
        assert item.content == "new content"
        assert item.tags == ["original"]  # tags unchanged

    def test_delete_soft(self, mock_context):
        store = MemoryStore(mock_context.settings)
        item_id = store.store("to be archived")
        result = store.delete(item_id)
        assert result is True
        assert store.search("archived") == []
        assert item_id in store._items
        assert store._items[item_id].archived is True

    def test_delete_purge(self, mock_context):
        store = MemoryStore(mock_context.settings)
        item_id = store.store("to be purged")
        result = store.delete(item_id, purge=True)
        assert result is True
        assert item_id not in store._items

    def test_delete_nonexistent(self, mock_context):
        store = MemoryStore(mock_context.settings)
        result = store.delete("nonexistent-id")
        assert result is False

    def test_search_excludes_archived(self, mock_context):
        store = MemoryStore(mock_context.settings)
        store.store("visible memory")
        archived_id = store.store("archived memory")
        store.delete(archived_id)
        results = store.search("")
        assert len(results) == 1
        assert results[0].content == "visible memory"

    def test_search_include_archived(self, mock_context):
        store = MemoryStore(mock_context.settings)
        store.store("visible memory")
        archived_id = store.store("archived memory")
        store.delete(archived_id)
        results = store.search("", include_archived=True)
        assert len(results) == 2


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
        assert item.tags is None
        assert item.created_at == ""

    def test_new_fields_defaults(self):
        """New fields have sensible defaults when created via from_dict with old data."""
        from agentic_cli.tools.memory_tools import MemoryItem

        old_data = {
            "id": "abc-123",
            "content": "test content",
            "tags": ["tag1"],
            "created_at": "2026-01-01T00:00:00",
        }
        item = MemoryItem.from_dict(old_data)
        assert item.updated_at == "2026-01-01T00:00:00"  # falls back to created_at
        assert item.last_accessed_at == "2026-01-01T00:00:00"
        assert item.access_count == 0
        assert item.importance == 5
        assert item.embedding is None
        assert item.archived is False

    def test_new_fields_roundtrip(self):
        """New fields survive serialization roundtrip."""
        from agentic_cli.tools.memory_tools import MemoryItem

        item = MemoryItem(
            id="abc-123",
            content="test",
            tags=None,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-02T00:00:00",
            last_accessed_at="2026-01-03T00:00:00",
            access_count=5,
            importance=8,
            embedding=[0.1, 0.2, 0.3],
            archived=True,
        )
        data = item.to_dict()
        restored = MemoryItem.from_dict(data)
        assert restored.updated_at == "2026-01-02T00:00:00"
        assert restored.last_accessed_at == "2026-01-03T00:00:00"
        assert restored.access_count == 5
        assert restored.importance == 8
        # embedding is excluded from to_dict (stored separately), so it won't round-trip
        assert restored.embedding is None
        assert restored.archived is True

    def test_to_dict_excludes_embedding(self):
        """to_dict does NOT include embedding (stored separately)."""
        from agentic_cli.tools.memory_tools import MemoryItem

        item = MemoryItem(
            id="abc-123",
            content="test",
            tags=None,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            last_accessed_at="2026-01-01T00:00:00",
            access_count=0,
            importance=5,
            embedding=[0.1, 0.2],
            archived=False,
        )
        data = item.to_dict()
        assert "embedding" not in data


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

    def test_save_and_search_with_context(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory, search_memory

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

    def test_save_memory_with_tags(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory, search_memory

        result = save_memory(content="Tagged item", tags=["a", "b"])
        assert result["success"] is True

        search_result = search_memory(query="Tagged")
        assert search_result["items"][0]["tags"] == ["a", "b"]

    def test_search_memory_with_limit(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory, search_memory

        for i in range(5):
            save_memory(content=f"Item {i}")

        result = search_memory(query="", limit=2)
        assert result["count"] == 2

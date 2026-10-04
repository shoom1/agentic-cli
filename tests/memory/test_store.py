"""The memory store, constructed from a directory."""

from __future__ import annotations

from agentic_cli.memory import ForgettingPolicy, MemoryItem, MemoryStore


def test_the_store_keeps_its_files_in_the_directory_it_is_given(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.store("User prefers markdown output", tags=["preference"])

    assert (tmp_path / "memory" / "memories.json").is_file()
    reloaded = MemoryStore(tmp_path / "memory")
    assert [item.content for item in reloaded.search("markdown")] == [
        "User prefers markdown output"
    ]


def test_the_public_names_are_the_store_types():
    assert MemoryItem.__module__ == "agentic_cli.memory.store"
    assert ForgettingPolicy.__module__ == "agentic_cli.memory.store"


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_store_and_search(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("User prefers markdown output")
        assert item_id is not None

        results = store.search("markdown")
        assert len(results) == 1
        assert results[0].content == "User prefers markdown output"
        assert results[0].id == item_id

    def test_store_with_tags(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("Important fact", tags=["fact", "finance"])

        results = store.search("Important")
        assert len(results) == 1
        assert results[0].tags == ["fact", "finance"]

    def test_search_case_insensitive(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Basel III requires 99% confidence")

        results = store.search("BASEL")
        assert len(results) == 1
        assert "Basel" in results[0].content

    def test_search_empty_query_returns_all(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Item 1")
        store.store("Item 2")
        store.store("Item 3")

        results = store.search("")
        assert len(results) == 3

    def test_search_with_limit(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        for i in range(5):
            store.store(f"Memory item {i}")

        results = store.search("", limit=3)
        assert len(results) == 3

    def test_search_no_match(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Something about Python")

        results = store.search("JavaScript")
        assert len(results) == 0

    def test_load_all_empty(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        assert store.load_all() == ""

    def test_load_all_with_items(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Fact one", tags=["fact"])
        store.store("Fact two")

        output = store.load_all()
        assert "Fact one" in output
        assert "Fact two" in output
        assert "[fact]" in output

    def test_persistence_across_instances(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store1 = MemoryStore(tmp_path / "memory")
        store1.store("Persistent data")

        store2 = MemoryStore(tmp_path / "memory")
        results = store2.search("Persistent")
        assert len(results) == 1
        assert results[0].content == "Persistent data"

    def test_corrupted_file_starts_fresh(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        # Create a store and add data
        store = MemoryStore(tmp_path / "memory")
        store.store("Some data")

        # Corrupt the file
        storage_path = tmp_path / "memory" / "memories.json"
        storage_path.write_text("{invalid json")

        # New instance should start fresh
        store2 = MemoryStore(tmp_path / "memory")
        assert store2.search("") == []

    def test_atomic_write(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Test data")

        # The writer's temp file (``.memories.json.<random>.tmp``) must not
        # survive; without an embedding service there's no embeddings file.
        assert sorted(p.name for p in (tmp_path / "memory").iterdir()) == ["memories.json"]

    def test_created_at_is_set(self, tmp_path):
        from agentic_cli.memory import MemoryStore

        store = MemoryStore(tmp_path / "memory")
        store.store("Timestamped item")

        results = store.search("Timestamped")
        assert results[0].created_at != ""

    def test_update_existing_memory(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("original content", tags=["tag1"])
        result = store.update(item_id, content="updated content", tags=["tag2"])
        assert result is True
        items = store.search("updated")
        assert len(items) == 1
        assert items[0].content == "updated content"
        assert items[0].tags == ["tag2"]
        assert items[0].updated_at >= items[0].created_at

    def test_update_nonexistent_memory(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        result = store.update("nonexistent-id", content="new content")
        assert result is False

    def test_update_partial_fields(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("content", tags=["original"])
        store.update(item_id, content="new content")
        item = store._items[item_id]
        assert item.content == "new content"
        assert item.tags == ["original"]  # tags unchanged

    def test_delete_soft(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("to be archived")
        result = store.delete(item_id)
        assert result is True
        assert store.search("archived") == []
        assert item_id in store._items
        assert store._items[item_id].archived is True

    def test_delete_purge(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("to be purged")
        result = store.delete(item_id, purge=True)
        assert result is True
        assert item_id not in store._items

    def test_delete_nonexistent(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        result = store.delete("nonexistent-id")
        assert result is False

    def test_search_excludes_archived(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        store.store("visible memory")
        archived_id = store.store("archived memory")
        store.delete(archived_id)
        results = store.search("")
        assert len(results) == 1
        assert results[0].content == "visible memory"

    def test_search_include_archived(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        store.store("visible memory")
        archived_id = store.store("archived memory")
        store.delete(archived_id)
        results = store.search("", include_archived=True)
        assert len(results) == 2


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_to_dict_and_from_dict(self):
        from agentic_cli.memory import MemoryItem

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
        from agentic_cli.memory import MemoryItem

        data = {"id": "abc", "content": "Hello"}
        item = MemoryItem.from_dict(data)
        assert item.tags is None
        assert item.created_at == ""

    def test_new_fields_defaults(self):
        """New fields have sensible defaults when created via from_dict with old data."""
        from agentic_cli.memory import MemoryItem

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
        from agentic_cli.memory import MemoryItem

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
        from agentic_cli.memory import MemoryItem

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


class TestForgettingPolicy:

    def test_max_age_days(self, tmp_path):
        from datetime import datetime, timedelta
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("old memory")
        old_time = (datetime.now() - timedelta(days=100)).isoformat()
        store._items[item_id].created_at = old_time
        store._items[item_id].last_accessed_at = old_time
        store.store("recent memory")
        result = store.apply_forgetting(ForgettingPolicy(max_age_days=90))
        assert result["archived_count"] == 1
        assert result["remaining_count"] == 1
        assert store._items[item_id].archived is True

    def test_max_inactive_days(self, tmp_path):
        from datetime import datetime, timedelta
        store = MemoryStore(tmp_path / "memory")
        item_id = store.store("inactive memory")
        old_time = (datetime.now() - timedelta(days=40)).isoformat()
        store._items[item_id].last_accessed_at = old_time
        store.store("active memory")
        result = store.apply_forgetting(ForgettingPolicy(max_inactive_days=30))
        assert result["archived_count"] == 1
        assert store._items[item_id].archived is True

    def test_min_importance(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        store.store("low importance", importance=2)
        store.store("high importance", importance=8)
        result = store.apply_forgetting(ForgettingPolicy(min_importance=5))
        assert result["archived_count"] == 1

    def test_budget_top_n(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        for i in range(5):
            store.store(f"memory {i}", importance=i + 1)
        result = store.apply_forgetting(ForgettingPolicy(budget_top_n=3))
        assert result["archived_count"] == 2
        assert result["remaining_count"] == 3

    def test_no_policy_no_changes(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        store.store("memory")
        result = store.apply_forgetting(ForgettingPolicy())
        assert result["archived_count"] == 0

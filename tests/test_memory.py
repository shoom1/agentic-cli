"""Tests for simplified memory module."""

import pytest

from agentic_cli.memory import MemoryStore
from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService


@pytest.fixture
def memory_store_ctx(mock_context):
    """Provide a MemoryStore with context set, auto-cleanup."""
    from agentic_cli.workflow.service_registry import set_service_registry

    store = MemoryStore(mock_context.settings.workspace_dir / "memory")
    token = set_service_registry({"memory_store": store})
    yield store
    token.var.reset(token)


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

    def test_update_memory_tool(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory, update_memory
        result = save_memory(content="original")
        item_id = result["item_id"]
        update_result = update_memory(item_id=item_id, content="updated")
        assert update_result["success"] is True
        assert update_result["updated"] is True

    def test_update_memory_not_found(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import update_memory
        result = update_memory(item_id="nonexistent", content="new")
        assert result["success"] is True
        assert result["updated"] is False

    def test_delete_memory_tool(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory, delete_memory
        result = save_memory(content="to delete")
        item_id = result["item_id"]
        delete_result = delete_memory(item_id=item_id)
        assert delete_result["success"] is True
        assert delete_result["deleted"] is True

    def test_save_memory_with_importance(self, memory_store_ctx):
        from agentic_cli.tools.memory_tools import save_memory
        result = save_memory(content="important fact", importance=9)
        assert result["success"] is True


class TestMemoryToolParity:
    """The @register_tool wrappers in memory_tools.py and the closures
    returned by make_memory_tools() must produce identical results for
    identical inputs — both delegate to the same _with_store helpers,
    and this test locks that invariant in place.
    """

    def _ctx(self, mock_context):
        """Build both entry-point variants on top of one fresh MemoryStore."""
        from agentic_cli.memory import MemoryStore
        from agentic_cli.tools.factories import make_memory_tools
        from agentic_cli.tools import memory_tools as mt
        from agentic_cli.workflow.service_registry import set_service_registry

        store = MemoryStore(mock_context.settings.workspace_dir / "memory")
        factory_tools = make_memory_tools(store)
        token = set_service_registry({"memory_store": store})
        return store, factory_tools, mt, token

    def test_save_output_parity(self, mock_context):
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            registry_out = mt.save_memory(content="fact A", tags=["x"], importance=7)
            factory_out = factory_tools[0](content="fact A", tags=["x"], importance=7)
        finally:
            token.var.reset(token)
        # IDs differ (fresh UUIDs), so compare the contract-bearing keys only.
        assert registry_out.keys() == factory_out.keys()
        for k in ("success", "message"):
            assert registry_out[k] == factory_out[k]

    def test_search_output_parity(self, mock_context):
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            mt.save_memory(content="alpha beta", tags=["t"], importance=6)
            mt.save_memory(content="gamma delta")
            registry_out = mt.search_memory(query="alpha", limit=5)
            factory_out = factory_tools[1](query="alpha", limit=5)
        finally:
            token.var.reset(token)
        assert registry_out == factory_out

    def test_update_output_parity_tags_unchanged(self, mock_context):
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            r = mt.save_memory(content="orig", tags=["keep"])
            item_id = r["item_id"]
            # Neither caller passes tags — both must leave them alone.
            registry_out = mt.update_memory(item_id=item_id, content="new1")
            assert store._items[item_id].tags == ["keep"]
            factory_out = factory_tools[2](item_id=item_id, content="new2")
            assert store._items[item_id].tags == ["keep"]
        finally:
            token.var.reset(token)
        assert registry_out == factory_out == {"success": True, "updated": True}

    def test_update_output_parity_tags_cleared_with_empty_list(self, mock_context):
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            r = mt.save_memory(content="orig", tags=["clear-me"])
            item_id = r["item_id"]
            # Both entry points clear tags when passed an empty list.
            mt.update_memory(item_id=item_id, tags=[])
            assert not store._items[item_id].tags

            r2 = mt.save_memory(content="orig2", tags=["clear-me-2"])
            factory_tools[2](item_id=r2["item_id"], tags=[])
            assert not store._items[r2["item_id"]].tags
        finally:
            token.var.reset(token)

    def test_update_with_null_tags_leaves_them_unchanged(self, mock_context):
        """Models often send null for an optional argument they mean to omit,
        so None must not wipe the tags."""
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            r = mt.save_memory(content="orig", tags=["keep"])
            mt.update_memory(item_id=r["item_id"], content="new", tags=None)
            assert store._items[r["item_id"]].tags == ["keep"]

            r2 = mt.save_memory(content="orig2", tags=["keep-2"])
            factory_tools[2](item_id=r2["item_id"], content="new", tags=None)
            assert store._items[r2["item_id"]].tags == ["keep-2"]
        finally:
            token.var.reset(token)

    def test_delete_output_parity(self, mock_context):
        store, factory_tools, mt, token = self._ctx(mock_context)
        try:
            r1 = mt.save_memory(content="one")
            r2 = mt.save_memory(content="two")
            registry_out = mt.delete_memory(item_id=r1["item_id"])
            factory_out = factory_tools[3](item_id=r2["item_id"])
        finally:
            token.var.reset(token)
        assert registry_out == factory_out == {"success": True, "deleted": True}


class TestMemorySemanticSearch:
    """Tests for semantic search in MemoryStore."""

    def test_semantic_search_returns_results(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        store.store("Python is a programming language")
        store.store("The weather is sunny today")
        store.store("Machine learning uses Python")
        results = store.search("Python programming", limit=10)
        assert len(results) > 0
        # All results should have embeddings now
        for item in store._items.values():
            assert item.embedding is not None

    def test_semantic_search_updates_access_tracking(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        item_id = store.store("important fact")
        results = store.search("fact")
        assert len(results) == 1
        item = store._items[item_id]
        assert item.access_count == 1
        assert item.last_accessed_at >= item.created_at

    def test_fallback_to_substring_without_embeddings(self, mock_context):
        store = MemoryStore(mock_context.settings.workspace_dir / "memory")  # no embedding service
        store.store("Python is great")
        store.store("Java is also good")
        results = store.search("Python")
        assert len(results) == 1
        assert results[0].content == "Python is great"

    def test_importance_and_recency_affect_ranking(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        store.store("important Python fact", importance=9)
        store.store("trivial Python fact", importance=1)
        results = store.search("Python fact", limit=2)
        assert results[0].importance >= results[1].importance

    def test_embeddings_persisted_separately(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        store.store("test content")
        emb_path = store._embeddings_path
        assert emb_path.exists()
        import json
        main_data = json.loads(store._path.read_text())
        for item_data in main_data:
            assert "embedding" not in item_data

    def test_embedding_migration_on_load(self, mock_context):
        """Existing memories without embeddings get embedded on load."""
        store1 = MemoryStore(mock_context.settings.workspace_dir / "memory")
        store1.store("old memory without embedding")
        store2 = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=MockEmbeddingService())
        items = list(store2._items.values())
        assert len(items) == 1
        assert items[0].embedding is not None


class TestMemoryContradictionDetection:

    def test_find_similar_on_store(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        store.store("The user prefers dark mode")
        # MockEmbeddingService uses MD5 hashing — use the same text so
        # similarity is 1.0 (well above threshold), which is enough to
        # verify the detection logic without requiring real semantic embeddings.
        result = store.store_with_similarity_check(
            "The user prefers dark mode",
            similarity_threshold=0.5,
        )
        assert result["stored"] is True
        assert result["item_id"] is not None
        assert len(result["similar_existing"]) > 0
        assert result["similar_existing"][0]["content"] == "The user prefers dark mode"

    def test_no_similar_when_different(self, mock_context):
        emb = MockEmbeddingService()
        store = MemoryStore(mock_context.settings.workspace_dir / "memory", embedding_service=emb)
        store.store("The user prefers dark mode")
        result = store.store_with_similarity_check(
            "Python is a programming language",
            similarity_threshold=0.99,
        )
        assert result["stored"] is True
        assert result["similar_existing"] == []

    def test_no_similarity_check_without_embeddings(self, mock_context):
        store = MemoryStore(mock_context.settings.workspace_dir / "memory")
        result = store.store_with_similarity_check("some content")
        assert result["stored"] is True
        assert result["similar_existing"] == []



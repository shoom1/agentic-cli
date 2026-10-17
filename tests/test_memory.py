"""Tests for simplified memory module."""

import pytest

from agentic_cli.memory import MemoryStore


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



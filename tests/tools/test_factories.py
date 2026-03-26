"""Tests for agentic_cli.tools.factories — service tool factories."""

import pytest

from agentic_cli.tools.factories import (
    make_memory_tools,
    make_kb_tools,
    make_webfetch_tool,
    make_sandbox_tool,
    make_interaction_tools,
)


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


class MockMemoryItem:
    """Minimal MemoryItem stand-in."""

    def __init__(self, id: str, content: str, tags: list[str] | None = None):
        self.id = id
        self.content = content
        self.tags = tags or []


class MockMemoryStore:
    """Mock MemoryStore for factory tests."""

    def __init__(self):
        self._items: dict[str, MockMemoryItem] = {}
        self._counter = 0

    def store(self, content: str, tags: list[str] | None = None) -> str:
        self._counter += 1
        item_id = f"mem_{self._counter}"
        self._items[item_id] = MockMemoryItem(item_id, content, tags)
        return item_id

    def search(self, query: str, limit: int = 10) -> list[MockMemoryItem]:
        query_lower = query.lower()
        results = []
        for item in self._items.values():
            if not query or query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results


# ---------------------------------------------------------------------------
# make_memory_tools
# ---------------------------------------------------------------------------


class TestMakeMemoryTools:
    """Tests for make_memory_tools factory."""

    def test_returns_two_functions(self):
        tools = make_memory_tools(MockMemoryStore())
        assert len(tools) == 2

    def test_function_names(self):
        tools = make_memory_tools(MockMemoryStore())
        assert tools[0].__name__ == "save_memory"
        assert tools[1].__name__ == "search_memory"

    def test_save_memory_stores_content(self):
        store = MockMemoryStore()
        save_memory, _ = make_memory_tools(store)
        result = save_memory(content="Remember this fact")
        assert result["success"] is True
        assert "item_id" in result
        assert result["item_id"].startswith("mem_")
        assert len(store._items) == 1

    def test_save_memory_with_tags(self):
        store = MockMemoryStore()
        save_memory, _ = make_memory_tools(store)
        result = save_memory(content="Preference", tags=["pref"])
        assert result["success"] is True
        item = store._items[result["item_id"]]
        assert item.tags == ["pref"]

    def test_search_memory_finds_match(self):
        store = MockMemoryStore()
        save_memory, search_memory = make_memory_tools(store)
        save_memory(content="User likes markdown output")
        save_memory(content="User works with Python")
        result = search_memory(query="markdown")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["items"][0]["content"] == "User likes markdown output"

    def test_search_memory_empty_query_returns_all(self):
        store = MockMemoryStore()
        save_memory, search_memory = make_memory_tools(store)
        save_memory(content="Item 1")
        save_memory(content="Item 2")
        result = search_memory(query="")
        assert result["count"] == 2

    def test_search_memory_respects_limit(self):
        store = MockMemoryStore()
        save_memory, search_memory = make_memory_tools(store)
        for i in range(10):
            save_memory(content=f"Item {i}")
        result = search_memory(query="", limit=3)
        assert result["count"] == 3

    def test_search_memory_no_match(self):
        store = MockMemoryStore()
        save_memory, search_memory = make_memory_tools(store)
        save_memory(content="Something")
        result = search_memory(query="nonexistent")
        assert result["count"] == 0

    def test_closure_binds_to_specific_store(self):
        """Two factories with different stores are independent."""
        store_a = MockMemoryStore()
        store_b = MockMemoryStore()
        save_a, search_a = make_memory_tools(store_a)
        save_b, search_b = make_memory_tools(store_b)

        save_a(content="In store A")
        save_b(content="In store B")

        result_a = search_a(query="store")
        result_b = search_b(query="store")

        assert result_a["count"] == 1
        assert result_a["items"][0]["content"] == "In store A"
        assert result_b["count"] == 1
        assert result_b["items"][0]["content"] == "In store B"


# ---------------------------------------------------------------------------
# make_kb_tools
# ---------------------------------------------------------------------------


class TestMakeKBTools:
    """Tests for make_kb_tools factory."""

    def test_returns_five_functions(self):
        # We don't need a real KB manager for checking the return value
        tools = make_kb_tools(kb_manager=object())
        assert len(tools) == 5

    def test_function_names(self):
        tools = make_kb_tools(kb_manager=object())
        names = [t.__name__ for t in tools]
        assert names == [
            "search_knowledge_base",
            "ingest_document",
            "read_document",
            "list_documents",
            "open_document",
        ]

    def test_functions_have_docstrings(self):
        tools = make_kb_tools(kb_manager=object())
        for tool in tools:
            assert tool.__doc__ is not None
            assert len(tool.__doc__) > 0


# ---------------------------------------------------------------------------
# make_webfetch_tool
# ---------------------------------------------------------------------------


class TestMakeWebfetchTool:
    """Tests for make_webfetch_tool factory."""

    def test_returns_single_function(self):
        tool = make_webfetch_tool(summarizer=object())
        assert callable(tool)
        assert tool.__name__ == "web_fetch"

    def test_is_async(self):
        import asyncio
        tool = make_webfetch_tool(summarizer=object())
        assert asyncio.iscoroutinefunction(tool)


# ---------------------------------------------------------------------------
# make_sandbox_tool
# ---------------------------------------------------------------------------


class TestMakeSandboxTool:
    """Tests for make_sandbox_tool factory."""

    def test_returns_single_function(self):
        tool = make_sandbox_tool(sandbox_manager=object())
        assert callable(tool)
        assert tool.__name__ == "sandbox_execute"


# ---------------------------------------------------------------------------
# make_interaction_tools
# ---------------------------------------------------------------------------


class TestMakeInteractionTools:
    """Tests for make_interaction_tools factory."""

    def test_returns_one_function(self):
        tools = make_interaction_tools(workflow_manager=object())
        assert len(tools) == 1

    def test_function_name(self):
        tools = make_interaction_tools(workflow_manager=object())
        assert tools[0].__name__ == "ask_clarification"

    def test_is_async(self):
        import asyncio
        tools = make_interaction_tools(workflow_manager=object())
        assert asyncio.iscoroutinefunction(tools[0])

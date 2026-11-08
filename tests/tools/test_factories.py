"""Tests for agentic_cli.tools.factories — service tool factories."""

import pytest

from agentic_cli.tools.factories import (
    make_kb_tools,
    make_webfetch_tool,
    make_sandbox_tool,
    make_interaction_tools,
)


# ---------------------------------------------------------------------------
# make_kb_tools
# ---------------------------------------------------------------------------


class TestMakeKBTools:
    """Tests for make_kb_tools factory."""

    def test_returns_seven_functions(self):
        """make_kb_tools returns 7 tools (kb_ingest split into text/file)."""
        from agentic_cli.tools.factories import make_kb_tools
        from agentic_cli.memory.kb import KnowledgeBase
        tools = make_kb_tools(KnowledgeBase({"project": object()}))
        assert len(tools) == 7
        names = [t.__name__ for t in tools]
        assert "kb_write_concept" in names
        assert "kb_search_concepts" in names

    def test_function_names(self):
        from agentic_cli.memory.kb import KnowledgeBase
        tools = make_kb_tools(KnowledgeBase({"project": object()}))
        names = [t.__name__ for t in tools]
        assert names == [
            "kb_search",
            "kb_ingest_text",
            "kb_ingest_file",
            "kb_read",
            "kb_list",
            "kb_write_concept",
            "kb_search_concepts",
        ]

    def test_functions_have_docstrings(self):
        from agentic_cli.memory.kb import KnowledgeBase
        tools = make_kb_tools(KnowledgeBase({"project": object()}))
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

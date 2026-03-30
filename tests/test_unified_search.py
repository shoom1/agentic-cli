"""Tests for unified_search tool."""

import pytest

from agentic_cli.knowledge_base._mocks import MockEmbeddingService
from agentic_cli.knowledge_base.models import SourceType
from agentic_cli.tools.memory_tools import MemoryStore
from agentic_cli.workflow.service_registry import set_service_registry


@pytest.fixture
def unified_ctx(mock_context):
    """Set up both KB and memory for unified search."""
    from agentic_cli.knowledge_base import KnowledgeBaseManager

    emb = MockEmbeddingService()
    kb = KnowledgeBaseManager(
        settings=mock_context.settings,
        use_mock=True,
        base_dir=mock_context.workspace_dir / "kb",
    )
    kb.ingest_document(
        title="Python Guide",
        content="Python is a high-level programming language used for web development",
        source_type=SourceType.USER,
    )
    memory = MemoryStore(mock_context.settings, embedding_service=emb)
    memory.store("The user prefers Python over Java")

    token = set_service_registry({
        "kb_manager": kb,
        "memory_store": memory,
    })
    yield {"kb": kb, "memory": memory}
    token.var.reset(token)


class TestUnifiedSearch:

    def test_searches_both_sources(self, unified_ctx):
        from agentic_cli.tools.knowledge_tools import unified_search
        result = unified_search(query="Python")
        assert result["success"] is True
        assert len(result["results"]) > 0

    def test_filter_by_source(self, unified_ctx):
        from agentic_cli.tools.knowledge_tools import unified_search
        result = unified_search(query="Python", sources=["memory"])
        assert result["success"] is True
        for r in result["results"]:
            assert r["source"] == "memory"

    def test_empty_query(self, unified_ctx):
        from agentic_cli.tools.knowledge_tools import unified_search
        result = unified_search(query="completely unrelated xyzzy")
        assert result["success"] is True

    def test_no_services_available(self, mock_context):
        from agentic_cli.workflow.service_registry import set_service_registry
        token = set_service_registry({})
        from agentic_cli.tools.knowledge_tools import unified_search
        result = unified_search(query="test")
        assert result["success"] is True
        assert result["results"] == []
        token.var.reset(token)

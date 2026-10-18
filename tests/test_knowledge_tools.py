from tests.kb_registry import kb_services
"""Tests for knowledge_tools.

Covers:
- E1: Error handling in kb_ingest
- E2: success key in kb_search
- E4: ContextVar-based KB manager lifecycle
- New tools: kb_read, kb_list
- Summaries written by the ingest tools
- Paper store migration

How the knowledge base itself stores documents is tested in
``tests/memory/test_kb_manager.py``.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import Document, DocumentChunk, SourceType


# ============================================================================
# Helper to create a test KB manager
# ============================================================================


def _make_kb(tmp_path, summarizer=None):
    """Create a KB manager with mock services in a temp dir."""
    return KnowledgeBaseManager(tmp_path / "kb", summarizer=summarizer, use_mock=True)


# ============================================================================
# Summary generation tests
# ============================================================================


class TestKBSummaryGeneration:
    """Tests for summary generation in KnowledgeBaseManager."""

    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_ingest_tool_uses_the_given_summarizer_when_available(self, kb):
        """Tool-level ingest should invoke the knowledge base's summarizer
        and store its output in Document.summary (not the truncate fallback)."""
        from agentic_cli.tools.knowledge_tools import _ingest_text_with_kb

        received: dict = {}

        class FakeSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                received["content"] = content
                received["prompt"] = prompt
                return "SUMMARY: FAKE_LLM_SUMMARY"

        kb._summarizer = FakeSummarizer()
        long_content = "This is a lengthy body. " * 50
        result = await _ingest_text_with_kb(
            kb,
            content=long_content,
            title="Test Paper",
            source_type="user",
        )

        assert result["success"] is True
        assert result["summary"] == "FAKE_LLM_SUMMARY"
        doc = kb.get_document(result["document_id"])
        assert doc.summary == "FAKE_LLM_SUMMARY"
        # The prompt should mention the title so the LLM has context.
        assert "Test Paper" in received["prompt"]
        # The raw content should be passed through to the summarizer.
        assert long_content in received["content"]

    async def test_ingest_tool_falls_back_when_summarizer_errors(self, kb):
        """If the LLM summarizer raises, ingest should fall back to truncation
        rather than losing the document."""
        from agentic_cli.tools.knowledge_tools import _ingest_text_with_kb

        class BrokenSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                raise RuntimeError("LLM unavailable")

        kb._summarizer = BrokenSummarizer()
        content = "Fallback test content that should survive summarizer errors."
        result = await _ingest_text_with_kb(
            kb,
            content=content,
            title="Fallback Test",
            source_type="user",
        )

        assert result["success"] is True
        # Fallback is the first 500 chars of content.
        assert result["summary"] == content


# ============================================================================
# Paper store migration tests
# ============================================================================


# ============================================================================
# Tool error handling tests (E1, E2)
# ============================================================================


class TestSearchKnowledgeBaseTool:
    """Tests for kb_search error handling and success key."""

    def test_invalid_json_filters_returns_error(self):
        """Invalid JSON in filters returns error dict."""
        from agentic_cli.tools.knowledge_tools import kb_search

        result = kb_search("query", filters="not json")
        assert result["success"] is False

    def test_search_returns_success_key(self):
        """Search results include success=True when context is set."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_search

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = kb_search("test query")
                assert result["success"] is True
                assert "results" in result
                assert "total_matches" in result
            finally:
                token.var.reset(token)

    def test_no_context_returns_error(self):
        """kb_search returns error when KB context is not set."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_search

        token = set_service_registry({})
        try:
            result = kb_search("query")
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)


class TestIngestDocumentTool:
    """Tests for kb_ingest_text / kb_ingest_file error handling."""

    @pytest.mark.asyncio
    async def test_invalid_source_type_returns_error(self):
        """Invalid source_type returns error dict."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_text

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_ingest_text(
                    content="Test",
                    title="Test",
                    source_type="invalid_type",
                )
                assert result["success"] is False
                assert "Invalid source_type" in result["error"]
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_valid_ingest_returns_success(self):
        """Valid ingestion returns success=True with document info."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_text

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_ingest_text(
                    content="Test content for ingestion.",
                    title="Test Document",
                    source_type="user",
                )
                assert result["success"] is True
                assert "document_id" in result
                assert result["title"] == "Test Document"
                assert result["chunks_created"] > 0
                assert "summary" in result
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self):
        """kb_ingest_text returns error when KB context is not set."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_text

        token = set_service_registry({})
        try:
            result = await kb_ingest_text(
                content="Test",
                title="Test",
            )
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_no_content_returns_error(self):
        """Empty text content returns error."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_text

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_ingest_text(content="", title="Empty")
                assert result["success"] is False
                assert "No content" in result["error"]
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_local_file(self):
        """Ingesting a local PDF copies and extracts text."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_file

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                # Create a fake PDF file
                pdf_path = Path(tmp) / "test.pdf"
                pdf_path.write_bytes(b"%PDF-1.4 fake content")

                result = await kb_ingest_file(
                    path=str(pdf_path),
                    title="Local PDF",
                )
                assert result["success"] is True
                assert result["title"] == "Local PDF"
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_local_file_not_found(self):
        """Ingesting a nonexistent file returns error."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_file

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_ingest_file(
                    path="/nonexistent/file.pdf",
                )
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_url_routes_through_hardened_fetcher(self, monkeypatch):
        """kb_ingest_url uses ContentFetcher (SSRF protection) instead of raw httpx."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_url
        from agentic_cli.tools.webfetch.fetcher import FetchResult

        called_urls: list[str] = []

        class _StubFetcher:
            async def fetch(self, url, timeout=30):
                called_urls.append(url)
                return FetchResult(
                    success=True,
                    content="<html>Hi</html>",
                    content_type="text/html",
                )

        # The import inside the helper happens lazily; patch the source.
        import agentic_cli.tools.webfetch_tool as wft
        monkeypatch.setattr(wft, "get_or_create_fetcher", lambda: _StubFetcher())

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.chdir(tmp)
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                with pytest.warns(DeprecationWarning, match="kb_ingest_url"):
                    result = await kb_ingest_url(url="https://example.com/article")
                assert result["success"] is True
                assert called_urls == ["https://example.com/article"]
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_url_rejects_non_http_scheme(self):
        """kb_ingest_url refuses file:// and other non-http(s) schemes."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_ingest_url

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                with pytest.warns(DeprecationWarning, match="kb_ingest_url"):
                    result = await kb_ingest_url(url="file:///etc/passwd")
                assert result["success"] is False
                assert "http" in result["error"].lower()
            finally:
                token.var.reset(token)


# ============================================================================
# kb_read tests
# ============================================================================


class TestKbRead:
    """Tests for kb_read tool."""

    async def test_read_by_id(self):
        """Read document by exact ID (full=True)."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_read

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            doc = kb.ingest_document(
                content="Full document content here.",
                title="Readable Doc",
                source_type=SourceType.USER,
            )
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_read(doc.id, full=True)
                assert result["success"] is True
                assert result["content"] == "Full document content here."
                assert result["title"] == "Readable Doc"
                assert result["truncated"] is False
            finally:
                token.var.reset(token)

    async def test_read_by_title(self):
        """Read document by title substring."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_read

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(
                content="Content here.",
                title="Unique Title ABC",
                source_type=SourceType.USER,
            )
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_read("Unique Title")
                assert result["success"] is True
                assert result["title"] == "Unique Title ABC"
            finally:
                token.var.reset(token)

    async def test_read_not_found(self):
        """Read nonexistent document returns error."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_read

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_read("nonexistent")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    async def test_read_truncation(self):
        """Long content is truncated to max_chars when full=True."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_read

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            long_content = "x" * 50000
            doc = kb.ingest_document(
                content=long_content,
                title="Long Doc",
                source_type=SourceType.USER,
            )
            token = set_service_registry(kb_services(kb))
            try:
                result = await kb_read(doc.id, full=True, max_chars=1000)
                assert result["success"] is True
                assert len(result["content"]) == 1000
                assert result["truncated"] is True
            finally:
                token.var.reset(token)


# ============================================================================
# kb_list tests
# ============================================================================


class TestListDocuments:
    """Tests for kb_list tool."""

    def test_list_empty(self):
        """List returns empty when no documents."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_list

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_service_registry(kb_services(kb))
            try:
                result = kb_list()
                assert result["success"] is True
                assert result["count"] == 0
                assert result["documents"] == []
            finally:
                token.var.reset(token)

    def test_list_with_documents(self):
        """List returns documents with summaries."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_list

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(
                content="Document A content.",
                title="Document A",
                source_type=SourceType.ARXIV,
                metadata={"authors": ["Alice"]},
            )
            kb.ingest_document(
                content="Document B content.",
                title="Document B",
                source_type=SourceType.USER,
            )
            token = set_service_registry(kb_services(kb))
            try:
                result = kb_list()
                assert result["success"] is True
                assert result["count"] == 2
                # Check document fields
                doc_a = next(d for d in result["documents"] if d["title"] == "Document A")
                assert doc_a["source_type"] == "arxiv"
                assert "summary" in doc_a
            finally:
                token.var.reset(token)

    def test_list_filter_source_type(self):
        """List filters by source type."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_list

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(content="A.", title="A", source_type=SourceType.ARXIV)
            kb.ingest_document(content="B.", title="B", source_type=SourceType.USER)
            token = set_service_registry(kb_services(kb))
            try:
                result = kb_list(source_type="arxiv")
                assert result["count"] == 1
                assert result["documents"][0]["source_type"] == "arxiv"
            finally:
                token.var.reset(token)

    def test_list_filter_query(self):
        """List filters by title query."""
        from agentic_cli.workflow.service_registry import set_service_registry
        from agentic_cli.tools.knowledge_tools import kb_list

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(content="A.", title="Attention Paper", source_type=SourceType.USER)
            kb.ingest_document(content="B.", title="BERT Paper", source_type=SourceType.USER)
            token = set_service_registry(kb_services(kb))
            try:
                result = kb_list(query="attention")
                assert result["count"] == 1
                assert result["documents"][0]["title"] == "Attention Paper"
            finally:
                token.var.reset(token)


# ============================================================================
# ContextVar lifecycle tests (E4)
# ============================================================================


class TestKBManagerContextVar:
    """Tests for KB manager ContextVar integration."""

    def test_service_registry_accessors_exist(self):
        from agentic_cli.workflow.service_registry import (
            get_service,
            set_service_registry,
        )
        assert callable(get_service)
        assert callable(set_service_registry)

    def test_service_default_is_none(self):
        from agentic_cli.workflow.service_registry import (
            get_service,
            set_service_registry,
        )
        token = set_service_registry({})
        try:
            assert get_service("kb_manager") is None
        finally:
            token.var.reset(token)

    def test_service_roundtrip(self):
        from agentic_cli.workflow.service_registry import (
            get_service,
            set_service_registry,
        )
        sentinel = object()
        token = set_service_registry({"kb_manager": sentinel})
        try:
            assert get_service("kb_manager") is sentinel
        finally:
            token.var.reset(token)

    def test_kb_tools_declare_the_knowledge_base_service(self):
        """KB tools carry their own service requirement in the registry."""
        from agentic_cli.tools.registry import get_registry

        assert get_registry().get("kb_search").requires == ("knowledge_base",)

    def test_base_manager_has_kb_manager_slot(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        assert hasattr(BaseWorkflowManager, "kb_manager")

    def test_base_manager_has_user_kb_manager_slot(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        assert hasattr(BaseWorkflowManager, "user_kb_manager")

    def test_user_kb_manager_registry_key_exists(self):
        from agentic_cli.workflow.service_registry import (
            get_service,
            set_service_registry,
            USER_KB_MANAGER,
        )
        sentinel = object()
        token = set_service_registry({USER_KB_MANAGER: sentinel})
        try:
            assert get_service(USER_KB_MANAGER) is sentinel
        finally:
            token.var.reset(token)


# ============================================================================
# Two-tier knowledge base tests
# ============================================================================


class TestTwoTierKnowledgeBase:
    """Tests for two-tier (project + user) knowledge base access."""

    @pytest.fixture
    def project_kb(self, tmp_path):
        """Create a project-scoped KB with a document."""
        kb = _make_kb(tmp_path / "project")
        kb.ingest_document(
            content="Project document about transformers.",
            title="Project Transformer Paper",
            source_type=SourceType.ARXIV,
        )
        return kb

    @pytest.fixture
    def user_kb(self, tmp_path):
        """Create a user-scoped KB with a different document."""
        kb = _make_kb(tmp_path / "user")
        kb.ingest_document(
            content="User curated document about attention mechanisms.",
            title="User Attention Paper",
            source_type=SourceType.USER,
        )
        return kb

    def _set_both_contexts(self, project_kb, user_kb):
        """Set both KB contexts via service registry, return token for cleanup."""
        from agentic_cli.workflow.service_registry import set_service_registry
        token = set_service_registry(kb_services(project_kb, user_kb))
        return token

    def _reset_token(self, token):
        token.var.reset(token)

    def test_search_merges_both_kbs(self, project_kb, user_kb):
        """Search returns results from both KBs with scope tags."""
        from agentic_cli.tools.knowledge_tools import kb_search

        token = self._set_both_contexts(project_kb, user_kb)
        try:
            result = kb_search("attention transformers")
            assert result["success"] is True
            scopes = {r.get("scope") for r in result["results"]}
            assert scopes == {"project", "user"}
            # Both KBs should contribute results
            assert result["total_matches"] >= 1
        finally:
            self._reset_token(token)

    @pytest.mark.asyncio
    async def test_ingest_writes_to_project_only(self, project_kb, user_kb):
        """Ingestion only writes to project KB, not user KB."""
        from agentic_cli.tools.knowledge_tools import kb_ingest_text

        token = self._set_both_contexts(project_kb, user_kb)
        try:
            user_count_before = len(user_kb._documents)
            project_count_before = len(project_kb._documents)

            result = await kb_ingest_text(
                content="New ingested document.",
                title="New Doc",
                source_type="user",
            )
            assert result["success"] is True

            # Project KB gains a document
            assert len(project_kb._documents) == project_count_before + 1
            # User KB unchanged
            assert len(user_kb._documents) == user_count_before
        finally:
            self._reset_token(token)

    def test_list_documents_shows_scope(self, project_kb, user_kb):
        """kb_list includes scope indicator for both tiers."""
        from agentic_cli.tools.knowledge_tools import kb_list

        token = self._set_both_contexts(project_kb, user_kb)
        try:
            result = kb_list()
            assert result["success"] is True
            assert result["count"] == 2

            scopes = {d["scope"] for d in result["documents"]}
            assert scopes == {"project", "user"}
        finally:
            self._reset_token(token)

    async def test_read_document_project_first(self, tmp_path):
        """When both KBs have a matching doc, project is returned first."""
        from agentic_cli.tools.knowledge_tools import kb_read

        pkb = _make_kb(tmp_path / "proj")
        ukb = _make_kb(tmp_path / "usr")
        pkb.ingest_document(
            content="Project version of the doc.",
            title="Shared Title",
            source_type=SourceType.USER,
        )
        ukb.ingest_document(
            content="User version of the doc.",
            title="Shared Title",
            source_type=SourceType.USER,
        )

        token = self._set_both_contexts(pkb, ukb)
        try:
            result = await kb_read("Shared Title", full=True)
            assert result["success"] is True
            assert result["content"] == "Project version of the doc."
        finally:
            self._reset_token(token)

    async def test_read_document_falls_back_to_user(self, project_kb, user_kb):
        """Doc only in user KB is still found via fallback."""
        from agentic_cli.tools.knowledge_tools import kb_read

        token = self._set_both_contexts(project_kb, user_kb)
        try:
            result = await kb_read("User Attention Paper", full=True)
            assert result["success"] is True
            assert result["title"] == "User Attention Paper"
            assert "attention" in result["content"].lower()
        finally:
            self._reset_token(token)

    def test_same_dir_single_instance(self, tmp_path):
        """When project and user paths resolve to same dir, one manager is reused."""
        from agentic_cli.tools.knowledge_tools import kb_search

        kb = _make_kb(tmp_path / "shared")
        kb.ingest_document(
            content="Shared document.",
            title="Shared Doc",
            source_type=SourceType.USER,
        )

        # Set same KB for both
        from agentic_cli.workflow.service_registry import set_service_registry
        token = set_service_registry(kb_services(kb, kb))
        try:
            result = kb_search("shared")
            assert result["success"] is True
            # Should not have duplicates — user KB is same instance, so merge is skipped
            doc_ids = [r["document_id"] for r in result["results"]]
            assert len(doc_ids) == len(set(doc_ids))
        finally:
            token.var.reset(token)

    def test_user_kb_search_failure_non_fatal(self, project_kb):
        """Project results returned even if user KB search raises."""
        from agentic_cli.tools.knowledge_tools import kb_search
        from agentic_cli.workflow.service_registry import set_service_registry

        # Create a mock user KB that raises on search
        failing_kb = MagicMock()
        failing_kb.search.side_effect = RuntimeError("User KB broken")

        token = set_service_registry(kb_services(project_kb, failing_kb))
        try:
            result = kb_search("transformers")
            assert result["success"] is True
            # Project results should still be present
            assert result["total_matches"] >= 1
        finally:
            token.var.reset(token)

    def test_search_cross_kb_merge_uses_rrf(self, tmp_path):
        """Cross-KB merge uses Reciprocal Rank Fusion, not absolute scores.

        Scores from separate FAISS / BM25 indexes are not comparable, so
        a "concat + sort by score" merge produces a degenerate ordering
        whenever the two KBs happen to score on different scales. RRF
        works on rank position instead, so a project rank-0 result and
        a user rank-0 result should both end up above either KB's
        rank-1 result, regardless of absolute score magnitudes.
        """
        from agentic_cli.tools.knowledge_tools import kb_search
        from agentic_cli.workflow.service_registry import set_service_registry

        project_kb = MagicMock()
        project_kb.search.return_value = {
            "results": [
                {"document_id": "p1", "score": 0.5, "document_title": "Project Top"},
                {"document_id": "p2", "score": 0.4, "document_title": "Project Mid"},
            ],
            "total_matches": 2,
        }
        user_kb = MagicMock()
        user_kb.search.return_value = {
            "results": [
                {"document_id": "u1", "score": 0.9, "document_title": "User Top"},
                {"document_id": "u2", "score": 0.8, "document_title": "User Mid"},
            ],
            "total_matches": 2,
        }

        token = set_service_registry(kb_services(project_kb, user_kb))
        try:
            result = kb_search("query")
            assert result["success"] is True

            merged_ids = [r["document_id"] for r in result["results"]]
            assert set(merged_ids) == {"p1", "p2", "u1", "u2"}

            # Both rank-0 results (p1, u1) must come before both rank-1
            # results (p2, u2). The old "concat + sort by raw score"
            # would have produced [u1, u2, p1, p2] because user scores
            # happened to be larger — putting u2 (rank 1) above p1
            # (rank 0), which is the bug.
            p1_idx = merged_ids.index("p1")
            p2_idx = merged_ids.index("p2")
            u1_idx = merged_ids.index("u1")
            u2_idx = merged_ids.index("u2")
            assert max(p1_idx, u1_idx) < min(p2_idx, u2_idx)

            # Output scores are RRF-fused, not raw KB scores. RRF rank-0
            # with k=60 is 1/61 ≈ 0.016 — well below the raw 0.4-0.9 input.
            for r in result["results"]:
                assert r["score"] < 0.1
        finally:
            token.var.reset(token)

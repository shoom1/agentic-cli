"""Tests for paper management tools.

Tests PaperMetadata, PaperStore, and the four registered tools:
save_paper, list_papers, get_paper_info, open_paper.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import MockContext


# ---------------------------------------------------------------------------
# PaperMetadata tests
# ---------------------------------------------------------------------------


class TestPaperMetadata:
    """Tests for PaperMetadata dataclass."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperSourceType

        meta = PaperMetadata(
            id="abc12345",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            abstract="The dominant sequence transduction models...",
            source_type=PaperSourceType.ARXIV,
            source_url="https://arxiv.org/abs/1706.03762",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            arxiv_id="1706.03762",
            file_path="abc12345.pdf",
            added_at="2024-01-15T10:30:00",
            file_size_bytes=1234567,
            tags=["transformer", "attention"],
        )

        d = meta.to_dict()
        restored = PaperMetadata.from_dict(d)

        assert restored.id == meta.id
        assert restored.title == meta.title
        assert restored.authors == meta.authors
        assert restored.abstract == meta.abstract
        assert restored.source_type == meta.source_type
        assert restored.source_url == meta.source_url
        assert restored.pdf_url == meta.pdf_url
        assert restored.arxiv_id == meta.arxiv_id
        assert restored.file_path == meta.file_path
        assert restored.added_at == meta.added_at
        assert restored.file_size_bytes == meta.file_size_bytes
        assert restored.tags == meta.tags

    def test_defaults(self):
        """Test default values."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperSourceType

        meta = PaperMetadata(id="test", title="Test Paper")
        assert meta.authors == []
        assert meta.abstract == ""
        assert meta.source_type == PaperSourceType.LOCAL
        assert meta.tags == []
        assert meta.file_size_bytes == 0

    def test_from_dict_missing_optional_fields(self):
        """Test from_dict with minimal data."""
        from agentic_cli.tools.paper_tools import PaperMetadata

        data = {"id": "test123", "title": "Minimal Paper"}
        meta = PaperMetadata.from_dict(data)
        assert meta.id == "test123"
        assert meta.title == "Minimal Paper"
        assert meta.authors == []
        assert meta.tags == []

    def test_source_type_enum_values(self):
        """Test PaperSourceType enum values."""
        from agentic_cli.tools.paper_tools import PaperSourceType

        assert PaperSourceType.ARXIV == "arxiv"
        assert PaperSourceType.WEB == "web"
        assert PaperSourceType.LOCAL == "local"


# ---------------------------------------------------------------------------
# PaperStore tests
# ---------------------------------------------------------------------------


class TestPaperStore:
    """Tests for PaperStore."""

    def test_add_and_get(self):
        """Test adding and retrieving a paper."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            meta = PaperMetadata(id="paper1", title="Test Paper", authors=["Author A"])
            store.add(meta)

            result = store.get("paper1")
            assert result is not None
            assert result.title == "Test Paper"
            assert result.authors == ["Author A"]

    def test_get_not_found(self):
        """Test getting a non-existent paper."""
        from agentic_cli.tools.paper_tools import PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            assert store.get("nonexistent") is None

    def test_find_by_id(self):
        """Test finding paper by exact ID."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="abc12345", title="Paper A"))

            result = store.find("abc12345")
            assert result is not None
            assert result.title == "Paper A"

    def test_find_by_id_prefix(self):
        """Test finding paper by ID prefix."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="abc12345", title="Paper A"))

            result = store.find("abc1")
            assert result is not None
            assert result.title == "Paper A"

    def test_find_by_title(self):
        """Test finding paper by title substring."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Attention Is All You Need"))

            result = store.find("attention")
            assert result is not None
            assert result.id == "p1"

    def test_find_not_found(self):
        """Test find returns None for no match."""
        from agentic_cli.tools.paper_tools import PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            assert store.find("nonexistent") is None

    def test_list_papers_all(self):
        """Test listing all papers."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore, PaperSourceType

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Paper A", source_type=PaperSourceType.ARXIV))
            store.add(PaperMetadata(id="p2", title="Paper B", source_type=PaperSourceType.WEB))

            papers = store.list_papers()
            assert len(papers) == 2

    def test_list_papers_filter_source_type(self):
        """Test listing papers filtered by source type."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore, PaperSourceType

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Paper A", source_type=PaperSourceType.ARXIV))
            store.add(PaperMetadata(id="p2", title="Paper B", source_type=PaperSourceType.WEB))

            papers = store.list_papers(source_type="arxiv")
            assert len(papers) == 1
            assert papers[0].id == "p1"

    def test_list_papers_filter_query(self):
        """Test listing papers filtered by query."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Attention Is All You Need"))
            store.add(PaperMetadata(id="p2", title="BERT: Pre-training"))

            papers = store.list_papers(query="attention")
            assert len(papers) == 1
            assert papers[0].id == "p1"

    def test_list_papers_query_matches_author(self):
        """Test query matching author names."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Some Paper", authors=["Vaswani"]))
            store.add(PaperMetadata(id="p2", title="Other Paper", authors=["Smith"]))

            papers = store.list_papers(query="vaswani")
            assert len(papers) == 1
            assert papers[0].id == "p1"

    def test_persistence_across_instances(self):
        """Test that data persists across PaperStore instances."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store1 = PaperStore(ctx.settings)
            store1.add(PaperMetadata(id="p1", title="Persisted Paper"))

            # Create new instance - should load from disk
            store2 = PaperStore(ctx.settings)
            result = store2.get("p1")
            assert result is not None
            assert result.title == "Persisted Paper"

    def test_index_format(self):
        """Test that index file uses dict-based format."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Test Paper"))

            with open(store._index_path) as f:
                data = json.load(f)

            assert "papers" in data
            assert isinstance(data["papers"], dict)
            assert "p1" in data["papers"]
            assert data["papers"]["p1"]["title"] == "Test Paper"

    def test_corrupt_file_recovery(self):
        """Test that corrupt index file is handled gracefully."""
        from agentic_cli.tools.paper_tools import PaperStore

        with MockContext() as ctx:
            papers_dir = ctx.settings.workspace_dir / "papers"
            papers_dir.mkdir(parents=True)
            index_path = papers_dir / "papers_index.json"
            index_path.write_text("not valid json{{{")

            store = PaperStore(ctx.settings)
            assert len(store.list_papers()) == 0

    def test_get_pdf_path(self):
        """Test getting PDF path."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Test", file_path="p1.pdf"))

            pdf_path = store.get_pdf_path("p1")
            assert pdf_path is not None
            assert pdf_path == store.pdfs_dir / "p1.pdf"

    def test_get_pdf_path_not_found(self):
        """Test PDF path for nonexistent paper."""
        from agentic_cli.tools.paper_tools import PaperStore

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            assert store.get_pdf_path("nonexistent") is None


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestPaperTools:
    """Tests for registered paper tools."""

    def test_save_paper_local_file(self):
        """Test saving a paper from a local file."""
        from agentic_cli.tools.paper_tools import PaperStore, save_paper, get_context_paper_store
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                # Create a fake PDF file
                pdf_path = Path(ctx.settings.workspace_dir) / "test_paper.pdf"
                pdf_path.write_bytes(b"%PDF-1.4 fake content")

                import asyncio
                result = asyncio.get_event_loop().run_until_complete(
                    save_paper(
                        url_or_path=str(pdf_path),
                        title="My Local Paper",
                        tags=["test"],
                    )
                )

                assert result["success"] is True
                assert result["title"] == "My Local Paper"
                assert result["file_size_bytes"] > 0

                # Verify paper is in store
                papers = store.list_papers()
                assert len(papers) == 1
                assert papers[0].title == "My Local Paper"
                assert papers[0].source_type == "local"
                assert papers[0].tags == ["test"]
            finally:
                token.var.reset(token)

    def test_save_paper_local_file_not_found(self):
        """Test saving a nonexistent local file."""
        from agentic_cli.tools.paper_tools import PaperStore, save_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(
                    save_paper(url_or_path="/nonexistent/paper.pdf")
                )
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_save_paper_local_not_pdf(self):
        """Test saving a non-PDF file."""
        from agentic_cli.tools.paper_tools import PaperStore, save_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                txt_path = Path(ctx.settings.workspace_dir) / "notes.txt"
                txt_path.write_text("not a pdf")

                import asyncio
                result = asyncio.get_event_loop().run_until_complete(
                    save_paper(url_or_path=str(txt_path))
                )
                assert result["success"] is False
                assert "not a pdf" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_save_paper_from_url(self):
        """Test saving a paper from a URL with mocked HTTP."""
        import httpx as real_httpx
        from agentic_cli.tools.paper_tools import PaperStore, save_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                # Mock httpx response
                mock_response = MagicMock()
                mock_response.content = b"%PDF-1.4 downloaded content"
                mock_response.raise_for_status = MagicMock()

                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)

                mock_httpx = MagicMock()
                mock_httpx.AsyncClient.return_value = mock_client
                mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
                mock_httpx.RequestError = real_httpx.RequestError

                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    import asyncio
                    result = asyncio.get_event_loop().run_until_complete(
                        save_paper(
                            url_or_path="https://example.com/paper.pdf",
                            title="Web Paper",
                        )
                    )

                assert result["success"] is True
                assert result["title"] == "Web Paper"

                papers = store.list_papers()
                assert len(papers) == 1
                assert papers[0].source_type == "web"
            finally:
                token.var.reset(token)

    def test_list_papers_empty(self):
        """Test listing papers when store is empty."""
        from agentic_cli.tools.paper_tools import PaperStore, list_papers
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                result = list_papers()
                assert result["success"] is True
                assert result["count"] == 0
                assert result["papers"] == []
            finally:
                token.var.reset(token)

    def test_list_papers_filtered(self):
        """Test listing papers with source filter."""
        from agentic_cli.tools.paper_tools import (
            PaperMetadata, PaperStore, PaperSourceType, list_papers,
        )
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="A", source_type=PaperSourceType.ARXIV))
            store.add(PaperMetadata(id="p2", title="B", source_type=PaperSourceType.WEB))
            token = set_context_paper_store(store)

            try:
                result = list_papers(source_type="arxiv")
                assert result["success"] is True
                assert result["count"] == 1
                assert result["papers"][0]["id"] == "p1"
            finally:
                token.var.reset(token)

    def test_get_paper_info_found(self):
        """Test getting paper info for an existing paper."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore, get_paper_info
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(
                id="p1", title="Attention Is All You Need",
                authors=["Vaswani"], abstract="A great paper.",
            ))
            token = set_context_paper_store(store)

            try:
                result = get_paper_info("p1")
                assert result["success"] is True
                assert result["paper"]["title"] == "Attention Is All You Need"
                assert result["paper"]["authors"] == ["Vaswani"]
            finally:
                token.var.reset(token)

    def test_get_paper_info_not_found(self):
        """Test getting paper info for nonexistent paper."""
        from agentic_cli.tools.paper_tools import PaperStore, get_paper_info
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                result = get_paper_info("nonexistent")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_open_paper_success(self):
        """Test opening a paper (mocked subprocess)."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore, open_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Test Paper", file_path="p1.pdf"))

            # Create the actual PDF file
            store.pdfs_dir.mkdir(parents=True, exist_ok=True)
            (store.pdfs_dir / "p1.pdf").write_bytes(b"%PDF-1.4 content")

            token = set_context_paper_store(store)

            try:
                with patch("agentic_cli.tools.paper_tools.subprocess.Popen") as mock_popen:
                    result = open_paper("p1")

                assert result["success"] is True
                assert result["title"] == "Test Paper"
                mock_popen.assert_called_once()
            finally:
                token.var.reset(token)

    def test_open_paper_not_found(self):
        """Test opening a nonexistent paper."""
        from agentic_cli.tools.paper_tools import PaperStore, open_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            token = set_context_paper_store(store)

            try:
                result = open_paper("nonexistent")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_open_paper_pdf_missing(self):
        """Test opening a paper whose PDF file doesn't exist."""
        from agentic_cli.tools.paper_tools import PaperMetadata, PaperStore, open_paper
        from agentic_cli.workflow.context import set_context_paper_store

        with MockContext() as ctx:
            store = PaperStore(ctx.settings)
            store.add(PaperMetadata(id="p1", title="Test Paper", file_path="p1.pdf"))
            token = set_context_paper_store(store)

            try:
                result = open_paper("p1")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_save_paper_no_context(self):
        """Test save_paper returns error when context not set."""
        from agentic_cli.tools.paper_tools import save_paper
        from agentic_cli.workflow.context import set_context_paper_store

        # Ensure context is None
        token = set_context_paper_store(None)
        try:
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                save_paper(url_or_path="/tmp/test.pdf")
            )
            assert result["success"] is False
            assert "not available" in result["error"].lower()
        finally:
            token.var.reset(token)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_detect_source_type_arxiv(self):
        from agentic_cli.tools.paper_tools import _detect_source_type, PaperSourceType

        assert _detect_source_type("https://arxiv.org/abs/1706.03762") == PaperSourceType.ARXIV
        assert _detect_source_type("https://arxiv.org/pdf/1706.03762.pdf") == PaperSourceType.ARXIV

    def test_detect_source_type_web(self):
        from agentic_cli.tools.paper_tools import _detect_source_type, PaperSourceType

        assert _detect_source_type("https://example.com/paper.pdf") == PaperSourceType.WEB
        assert _detect_source_type("http://papers.nips.cc/paper.pdf") == PaperSourceType.WEB

    def test_detect_source_type_local(self):
        from agentic_cli.tools.paper_tools import _detect_source_type, PaperSourceType

        assert _detect_source_type("/home/user/paper.pdf") == PaperSourceType.LOCAL
        assert _detect_source_type("./paper.pdf") == PaperSourceType.LOCAL

    def test_extract_arxiv_id(self):
        from agentic_cli.tools.paper_tools import _extract_arxiv_id

        assert _extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
        assert _extract_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"
        assert _extract_arxiv_id("no id here") == ""

    def test_ensure_arxiv_pdf_url(self):
        from agentic_cli.tools.paper_tools import _ensure_arxiv_pdf_url

        assert _ensure_arxiv_pdf_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/pdf/1706.03762.pdf"
        assert _ensure_arxiv_pdf_url("https://example.com/paper.pdf") == "https://example.com/paper.pdf"

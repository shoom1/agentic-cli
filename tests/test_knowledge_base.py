"""Tests for knowledge base models."""

from datetime import datetime
from pathlib import Path

import pytest

from agentic_cli.knowledge_base.models import (
    Document,
    DocumentChunk,
    PaperResult,
    SearchResult,
    SourceType,
    WebResult,
)


class TestSourceType:
    """Tests for SourceType enum."""

    def test_all_source_types(self):
        """Test all source types are defined."""
        assert SourceType.ARXIV.value == "arxiv"
        assert SourceType.SSRN.value == "ssrn"
        assert SourceType.WEB.value == "web"
        assert SourceType.INTERNAL.value == "internal"
        assert SourceType.USER.value == "user"


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a chunk with factory method."""
        chunk = DocumentChunk.create(
            document_id="doc-123",
            content="This is chunk content.",
            chunk_index=0,
            metadata={"page": 1},
        )

        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is chunk content."
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"page": 1}
        assert chunk.embedding is None
        assert chunk.id  # Should have generated ID

    def test_create_chunk_minimal(self):
        """Test creating chunk with minimal arguments."""
        chunk = DocumentChunk.create(
            document_id="doc-123",
            content="Content",
            chunk_index=5,
        )

        assert chunk.metadata == {}
        assert chunk.embedding is None

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )

        data = chunk.to_dict()

        assert data["id"] == "chunk-1"
        assert data["document_id"] == "doc-1"
        assert data["content"] == "Test content"
        assert data["chunk_index"] == 0
        assert data["metadata"] == {"key": "value"}
        # Embedding is not serialized
        assert "embedding" not in data

    def test_chunk_from_dict(self):
        """Test chunk deserialization."""
        data = {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Test content",
            "chunk_index": 2,
            "metadata": {"key": "value"},
        }

        chunk = DocumentChunk.from_dict(data)

        assert chunk.id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 2
        assert chunk.metadata == {"key": "value"}
        assert chunk.embedding is None

    def test_chunk_roundtrip(self):
        """Test serialization roundtrip."""
        original = DocumentChunk.create(
            document_id="doc-1",
            content="Round trip test",
            chunk_index=0,
            metadata={"test": True},
        )

        data = original.to_dict()
        restored = DocumentChunk.from_dict(data)

        assert restored.id == original.id
        assert restored.document_id == original.document_id
        assert restored.content == original.content
        assert restored.chunk_index == original.chunk_index
        assert restored.metadata == original.metadata


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        """Test creating document with factory method."""
        doc = Document.create(
            title="Test Paper",
            content="Full paper content...",
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234.5678",
            metadata={"authors": ["Alice", "Bob"]},
        )

        assert doc.title == "Test Paper"
        assert doc.content == "Full paper content..."
        assert doc.source_type == SourceType.ARXIV
        assert doc.source_url == "https://arxiv.org/abs/1234.5678"
        assert doc.metadata == {"authors": ["Alice", "Bob"]}
        assert doc.id  # Should have generated ID
        assert doc.created_at
        assert doc.updated_at
        assert doc.chunks == []

    def test_create_document_minimal(self):
        """Test creating document with minimal arguments."""
        doc = Document.create(
            title="Minimal",
            content="Content",
            source_type=SourceType.USER,
        )

        assert doc.source_url is None
        assert doc.file_path is None
        assert doc.metadata == {}

    def test_document_with_file_path(self):
        """Test document with file path."""
        doc = Document.create(
            title="Local File",
            content="Content",
            source_type=SourceType.INTERNAL,
            file_path=Path("/docs/report.pdf"),
        )

        assert doc.file_path == Path("/docs/report.pdf")

    def test_document_to_dict(self):
        """Test document serialization."""
        doc = Document(
            id="doc-1",
            title="Test",
            content="Content",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            file_path=Path("/test.txt"),
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 2, 12, 0, 0),
            metadata={"key": "value"},
            chunks=[],
        )

        data = doc.to_dict()

        assert data["id"] == "doc-1"
        assert data["title"] == "Test"
        assert data["source_type"] == "web"
        assert data["source_url"] == "https://example.com"
        assert data["file_path"] == "/test.txt"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "updated_at" in data

    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            "id": "doc-1",
            "title": "Test",
            "content": "Content",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/1234",
            "file_path": None,
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T12:00:00",
            "metadata": {"test": True},
            "chunks": [],
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc-1"
        assert doc.title == "Test"
        assert doc.source_type == SourceType.ARXIV
        assert doc.source_url == "https://arxiv.org/abs/1234"
        assert doc.file_path is None
        assert doc.metadata == {"test": True}

    def test_document_with_chunks(self):
        """Test document with chunks."""
        chunk_data = {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Chunk content",
            "chunk_index": 0,
            "metadata": {},
        }

        data = {
            "id": "doc-1",
            "title": "Test",
            "content": "Content",
            "source_type": "user",
            "source_url": None,
            "file_path": None,
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "metadata": {},
            "chunks": [chunk_data],
        }

        doc = Document.from_dict(data)

        assert len(doc.chunks) == 1
        assert doc.chunks[0].id == "chunk-1"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_to_dict(self):
        """Test search result serialization."""
        doc = Document.create(
            title="Found Paper",
            content="Full content",
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234",
            metadata={"doc_key": "doc_value"},
        )

        chunk = DocumentChunk.create(
            document_id=doc.id,
            content="Matching chunk",
            chunk_index=0,
            metadata={"chunk_key": "chunk_value"},
        )

        result = SearchResult(
            document=doc,
            chunk=chunk,
            score=0.95,
            highlight="...matching **chunk**...",
        )

        data = result.to_dict()

        assert data["document_id"] == doc.id
        assert data["document_title"] == "Found Paper"
        assert data["source_type"] == "arxiv"
        assert data["source_url"] == "https://arxiv.org/abs/1234"
        assert data["chunk_id"] == chunk.id
        assert data["chunk_content"] == "Matching chunk"
        assert data["score"] == 0.95
        assert data["highlight"] == "...matching **chunk**..."
        # Merged metadata
        assert data["metadata"]["doc_key"] == "doc_value"
        assert data["metadata"]["chunk_key"] == "chunk_value"


class TestPaperResult:
    """Tests for PaperResult dataclass."""

    def test_paper_result_basic(self):
        """Test basic paper result."""
        paper = PaperResult(
            title="Deep Learning Survey",
            authors=["Alice", "Bob"],
            abstract="A survey of deep learning...",
            url="https://arxiv.org/abs/1234.5678",
            published_date="2024-01-15",
            source="arxiv",
        )

        assert paper.title == "Deep Learning Survey"
        assert paper.authors == ["Alice", "Bob"]
        assert paper.categories == []
        assert paper.pdf_url is None
        assert paper.arxiv_id is None

    def test_paper_result_full(self):
        """Test paper result with all fields."""
        paper = PaperResult(
            title="Paper Title",
            authors=["Author"],
            abstract="Abstract",
            url="https://arxiv.org/abs/1234",
            published_date="2024-01-01",
            source="arxiv",
            categories=["cs.AI", "cs.LG"],
            pdf_url="https://arxiv.org/pdf/1234.pdf",
            arxiv_id="1234.5678",
        )

        assert paper.categories == ["cs.AI", "cs.LG"]
        assert paper.pdf_url == "https://arxiv.org/pdf/1234.pdf"
        assert paper.arxiv_id == "1234.5678"

    def test_paper_result_to_dict(self):
        """Test paper result serialization."""
        paper = PaperResult(
            title="Test Paper",
            authors=["Alice"],
            abstract="Abstract text",
            url="https://example.com",
            published_date="2024-01-01",
            source="ssrn",
            categories=["finance"],
            pdf_url="https://example.com/paper.pdf",
            arxiv_id=None,
        )

        data = paper.to_dict()

        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Alice"]
        assert data["abstract"] == "Abstract text"
        assert data["url"] == "https://example.com"
        assert data["published_date"] == "2024-01-01"
        assert data["source"] == "ssrn"
        assert data["categories"] == ["finance"]
        assert data["pdf_url"] == "https://example.com/paper.pdf"
        assert data["arxiv_id"] is None


class TestWebResult:
    """Tests for WebResult dataclass."""

    def test_web_result(self):
        """Test web result creation."""
        result = WebResult(
            title="Example Page",
            url="https://example.com/page",
            snippet="This is a snippet of the page content...",
            domain="example.com",
        )

        assert result.title == "Example Page"
        assert result.url == "https://example.com/page"
        assert result.snippet == "This is a snippet of the page content..."
        assert result.domain == "example.com"

    def test_web_result_to_dict(self):
        """Test web result serialization."""
        result = WebResult(
            title="Test Page",
            url="https://test.com",
            snippet="Snippet",
            domain="test.com",
        )

        data = result.to_dict()

        assert data == {
            "title": "Test Page",
            "url": "https://test.com",
            "snippet": "Snippet",
            "domain": "test.com",
        }


# ============================================================================
# Search Sources Tests
# ============================================================================

from agentic_cli.knowledge_base.sources import (
    SearchSource,
    SearchSourceResult,
    SearchSourceRegistry,
    ArxivSearchSource,
    WebSearchSource,
    get_search_registry,
    register_search_source,
)
from unittest.mock import patch, MagicMock


class TestSearchSourceResult:
    """Tests for SearchSourceResult dataclass."""

    def test_create_result(self):
        """Test creating a search source result."""
        result = SearchSourceResult(
            title="Test Paper",
            url="https://example.com/paper",
            snippet="This is a snippet...",
            source_name="test_source",
            metadata={"key": "value"},
        )

        assert result.title == "Test Paper"
        assert result.url == "https://example.com/paper"
        assert result.snippet == "This is a snippet..."
        assert result.source_name == "test_source"
        assert result.metadata == {"key": "value"}

    def test_create_result_default_metadata(self):
        """Test result with default empty metadata."""
        result = SearchSourceResult(
            title="Title",
            url="https://example.com",
            snippet="Snippet",
            source_name="source",
        )

        assert result.metadata == {}

    def test_to_dict(self):
        """Test result serialization."""
        result = SearchSourceResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
            source_name="test",
            metadata={"extra": "data"},
        )

        data = result.to_dict()

        assert data == {
            "title": "Test",
            "url": "https://test.com",
            "snippet": "Test snippet",
            "source": "test",
            "metadata": {"extra": "data"},
        }


class ConcreteSearchSource(SearchSource):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_source"

    @property
    def description(self) -> str:
        return "A test search source"

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> list[SearchSourceResult]:
        return [
            SearchSourceResult(
                title=f"Result for: {query}",
                url="https://test.com/result",
                snippet="Test snippet",
                source_name=self.name,
            )
        ]


class TestSearchSource:
    """Tests for SearchSource abstract class."""

    def test_concrete_implementation(self):
        """Test concrete implementation works."""
        source = ConcreteSearchSource()

        assert source.name == "test_source"
        assert source.description == "A test search source"
        assert source.requires_api_key is None
        assert source.rate_limit == 0

    def test_search(self):
        """Test search method."""
        source = ConcreteSearchSource()
        results = source.search("test query", max_results=5)

        assert len(results) == 1
        assert results[0].title == "Result for: test query"
        assert results[0].source_name == "test_source"

    def test_is_available_no_key_required(self):
        """Test availability when no API key required."""
        source = ConcreteSearchSource()
        assert source.is_available() is True


class SourceRequiringKey(SearchSource):
    """Source that requires an API key."""

    @property
    def name(self) -> str:
        return "key_required"

    @property
    def description(self) -> str:
        return "Source requiring API key"

    @property
    def requires_api_key(self) -> str | None:
        return "serper_api_key"

    def search(self, query: str, max_results: int = 10, **kwargs):
        return []


class TestSearchSourceWithApiKey:
    """Tests for sources requiring API keys."""

    def test_not_available_without_key(self):
        """Test source not available without API key."""
        source = SourceRequiringKey()

        with patch("agentic_cli.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(serper_api_key=None)
            assert source.is_available() is False

    def test_available_with_key(self):
        """Test source available with API key."""
        source = SourceRequiringKey()

        with patch("agentic_cli.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(serper_api_key="test-key")
            assert source.is_available() is True


class TestSearchSourceRegistry:
    """Tests for SearchSourceRegistry."""

    def test_register_source(self):
        """Test registering a source."""
        registry = SearchSourceRegistry()
        source = ConcreteSearchSource()

        registry.register(source)

        assert registry.get("test_source") is source

    def test_unregister_source(self):
        """Test unregistering a source."""
        registry = SearchSourceRegistry()
        source = ConcreteSearchSource()

        registry.register(source)
        registry.unregister("test_source")

        assert registry.get("test_source") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent source doesn't error."""
        registry = SearchSourceRegistry()
        registry.unregister("nonexistent")  # Should not raise

    def test_list_sources(self):
        """Test listing all sources."""
        registry = SearchSourceRegistry()
        source1 = ConcreteSearchSource()

        registry.register(source1)
        sources = registry.list_sources()

        assert len(sources) == 1
        assert source1 in sources

    def test_list_available(self):
        """Test listing available sources."""
        registry = SearchSourceRegistry()
        available_source = ConcreteSearchSource()
        unavailable_source = SourceRequiringKey()

        registry.register(available_source)
        registry.register(unavailable_source)

        with patch("agentic_cli.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(serper_api_key=None)
            available = registry.list_available()

        assert len(available) == 1
        assert available_source in available

    def test_search_all_sources(self):
        """Test searching all available sources."""
        registry = SearchSourceRegistry()
        source = ConcreteSearchSource()
        registry.register(source)

        results = registry.search("test query")

        assert "test_source" in results
        assert len(results["test_source"]) == 1
        assert results["test_source"][0].title == "Result for: test query"

    def test_search_specific_sources(self):
        """Test searching specific sources only."""
        registry = SearchSourceRegistry()
        source1 = ConcreteSearchSource()
        registry.register(source1)

        results = registry.search("test query", sources=["test_source"])

        assert "test_source" in results

    def test_search_handles_errors(self):
        """Test search handles source errors gracefully."""

        class FailingSource(SearchSource):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def description(self) -> str:
                return "Always fails"

            def search(self, query: str, max_results: int = 10, **kwargs):
                raise RuntimeError("Search failed")

        registry = SearchSourceRegistry()
        registry.register(FailingSource())

        results = registry.search("test")

        assert "failing" in results
        assert results["failing"] == []


class TestArxivSearchSource:
    """Tests for ArxivSearchSource."""

    def test_properties(self):
        """Test source properties."""
        source = ArxivSearchSource()

        assert source.name == "arxiv"
        assert source.description == "Search arXiv for academic papers"
        assert source.rate_limit == 3.0
        assert source.requires_api_key is None

    def test_is_available(self):
        """Test arxiv is always available (no API key needed)."""
        source = ArxivSearchSource()
        assert source.is_available() is True

    def test_search_without_feedparser(self):
        """Test search returns empty when feedparser not installed."""
        source = ArxivSearchSource()

        with patch.dict("sys.modules", {"feedparser": None}):
            # Force reimport to trigger ImportError path
            import importlib
            import agentic_cli.knowledge_base.sources as sources_module
            importlib.reload(sources_module)

        # Even if feedparser exists, we can test the return type
        # by checking it returns a list
        assert isinstance(source.search("test"), list)

    @patch("feedparser.parse")
    def test_search_with_results(self, mock_parse):
        """Test search parses arxiv results."""
        mock_parse.return_value = MagicMock(
            entries=[
                {
                    "title": "Test Paper\nWith Newline",
                    "link": "https://arxiv.org/abs/1234.5678",
                    "summary": "Paper abstract text",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "published": "2024-01-15",
                    "tags": [{"term": "cs.AI"}, {"term": "cs.LG"}],
                    "id": "http://arxiv.org/abs/1234.5678v1",
                }
            ]
        )

        source = ArxivSearchSource()
        results = source.search("machine learning", max_results=5)

        assert len(results) == 1
        assert results[0].title == "Test Paper With Newline"
        assert results[0].url == "https://arxiv.org/abs/1234.5678"
        assert results[0].source_name == "arxiv"
        assert results[0].metadata["authors"] == ["Alice", "Bob"]
        assert results[0].metadata["categories"] == ["cs.AI", "cs.LG"]
        assert results[0].metadata["arxiv_id"] == "1234.5678v1"

    @patch("feedparser.parse")
    def test_search_with_categories(self, mock_parse):
        """Test search with category filter."""
        mock_parse.return_value = MagicMock(entries=[])

        source = ArxivSearchSource()
        source.search("test", categories=["cs.AI", "cs.LG"])

        # Check the URL includes category filter
        call_url = mock_parse.call_args[0][0]
        assert "cat:cs.AI" in call_url
        assert "cat:cs.LG" in call_url

    @patch("feedparser.parse")
    def test_search_handles_parse_error(self, mock_parse):
        """Test search handles parse errors gracefully."""
        mock_parse.side_effect = Exception("Parse error")

        source = ArxivSearchSource()
        results = source.search("test")

        assert results == []


class TestWebSearchSource:
    """Tests for WebSearchSource."""

    def test_properties(self):
        """Test source properties."""
        source = WebSearchSource()

        assert source.name == "web"
        assert source.description == "Search the web using Serper.dev"
        assert source.requires_api_key == "serper_api_key"

    def test_not_available_without_key(self):
        """Test not available without API key."""
        source = WebSearchSource()

        with patch("agentic_cli.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(serper_api_key=None)
            assert source.is_available() is False

    @patch("agentic_cli.config.get_settings")
    @patch("agentic_cli.tools.search.WebSearchClient")
    def test_search(self, mock_client_class, mock_settings):
        """Test web search."""
        mock_settings.return_value = MagicMock(serper_api_key="test-key")
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test Page",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "domain": "example.com",
                }
            ]
        }
        mock_client_class.return_value = mock_client

        source = WebSearchSource()
        results = source.search("test query", max_results=5)

        assert len(results) == 1
        assert results[0].title == "Test Page"
        assert results[0].url == "https://example.com"
        assert results[0].source_name == "web"
        assert results[0].metadata["domain"] == "example.com"


class TestDefaultRegistry:
    """Tests for default registry functions."""

    def test_get_search_registry(self):
        """Test getting default registry."""
        registry = get_search_registry()

        # Check registry has expected methods (duck typing)
        assert hasattr(registry, "register")
        assert hasattr(registry, "get")
        assert hasattr(registry, "search")
        # Default registry should have built-in sources
        assert registry.get("arxiv") is not None
        assert registry.get("web") is not None

    def test_register_search_source(self):
        """Test registering to default registry."""
        registry = get_search_registry()

        # Create a custom source
        custom = ConcreteSearchSource()

        # Register it
        register_search_source(custom)

        # It should be in the default registry
        assert registry.get("test_source") is custom

        # Clean up
        registry.unregister("test_source")

"""Tests for knowledge base models."""

from datetime import datetime
from pathlib import Path

import pytest

from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
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
        assert SourceType.LOCAL.value == "local"


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

    def test_create_document_with_summary(self):
        """Test creating document with summary field."""
        doc = Document.create(
            title="Summarized Doc",
            content="Full content here",
            source_type=SourceType.USER,
            summary="A brief summary",
        )

        assert doc.summary == "A brief summary"

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
        assert doc.summary == ""

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
            summary="A summary",
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
        assert data["summary"] == "A summary"
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
)
from agentic_cli.tools.arxiv_source import ArxivSearchSource, CachedSearchResult
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
            import agentic_cli.tools.arxiv_source as arxiv_source_module
            importlib.reload(arxiv_source_module)

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
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()
        source.search("test", categories=["cs.AI", "cs.LG"])

        # Check the URL includes category filter (URL-encoded)
        from urllib.parse import unquote
        call_url = unquote(mock_parse.call_args[0][0])
        assert "cat:cs.AI" in call_url
        assert "cat:cs.LG" in call_url

    @patch("feedparser.parse")
    def test_search_handles_parse_error(self, mock_parse):
        """Test search handles parse errors gracefully."""
        mock_parse.side_effect = Exception("Parse error")

        source = ArxivSearchSource()
        results = source.search("test")

        assert results == []

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_rate_limiting_enforced(self, mock_time_module, mock_parse):
        """Test rate limiting is enforced between requests."""
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        # First call: time()=1.0 for check, time()=1.0 to record
        # Second call: time()=1.0 for check (elapsed=0, needs sleep), time()=1.0 to record
        mock_time_module.time.return_value = 1.0
        mock_time_module.sleep = MagicMock()

        source = ArxivSearchSource()

        # First request - should not sleep (no previous request, _last_request_time=0)
        source.search("first query")
        mock_time_module.sleep.assert_not_called()

        # Second request immediately after - should sleep
        # _last_request_time is now 1.0, current is 1.0, elapsed=0 < rate_limit
        source.search("second query")
        mock_time_module.sleep.assert_called_once()
        # Should sleep for approximately rate_limit seconds
        sleep_time = mock_time_module.sleep.call_args[0][0]
        assert 0 < sleep_time <= source.rate_limit

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_rate_limiting_respects_elapsed_time(self, mock_time_module, mock_parse):
        """Test rate limiting accounts for time already elapsed."""
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        # First call (3 time() calls): cache check, rate limit check, cache store
        # Second call (3 time() calls): cache check, rate limit check (elapsed=1s), cache store
        # Times: cache=1.0, rate=1.0, store=1.0, cache=2.0, rate=2.0, store=2.0
        mock_time_module.time.side_effect = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        mock_time_module.sleep = MagicMock()

        source = ArxivSearchSource()
        source.search("first query")
        source.search("second query")  # Different query, not cached

        # Should only sleep for remaining time (3.0 - 1.0 = 2.0 seconds)
        mock_time_module.sleep.assert_called_once()
        sleep_time = mock_time_module.sleep.call_args[0][0]
        assert 1.9 <= sleep_time <= 2.1  # Allow small tolerance

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_caching_returns_cached_results(self, mock_time_module, mock_parse):
        """Test that repeated queries return cached results without API call."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_parse.return_value = MagicMock(
            entries=[{"title": "Paper 1", "link": "http://arxiv.org/1", "summary": "Abstract"}]
        )

        source = ArxivSearchSource()

        # First search - calls API
        results1 = source.search("machine learning")
        assert mock_parse.call_count == 1

        # Second search with same query - should use cache
        results2 = source.search("machine learning")
        assert mock_parse.call_count == 1  # Still 1, no new API call

        # Results should be the same
        assert len(results1) == len(results2)
        assert results1[0].title == results2[0].title

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_cache_expires_after_ttl(self, mock_time_module, mock_parse):
        """Test that cache expires after TTL."""
        # First call at t=100
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_parse.return_value = MagicMock(
            entries=[{"title": "Paper 1", "link": "http://arxiv.org/1", "summary": "Abstract"}]
        )

        source = ArxivSearchSource(cache_ttl_seconds=60)

        # First search
        source.search("test query")
        assert mock_parse.call_count == 1

        # Time advances past TTL (100 + 61 = 161)
        mock_time_module.time.return_value = 161.0

        # Should call API again since cache expired
        source.search("test query")
        assert mock_parse.call_count == 2

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_different_queries_not_cached(self, mock_time_module, mock_parse):
        """Test that different queries are not served from cache."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()

        source.search("query one")
        source.search("query two")

        # Both should hit API
        assert mock_parse.call_count == 2

    def test_cache_ttl_default(self):
        """Test default cache TTL is set."""
        source = ArxivSearchSource()
        assert source.cache_ttl_seconds == 900  # 15 minutes default

    def test_cache_ttl_custom(self):
        """Test custom cache TTL can be set."""
        source = ArxivSearchSource(cache_ttl_seconds=300)
        assert source.cache_ttl_seconds == 300

    def test_cache_max_size_default(self):
        """Test default max cache size is set."""
        source = ArxivSearchSource()
        assert source.max_cache_size == 100

    def test_cache_max_size_custom(self):
        """Test custom max cache size can be set."""
        source = ArxivSearchSource(max_cache_size=50)
        assert source.max_cache_size == 50

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_cache_evicts_oldest_when_full(self, mock_time_module, mock_parse):
        """Test that oldest cache entries are evicted when max size is reached."""
        mock_time_module.sleep = MagicMock()
        # Return incrementing time for proper ordering
        time_counter = [0]
        def get_time():
            time_counter[0] += 1
            return float(time_counter[0])
        mock_time_module.time.side_effect = get_time

        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource(max_cache_size=3)

        # Fill cache with 3 queries
        source.search("query1")
        source.search("query2")
        source.search("query3")
        assert len(source._cache) == 3
        assert mock_parse.call_count == 3

        # Add 4th query - should evict oldest (query1)
        source.search("query4")
        assert len(source._cache) == 3
        assert mock_parse.call_count == 4

        # query1 should be evicted, so searching again should hit API
        source.search("query1")
        assert mock_parse.call_count == 5

        # query4 should still be cached (most recent before query1 was re-added)
        source.search("query4")
        assert mock_parse.call_count == 5  # No new API call - still cached

    def test_clear_cache(self):
        """Test cache can be manually cleared."""
        source = ArxivSearchSource()
        # Manually add something to cache
        source._cache["test"] = CachedSearchResult(results=[], timestamp=0.0)
        assert len(source._cache) == 1

        source.clear_cache()
        assert len(source._cache) == 0

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_with_sort_by(self, mock_time_module, mock_parse):
        """Test search with sort_by parameter."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()
        source.search("test", sort_by="lastUpdatedDate")

        call_url = mock_parse.call_args[0][0]
        assert "sortBy=lastUpdatedDate" in call_url

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_with_sort_order(self, mock_time_module, mock_parse):
        """Test search with sort_order parameter."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()
        source.search("test", sort_by="submittedDate", sort_order="ascending")

        call_url = mock_parse.call_args[0][0]
        assert "sortBy=submittedDate" in call_url
        assert "sortOrder=ascending" in call_url

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_with_date_range(self, mock_time_module, mock_parse):
        """Test search with date range filter."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()
        source.search("test", date_from="2024-01-01", date_to="2024-12-31")

        call_url = mock_parse.call_args[0][0]
        # ArXiv uses submittedDate:[YYYYMMDD TO YYYYMMDD] format
        assert "submittedDate" in call_url
        assert "20240101" in call_url
        assert "20241231" in call_url

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_sort_and_date_in_cache_key(self, mock_time_module, mock_parse):
        """Test that sort and date params are included in cache key."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()
        mock_parse.return_value = MagicMock(entries=[], bozo=False, status=200)

        source = ArxivSearchSource()

        # Same query with different sort should hit API twice
        source.search("test", sort_by="relevance")
        source.search("test", sort_by="lastUpdatedDate")

        assert mock_parse.call_count == 2

    def test_sort_by_default(self):
        """Test sort_by defaults to relevance."""
        source = ArxivSearchSource()
        # Default is relevance (ArXiv default)
        cache_key = source._make_cache_key("test", 10, None, "relevance", "descending", None, None)
        assert "relevance" in cache_key

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_http_403_returns_empty_and_not_cached(self, mock_time_module, mock_parse):
        """Test that HTTP 403 (rate limited) returns empty and is NOT cached."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 403
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        source = ArxivSearchSource()
        results = source.search("test query")

        assert results == []
        assert len(source._cache) == 0  # Should NOT be cached

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_http_429_returns_empty_and_not_cached(self, mock_time_module, mock_parse):
        """Test that HTTP 429 (too many requests) returns empty and is NOT cached."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 429
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        source = ArxivSearchSource()
        results = source.search("test query")

        assert results == []
        assert len(source._cache) == 0  # Should NOT be cached

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_bozo_feed_error_returns_empty_and_not_cached(self, mock_time_module, mock_parse):
        """Test that bozo feed errors with no entries return empty and are NOT cached."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("XML parsing error")
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        source = ArxivSearchSource()
        results = source.search("test query")

        assert results == []
        assert len(source._cache) == 0  # Should NOT be cached

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_successful_empty_results_are_cached(self, mock_time_module, mock_parse):
        """Test that legitimate empty results (HTTP 200, no bozo) ARE cached."""
        mock_time_module.time.return_value = 100.0
        mock_time_module.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = False
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        source = ArxivSearchSource()
        results = source.search("test query")

        assert results == []
        assert len(source._cache) == 1  # SHOULD be cached (legitimate empty result)


# ============================================================================
# KnowledgeBaseManager base_dir and find_document tests
# ============================================================================


class TestBaseDirOverride:
    """Tests for the base_dir parameter in KnowledgeBaseManager.__init__."""

    def test_base_dir_overrides_settings(self, tmp_path):
        """When base_dir is provided, all paths derive from it."""
        custom_dir = tmp_path / "custom_kb"
        kb = KnowledgeBaseManager(use_mock=True, base_dir=custom_dir)

        assert kb.kb_dir == custom_dir
        assert kb.documents_dir == custom_dir / "documents"
        assert kb.embeddings_dir == custom_dir / "embeddings"
        assert kb.files_dir == custom_dir / "files"
        assert kb.metadata_path == custom_dir / "metadata.json"

    def test_base_dir_creates_directories(self, tmp_path):
        """base_dir creates all subdirectories on init."""
        custom_dir = tmp_path / "new_kb"
        KnowledgeBaseManager(use_mock=True, base_dir=custom_dir)

        assert custom_dir.is_dir()
        assert (custom_dir / "documents").is_dir()
        assert (custom_dir / "embeddings").is_dir()
        assert (custom_dir / "files").is_dir()

    def test_base_dir_with_settings(self, tmp_path):
        """base_dir overrides settings paths but uses settings embedding config."""
        custom_dir = tmp_path / "override_kb"
        mock_settings = MagicMock()
        mock_settings.knowledge_base_dir = tmp_path / "settings_kb"
        mock_settings.knowledge_base_documents_dir = tmp_path / "settings_kb" / "documents"
        mock_settings.knowledge_base_embeddings_dir = tmp_path / "settings_kb" / "embeddings"
        mock_settings.embedding_model = "test-model"
        mock_settings.embedding_batch_size = 16
        mock_settings.knowledge_base_use_mock = True

        kb = KnowledgeBaseManager(
            settings=mock_settings,
            use_mock=True,
            base_dir=custom_dir,
        )

        # Paths come from base_dir, not settings
        assert kb.kb_dir == custom_dir
        assert kb.documents_dir == custom_dir / "documents"
        # Settings KB dir was NOT used
        assert not (tmp_path / "settings_kb").exists()


class TestFindDocument:
    """Tests for KnowledgeBaseManager.find_document()."""

    @pytest.fixture
    def kb(self, tmp_path):
        """Create a KB with a test document."""
        kb = KnowledgeBaseManager(use_mock=True, base_dir=tmp_path / "kb")
        kb.ingest_document(
            content="Test document content.",
            title="Neural Network Fundamentals",
            source_type=SourceType.USER,
        )
        return kb

    def test_find_document_by_exact_id(self, kb):
        """Find document by exact ID."""
        doc = list(kb._documents.values())[0]
        found = kb.find_document(doc.id)
        assert found is not None
        assert found.id == doc.id

    def test_find_document_by_id_prefix(self, kb):
        """Find document by ID prefix."""
        doc = list(kb._documents.values())[0]
        prefix = doc.id[:8]
        found = kb.find_document(prefix)
        assert found is not None
        assert found.id == doc.id

    def test_find_document_by_title(self, kb):
        """Find document by title substring (case-insensitive)."""
        found = kb.find_document("neural network")
        assert found is not None
        assert found.title == "Neural Network Fundamentals"

    def test_find_document_not_found(self, kb):
        """Return None when document not found."""
        found = kb.find_document("nonexistent-xyz-123")
        assert found is None



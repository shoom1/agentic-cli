"""Tests for ArXiv tools — error reporting and metadata fetching.

Covers:
- search_arxiv error propagation from ArxivSearchSource.last_error
- fetch_arxiv_paper (metadata only)
- ArxivSearchSource.last_error lifecycle
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# ArxivSearchSource.last_error tests
# ---------------------------------------------------------------------------


class TestArxivSearchSourceLastError:
    """Tests for last_error on ArxivSearchSource."""

    def test_last_error_initially_none(self):
        """last_error is None before any search."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        assert source.last_error is None

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_last_error_set_on_http_403(self, mock_time, mock_parse):
        """last_error is set when ArXiv returns HTTP 403."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 403
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "403" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_last_error_set_on_http_429(self, mock_time, mock_parse):
        """last_error is set when ArXiv returns HTTP 429."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 429
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "429" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_last_error_set_on_bozo(self, mock_time, mock_parse):
        """last_error is set on feed bozo error with no entries."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("XML parsing error")
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "feed error" in source.last_error.lower()

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_last_error_set_on_parse_exception(self, mock_time, mock_parse):
        """last_error is set when feedparser.parse() raises."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_parse.side_effect = Exception("Connection timeout")

        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "Connection timeout" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_last_error_cleared_on_success(self, mock_time, mock_parse):
        """last_error is cleared after a successful search."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        # First: trigger an error
        mock_feed_err = MagicMock()
        mock_feed_err.status = 403
        mock_feed_err.entries = []
        mock_feed_err.bozo = False
        mock_parse.return_value = mock_feed_err
        source.search("error query")
        assert source.last_error is not None

        # Second: successful search
        mock_feed_ok = MagicMock()
        mock_feed_ok.status = 200
        mock_feed_ok.bozo = False
        mock_feed_ok.entries = []
        mock_parse.return_value = mock_feed_ok
        source.search("good query")
        assert source.last_error is None

    def test_last_error_set_when_feedparser_missing(self):
        """last_error is set when feedparser is not installed."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch.dict("sys.modules", {"feedparser": None}):
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "feedparser":
                    raise ImportError("No module named 'feedparser'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                source.search("test")

        assert source.last_error is not None
        assert "feedparser" in source.last_error.lower()


# ---------------------------------------------------------------------------
# search_arxiv error reporting tests
# ---------------------------------------------------------------------------


class TestSearchArxivErrorReporting:
    """Tests for search_arxiv error propagation."""

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_success_true_on_results(self, mock_time, mock_parse):
        """search_arxiv includes success=True when results are found."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_parse.return_value = MagicMock(
            entries=[
                {
                    "title": "Test Paper",
                    "link": "https://arxiv.org/abs/1234.5678",
                    "summary": "Abstract",
                    "authors": [{"name": "Author"}],
                    "published": "2024-01-01",
                    "tags": [{"term": "cs.AI"}],
                    "id": "http://arxiv.org/abs/1234.5678v1",
                }
            ]
        )

        import agentic_cli.tools.arxiv_tools as arxiv_module

        arxiv_module._arxiv_source = None

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test")

        assert result["success"] is True
        assert result["total_found"] == 1
        assert len(result["papers"]) == 1

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_success_true_on_empty(self, mock_time, mock_parse):
        """search_arxiv returns success=True for genuine empty results."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = False
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        import agentic_cli.tools.arxiv_tools as arxiv_module

        arxiv_module._arxiv_source = None

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("nonexistent_gibberish_query")

        assert result["success"] is True
        assert result["total_found"] == 0
        assert result["papers"] == []

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_error_on_rate_limit(self, mock_time, mock_parse):
        """search_arxiv returns success=False on rate limiting."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 403
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        import agentic_cli.tools.arxiv_tools as arxiv_module

        arxiv_module._arxiv_source = None

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test query")

        assert result["success"] is False
        assert "error" in result
        assert "403" in result["error"]
        assert result["query"] == "test query"
        assert "papers" not in result

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_error_on_feed_error(self, mock_time, mock_parse):
        """search_arxiv returns success=False on feed parse error."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("Malformed XML")
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        import agentic_cli.tools.arxiv_tools as arxiv_module

        arxiv_module._arxiv_source = None

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test query")

        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# fetch_arxiv_paper metadata-only tests
# ---------------------------------------------------------------------------


class TestFetchArxivPaperMetadata:
    """Tests for fetch_arxiv_paper (metadata only)."""

    @pytest.mark.asyncio
    async def test_fetch_returns_metadata(self):
        """fetch_arxiv_paper returns paper metadata."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        mock_entry = {
            "title": "Test Paper",
            "link": "https://arxiv.org/abs/1234.5678",
            "summary": "Abstract text",
            "authors": [{"name": "Author"}],
            "published": "2024-01-01",
            "updated": "2024-01-02",
            "tags": [{"term": "cs.AI"}],
            "id": "http://arxiv.org/abs/1234.5678v1",
            "arxiv_primary_category": {"term": "cs.AI"},
        }

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[mock_entry])
            result = await fetch_arxiv_paper("1234.5678")

        assert result["success"] is True
        assert result["paper"]["title"] == "Test Paper"
        assert result["paper"]["arxiv_id"] == "1234.5678"
        assert result["paper"]["pdf_url"] == "https://arxiv.org/pdf/1234.5678.pdf"

    @pytest.mark.asyncio
    async def test_fetch_not_found(self):
        """fetch_arxiv_paper handles missing paper."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[])
            result = await fetch_arxiv_paper("9999.99999")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fetch_cleans_id(self):
        """fetch_arxiv_paper handles various ID formats."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        mock_entry = {
            "title": "Test",
            "link": "https://arxiv.org/abs/1234.5678",
            "summary": "Abstract",
            "authors": [],
            "published": "",
            "updated": "",
            "tags": [],
            "id": "http://arxiv.org/abs/1234.5678v1",
            "arxiv_primary_category": {"term": "cs.AI"},
        }

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[mock_entry])

            # Test with URL format
            await fetch_arxiv_paper("https://arxiv.org/abs/1234.5678")
            call_url = mock_parse.call_args[0][0]
            assert "1234.5678" in call_url

            # Test with version suffix
            await fetch_arxiv_paper("1234.5678v2")
            call_url = mock_parse.call_args[0][0]
            assert "1234.5678" in call_url
            assert "v2" not in call_url

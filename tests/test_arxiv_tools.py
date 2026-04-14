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
# search_arxiv validation tests (C4: return error dict, not raise)
# ---------------------------------------------------------------------------


class TestSearchArxivValidation:
    """Tests for search_arxiv input validation returning error dicts."""

    def test_search_arxiv_invalid_sort_by_returns_error(self):
        """Invalid sort_by returns {success: False} instead of raising."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test", sort_by="invalid")
        assert result["success"] is False
        assert "sort_by" in result["error"]
        assert "invalid" in result["error"]

    def test_search_arxiv_invalid_sort_order_returns_error(self):
        """Invalid sort_order returns {success: False} instead of raising."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test", sort_order="invalid")
        assert result["success"] is False
        assert "sort_order" in result["error"]
        assert "invalid" in result["error"]


# ---------------------------------------------------------------------------
# search_arxiv error reporting tests
# ---------------------------------------------------------------------------


class TestSearchArxivErrorReporting:
    """Tests for search_arxiv error propagation."""

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_success_true_on_results(self, mock_time, mock_parse, arxiv_source_ctx):
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

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test")

        assert result["success"] is True
        assert result["total_found"] == 1
        assert len(result["papers"]) == 1

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_success_true_on_empty(self, mock_time, mock_parse, arxiv_source_ctx):
        """search_arxiv returns success=True for genuine empty results."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = False
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("nonexistent_gibberish_query")

        assert result["success"] is True
        assert result["total_found"] == 0
        assert result["papers"] == []

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_error_on_rate_limit(self, mock_time, mock_parse, arxiv_source_ctx):
        """search_arxiv returns success=False on rate limiting."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 403
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = search_arxiv("test query")

        assert result["success"] is False
        assert "error" in result
        assert "403" in result["error"]
        assert result["query"] == "test query"
        assert "papers" not in result

    @patch("feedparser.parse")
    @patch("agentic_cli.tools.arxiv_source.time")
    def test_search_arxiv_returns_error_on_feed_error(self, mock_time, mock_parse, arxiv_source_ctx):
        """search_arxiv returns success=False on feed parse error."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("Malformed XML")
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

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
    async def test_fetch_returns_metadata(self, arxiv_source_ctx):
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
        assert result["paper"]["pdf_url"] == "https://arxiv.org/pdf/1234.5678"
        assert result["paper"]["abs_url"] == "https://arxiv.org/abs/1234.5678"
        assert result["paper"]["src_url"] == "https://arxiv.org/e-print/1234.5678"

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, arxiv_source_ctx):
        """fetch_arxiv_paper handles missing paper."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[])
            result = await fetch_arxiv_paper("9999.99999")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fetch_cleans_id(self, arxiv_source_ctx):
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


# ---------------------------------------------------------------------------
# _parse_entry tests — Atom feed link extraction
# ---------------------------------------------------------------------------


class TestArxivParseEntry:
    """Tests for ArxivSearchSource._parse_entry link extraction."""

    def test_parse_entry_uses_feed_pdf_link(self):
        """When the feed advertises a pdf link, use it directly."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        entry = {
            "title": "Test Paper",
            "summary": "Abstract",
            "authors": [{"name": "Author"}],
            "published": "2024-01-01",
            "updated": "2024-01-02",
            "tags": [{"term": "cs.AI"}],
            "id": "http://arxiv.org/abs/1706.03762v5",
            "arxiv_primary_category": {"term": "cs.AI"},
            "links": [
                {"rel": "alternate", "type": "text/html", "href": "http://arxiv.org/abs/1706.03762v5"},
                {"rel": "related", "type": "application/pdf", "href": "http://arxiv.org/pdf/1706.03762v5", "title": "pdf"},
            ],
        }
        paper = source._parse_entry(entry)

        # Version stripped from arxiv_id
        assert paper["arxiv_id"] == "1706.03762"
        # pdf_url comes from the feed (still has version because that's what the feed said)
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1706.03762v5"
        # abs_url upgraded to https
        assert paper["abs_url"] == "https://arxiv.org/abs/1706.03762v5"
        # src_url synthesized from version-stripped id
        assert paper["src_url"] == "https://arxiv.org/e-print/1706.03762"

    def test_parse_entry_synthesizes_links_when_feed_omits_them(self):
        """When the feed has only a single .link, synthesize pdf and src URLs."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        entry = {
            "title": "Test Paper",
            "summary": "Abstract",
            "authors": [],
            "published": "",
            "updated": "",
            "tags": [],
            "id": "http://arxiv.org/abs/1234.5678v1",
            "arxiv_primary_category": {"term": ""},
            "link": "https://arxiv.org/abs/1234.5678v1",
            "links": [],
        }
        paper = source._parse_entry(entry)

        assert paper["arxiv_id"] == "1234.5678"
        assert paper["abs_url"] == "https://arxiv.org/abs/1234.5678v1"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1234.5678"
        assert paper["src_url"] == "https://arxiv.org/e-print/1234.5678"

    def test_parse_entry_with_explicit_arxiv_id(self):
        """fetch_by_id passes arxiv_id explicitly; _parse_entry uses it."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        entry = {
            "title": "T",
            "summary": "",
            "authors": [],
            "published": "",
            "updated": "",
            "tags": [],
            "id": "",
            "arxiv_primary_category": {"term": ""},
            "links": [],
        }
        paper = source._parse_entry(entry, arxiv_id="2301.07041")
        assert paper["arxiv_id"] == "2301.07041"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/2301.07041"
        assert paper["src_url"] == "https://arxiv.org/e-print/2301.07041"
        assert paper["abs_url"] == "https://arxiv.org/abs/2301.07041"

    def test_parse_entry_old_format_id(self):
        """Pre-2007 id format (subject/NNNNNNN) is handled."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        entry = {
            "title": "Old",
            "summary": "",
            "authors": [],
            "published": "",
            "updated": "",
            "tags": [],
            "id": "http://arxiv.org/abs/math/0607733v1",
            "arxiv_primary_category": {"term": ""},
            "links": [],
        }
        paper = source._parse_entry(entry)
        assert paper["arxiv_id"] == "math/0607733"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/math/0607733"

    def test_search_metadata_includes_link_fields(self, arxiv_source_ctx):
        """search() writes abs_url/pdf_url/src_url into SearchSourceResult metadata."""
        from unittest.mock import patch, MagicMock
        from agentic_cli.tools.arxiv_tools import search_arxiv

        with patch("agentic_cli.tools.arxiv_source.time") as mock_time, \
             patch("feedparser.parse") as mock_parse:
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = MagicMock(
                status=200,
                bozo=False,
                entries=[
                    {
                        "title": "Attention",
                        "summary": "abs",
                        "authors": [{"name": "Vaswani"}],
                        "published": "2017-06-12",
                        "tags": [{"term": "cs.CL"}],
                        "id": "http://arxiv.org/abs/1706.03762v5",
                        "links": [
                            {"rel": "alternate", "type": "text/html", "href": "http://arxiv.org/abs/1706.03762v5"},
                            {"rel": "related", "type": "application/pdf", "href": "http://arxiv.org/pdf/1706.03762v5", "title": "pdf"},
                        ],
                    }
                ],
            )

            result = search_arxiv("attention")

        assert result["success"] is True
        paper = result["papers"][0]
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["abs_url"] == "https://arxiv.org/abs/1706.03762v5"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1706.03762v5"
        assert paper["src_url"] == "https://arxiv.org/e-print/1706.03762"

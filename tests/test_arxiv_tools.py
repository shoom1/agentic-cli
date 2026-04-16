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


# ---------------------------------------------------------------------------
# Id-indexed entry cache tests
# ---------------------------------------------------------------------------


class TestArxivEntryCache:
    """Tests for ArxivSearchSource._entry_cache populated via search()."""

    def _make_search_feed(self, arxiv_id: str = "1706.03762"):
        from unittest.mock import MagicMock
        return MagicMock(
            status=200,
            bozo=False,
            entries=[
                {
                    "title": "Attention",
                    "summary": "abs",
                    "authors": [{"name": "Vaswani"}],
                    "published": "2017-06-12",
                    "tags": [{"term": "cs.CL"}],
                    "id": f"http://arxiv.org/abs/{arxiv_id}v5",
                    "links": [
                        {"rel": "alternate", "type": "text/html", "href": f"http://arxiv.org/abs/{arxiv_id}v5"},
                        {"rel": "related", "type": "application/pdf", "href": f"http://arxiv.org/pdf/{arxiv_id}v5", "title": "pdf"},
                    ],
                }
            ],
        )

    @pytest.mark.asyncio
    async def test_fetch_by_id_hits_entry_cache_after_search(self):
        """search() populates entry cache; subsequent fetch_by_id is free."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time:
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = self._make_search_feed("1706.03762")

            source.search("attention")
            assert mock_parse.call_count == 1

            paper = await source.fetch_by_id("1706.03762")

        # Second feedparser call did not happen — cache hit
        assert mock_parse.call_count == 1
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1706.03762v5"

    @pytest.mark.asyncio
    async def test_fetch_by_id_normalizes_version_for_cache_lookup(self):
        """fetch_by_id('Xv2') hits cache populated by search returning X."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time:
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = self._make_search_feed("1706.03762")

            source.search("attention")

            # User asks for a different version — should still cache hit
            paper = await source.fetch_by_id("1706.03762v3")

        assert mock_parse.call_count == 1
        assert paper["arxiv_id"] == "1706.03762"

    @pytest.mark.asyncio
    async def test_fetch_by_id_misses_when_id_not_seen(self):
        """fetch_by_id for an id that no search returned still hits the API."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()

            # First populate cache with one paper via search
            mock_parse.return_value = self._make_search_feed("1706.03762")
            source.search("attention")
            assert mock_parse.call_count == 1

            # Now fetch a different id — feed mock is reused but the call should fire
            mock_parse.return_value = MagicMock(
                entries=[
                    {
                        "title": "Other",
                        "summary": "x",
                        "authors": [],
                        "published": "",
                        "updated": "",
                        "tags": [],
                        "id": "http://arxiv.org/abs/2301.07041v1",
                        "links": [],
                        "arxiv_primary_category": {"term": ""},
                    }
                ]
            )
            paper = await source.fetch_by_id("2301.07041")

        assert mock_parse.call_count == 2  # search + fetch_by_id miss
        assert paper["arxiv_id"] == "2301.07041"

    @pytest.mark.asyncio
    async def test_fetch_by_id_caches_its_own_result_on_miss(self):
        """A second fetch_by_id for the same id is a cache hit."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = MagicMock(
                entries=[
                    {
                        "title": "Test",
                        "summary": "",
                        "authors": [],
                        "published": "",
                        "updated": "",
                        "tags": [],
                        "id": "http://arxiv.org/abs/1234.5678v1",
                        "links": [],
                        "arxiv_primary_category": {"term": ""},
                    }
                ]
            )

            await source.fetch_by_id("1234.5678")
            assert mock_parse.call_count == 1

            await source.fetch_by_id("1234.5678")

        assert mock_parse.call_count == 1  # second call was cached

    def test_clear_cache_clears_entry_cache(self):
        """clear_cache() empties both query and entry caches."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time:
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = self._make_search_feed("1706.03762")

            source.search("attention")
            assert "1706.03762" in source._entry_cache
            assert len(source._cache) == 1

            source.clear_cache()
            assert source._entry_cache == {}
            assert source._cache == {}

    @pytest.mark.asyncio
    async def test_entry_cache_respects_ttl(self):
        """Entry cache entries expire after cache_ttl_seconds."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource(cache_ttl_seconds=60)

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.time") as mock_time, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            # Initial population at t=100
            mock_time.time.return_value = 100.0
            mock_time.sleep = MagicMock()
            mock_parse.return_value = self._make_search_feed("1706.03762")
            source.search("attention")
            assert mock_parse.call_count == 1

            # Jump past TTL
            mock_time.time.return_value = 200.0
            await source.fetch_by_id("1706.03762")

        # Cache expired, second call hit the API again
        assert mock_parse.call_count == 2


# ---------------------------------------------------------------------------
# ArxivSearchSource.download_pdf tests
# ---------------------------------------------------------------------------


class TestArxivDownloadPdf:
    """Direct unit tests for the encapsulated PDF downloader."""

    @pytest.mark.asyncio
    async def test_returns_pdf_bytes_on_success(self):
        """download_pdf returns response.content from a successful GET."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        fake_bytes = b"%PDF-1.4 fake content"

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_response = MagicMock()
            mock_response.content = fake_bytes
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await source.download_pdf("https://arxiv.org/pdf/1706.03762")

        assert result == fake_bytes
        mock_client.get.assert_called_once_with("https://arxiv.org/pdf/1706.03762")

    @pytest.mark.asyncio
    async def test_propagates_http_error(self):
        """download_pdf re-raises errors from the HTTP layer."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(RuntimeError, match="boom"):
                await source.download_pdf("https://arxiv.org/pdf/1706.03762")

    @pytest.mark.asyncio
    async def test_waits_for_rate_limit(self):
        """download_pdf calls _wait_for_rate_limit_async before the GET."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource()
        rate_wait = AsyncMock()

        with patch.object(source, "_wait_for_rate_limit_async", rate_wait), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.content = b"x"
            mock_response.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            await source.download_pdf("https://arxiv.org/pdf/1706.03762")

        rate_wait.assert_awaited_once()


# ---------------------------------------------------------------------------
# ingest_arxiv_paper tool tests
# ---------------------------------------------------------------------------


class _FakeKB:
    """Minimal KnowledgeBaseManager stub for ingestion tests.

    Records the kwargs passed to ingest_document and returns a fake
    Document. Avoids the full KB pipeline and any disk I/O.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.summary_calls: list[dict] = []

    async def generate_summary(self, content: str, title: str = "") -> str:
        self.summary_calls.append({"content": content, "title": title})
        return "fake summary"

    def ingest_document(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeDoc(
            id="doc-123",
            title=kwargs.get("title", ""),
            chunks=[object(), object(), object()],
            summary=kwargs.get("summary", "fake summary"),
        )


class _FakeDoc:
    def __init__(self, id, title, chunks, summary):
        self.id = id
        self.title = title
        self.chunks = chunks
        self.summary = summary


@pytest.fixture
def ingest_ctx():
    """Publish ArxivSearchSource and a fake KB into the registry."""
    from agentic_cli.tools.arxiv_source import ArxivSearchSource
    from agentic_cli.workflow.service_registry import (
        ARXIV_SOURCE,
        KB_MANAGER,
        set_service_registry,
    )

    source = ArxivSearchSource()
    kb = _FakeKB()
    token = set_service_registry({ARXIV_SOURCE: source, KB_MANAGER: kb})
    try:
        yield source, kb
    finally:
        token.var.reset(token)


class TestIngestArxivPaper:
    """Tests for the composed ingest_arxiv_paper tool."""

    @pytest.mark.asyncio
    async def test_happy_path_downloads_pdf_and_ingests(self, ingest_ctx):
        """Fetches metadata, downloads PDF, extracts text, ingests into KB."""
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        # Pre-populate entry cache so fetch_by_id is a hit (no feedparser needed)
        source._store_entry(
            "1706.03762",
            {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani"],
                "abstract": "We propose a new architecture",
                "abs_url": "https://arxiv.org/abs/1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
                "src_url": "https://arxiv.org/e-print/1706.03762",
                "published_date": "2017-06-12",
                "updated_date": "2017-12-06",
                "categories": ["cs.CL", "cs.LG"],
                "primary_category": "cs.CL",
            },
        )

        fake_pdf_bytes = b"%PDF-fake-bytes"

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("agentic_cli.tools.arxiv_tools.extract_pdf_text") as mock_extract, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_response = MagicMock()
            mock_response.content = fake_pdf_bytes
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_extract.return_value = "Extracted PDF text content"

            result = await ingest_arxiv_paper("1706.03762", tags=["transformer"])

        assert result["success"] is True
        assert result["document_id"] == "doc-123"
        assert result["title"] == "Attention Is All You Need"
        assert result["chunks_created"] == 3
        assert result["pdf_downloaded"] is True

        # Verify KB was called with the right shape
        assert len(kb.calls) == 1
        call = kb.calls[0]
        assert call["title"] == "Attention Is All You Need"
        assert call["content"] == "Extracted PDF text content"
        assert call["file_bytes"] == fake_pdf_bytes
        assert call["source_url"] == "https://arxiv.org/abs/1706.03762"
        assert call["metadata"]["arxiv_id"] == "1706.03762"
        assert call["metadata"]["pdf_url"] == "https://arxiv.org/pdf/1706.03762"
        assert call["metadata"]["src_url"] == "https://arxiv.org/e-print/1706.03762"
        assert call["metadata"]["tags"] == ["transformer"]
        assert call["metadata"]["authors"] == ["Vaswani"]

        # Verify the PDF was downloaded from the cached pdf_url, not synthesized
        mock_client.get.assert_called_once_with("https://arxiv.org/pdf/1706.03762")

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_on_pdf_download_failure(self, ingest_ctx):
        """When PDF download fails, ingest the abstract as fallback content."""
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        source._store_entry(
            "2301.07041",
            {
                "arxiv_id": "2301.07041",
                "title": "A paper",
                "authors": [],
                "abstract": "This is the abstract used as fallback",
                "abs_url": "https://arxiv.org/abs/2301.07041",
                "pdf_url": "https://arxiv.org/pdf/2301.07041",
                "src_url": "https://arxiv.org/e-print/2301.07041",
                "published_date": "",
                "updated_date": "",
                "categories": [],
                "primary_category": "",
            },
        )

        with patch("httpx.AsyncClient") as mock_client_cls, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("network error"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await ingest_arxiv_paper("2301.07041")

        assert result["success"] is True
        assert result["pdf_downloaded"] is False
        assert kb.calls[0]["content"] == "This is the abstract used as fallback"
        assert kb.calls[0]["file_bytes"] is None
        assert "pdf_download_error" in kb.calls[0]["metadata"]

    @pytest.mark.asyncio
    async def test_returns_error_when_paper_not_found(self, ingest_ctx):
        """source.fetch_by_id raises LookupError → wrapper returns error dict."""
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        with patch("feedparser.parse") as mock_parse, \
             patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            mock_parse.return_value = MagicMock(entries=[])
            result = await ingest_arxiv_paper("9999.99999")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert kb.calls == []

    @pytest.mark.asyncio
    async def test_returns_error_when_arxiv_source_missing(self):
        """Without a registry, the tool surfaces the require_service error."""
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper
        from agentic_cli.workflow.service_registry import set_service_registry

        token = set_service_registry({})
        try:
            result = await ingest_arxiv_paper("1706.03762")
        finally:
            token.var.reset(token)

        assert result["success"] is False
        assert "arxiv source" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_kb_manager_missing(self):
        """Source present but KB missing returns require_service KB error."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper
        from agentic_cli.workflow.service_registry import (
            ARXIV_SOURCE,
            set_service_registry,
        )

        source = ArxivSearchSource()
        token = set_service_registry({ARXIV_SOURCE: source})
        try:
            result = await ingest_arxiv_paper("1706.03762")
        finally:
            token.var.reset(token)

        assert result["success"] is False
        assert "kb manager" in result["error"].lower()

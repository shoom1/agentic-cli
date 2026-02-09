"""Tests for ArXiv tools — error reporting, PDF download, and save_paper integration.

Covers:
- search_arxiv error propagation from ArxivSearchSource.last_error
- fetch_arxiv_paper with download="pdf"
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
        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()
        assert source.last_error is None

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
    def test_last_error_set_on_http_403(self, mock_time, mock_parse):
        """last_error is set when ArXiv returns HTTP 403."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 403
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "403" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
    def test_last_error_set_on_http_429(self, mock_time, mock_parse):
        """last_error is set when ArXiv returns HTTP 429."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        mock_feed = MagicMock()
        mock_feed.status = 429
        mock_feed.entries = []
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "429" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
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

        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "feed error" in source.last_error.lower()

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
    def test_last_error_set_on_parse_exception(self, mock_time, mock_parse):
        """last_error is set when feedparser.parse() raises."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()
        mock_parse.side_effect = Exception("Connection timeout")

        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()
        source.search("test")

        assert source.last_error is not None
        assert "Connection timeout" in source.last_error

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
    def test_last_error_cleared_on_success(self, mock_time, mock_parse):
        """last_error is cleared after a successful search."""
        mock_time.time.return_value = 100.0
        mock_time.sleep = MagicMock()

        from agentic_cli.knowledge_base.sources import ArxivSearchSource

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
        from agentic_cli.knowledge_base.sources import ArxivSearchSource

        source = ArxivSearchSource()

        with patch.dict("sys.modules", {"feedparser": None}):
            # Need to force the import to fail inside .search()
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
    @patch("agentic_cli.knowledge_base.sources.time")
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
    @patch("agentic_cli.knowledge_base.sources.time")
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
    @patch("agentic_cli.knowledge_base.sources.time")
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
        # No papers key in error response
        assert "papers" not in result

    @patch("feedparser.parse")
    @patch("agentic_cli.knowledge_base.sources.time")
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
# fetch_arxiv_paper download="pdf" tests
# ---------------------------------------------------------------------------


class TestFetchArxivPaperDownloadPdf:
    """Tests for fetch_arxiv_paper with download='pdf'."""

    @pytest.mark.asyncio
    async def test_download_pdf_returns_text_without_bytes(self):
        """download='pdf' returns pdf_text but NOT pdf_bytes (context window fix)."""
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

        import base64

        fake_pdf_bytes = b"%PDF-1.4 fake content"
        fake_pdf_b64 = base64.b64encode(fake_pdf_bytes).decode("ascii")

        mock_download_result = {
            "pdf_text": "Extracted text from PDF",
            "pdf_bytes": fake_pdf_b64,
            "pdf_size_bytes": len(fake_pdf_bytes),
        }

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[mock_entry])

            with patch(
                "agentic_cli.tools.arxiv_tools._download_arxiv_pdf",
                new_callable=AsyncMock,
                return_value=mock_download_result,
            ):
                result = await fetch_arxiv_paper("1234.5678", download="pdf")

        assert result["success"] is True
        assert result["paper"]["title"] == "Test Paper"
        assert result["pdf_text"] == "Extracted text from PDF"
        assert result["pdf_text_truncated"] is False
        assert "pdf_bytes" not in result  # Must NOT leak into LLM context
        assert result["pdf_size_bytes"] == len(fake_pdf_bytes)

    @pytest.mark.asyncio
    async def test_download_pdf_text_truncated(self):
        """Long pdf_text is truncated to PDF_TEXT_MAX_CHARS."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper, PDF_TEXT_MAX_CHARS

        mock_entry = {
            "title": "Long Paper",
            "link": "https://arxiv.org/abs/1234.5678",
            "summary": "Abstract",
            "authors": [],
            "published": "",
            "updated": "",
            "tags": [],
            "id": "http://arxiv.org/abs/1234.5678v1",
            "arxiv_primary_category": {"term": "cs.AI"},
        }

        import base64

        fake_pdf_bytes = b"%PDF-1.4 content"
        fake_pdf_b64 = base64.b64encode(fake_pdf_bytes).decode("ascii")
        long_text = "x" * (PDF_TEXT_MAX_CHARS + 5000)

        mock_download_result = {
            "pdf_text": long_text,
            "pdf_bytes": fake_pdf_b64,
            "pdf_size_bytes": len(fake_pdf_bytes),
        }

        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[mock_entry])

            with patch(
                "agentic_cli.tools.arxiv_tools._download_arxiv_pdf",
                new_callable=AsyncMock,
                return_value=mock_download_result,
            ):
                result = await fetch_arxiv_paper("1234.5678", download="pdf")

        assert result["success"] is True
        assert len(result["pdf_text"]) == PDF_TEXT_MAX_CHARS
        assert result["pdf_text_truncated"] is True

    @pytest.mark.asyncio
    async def test_download_pdf_error_non_fatal(self):
        """PDF download error is non-fatal — metadata still returned."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        mock_entry = {
            "title": "Test Paper",
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

            with patch(
                "agentic_cli.tools.arxiv_tools._download_arxiv_pdf",
                new_callable=AsyncMock,
                return_value={"error": "pypdf not installed"},
            ):
                result = await fetch_arxiv_paper("1234.5678", download="pdf")

        assert result["success"] is True
        assert result["paper"]["title"] == "Test Paper"
        assert result["download_error"] == "pypdf not installed"
        assert "pdf_text" not in result

    @pytest.mark.asyncio
    async def test_no_download_by_default(self):
        """Without download param, no PDF download occurs."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        mock_entry = {
            "title": "Test Paper",
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

            with patch(
                "agentic_cli.tools.arxiv_tools._download_arxiv_pdf",
                new_callable=AsyncMock,
            ) as mock_download:
                result = await fetch_arxiv_paper("1234.5678")

        assert result["success"] is True
        mock_download.assert_not_called()
        assert "pdf_text" not in result
        assert "download_error" not in result


# ---------------------------------------------------------------------------
# _download_arxiv_pdf unit tests
# ---------------------------------------------------------------------------


class TestDownloadArxivPdf:
    """Tests for the _download_arxiv_pdf helper."""

    @pytest.mark.asyncio
    async def test_successful_download(self):
        """Test successful PDF download and text extraction."""
        from agentic_cli.tools.arxiv_tools import _download_arxiv_pdf

        fake_pdf_bytes = b"%PDF-1.4 fake content"

        mock_response = MagicMock()
        mock_response.content = fake_pdf_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_source = MagicMock()
        mock_source.wait_for_rate_limit = MagicMock()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text"
        mock_reader.pages = [mock_page]

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("pypdf.PdfReader", return_value=mock_reader):
                result = await _download_arxiv_pdf("1234.5678", mock_source)

        assert "error" not in result
        assert result["pdf_text"] == "Page 1 text"
        assert result["pdf_size_bytes"] == len(fake_pdf_bytes)
        assert result["pdf_bytes"]  # base64 string
        mock_source.wait_for_rate_limit.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test HTTP error during download."""
        import httpx

        from agentic_cli.tools.arxiv_tools import _download_arxiv_pdf

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_source = MagicMock()
        mock_source.wait_for_rate_limit = MagicMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _download_arxiv_pdf("9999.99999", mock_source)

        assert "error" in result
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_pypdf_not_installed(self):
        """Test error when pypdf is not installed."""
        from agentic_cli.tools.arxiv_tools import _download_arxiv_pdf

        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 content"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_source = MagicMock()
        mock_source.wait_for_rate_limit = MagicMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch.dict("sys.modules", {"pypdf": None}):
                # Force ImportError for pypdf
                original_import = __import__

                def mock_import(name, *args, **kwargs):
                    if name == "pypdf":
                        raise ImportError("No module named 'pypdf'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = await _download_arxiv_pdf("1234.5678", mock_source)

        assert "error" in result
        assert "pypdf" in result["error"].lower()

"""Tests for ArXiv tools — error reporting and metadata fetching.

Covers:
- search_arxiv's result, and its error from ArxivSearchSource.asearch
- fetch_arxiv_paper (metadata only)
- the synchronous ArxivSearchSource.search's last_error lifecycle

arXiv is faked (``tests/tools/arxiv_fakes.py``, the ``arxiv_api`` fixture);
how requests are paced is ``tests/tools/test_arxiv_rate_limit.py``.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest

from tests.tools.arxiv_fakes import ArxivAPI, Paper


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    """No gap between requests and no wait before a retry: these tests are not
    about pacing (tests/tools/test_arxiv_rate_limit.py is)."""
    from agentic_cli.tools import arxiv_source

    monkeypatch.setattr(arxiv_source.ArxivSearchSource, "rate_limit", property(lambda self: 0.0))
    monkeypatch.setattr(arxiv_source, "_RETRY_DELAYS", (0.0, 0.0, 0.0))


def _source(api: ArxivAPI):
    from agentic_cli.tools.arxiv_source import ArxivSearchSource

    return ArxivSearchSource(transport=api.transport)


# ---------------------------------------------------------------------------
# ArxivSearchSource.last_error tests (the synchronous search)
# ---------------------------------------------------------------------------


class TestArxivSearchSourceLastError:
    """Tests for last_error on ArxivSearchSource."""

    def test_last_error_initially_none(self, arxiv_api):
        """last_error is None before any search."""
        assert _source(arxiv_api).last_error is None

    def test_last_error_set_on_http_403(self, arxiv_api):
        """last_error is set when ArXiv returns HTTP 403."""
        arxiv_api.refuse(403)
        source = _source(arxiv_api)

        assert source.search("test") == []
        assert "403" in source.last_error

    def test_last_error_set_on_http_429(self, arxiv_api):
        """last_error is set when ArXiv still refuses with HTTP 429 after its retries."""
        arxiv_api.refuse(429, times=4)
        source = _source(arxiv_api)

        assert source.search("test") == []
        assert "429" in source.last_error

    def test_last_error_set_on_bozo(self, arxiv_api):
        """last_error is set on a feed that does not parse and has no entries."""
        arxiv_api.garble()
        source = _source(arxiv_api)

        assert source.search("test") == []
        assert "feed error" in source.last_error.lower()

    def test_last_error_set_on_a_failed_request(self, arxiv_api):
        """last_error is set when the request fails."""
        arxiv_api.fail(httpx.ConnectError("Connection refused"))
        source = _source(arxiv_api)

        assert source.search("test") == []
        assert "Connection refused" in source.last_error

    def test_last_error_cleared_on_success(self, arxiv_api):
        """last_error is cleared after a successful search."""
        arxiv_api.refuse(403)
        source = _source(arxiv_api)
        source.search("error query")
        assert source.last_error is not None

        source.search("good query")
        assert source.last_error is None

    def test_last_error_set_when_feedparser_missing(self, arxiv_api):
        """last_error is set when feedparser is not installed."""
        source = _source(arxiv_api)

        with patch.dict("sys.modules", {"feedparser": None}):
            source.search("test")

        assert "feedparser" in source.last_error.lower()
        assert arxiv_api.requests == []


# ---------------------------------------------------------------------------
# search_arxiv validation tests (C4: return error dict, not raise)
# ---------------------------------------------------------------------------


class TestSearchArxivValidation:
    """Tests for search_arxiv input validation returning error dicts."""

    async def test_search_arxiv_invalid_sort_by_returns_error(self):
        """Invalid sort_by returns {success: False} instead of raising."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("test", sort_by="invalid")
        assert result["success"] is False
        assert "sort_by" in result["error"]
        assert "invalid" in result["error"]

    async def test_search_arxiv_invalid_sort_order_returns_error(self):
        """Invalid sort_order returns {success: False} instead of raising."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("test", sort_order="invalid")
        assert result["success"] is False
        assert "sort_order" in result["error"]
        assert "invalid" in result["error"]


# ---------------------------------------------------------------------------
# search_arxiv error reporting tests
# ---------------------------------------------------------------------------


class TestSearchArxivErrorReporting:
    """Tests for search_arxiv error propagation."""

    async def test_search_arxiv_returns_success_true_on_results(self, arxiv_api, arxiv_source_ctx):
        """search_arxiv includes success=True when results are found."""
        arxiv_api.papers.append(Paper("1234.5678", title="Test Paper"))

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("test")

        assert result["success"] is True
        assert result["total_found"] == 1
        assert [paper["title"] for paper in result["papers"]] == ["Test Paper"]

    async def test_search_arxiv_returns_success_true_on_empty(self, arxiv_source_ctx):
        """search_arxiv returns success=True for genuine empty results."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("nonexistent_gibberish_query")

        assert result["success"] is True
        assert result["total_found"] == 0
        assert result["papers"] == []

    async def test_search_arxiv_returns_error_on_a_refusal(self, arxiv_api, arxiv_source_ctx):
        """search_arxiv returns success=False when arXiv refuses the request."""
        arxiv_api.refuse(403)

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("test query")

        assert result["success"] is False
        assert "403" in result["error"]
        assert result["query"] == "test query"
        assert "papers" not in result

    async def test_search_arxiv_returns_error_on_feed_error(self, arxiv_api, arxiv_source_ctx):
        """search_arxiv returns success=False on a feed that does not parse."""
        arxiv_api.garble()

        from agentic_cli.tools.arxiv_tools import search_arxiv

        result = await search_arxiv("test query")

        assert result["success"] is False
        assert "feed error" in result["error"].lower()


# ---------------------------------------------------------------------------
# fetch_arxiv_paper metadata-only tests
# ---------------------------------------------------------------------------


class TestFetchArxivPaperMetadata:
    """Tests for fetch_arxiv_paper (metadata only)."""

    async def test_fetch_returns_metadata(self, arxiv_api, arxiv_source_ctx):
        """fetch_arxiv_paper returns paper metadata."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        arxiv_api.papers.append(Paper("1234.5678", title="Test Paper"))

        result = await fetch_arxiv_paper("1234.5678")

        assert result["success"] is True
        assert result["paper"]["title"] == "Test Paper"
        assert result["paper"]["arxiv_id"] == "1234.5678"
        assert result["paper"]["pdf_url"] == "https://arxiv.org/pdf/1234.5678v1"
        assert result["paper"]["abs_url"] == "https://arxiv.org/abs/1234.5678v1"
        assert result["paper"]["src_url"] == "https://arxiv.org/e-print/1234.5678"

    async def test_fetch_not_found(self, arxiv_source_ctx):
        """fetch_arxiv_paper handles missing paper."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        result = await fetch_arxiv_paper("9999.99999")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_fetch_reports_a_failed_request(self, arxiv_api, arxiv_source_ctx):
        """fetch_arxiv_paper reports arXiv's refusal as its error."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        arxiv_api.refuse(403)

        result = await fetch_arxiv_paper("1234.5678")

        assert result["success"] is False
        assert "403" in result["error"]

    async def test_fetch_cleans_id(self, arxiv_api, arxiv_source_ctx):
        """fetch_arxiv_paper handles various ID formats."""
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper

        arxiv_api.papers.append(Paper("1234.5678"))

        await fetch_arxiv_paper("https://arxiv.org/abs/1234.5678")
        arxiv_source_ctx.clear_cache()
        await fetch_arxiv_paper("1234.5678v2")

        ids = [httpx.URL(url).params["id_list"] for url in arxiv_api.requests]
        assert ids == ["1234.5678", "1234.5678"]


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

    async def test_search_metadata_includes_link_fields(self, arxiv_api, arxiv_source_ctx):
        """search writes abs_url/pdf_url/src_url into SearchSourceResult metadata."""
        from agentic_cli.tools.arxiv_tools import search_arxiv

        arxiv_api.papers.append(Paper("1706.03762", title="Attention"))

        result = await search_arxiv("attention")

        assert result["success"] is True
        paper = result["papers"][0]
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["abs_url"] == "https://arxiv.org/abs/1706.03762v1"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1706.03762v1"
        assert paper["src_url"] == "https://arxiv.org/e-print/1706.03762"


# ---------------------------------------------------------------------------
# Id-indexed entry cache tests
# ---------------------------------------------------------------------------


class TestArxivEntryCache:
    """Tests for ArxivSearchSource._entry_cache populated via search."""

    @pytest.fixture
    def api(self, arxiv_api) -> ArxivAPI:
        arxiv_api.papers[:] = [Paper("1706.03762", title="Attention"), Paper("2301.07041")]
        return arxiv_api

    async def test_fetch_by_id_hits_entry_cache_after_search(self, api):
        """A search fills the entry cache; a later fetch_by_id is free."""
        source = _source(api)
        api.papers[:] = api.papers[:1]

        await source.asearch("attention")
        paper = await source.fetch_by_id("1706.03762")

        assert len(api.requests) == 1  # the search's
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1706.03762v1"

    async def test_fetch_by_id_normalizes_version_for_cache_lookup(self, api):
        """fetch_by_id('Xv3') hits the cache a search returning X filled."""
        source = _source(api)

        await source.asearch("attention")
        paper = await source.fetch_by_id("1706.03762v3")

        assert len(api.requests) == 1
        assert paper["arxiv_id"] == "1706.03762"

    async def test_fetch_by_id_misses_when_id_not_seen(self, api):
        """fetch_by_id for an id no search returned still asks arXiv."""
        source = _source(api)
        api.papers[:] = api.papers[:1]
        await source.asearch("attention")

        api.papers.append(Paper("2301.07041", title="Other"))
        paper = await source.fetch_by_id("2301.07041")

        assert len(api.requests) == 2  # the search, then the miss
        assert paper["arxiv_id"] == "2301.07041"

    async def test_fetch_by_id_caches_its_own_result_on_miss(self, api):
        """A second fetch_by_id for the same id is a cache hit."""
        source = _source(api)

        await source.fetch_by_id("2301.07041")
        await source.fetch_by_id("2301.07041")

        assert len(api.requests) == 1

    def test_clear_cache_clears_entry_cache(self, api):
        """clear_cache() empties both query and entry caches."""
        source = _source(api)
        source.search("attention")
        assert "1706.03762" in source._entry_cache
        assert len(source._cache) == 1

        source.clear_cache()

        assert source._entry_cache == {}
        assert source._cache == {}

    async def test_entry_cache_respects_ttl(self, api):
        """Entry cache entries expire after cache_ttl_seconds."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        source = ArxivSearchSource(cache_ttl_seconds=60, transport=api.transport)
        await source.asearch("attention")
        source._entry_cache["1706.03762"].timestamp -= 120  # past the TTL

        await source.fetch_by_id("1706.03762")

        assert len(api.requests) == 2  # expired: asked again


# ---------------------------------------------------------------------------
# ArxivSearchSource.download_pdf tests
# ---------------------------------------------------------------------------


class TestArxivDownloadPdf:
    """Direct unit tests for the encapsulated PDF downloader."""

    @pytest.mark.asyncio
    async def test_returns_pdf_bytes_on_success(self):
        """download_pdf returns the bytes fetched via the hardened ContentFetcher."""
        from agentic_cli.tools.arxiv_source import ArxivSearchSource
        from agentic_cli.tools.webfetch.fetcher import FetchResult

        source = ArxivSearchSource()
        fake_bytes = b"%PDF-1.4 fake content"

        fake_fetcher = MagicMock()
        fake_fetcher.fetch = AsyncMock(
            return_value=FetchResult(success=True, content=fake_bytes)
        )

        with patch(
            "agentic_cli.tools.webfetch_tool.get_or_create_fetcher",
            return_value=fake_fetcher,
        ), patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            result = await source.download_pdf("https://arxiv.org/pdf/1706.03762")

        assert result == fake_bytes
        fake_fetcher.fetch.assert_awaited_once()
        assert fake_fetcher.fetch.await_args.args[0] == "https://arxiv.org/pdf/1706.03762"

    @pytest.mark.asyncio
    async def test_raises_when_fetch_blocked_or_fails(self):
        """download_pdf raises RuntimeError when the hardened fetch is rejected.

        The SSRF guard / byte cap surfaces as FetchResult(success=False); the
        caller catches RuntimeError and falls back to the abstract.
        """
        from agentic_cli.tools.arxiv_source import ArxivSearchSource
        from agentic_cli.tools.webfetch.fetcher import FetchResult

        source = ArxivSearchSource()
        fake_fetcher = MagicMock()
        fake_fetcher.fetch = AsyncMock(
            return_value=FetchResult(success=False, error="boom")
        )

        with patch(
            "agentic_cli.tools.webfetch_tool.get_or_create_fetcher",
            return_value=fake_fetcher,
        ), patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="blocked or failed"):
                await source.download_pdf("https://arxiv.org/pdf/1706.03762")

    # That a download waits its turn with every other arXiv request:
    # tests/tools/test_arxiv_rate_limit.py::test_a_pdf_download_waits_its_turn_too


# ---------------------------------------------------------------------------
# ingest_arxiv_paper tool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ingest_ctx(tmp_path, arxiv_api):
    """Publish ArxivSearchSource (on ``arxiv_api``) and a mock-backed knowledge
    base into the registry."""
    from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager
    from agentic_cli.tools.arxiv_source import ArxivSearchSource
    from agentic_cli.workflow.service_registry import (
        ARXIV_SOURCE,
        KNOWLEDGE_BASE,
        set_service_registry,
    )

    source = ArxivSearchSource(transport=arxiv_api.transport)
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    token = set_service_registry({ARXIV_SOURCE: source, KNOWLEDGE_BASE: KnowledgeBase({"project": kb})})
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

        from tests.memory.pdf_bytes import pdf_with_text

        fake_pdf_bytes = pdf_with_text("Extracted PDF text content")

        from agentic_cli.tools.webfetch.fetcher import FetchResult

        fake_fetcher = MagicMock()
        fake_fetcher.fetch = AsyncMock(
            return_value=FetchResult(success=True, content=fake_pdf_bytes)
        )

        with patch(
            "agentic_cli.tools.webfetch_tool.get_or_create_fetcher",
            return_value=fake_fetcher,
        ), patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            result = await ingest_arxiv_paper("1706.03762", tags=["transformer"])

        assert result["success"] is True
        assert result["title"] == "Attention Is All You Need"
        assert result["chunks_created"] == 1
        assert result["pdf_downloaded"] is True

        # Verify what the knowledge base stored
        doc = kb.get_document(result["document_id"])
        assert doc.title == "Attention Is All You Need"
        assert doc.content == "Extracted PDF text content"
        assert kb.get_file_path(doc.id).read_bytes() == fake_pdf_bytes
        assert doc.source_url == "https://arxiv.org/abs/1706.03762"
        assert doc.metadata["arxiv_id"] == "1706.03762"
        assert doc.metadata["pdf_url"] == "https://arxiv.org/pdf/1706.03762"
        assert doc.metadata["src_url"] == "https://arxiv.org/e-print/1706.03762"
        assert doc.metadata["tags"] == ["transformer"]
        assert doc.metadata["authors"] == ["Vaswani"]

        # Verify the PDF was fetched from the cached pdf_url, not synthesized,
        # and that it went through the hardened fetcher.
        fake_fetcher.fetch.assert_awaited_once()
        assert fake_fetcher.fetch.await_args.args[0] == "https://arxiv.org/pdf/1706.03762"

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
        doc = kb.get_document(result["document_id"])
        assert doc.content == "This is the abstract used as fallback"
        assert doc.file_path is None
        assert "pdf_download_error" in doc.metadata

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
        assert kb.list_documents() == []

    @pytest.mark.asyncio
    async def test_the_summary_is_the_synthesis_other_documents_get(self, ingest_ctx):
        """One summarizer call, with the sidecar prompt (it was 2-3 sentences)."""
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        prompts = []

        class _Summarizer:
            async def summarize(self, content, prompt):
                prompts.append(prompt)
                return "SUMMARY:\nA synthesis of the paper.\n\nCLAIMS:\n- Attention suffices.\n"

        kb._summarizer = _Summarizer()
        source._store_entry("2301.07041", {
            "arxiv_id": "2301.07041", "title": "A paper", "authors": [],
            "abstract": "The abstract.", "abs_url": "https://arxiv.org/abs/2301.07041",
            "pdf_url": "", "src_url": "", "published_date": "", "updated_date": "",
            "categories": [], "primary_category": "",
        })

        result = await ingest_arxiv_paper("2301.07041")

        assert result["summary"] == "A synthesis of the paper."
        assert len(prompts) == 1 and prompts[0].startswith("Extract a structured payload")
        assert "- Attention suffices." in kb.sidecar_path(result["document_id"]).read_text()

    @pytest.mark.asyncio
    async def test_a_summarizer_answer_that_is_not_text_stores_the_preview(self, ingest_ctx):
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        class _Summarizer:
            async def summarize(self, content, prompt):
                return 42

        kb._summarizer = _Summarizer()
        source._store_entry("2301.07041", {
            "arxiv_id": "2301.07041", "title": "A paper", "authors": [],
            "abstract": "The abstract.", "abs_url": "https://arxiv.org/abs/2301.07041",
            "pdf_url": "", "src_url": "", "published_date": "", "updated_date": "",
            "categories": [], "primary_category": "",
        })

        result = await ingest_arxiv_paper("2301.07041")

        assert result["success"] is True
        assert result["summary"] == "The abstract."

    @pytest.mark.asyncio
    async def test_an_unexpected_storage_error_is_reported_not_raised(self, ingest_ctx):
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        def full(*args, **kwargs):
            raise OSError("disk full")

        kb.ingest_document = full
        source._store_entry("2301.07041", {
            "arxiv_id": "2301.07041", "title": "A paper", "authors": [],
            "abstract": "The abstract.", "abs_url": "https://arxiv.org/abs/2301.07041",
            "pdf_url": "", "src_url": "", "published_date": "", "updated_date": "",
            "categories": [], "primary_category": "",
        })

        result = await ingest_arxiv_paper("2301.07041")

        assert result == {"success": False, "error": "Ingestion failed: disk full"}

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


# ---------------------------------------------------------------------------
# ingest_arxiv_paper's PDF: its own limit, and why a download failed
# ---------------------------------------------------------------------------

PAPER_ID = "2301.07041"
PDF_PATH = f"/pdf/{PAPER_ID}"


def _store_paper(source):
    """``PAPER_ID``'s metadata in the source's cache, so no API request is made."""
    source._store_entry(PAPER_ID, {
        "arxiv_id": PAPER_ID, "title": "A paper", "authors": [],
        "abstract": "The abstract.", "abs_url": f"https://arxiv.org/abs/{PAPER_ID}",
        "pdf_url": f"https://arxiv.org{PDF_PATH}", "src_url": "", "published_date": "",
        "updated_date": "", "categories": [], "primary_category": "",
    })


def _pdf(text: str, size: int = 0) -> bytes:
    """A one-page PDF whose text is ``text``, padded with a comment to ``size`` bytes."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    if size:
        out += b"%" + b"x" * max(size - 600, 0) + b"\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


class _ArxivOrg:
    """arxiv.org, faked behind web_fetch's real fetcher: ``pdfs`` maps a path to
    its PDF, ``status`` a path to the status it answers instead; any other path
    is a 404. ``requests`` holds each path asked for."""

    def __init__(self) -> None:
        self.pdfs: dict[str, bytes] = {}
        self.status: dict[str, int] = {}
        self.requests: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        if path in self.status:
            return httpx.Response(self.status[path], text="refused")
        if path in self.pdfs:
            return httpx.Response(200, content=self.pdfs[path], headers={"content-type": "application/pdf"})
        return httpx.Response(404, text="not found")


@pytest.fixture
def arxiv_org(monkeypatch):
    """``get_or_create_fetcher`` builds its fetcher from the settings as it
    does in an app; only DNS and the network under the pinned transport are
    fakes."""
    import socket

    from agentic_cli.tools import webfetch_tool
    from agentic_cli.tools.webfetch.transport import PinnedTransport

    site = _ArxivOrg()

    def _gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("151.101.1.42", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _gai)
    monkeypatch.setattr(
        webfetch_tool, "PinnedTransport",
        lambda validator: PinnedTransport(validator, inner=httpx.MockTransport(site.handle)),
    )
    monkeypatch.setattr(webfetch_tool, "_fetcher", None)
    monkeypatch.setattr(webfetch_tool, "_fetcher_settings_snapshot", None)
    return site


class TestTheDownloadedPdf:
    """A paper's PDF is read up to ``arxiv_max_pdf_bytes`` (50 MB; web_fetch's
    5 MB limit was cutting papers with many figures, and the cut PDF was
    stored as the paper). One over the limit fails the call; a download that
    failed otherwise says why in the result, and a temporary refusal fails
    the call rather than storing the abstract alone."""

    def test_the_limit_setting(self):
        from pydantic import ValidationError

        from agentic_cli.config import BaseSettings
        from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS

        assert BaseSettings().arxiv_max_pdf_bytes == 52428800
        assert "arxiv_max_pdf_bytes" in PROJECT_SETTABLE_KEYS
        assert BaseSettings(arxiv_max_pdf_bytes=524288000).arxiv_max_pdf_bytes == 524288000
        with pytest.raises(ValidationError):
            BaseSettings(arxiv_max_pdf_bytes=0)
        with pytest.raises(ValidationError):
            BaseSettings(arxiv_max_pdf_bytes=524288000 + 1)

    @pytest.mark.asyncio
    async def test_a_pdf_over_web_fetch_s_limit_is_ingested_whole(self, ingest_ctx, arxiv_org):
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        _store_paper(source)
        pdf = _pdf("Attention replaces recurrence", size=6 * 1024 * 1024)
        arxiv_org.pdfs[PDF_PATH] = pdf

        result = await ingest_arxiv_paper(PAPER_ID)

        assert result["success"] is True
        assert result["pdf_downloaded"] is True
        doc = kb.get_document(result["document_id"])
        assert kb.get_file_path(doc.id).read_bytes() == pdf
        assert doc.content == "Attention replaces recurrence"

    @pytest.mark.asyncio
    async def test_a_pdf_over_the_limit_fails_the_call_naming_the_setting(self, ingest_ctx, arxiv_org):
        source, kb = ingest_ctx
        from agentic_cli.config import BaseSettings, SettingsContext
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        _store_paper(source)
        arxiv_org.pdfs[PDF_PATH] = _pdf("Attention replaces recurrence", size=3000)

        with SettingsContext(BaseSettings(arxiv_max_pdf_bytes=2000)):
            result = await ingest_arxiv_paper(PAPER_ID)

        assert result["success"] is False
        assert "arxiv_max_pdf_bytes" in result["error"]
        assert "2,000 bytes" in result["error"]
        assert kb.list_documents() == []

    @pytest.mark.asyncio
    async def test_a_missing_pdf_stores_the_abstract_and_says_why(self, ingest_ctx, arxiv_org):
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        _store_paper(source)  # arxiv.org answers 404 for its PDF

        result = await ingest_arxiv_paper(PAPER_ID)

        assert result["success"] is True
        assert result["pdf_downloaded"] is False
        assert "HTTP 404" in result["pdf_download_error"]
        assert kb.get_document(result["document_id"]).content == "The abstract."

    @pytest.mark.asyncio
    async def test_while_arxiv_refuses_requests_the_call_fails(self, ingest_ctx, arxiv_org):
        """The source fails a call at once for a while after arXiv's
        refusals: the paper is not stored as its abstract alone."""
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        _store_paper(source)
        arxiv_org.pdfs[PDF_PATH] = _pdf("Attention replaces recurrence")
        source._refuse(60, "ArXiv is rate limiting requests")

        result = await ingest_arxiv_paper(PAPER_ID)

        assert result["success"] is False
        assert "try again in" in result["error"]
        assert kb.list_documents() == []
        assert arxiv_org.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 503])
    async def test_a_pdf_download_refused_for_the_rate_fails_the_call(self, ingest_ctx, arxiv_org, status):
        source, kb = ingest_ctx
        from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper

        _store_paper(source)
        arxiv_org.status[PDF_PATH] = status

        result = await ingest_arxiv_paper(PAPER_ID)

        assert result["success"] is False
        assert f"HTTP {status}" in result["error"]
        assert kb.list_documents() == []

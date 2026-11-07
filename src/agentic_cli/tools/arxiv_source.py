"""ArXiv search source implementation.

Provides ArxivSearchSource with rate limiting and caching for use by
arxiv_tools and knowledge_tools. Relocated from knowledge_base/sources.py
because the KB manager never uses it directly.

arXiv asks for one request every three seconds, on one connection at a time,
and answers HTTP 429 (or 503) when a client goes faster. Every request the
source makes waits its turn behind the others; a refused one is retried
after the wait arXiv asks for (``Retry-After``), and the calls queued behind
it wait too.
"""

import asyncio
import email.utils
import io
import math
import re
import time
import weakref
from dataclasses import dataclass
from datetime import timezone
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from agentic_cli.tools.search_sources import SearchSource, SearchSourceResult
from agentic_cli.logging import Loggers

#: The arXiv API endpoint that search and metadata lookups query. The arXiv
#: tools declare it as their fixed ``http.read`` target.
ARXIV_API_URL = "https://export.arxiv.org/api/query"
#: Where paper PDFs are downloaded from; ``download_pdf`` refuses anything else,
#: so the tools' declared ``http.read`` target stays true.
ARXIV_PDF_URL = "https://arxiv.org/pdf"


def _is_arxiv_pdf_url(url: str) -> bool:
    """True only for ``https://arxiv.org/pdf/...`` with no credentials or port."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and parts.hostname == "arxiv.org"
        and port is None
        and parts.username is None
        and parts.password is None
        and parts.path.startswith("/pdf/")
    )

logger = Loggers.tools()

#: The statuses arXiv refuses a request with for going too fast; retried.
_RATE_LIMITED = (429, 503)
#: The waits before each retry of a refused request that names no wait.
_RETRY_DELAYS = (3.0, 6.0, 12.0)
#: The longest ``Retry-After`` a call waits out; a longer one fails it at once.
_MAX_RETRY_AFTER = 60.0
#: How long, after a call's retries ran out, a call fails without a request.
_REFUSED_FOR = 60.0
#: How long, after arXiv did not answer (or could not be reached), a call
#: fails without a request, so the calls queued behind do not each wait out
#: the timeout.
_UNANSWERED_FOR = 30.0
#: The longest a call is refused for, whatever ``Retry-After`` asks.
_MAX_REFUSAL = 3600.0
#: An API request's timeout, in seconds (httpx's: each of connecting, sending
#: and every read).
_TIMEOUT = 30.0
#: How the requests name their client, as arXiv asks an API client to.
_USER_AGENT = "agentic-cli (arXiv tools; +https://github.com/shoom1/agentic-cli)"


class ArxivRequestError(RuntimeError):
    """A request arXiv refused, failed or did not answer; the message says which."""


class PdfTooLargeError(RuntimeError):
    """A paper's PDF is larger than ``arxiv_max_pdf_bytes``; the message
    names the limit."""


def _retry_after(response: httpx.Response) -> float | None:
    """The wait a ``Retry-After`` header asks for, in seconds (it gives a
    number of seconds or an HTTP date, ``-0000`` read as UTC), or None
    without a readable, finite one."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = when.timestamp() - time.time()
    return max(0.0, seconds) if math.isfinite(seconds) else None

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_NEW_FORMAT_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
_OLD_FORMAT_ID_RE = re.compile(r"([a-zA-Z-]+/\d{7})")


def _clean_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arXiv id from any supported input form.

    Handles:
    - New plain ID:        '1706.03762'
    - With version:        '1706.03762v2'
    - Old format:          'math/0607733', 'hep-th/9901001v1'
    - Full abs URL:        'https://arxiv.org/abs/1706.03762'
    - Old abs URL:         'https://arxiv.org/abs/math/0607733'
    - PDF URL:             'https://arxiv.org/pdf/1706.03762.pdf'
    - Atom feed entry id:  'http://arxiv.org/abs/1706.03762v5'

    Both ``arxiv_tools._clean_arxiv_id`` (user-supplied strings) and
    ``ArxivSearchSource._extract_arxiv_id_from_entry`` (Atom feed
    entries) route through this single function, so the URL-parsing
    rules and version-strip behavior are guaranteed to match.

    Args:
        arxiv_id: The arXiv ID in any supported format. Empty input
            returns an empty string.

    Returns:
        Cleaned arXiv ID with no URL prefix and no version suffix
        (e.g. ``'1706.03762'`` or ``'math/0607733'``).
    """
    if "arxiv.org" in arxiv_id:
        match = _NEW_FORMAT_ID_RE.search(arxiv_id)
        if match:
            arxiv_id = match.group(1)
        else:
            match = _OLD_FORMAT_ID_RE.search(arxiv_id)
            if match:
                arxiv_id = match.group(1)

    return _VERSION_SUFFIX_RE.sub("", arxiv_id)


@dataclass
class CachedSearchResult:
    """Cached search result with timestamp."""

    results: list[SearchSourceResult]
    timestamp: float


@dataclass
class CachedEntry:
    """Cached single-paper entry with timestamp.

    Populated as a side effect of ``search()`` so a subsequent
    ``fetch_by_id`` for any returned paper hits this cache instead of
    re-querying the arxiv API.
    """

    paper: dict[str, Any]
    timestamp: float


class ArxivSearchSource(SearchSource):
    """ArXiv paper search source with rate limiting and caching.

    Its requests are paced together within one event loop: a manager has one
    source, and its tools run on one loop. Two sources, or a synchronous
    ``search()`` in another thread while a loop runs ``asearch``, pace
    their requests apart.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 900,
        max_cache_size: int = 100,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize with rate limiting and caching.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default: 900 = 15 minutes)
            max_cache_size: Maximum number of cached queries (default: 100)
            transport: The HTTP transport for API requests; httpx's own by
                default (tests pass a fake).
        """
        self._transport = transport
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._cache: dict[str, CachedSearchResult] = {}
        # Side index keyed on normalized (version-stripped) arxiv_id.
        # Populated as a side effect of search() so that fetch_by_id
        # returns cached results for any paper a prior search returned,
        # without a second round trip to the arxiv API.
        self._entry_cache: dict[str, CachedEntry] = {}
        self._last_error: str | None = None
        # The turn every request waits for, held for the whole request (and
        # any retry): one per event loop, as an asyncio.Lock belongs to one.
        self._gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        # When the next request may start (time.monotonic()), and until when
        # a call fails without one (arXiv's refusals ran out a call's retries,
        # or it did not answer), and why.
        self._next_start = 0.0
        self._refused_until = 0.0
        self._refused_why = ""

    @property
    def last_error(self) -> str | None:
        """The synchronous ``search()``'s last error, or None if it succeeded.
        ``asearch`` raises its error instead."""
        return self._last_error

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return "Search arXiv for academic papers"

    @property
    def rate_limit(self) -> float:
        return 3.0  # ArXiv requires 3 second delays

    def _make_cache_key(
        self,
        query: str,
        max_results: int,
        categories: list[str] | None,
        sort_by: str,
        sort_order: str,
        date_from: str | None,
        date_to: str | None,
    ) -> str:
        """Create a cache key from search parameters."""
        cat_str = ",".join(sorted(categories)) if categories else ""
        date_str = f"{date_from or ''}-{date_to or ''}"
        return f"{query}|{max_results}|{cat_str}|{sort_by}|{sort_order}|{date_str}"

    def _get_cached(self, cache_key: str) -> list[SearchSourceResult] | None:
        """Get cached results if available and not expired."""
        if cache_key not in self._cache:
            return None
        cached = self._cache[cache_key]
        if time.time() - cached.timestamp > self.cache_ttl_seconds:
            del self._cache[cache_key]
            return None
        return cached.results

    def _evict_oldest_if_full(self) -> None:
        """Evict oldest cache entry if cache is at max size."""
        if len(self._cache) >= self.max_cache_size:
            # Find oldest entry by timestamp
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

    def _get_cached_entry(self, arxiv_id: str) -> dict[str, Any] | None:
        """Return cached single-paper entry if present and fresh, else None."""
        cached = self._entry_cache.get(arxiv_id)
        if cached is None:
            return None
        if time.time() - cached.timestamp > self.cache_ttl_seconds:
            del self._entry_cache[arxiv_id]
            return None
        return cached.paper

    def _store_entry(self, arxiv_id: str, paper: dict[str, Any]) -> None:
        """Store a parsed paper in the id-indexed cache, evicting if full."""
        if len(self._entry_cache) >= self.max_cache_size and arxiv_id not in self._entry_cache:
            oldest_key = min(
                self._entry_cache.keys(),
                key=lambda k: self._entry_cache[k].timestamp,
            )
            del self._entry_cache[oldest_key]
        self._entry_cache[arxiv_id] = CachedEntry(paper=paper, timestamp=time.time())

    def clear_cache(self) -> None:
        """Clear all cached search results and id-indexed entries."""
        self._cache.clear()
        self._entry_cache.clear()

    def _gate(self) -> asyncio.Lock:
        """The running event loop's turn lock."""
        loop = asyncio.get_running_loop()
        gate = self._gates.get(loop)
        if gate is None:
            gate = self._gates[loop] = asyncio.Lock()
        return gate

    async def _take_turn(self) -> None:
        """Wait until the next request may start, and book the one after.
        Called holding the gate, so requests go one at a time."""
        remaining = self._refused_until - time.monotonic()
        if remaining > 0:
            raise ArxivRequestError(f"{self._refused_why}; try again in {math.ceil(remaining)} s")
        wait = self._next_start - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_start = time.monotonic() + self.rate_limit

    def _refuse(self, seconds: float, why: str) -> None:
        """Fail every call for ``seconds`` (an hour at most), saying ``why``."""
        self._refused_until = time.monotonic() + min(seconds, _MAX_REFUSAL)
        self._refused_why = why

    async def _get(self, url: str) -> httpx.Response:
        """GET ``url`` from arXiv's API, in turn: a request refused for the
        rate (HTTP 429 or 503) is retried after the wait arXiv asks for
        (``Retry-After``), else after 3, 6 and 12 s, all while holding the
        turn, so the calls queued behind it wait too. Once the retries run
        out, or arXiv did not answer, the calls queued behind fail at once.

        Raises:
            ArxivRequestError: arXiv refused the request (its retries spent,
                or it asks for a longer wait than a call gives), answered
                another error, failed or did not answer within 30 s.
        """
        async with self._gate():
            async with httpx.AsyncClient(
                transport=self._transport, timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
            ) as client:
                for delay in (*_RETRY_DELAYS, None):
                    await self._take_turn()
                    try:
                        response = await client.get(url)
                    except httpx.TimeoutException:
                        why = f"ArXiv did not answer within {_TIMEOUT:g} s"
                        self._refuse(_UNANSWERED_FOR, why)
                        raise ArxivRequestError(why) from None
                    except httpx.TransportError as exc:
                        self._refuse(_UNANSWERED_FOR, "ArXiv could not be reached")
                        raise ArxivRequestError(f"ArXiv request failed: {exc}") from None
                    except httpx.HTTPError as exc:
                        raise ArxivRequestError(f"ArXiv request failed: {exc}") from None
                    status = response.status_code
                    if status not in _RATE_LIMITED:
                        break
                    asked = _retry_after(response)
                    if asked is not None and asked > _MAX_RETRY_AFTER:
                        self._refuse(asked, "ArXiv is rate limiting requests")
                        raise ArxivRequestError(
                            f"ArXiv rate limited (HTTP {status}); it asks to wait {asked:.0f} s"
                        )
                    if delay is None:
                        self._refuse(_REFUSED_FOR, "ArXiv is rate limiting requests")
                        raise ArxivRequestError(
                            f"ArXiv rate limited (HTTP {status}), "
                            f"still after {len(_RETRY_DELAYS)} retries"
                        )
                    wait = delay if asked is None else asked
                    logger.info("arxiv_rate_limited", status=status, retry_in=wait)
                    self._next_start = max(self._next_start, time.monotonic() + wait)
        if status == 403:
            raise ArxivRequestError("ArXiv refused the request (HTTP 403)")
        if status != 200:
            raise ArxivRequestError(f"ArXiv answered HTTP {status}")
        return response

    async def _get_feed(self, url: str) -> Any:
        """``_get`` ``url``, and parse the reply, off the event loop (a long
        feed takes feedparser a while). Parsed from the bytes as a stream:
        feedparser opens a string naming a file, which a reply must not make
        it do."""
        import feedparser

        response = await self._get(url)
        return await asyncio.to_thread(
            feedparser.parse,
            io.BytesIO(response.content),
            response_headers={"content-type": response.headers.get("content-type", "")},
        )

    async def download_pdf(self, pdf_url: str, timeout: float = 60.0) -> bytes:
        """Download a paper PDF, in turn with every other arXiv request.

        Arxiv's crawl policy does not distinguish the API host from
        arxiv.org, so a download waits its turn as ``asearch`` and
        ``fetch_by_id`` do, and no other request runs meanwhile. The PDF is
        read up to ``arxiv_max_pdf_bytes`` (web_fetch's ``webfetch_max_pdf_bytes``
        is web_fetch's): a cut PDF is no paper, so one over it is an error.

        Args:
            pdf_url: The PDF URL to download (typically the ``pdf_url``
                field of a paper dict from ``fetch_by_id`` or ``search``).
            timeout: Request timeout in seconds.

        Returns:
            Raw PDF bytes.

        Raises:
            ArxivRequestError: A refusal that passes: the source fails calls
                for a while after arXiv's refusals (``_take_turn``), or
                arxiv.org answered HTTP 429 or 503.
            PdfTooLargeError: The PDF is larger than ``arxiv_max_pdf_bytes``.
            RuntimeError: If the fetch is blocked or fails otherwise. The
                ``pdf_url`` is taken from a remote Atom feed, so it is
                attacker-influenceable; routing the download through the shared
                ``ContentFetcher`` reuses the SSRF guard (private-IP/redirect
                validation), the byte cap, and DNS-rebinding checks rather than
                issuing a raw, unvalidated GET that follows redirects to
                arbitrary hosts.
        """
        from agentic_cli.config import get_settings
        from agentic_cli.tools.webfetch_tool import get_or_create_fetcher

        # The URL comes from the feed. ingest_arxiv_paper declares ARXIV_PDF_URL
        # as the location it reads, so anything else is refused, not fetched.
        if not _is_arxiv_pdf_url(pdf_url):
            raise RuntimeError(f"PDF URL must be under {ARXIV_PDF_URL}/: {pdf_url}")

        async def only_pdfs(next_url: str) -> str | None:
            # A redirect is followed only to another PDF URL: what the tool
            # declares it reads.
            return None if _is_arxiv_pdf_url(next_url) else f"outside {ARXIV_PDF_URL}/"

        limit = get_settings().arxiv_max_pdf_bytes
        fetcher = get_or_create_fetcher()
        async with self._gate():
            await self._take_turn()
            result = await fetcher.fetch(
                pdf_url, timeout=int(timeout), authorize=only_pdfs, max_pdf_bytes=limit,
            )
        if not result.success:
            if result.status_code in _RATE_LIMITED:
                raise ArxivRequestError(
                    f"ArXiv refused the PDF download (HTTP {result.status_code}); "
                    "try again later"
                )
            raise RuntimeError(f"PDF fetch blocked or failed: {result.error}")
        if result.raw_truncated:
            raise PdfTooLargeError(
                f"The PDF is larger than arxiv_max_pdf_bytes ({limit:,} bytes): "
                "the paper is not ingested. Raise the setting to ingest it."
            )

        content = result.content
        if content is None:
            raise RuntimeError("PDF fetch returned no content")
        return content if isinstance(content, bytes) else content.encode("utf-8", "replace")

    _API_BASE_URL = ARXIV_API_URL

    @staticmethod
    def _extract_arxiv_id_from_entry(entry: Any) -> str:
        """Pull a version-stripped arxiv id out of a feedparser entry.

        The Atom ``<id>`` element is a URL like
        ``http://arxiv.org/abs/1706.03762v5`` (or ``.../abs/math/0607733v2``
        for old-style ids). Delegating to ``_clean_arxiv_id`` keeps the
        URL-parsing rules consistent with the user-supplied id path.
        """
        return _clean_arxiv_id(entry.get("id", ""))

    @staticmethod
    def _extract_links(entry: Any, arxiv_id: str) -> dict[str, str]:
        """Extract abs/pdf/src URLs for an entry.

        The arxiv Atom feed emits multiple ``<link>`` elements per entry —
        feedparser exposes them as ``entry.links``, a list of dicts with
        ``rel``, ``type``, ``href``. We pick:

        - ``pdf_url`` from the link whose ``type`` is ``application/pdf``
        - ``abs_url`` from the ``rel="alternate"`` or ``type="text/html"``
          link, falling back to ``entry.link`` (feedparser's single-link
          shortcut) if no multi-link data is present
        - ``src_url`` is always synthesized from the id because arxiv's
          feed does not advertise e-print links, though the URL is stable

        All URLs use HTTPS. Falls back to id-synthesized URLs when the
        feed unexpectedly omits a link.
        """
        abs_url = ""
        pdf_url = ""
        for link in entry.get("links", []) or []:
            href = link.get("href", "") if isinstance(link, dict) else getattr(link, "href", "")
            ltype = link.get("type", "") if isinstance(link, dict) else getattr(link, "type", "")
            rel = link.get("rel", "") if isinstance(link, dict) else getattr(link, "rel", "")
            if ltype == "application/pdf" and not pdf_url:
                pdf_url = href
            elif (rel == "alternate" or ltype == "text/html") and not abs_url:
                abs_url = href

        if not abs_url:
            abs_url = entry.get("link", "") or f"https://arxiv.org/abs/{arxiv_id}"
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # Normalize http → https for consistency; arxiv serves both.
        if abs_url.startswith("http://arxiv.org"):
            abs_url = "https://" + abs_url[len("http://"):]
        if pdf_url.startswith("http://arxiv.org"):
            pdf_url = "https://" + pdf_url[len("http://"):]

        return {
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "src_url": f"https://arxiv.org/e-print/{arxiv_id}",
        }

    def _parse_entry(self, entry: Any, arxiv_id: str | None = None) -> dict[str, Any]:
        """Parse a feedparser entry into a complete paper dict.

        Shared by ``search()`` and ``fetch_by_id()`` so both code paths
        produce the same field set. If ``arxiv_id`` is not provided it
        is extracted from the entry (search path); callers that already
        have a normalized id (fetch_by_id path) pass it in.
        """
        if arxiv_id is None:
            arxiv_id = self._extract_arxiv_id_from_entry(entry)

        links = self._extract_links(entry, arxiv_id)

        return {
            "arxiv_id": arxiv_id,
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "authors": [author.get("name", "") for author in entry.get("authors", [])],
            "abstract": entry.get("summary", "").replace("\n", " ").strip(),
            "abs_url": links["abs_url"],
            "pdf_url": links["pdf_url"],
            "src_url": links["src_url"],
            "published_date": entry.get("published", ""),
            "updated_date": entry.get("updated", ""),
            "categories": [tag.get("term", "") for tag in entry.get("tags", [])],
            "primary_category": entry.get("arxiv_primary_category", {}).get("term", ""),
        }

    async def fetch_by_id(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch a single paper by arXiv ID.

        Async and rate-limited (``_get``). Returns a raw-paper dict (the same shape ``_parse_entry`` produces
        for search results), or raises ``LookupError`` / ``RuntimeError``
        for caller-handled error cases.

        The id is normalized (version suffix stripped) before lookup so
        ``fetch_by_id("1706.03762v3")`` hits a cache populated by a prior
        search returning ``1706.03762``. On cache hit, no API call and
        no rate-limit wait. On miss, the result is cached for next time.

        Raises:
            LookupError: arXiv has no paper with this id.
            ArxivRequestError: The request failed (a ``RuntimeError``).
        """
        normalized_id = _VERSION_SUFFIX_RE.sub("", arxiv_id)

        cached = self._get_cached_entry(normalized_id)
        if cached is not None:
            return cached

        try:
            import feedparser  # noqa: F401 - missing: fail before the request
        except ImportError as exc:
            raise RuntimeError("feedparser not installed") from exc

        feed = await self._get_feed(f"{self._API_BASE_URL}?id_list={normalized_id}")

        if not feed.entries:
            raise LookupError(f"Paper with ID '{normalized_id}' not found")

        paper = self._parse_entry(feed.entries[0], arxiv_id=normalized_id)
        self._store_entry(normalized_id, paper)
        return paper

    async def asearch(
        self,
        query: str,
        max_results: int = 10,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchSourceResult]:
        """Search ArXiv, in turn with every other arXiv request (``_get``).

        Args:
            query: Search query
            max_results: Maximum results
            categories: Optional ArXiv category filters (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort results by 'relevance', 'lastUpdatedDate', or 'submittedDate'
            sort_order: Sort order 'ascending' or 'descending'
            date_from: Filter papers submitted after this date (YYYY-MM-DD)
            date_to: Filter papers submitted before this date (YYYY-MM-DD)

        Raises:
            ArxivRequestError: The request failed, or arXiv's feed did not parse.
        """
        # Check cache first
        cache_key = self._make_cache_key(
            query, max_results, categories, sort_by, sort_order, date_from, date_to
        )
        cached_results = self._get_cached(cache_key)
        if cached_results is not None:
            return cached_results

        try:
            import feedparser  # noqa: F401 - missing: fail before the request
        except ImportError:
            raise ArxivRequestError("feedparser not installed") from None

        # Build search query
        query_parts = [f"all:{query}"]

        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            query_parts.append(f"({cat_query})")

        # Add date range filter if specified
        if date_from or date_to:
            from_date = date_from.replace("-", "") if date_from else "*"
            to_date = date_to.replace("-", "") if date_to else "*"
            query_parts.append(f"submittedDate:[{from_date} TO {to_date}]")

        raw_query = " AND ".join(f"({p})" if " " in p else p for p in query_parts)
        search_query = "search_query=" + quote(raw_query, safe="")

        # Build URL with sort parameters
        url = f"{ARXIV_API_URL}?{search_query}&start=0&max_results={max_results}"
        url += f"&sortBy={sort_by}&sortOrder={sort_order}"

        feed = await self._get_feed(url)
        if feed.bozo and not feed.entries:
            raise ArxivRequestError(f"ArXiv feed error: {feed.bozo_exception}")

        results = []
        for entry in feed.entries:
            paper = self._parse_entry(entry)
            # Populate id-indexed cache as a side effect so a subsequent
            # fetch_by_id for any of these papers is free.
            if paper["arxiv_id"]:
                self._store_entry(paper["arxiv_id"], paper)
            results.append(
                SearchSourceResult(
                    title=paper["title"],
                    url=paper["abs_url"],
                    snippet=paper["abstract"][:500],
                    source_name=self.name,
                    metadata={
                        "authors": paper["authors"],
                        "published": paper["published_date"],
                        "categories": paper["categories"],
                        "arxiv_id": paper["arxiv_id"],
                        "abs_url": paper["abs_url"],
                        "pdf_url": paper["pdf_url"],
                        "src_url": paper["src_url"],
                    },
                )
            )

        self._evict_oldest_if_full()
        self._cache[cache_key] = CachedSearchResult(results=results, timestamp=time.time())
        return results

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        date_from: str | None = None,
        date_to: str | None = None,
        **kwargs,
    ) -> list[SearchSourceResult]:
        """Search ArXiv synchronously (``SearchSource``): ``asearch``, run to
        its end. On a failure, ``[]``, and ``last_error`` names it.

        Not from a running event loop, which it would block: await
        ``asearch`` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "ArxivSearchSource.search() would block the running event loop; "
                "await asearch() instead"
            )
        try:
            results = asyncio.run(
                self.asearch(
                    query,
                    max_results=max_results,
                    categories=categories,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    date_from=date_from,
                    date_to=date_to,
                )
            )
        except ArxivRequestError as exc:
            self._last_error = str(exc)
            return []
        self._last_error = None
        return results

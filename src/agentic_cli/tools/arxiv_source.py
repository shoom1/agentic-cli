"""ArXiv search source implementation.

Provides ArxivSearchSource with rate limiting and caching for use by
arxiv_tools and knowledge_tools. Relocated from knowledge_base/sources.py
because the KB manager never uses it directly.
"""

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from agentic_cli.knowledge_base.sources import SearchSource, SearchSourceResult
from agentic_cli.logging import Loggers

logger = Loggers.knowledge_base()

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
    """ArXiv paper search source with rate limiting and caching."""

    def __init__(
        self,
        cache_ttl_seconds: int = 900,
        max_cache_size: int = 100,
    ) -> None:
        """Initialize with rate limiting and caching.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default: 900 = 15 minutes)
            max_cache_size: Maximum number of cached queries (default: 100)
        """
        self._last_request_time: float = 0.0
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._cache: dict[str, CachedSearchResult] = {}
        # Side index keyed on normalized (version-stripped) arxiv_id.
        # Populated as a side effect of search() so that fetch_by_id
        # returns cached results for any paper a prior search returned,
        # without a second round trip to the arxiv API.
        self._entry_cache: dict[str, CachedEntry] = {}
        self._last_error: str | None = None
        # Serialize access to _last_request_time across threads and async tasks.
        # The sync lock guards the timestamp for threaded callers; the async
        # lock does the same for async callers and also spans the full
        # request window so another task can't fire while we're mid-fetch.
        # The async lock is created lazily because a running event loop
        # is not guaranteed at construction time.
        self._rate_lock_sync = threading.Lock()
        self._rate_lock_async: asyncio.Lock | None = None

    @property
    def last_error(self) -> str | None:
        """Last error message from search(), or None if last search succeeded."""
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

    def wait_for_rate_limit(self) -> None:
        """Block until the rate limit allows another request.

        Thread-safe via ``_rate_lock_sync``. Do not call from async code —
        use ``_wait_for_rate_limit_async`` instead so the event loop is
        not blocked.
        """
        with self._rate_lock_sync:
            current_time = time.time()
            elapsed = current_time - self._last_request_time
            if self._last_request_time > 0 and elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
            self._last_request_time = time.time()

    async def _wait_for_rate_limit_async(self) -> None:
        """Async variant of ``wait_for_rate_limit``.

        Uses ``asyncio.Lock`` and ``asyncio.sleep`` so the event loop stays
        free and concurrent tasks actually serialize on the lock. The lock
        is created on first use so the source can be constructed outside a
        running event loop.
        """
        if self._rate_lock_async is None:
            self._rate_lock_async = asyncio.Lock()

        async with self._rate_lock_async:
            current_time = time.time()
            elapsed = current_time - self._last_request_time
            if self._last_request_time > 0 and elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            self._last_request_time = time.time()

    async def download_pdf(self, pdf_url: str, timeout: float = 60.0) -> bytes:
        """Download a paper PDF, respecting the shared rate limit.

        Encapsulates the "wait, then GET" pattern so callers don't need
        to reach into the rate-limit primitive. Arxiv's crawl policy
        does not distinguish the API host from arxiv.org, so PDF
        downloads must share the same per-source rate limiter as
        ``search`` and ``fetch_by_id``.

        Args:
            pdf_url: The PDF URL to download (typically the ``pdf_url``
                field of a paper dict from ``fetch_by_id`` or ``search``).
            timeout: Request timeout in seconds.

        Returns:
            Raw PDF bytes.

        Raises:
            RuntimeError: If httpx is not installed.
            httpx.HTTPError: For network failures or non-2xx responses
                (callers typically catch broadly and fall back).
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx not installed, cannot download PDF") from exc

        await self._wait_for_rate_limit_async()

        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            return response.content

    _API_BASE_URL = "http://export.arxiv.org/api/query"

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

        Async, rate-limited, and runs feedparser.parse off the event loop.
        Returns a raw-paper dict (the same shape ``_parse_entry`` produces
        for search results), or raises ``LookupError`` / ``RuntimeError``
        for caller-handled error cases.

        The id is normalized (version suffix stripped) before lookup so
        ``fetch_by_id("1706.03762v3")`` hits a cache populated by a prior
        search returning ``1706.03762``. On cache hit, no API call and
        no rate-limit wait. On miss, the result is cached for next time.
        """
        normalized_id = _VERSION_SUFFIX_RE.sub("", arxiv_id)

        cached = self._get_cached_entry(normalized_id)
        if cached is not None:
            return cached

        try:
            import feedparser
        except ImportError as exc:
            raise RuntimeError("feedparser not installed") from exc

        await self._wait_for_rate_limit_async()

        url = f"{self._API_BASE_URL}?id_list={normalized_id}"

        try:
            feed = await asyncio.to_thread(feedparser.parse, url)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch paper: {exc}") from exc

        if not feed.entries:
            raise LookupError(f"Paper with ID '{normalized_id}' not found")

        paper = self._parse_entry(feed.entries[0], arxiv_id=normalized_id)
        self._store_entry(normalized_id, paper)
        return paper

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
        """Search ArXiv.

        Args:
            query: Search query
            max_results: Maximum results
            categories: Optional ArXiv category filters (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort results by 'relevance', 'lastUpdatedDate', or 'submittedDate'
            sort_order: Sort order 'ascending' or 'descending'
            date_from: Filter papers submitted after this date (YYYY-MM-DD)
            date_to: Filter papers submitted before this date (YYYY-MM-DD)
        """
        # Check cache first
        cache_key = self._make_cache_key(
            query, max_results, categories, sort_by, sort_order, date_from, date_to
        )
        cached_results = self._get_cached(cache_key)
        if cached_results is not None:
            return cached_results

        try:
            import feedparser
        except ImportError:
            self._last_error = "feedparser not installed"
            return []

        # Enforce rate limiting
        self.wait_for_rate_limit()

        base_url = "http://export.arxiv.org/api/query?"

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

        from urllib.parse import quote

        raw_query = " AND ".join(f"({p})" if " " in p else p for p in query_parts)
        search_query = "search_query=" + quote(raw_query, safe="")

        # Build URL with sort parameters
        url = f"{base_url}{search_query}&start=0&max_results={max_results}"
        url += f"&sortBy={sort_by}&sortOrder={sort_order}"

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            self._last_error = f"Failed to parse ArXiv feed: {e}"
            return []

        # Detect ArXiv rate limiting / HTTP errors
        status = getattr(feed, "status", 200)
        if status in (403, 429):
            self._last_error = f"ArXiv rate limited (HTTP {status})"
            logger.warning("arxiv_rate_limited", status=status)
            return []
        if feed.bozo and not feed.entries:
            self._last_error = f"ArXiv feed error: {feed.bozo_exception}"
            logger.warning("arxiv_feed_error", error=str(feed.bozo_exception))
            return []

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

        # Clear error on successful parse
        self._last_error = None

        # Only cache non-error results (avoid caching empty results from rate limiting)
        if results or (status == 200 and not feed.bozo):
            self._evict_oldest_if_full()
            self._cache[cache_key] = CachedSearchResult(
                results=results,
                timestamp=time.time(),
            )

        return results

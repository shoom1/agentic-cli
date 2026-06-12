"""Security tests for ArxivSearchSource.download_pdf.

The pdf_url comes from a remote Atom feed and is therefore attacker-
influenceable. download_pdf must route through the hardened ContentFetcher so
it inherits SSRF validation (private-IP / redirect blocking) and the byte cap,
rather than issuing a raw GET that follows redirects to arbitrary hosts.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.tools.arxiv_source import ArxivSearchSource
from agentic_cli.tools.webfetch.fetcher import FetchResult


def test_api_endpoints_use_https():
    """The feed endpoint must be HTTPS so an on-path attacker can't rewrite it."""
    assert ArxivSearchSource._API_BASE_URL.startswith("https://")


async def test_download_pdf_routes_through_fetcher():
    """download_pdf returns the fetcher's bytes and never calls httpx directly."""
    fake_fetcher = MagicMock()
    fake_fetcher.fetch = AsyncMock(
        return_value=FetchResult(success=True, content=b"%PDF-1.7 data")
    )
    source = ArxivSearchSource()

    with patch(
        "agentic_cli.tools.webfetch_tool.get_or_create_fetcher",
        return_value=fake_fetcher,
    ):
        out = await source.download_pdf("https://arxiv.org/pdf/1706.03762")

    assert out == b"%PDF-1.7 data"
    fake_fetcher.fetch.assert_awaited_once()
    assert fake_fetcher.fetch.await_args.args[0] == "https://arxiv.org/pdf/1706.03762"


async def test_download_pdf_raises_when_fetch_blocked():
    """A blocked/failed fetch surfaces as RuntimeError (caller falls back)."""
    fake_fetcher = MagicMock()
    fake_fetcher.fetch = AsyncMock(
        return_value=FetchResult(success=False, error="Blocked: private IP")
    )
    source = ArxivSearchSource()

    with patch(
        "agentic_cli.tools.webfetch_tool.get_or_create_fetcher",
        return_value=fake_fetcher,
    ):
        with pytest.raises(RuntimeError, match="blocked or failed"):
            await source.download_pdf("https://arxiv.org/pdf/x")


async def test_download_pdf_blocks_internal_ip_end_to_end(mock_context):
    """End-to-end: a metadata-IP pdf_url is rejected by the real SSRF guard.

    No network is touched — URLValidator rejects 169.254.0.0/16 before any GET.
    """
    source = ArxivSearchSource()
    with pytest.raises(RuntimeError, match="blocked or failed"):
        await source.download_pdf("http://169.254.169.254/latest/meta-data/")

"""Declarations of the network tools, and the grant bleed they caused.

``web_search`` and the arXiv tools declared ``http.read`` with no target, so an
"Allow always" on any of them was stored as ``http.read *`` and silently covered
``web_fetch`` and ``kb_ingest_url`` for every URL.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import agentic_cli.tools  # noqa: F401  (registers the built-in tools)
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.prompt import ALLOW_ALWAYS_CHOICE
from agentic_cli.workflow.permissions.store import PermissionContext

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf"


def caps(tool: str) -> list[Capability]:
    return get_registry().get(tool).capabilities


class TestDeclarations:
    def test_web_search_has_its_own_capability(self):
        assert caps("web_search") == [Capability("search.web")]

    def test_search_arxiv_names_the_arxiv_api(self):
        assert caps("search_arxiv") == [Capability("http.read", target=ARXIV_API)]

    def test_fetch_arxiv_paper_names_the_arxiv_api(self):
        assert caps("fetch_arxiv_paper") == [Capability("http.read", target=ARXIV_API)]

    def test_ingest_arxiv_paper_names_both_arxiv_endpoints(self):
        assert caps("ingest_arxiv_paper") == [
            Capability("http.read", target=ARXIV_API),
            Capability("http.read", target=ARXIV_PDF),
            Capability("kb.write"),
        ]

    def test_sandbox_inputs_are_optional(self):
        import agentic_cli.tools.sandbox  # noqa: F401
        assert Capability("filesystem.read", target_arg="inputs", optional=True) in caps(
            "sandbox_execute"
        )


@pytest.fixture
def engine_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    (root / "home").mkdir()
    (root / "work").mkdir()
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.chdir(root / "work")

    def make(answer: str) -> PermissionEngine:
        s = MagicMock()
        s.permissions_enabled = True
        s.app_name = "bleedtest"
        w = MagicMock()
        w.request_user_input = AsyncMock(return_value=answer)
        ctx = PermissionContext(workdir=root / "work", home=root / "home", app_name="bleedtest")
        return PermissionEngine(settings=s, workflow=w, ctx=ctx)

    return make


@pytest.mark.parametrize(
    "tool, args",
    [
        ("web_search", {"query": "q"}),
        ("search_arxiv", {"query": "q"}),
        ("fetch_arxiv_paper", {"arxiv_id": "1706.03762"}),
        ("ingest_arxiv_paper", {"arxiv_id": "1706.03762"}),
    ],
)
def test_allow_always_on_a_search_tool_does_not_cover_web_fetch(engine_factory, tool, args):
    engine = engine_factory(ALLOW_ALWAYS_CHOICE)
    first = asyncio.run(engine.check(tool, caps(tool), args))
    assert first.allowed is True

    # A fresh engine reloads the saved grants, exactly as the next run would.
    fresh = engine_factory("Deny")
    fetch = asyncio.run(
        fresh.check("web_fetch", caps("web_fetch"), {"url": "https://example.com/page"})
    )
    assert fetch.allowed is False
    fresh._workflow.request_user_input.assert_awaited_once()


class TestArxivPdfDownloadIsPinned:
    """ingest_arxiv_paper declares https://arxiv.org/pdf, and the PDF URL comes
    from the remote feed, so the download must refuse anything else."""

    def _source(self, monkeypatch):
        from agentic_cli.tools import arxiv_source
        from agentic_cli.tools import webfetch_tool

        fetcher = MagicMock()
        result = MagicMock(success=True, content=b"%PDF-1.4", error=None)
        fetcher.fetch = AsyncMock(return_value=result)
        monkeypatch.setattr(webfetch_tool, "get_or_create_fetcher", lambda: fetcher)
        src = arxiv_source.ArxivSearchSource()
        monkeypatch.setattr(src, "_wait_for_rate_limit_async", AsyncMock())
        return src, fetcher

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/pdf/1706.03762",
            "http://arxiv.org/pdf/1706.03762",
            "https://arxiv.org.example.com/pdf/1706.03762",
            "https://arxiv.org/abs/1706.03762",
            "https://user@arxiv.org:8443/pdf/1706.03762",
        ],
    )
    def test_other_urls_are_refused_before_fetching(self, monkeypatch, url):
        src, fetcher = self._source(monkeypatch)
        with pytest.raises(RuntimeError, match="arxiv.org/pdf"):
            asyncio.run(src.download_pdf(url))
        fetcher.fetch.assert_not_awaited()

    def test_an_arxiv_pdf_url_is_fetched(self, monkeypatch):
        src, fetcher = self._source(monkeypatch)
        data = asyncio.run(src.download_pdf("https://arxiv.org/pdf/1706.03762v5"))
        assert data == b"%PDF-1.4"
        fetcher.fetch.assert_awaited_once()

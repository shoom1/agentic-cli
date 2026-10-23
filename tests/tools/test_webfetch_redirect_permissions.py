"""A redirect is fetched only where a direct fetch of its target would be.

The permission engine judged only the URL a ``web_fetch`` call named. The
fetcher then followed redirects on the same host after the SSRF and robots
checks alone, so with ``/public/**`` allowed and ``/private/**`` denied,
fetching a ``/public`` page that redirects to ``/private`` returned the
private page. "Allow once", which approves one URL, and an HTTPS grant, which
does not cover ``http://`` on the same host, were crossed the same way.

Each redirect target is now put to the engine before it is requested, as a
``web_fetch`` of that URL would be: denied, the fetch fails naming it; allowed,
it is followed; otherwise the user is asked. A page served from the cache is
checked again for the URL it was finally read from. The arXiv PDF download,
which declares ``https://arxiv.org/pdf`` as what it reads, follows a redirect
only to another PDF URL there.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import agentic_cli.tools.webfetch_tool  # noqa: F401  (registers web_fetch)
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE, DENY_CHOICE
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.service_registry import set_service_registry
from tests.test_webfetch import _pinned_fetcher

SITE = "https://docs.example.org"


class _Site:
    """A site whose ``/public/page`` redirects to ``target``; records the paths
    requested (the pinned transport connects to the address, not the name)."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.requested: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(request.url.path)
        if request.url.path == "/public/page":
            return httpx.Response(302, headers={"location": self.target})
        return httpx.Response(200, text="the page behind the redirect",
                              headers={"content-type": "text/plain"})


class _Approver:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def request_user_input(self, request) -> str:
        self.prompts.append(request.prompt)
        return self.answer


class _Summarizer:
    async def summarize(self, content: str, prompt: str) -> str:
        return f"summary of {content}"


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    return tmp_path / "project"


def _engine(approver: _Approver, *rules: tuple[str, Effect]) -> PermissionEngine:
    settings = BaseSettings(workspace_dir=Path.cwd() / "workspace")
    ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name)
    engine = PermissionEngine(settings=settings, workflow=approver, ctx=ctx)
    for target, effect in rules:
        engine._session_rules.append(Rule("http.read", target, effect, RuleSource.SESSION))
    return engine


async def _web_fetch(monkeypatch, site: _Site, engine: PermissionEngine | None) -> dict:
    from agentic_cli.tools.webfetch_tool import web_fetch

    fetcher = _pinned_fetcher(monkeypatch, site)
    services = {"llm_summarizer": _Summarizer()}
    if engine is not None:
        services["permission_engine"] = engine
    token = set_service_registry(services)
    try:
        with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=fetcher), \
                patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            return await web_fetch(url=f"{SITE}/public/page", prompt="summarize")
    finally:
        token.var.reset(token)


class TestWebFetch:
    async def test_a_redirect_to_a_denied_path_is_never_requested(self, monkeypatch):
        site = _Site("/private/page")
        approver = _Approver(ALLOW_ONCE_CHOICE)
        engine = _engine(approver, (f"{SITE}/public/**", Effect.ALLOW),
                         (f"{SITE}/private/**", Effect.DENY))

        result = await _web_fetch(monkeypatch, site, engine)

        assert result["success"] is False
        assert f"{SITE}/private/page" in result["error"]
        assert "not permitted" in result["error"]
        assert site.requested == ["/public/page"]
        assert approver.prompts == []

    async def test_a_redirect_to_an_allowed_path_is_followed(self, monkeypatch):
        site = _Site("/public/moved")
        approver = _Approver(DENY_CHOICE)
        engine = _engine(approver, (f"{SITE}/public/**", Effect.ALLOW))

        result = await _web_fetch(monkeypatch, site, engine)

        assert result["success"] is True
        assert site.requested == ["/public/page", "/public/moved"]
        assert approver.prompts == []

    @pytest.mark.parametrize("target", ["/elsewhere", "http://docs.example.org/public/page2"])
    async def test_a_redirect_no_rule_covers_is_asked_about(self, monkeypatch, target):
        # "/elsewhere" is outside the allowed path; plain HTTP on the same host
        # is outside an HTTPS rule.
        site = _Site(target)
        approver = _Approver(DENY_CHOICE)
        engine = _engine(approver, (f"{SITE}/public/**", Effect.ALLOW))

        result = await _web_fetch(monkeypatch, site, engine)

        assert result["success"] is False
        assert len(approver.prompts) == 1
        assert site.requested == ["/public/page"]

    async def test_an_approved_redirect_is_followed(self, monkeypatch):
        site = _Site("/elsewhere")
        approver = _Approver(ALLOW_ONCE_CHOICE)
        engine = _engine(approver, (f"{SITE}/public/**", Effect.ALLOW))

        result = await _web_fetch(monkeypatch, site, engine)

        assert result["success"] is True
        assert len(approver.prompts) == 1
        assert site.requested[-1] == "/elsewhere"

    async def test_outside_a_workflow_redirects_are_followed_as_before(self, monkeypatch):
        site = _Site("/private/page")

        result = await _web_fetch(monkeypatch, site, None)

        assert result["success"] is True
        assert site.requested[-1] == "/private/page"


class TestContentFetcher:
    async def test_each_hop_is_checked_before_it_is_requested(self, monkeypatch):
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(request.url.path)
            nxt = {"/a": "/b", "/b": "/c"}.get(request.url.path)
            if nxt:
                return httpx.Response(302, headers={"location": nxt})
            return httpx.Response(200, text="c", headers={"content-type": "text/plain"})

        checked: list[str] = []

        async def authorize(url: str) -> str | None:
            checked.append(url)
            assert not url.endswith(requested[-1])  # asked before it is requested
            return None

        fetcher = _pinned_fetcher(monkeypatch, handler)
        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            result = await fetcher.fetch(f"{SITE}/a", authorize=authorize)

        assert result.success is True
        assert checked == [f"{SITE}/b", f"{SITE}/c"]

    async def test_a_cached_page_reached_by_a_redirect_is_checked_again(self, monkeypatch):
        site = _Site("/private/page")
        fetcher = _pinned_fetcher(monkeypatch, site)

        async def allow(url: str) -> str | None:
            return None

        async def refuse(url: str) -> str | None:
            return "denied by rule"

        with patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True):
            first = await fetcher.fetch(f"{SITE}/public/page", authorize=allow)
            second = await fetcher.fetch(f"{SITE}/public/page", authorize=refuse)

        assert first.success is True
        assert second.success is False
        assert f"{SITE}/private/page" in second.error
        assert len(site.requested) == 2  # the second answer came from the cache


class TestArxivDownload:
    @pytest.mark.parametrize("target, followed", [
        ("/pdf/1706.03762v7", True),
        ("/abs/1706.03762", False),
    ])
    async def test_a_redirect_is_followed_only_to_another_pdf(self, monkeypatch, target, followed):
        from agentic_cli.tools.arxiv_source import ArxivSearchSource

        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(request.url.path)
            if request.url.path == "/pdf/1706.03762":
                return httpx.Response(302, headers={"location": target})
            return httpx.Response(200, content=b"%PDF-1.4 x",
                                  headers={"content-type": "application/pdf"})

        fetcher = _pinned_fetcher(monkeypatch, handler)
        source = ArxivSearchSource()
        with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=fetcher), \
                patch.object(fetcher._robots, "can_fetch", new_callable=AsyncMock, return_value=True), \
                patch("agentic_cli.tools.arxiv_source.asyncio.sleep", new=AsyncMock()):
            if followed:
                assert await source.download_pdf("https://arxiv.org/pdf/1706.03762") == b"%PDF-1.4 x"
            else:
                with pytest.raises(RuntimeError, match="arxiv.org/pdf"):
                    await source.download_pdf("https://arxiv.org/pdf/1706.03762")

        assert (target in requested) is followed

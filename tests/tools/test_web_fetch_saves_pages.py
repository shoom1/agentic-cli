"""``web_fetch`` saves every page it reads, so the page can be ingested later.

It returned only the summary, and the page stayed in an in-memory cache for 15
minutes. It now saves the body as received in ``./.<app>/fetched/`` and
returns ``saved_path``. A failed save never fails the fetch.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentic_cli.config import get_settings
from agentic_cli.tools.webfetch.fetcher import FetchResult
from agentic_cli.tools.webfetch.saved import page_name, saved_pages_dir
from agentic_cli.workflow.service_registry import LLM_SUMMARIZER, set_service_registry

URL = "https://example.com/guide"
PAGE = b"<html><body><p>The guide.</p></body></html>"


class _Summarizer:
    def __init__(self):
        self.contents: list[str] = []

    async def summarize(self, content: str, prompt: str) -> str:
        self.contents.append(content)
        return "summary"


class _Fetcher:
    def __init__(self, result):
        self.result = result

    async def fetch(self, url, timeout=30):
        return self.result


def _ok(raw=PAGE, content_type="text/html; charset=utf-8", from_cache=False):
    content = raw if "pdf" in content_type else raw.decode("utf-8", "replace")
    return FetchResult(
        success=True, content=content, content_type=content_type, raw=raw,
        charset="utf-8", final_url=URL, from_cache=from_cache,
    )


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def summarizer():
    s = _Summarizer()
    token = set_service_registry({LLM_SUMMARIZER: s})
    try:
        yield s
    finally:
        token.var.reset(token)


def _folder() -> Path:
    return saved_pages_dir(get_settings().app_name)


async def _web_fetch(result):
    from agentic_cli.tools.webfetch_tool import web_fetch

    with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=_Fetcher(result)):
        return await web_fetch(URL, "summarize")


async def test_a_fetched_page_is_saved(summarizer):
    result = await _web_fetch(_ok())

    assert result["success"] is True
    saved = Path(result["saved_path"])
    assert not saved.is_absolute()
    assert saved.resolve() == _folder() / f"{page_name(URL)}.html"
    assert saved.read_bytes() == PAGE
    assert json.loads((_folder() / f"{page_name(URL)}.meta.json").read_text())["url"] == URL
    assert "save_error" not in result


async def test_the_summary_is_unchanged(summarizer):
    result = await _web_fetch(_ok())

    assert result["summary"] == "summary"
    assert "The guide." in summarizer.contents[0]


async def test_a_type_that_is_not_saved_has_no_saved_path(summarizer):
    result = await _web_fetch(_ok(raw=b"\x89PNG", content_type="image/png"))

    assert result["success"] is True
    assert "saved_path" not in result and "save_error" not in result
    assert not _folder().exists()


async def test_a_result_without_the_raw_body_is_not_saved(summarizer):
    result = await _web_fetch(FetchResult(success=True, content="<p>x</p>", content_type="text/html"))

    assert result["success"] is True
    assert "saved_path" not in result
    assert not _folder().exists()


async def test_a_failed_save_still_returns_the_summary(summarizer):
    _folder().parent.write_text("a file where the app folder should be")

    result = await _web_fetch(_ok())

    assert result["success"] is True
    assert result["summary"] == "summary"
    assert "saved_path" not in result
    assert result["save_error"]


async def test_a_symlinked_folder_is_not_written_through(summarizer, project):
    elsewhere = project / "elsewhere"
    elsewhere.mkdir()
    _folder().parent.symlink_to(elsewhere, target_is_directory=True)

    result = await _web_fetch(_ok())

    assert "symlink" in result["save_error"]
    assert list(elsewhere.iterdir()) == []


async def test_a_cache_hit_saves_the_page_again(summarizer):
    first = await _web_fetch(_ok())
    Path(first["saved_path"]).unlink()

    second = await _web_fetch(_ok(from_cache=True))

    assert second["cached"] is True
    assert Path(second["saved_path"]).read_bytes() == PAGE


async def test_saving_runs_cleanup(summarizer, monkeypatch):
    from agentic_cli.tools import webfetch_tool

    calls = []
    monkeypatch.setattr(webfetch_tool, "cleanup_saved_pages", lambda folder, **kw: calls.append((folder, kw)))
    await _web_fetch(_ok())

    s = get_settings()
    assert calls == [(_folder(), {
        "max_age_days": s.webfetch_saved_max_age_days,
        "max_bytes": s.webfetch_saved_max_mb * 1024 * 1024,
        "keep": page_name(URL),
    })]


async def test_a_failed_summary_still_reports_the_saved_page():
    class _Broken:
        async def summarize(self, content, prompt):
            raise RuntimeError("model unavailable")

    token = set_service_registry({LLM_SUMMARIZER: _Broken()})
    try:
        result = await _web_fetch(_ok())
    finally:
        token.var.reset(token)

    assert result["success"] is False
    assert Path(result["saved_path"]).read_bytes() == PAGE


async def test_the_factory_variant_saves_too():
    from agentic_cli.tools.factories import make_webfetch_tool

    tool = make_webfetch_tool(summarizer=_Summarizer())
    with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=_Fetcher(_ok())):
        result = await tool(URL, "summarize")

    assert Path(result["saved_path"]).read_bytes() == PAGE


def test_the_descriptions_point_to_kb_ingest_file():
    from agentic_cli.tools.factories import make_webfetch_tool
    from agentic_cli.tools.registry import get_registry
    from agentic_cli.tools.webfetch_tool import web_fetch

    for tool in (web_fetch, make_webfetch_tool(summarizer=_Summarizer())):
        assert "saved_path" in tool.__doc__ and "kb_ingest_file" in tool.__doc__
    assert "kb_ingest_file" in get_registry().get("web_fetch").description


async def test_a_real_fetch_saves_the_whole_page(summarizer, monkeypatch):
    """Through the real fetcher, a page over the summary limit is saved whole."""
    from agentic_cli.tools.webfetch.fetcher import ContentFetcher
    from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
    from agentic_cli.tools.webfetch.transport import PinnedTransport
    from agentic_cli.tools.webfetch.validator import URLValidator
    from agentic_cli.tools.webfetch_tool import web_fetch

    def _gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _gai)
    body = b"<p>" + b"a" * 3000 + b"</p>"
    validator = URLValidator()
    transport = PinnedTransport(validator, inner=httpx.MockTransport(
        lambda req: httpx.Response(200, content=body, headers={"content-type": "text/html"})))
    fetcher = ContentFetcher(validator=validator, robots_checker=RobotsTxtChecker(),
                             transport=transport, max_content_bytes=1000)
    fetcher._robots.can_fetch = AsyncMock(return_value=True)

    with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=fetcher):
        result = await web_fetch(URL, "summarize")

    assert Path(result["saved_path"]).read_bytes() == body
    assert result["truncated"] is True
    # html2text may escape the brackets around the marker.
    assert "Content truncated at 1000 bytes" in summarizer.contents[0]

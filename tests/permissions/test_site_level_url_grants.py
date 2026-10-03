"""Approving a URL "for this session" or "always" covers its site.

``http.read`` grants (``web_fetch``, ``kb_ingest_url``) were stored for the exact
URL, so reading ten pages of one site asked ten times whatever the answer. A
session or "always" approval now covers the URL's site, ``https://host/**``:
the exact host (no other subdomain) and port, HTTPS only. A plain-HTTP URL, an
IP address, ``localhost`` or a single-label name keeps the exact-URL grant,
"Allow once" still approves one URL, and the prompt shows both the exact URL
being fetched and the site an approval would cover.

The URL matcher also dropped an IPv6 host's brackets, so a grant for an IPv6
URL could not be parsed again: re-checking that address raised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agentic_cli.tools.knowledge_tools  # noqa: F401  (registers kb_ingest_url)
import agentic_cli.tools.webfetch_tool  # noqa: F401  (registers web_fetch)
from agentic_cli.config import BaseSettings
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.matchers import URLMatcher
from agentic_cli.workflow.permissions.prompt import (
    ALLOW_ALWAYS_CHOICE,
    ALLOW_ONCE_CHOICE,
    ALLOW_SESSION_CHOICE,
)


class _Approver:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def request_user_input(self, request) -> str:
        self.prompts.append(request.prompt)
        return self.answer


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    return tmp_path / "project"


def _engine(approver: _Approver) -> PermissionEngine:
    settings = BaseSettings(workspace_dir=Path.cwd() / "workspace")
    ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name)
    return PermissionEngine(settings=settings, workflow=approver, ctx=ctx)


async def _asks(engine: PermissionEngine, approver: _Approver, url: str, tool="web_fetch") -> bool:
    before = len(approver.prompts)
    args = {"url": url, "prompt": "summarize"} if tool == "web_fetch" else {"url": url}
    result = await engine.check(tool, get_registry().get(tool).capabilities, args)
    assert result.allowed
    return len(approver.prompts) > before


async def test_a_session_approval_covers_the_site():
    approver = _Approver(ALLOW_SESSION_CHOICE)
    engine = _engine(approver)

    assert await _asks(engine, approver, "https://docs.example.org/guide/intro")
    assert not await _asks(engine, approver, "https://docs.example.org/api?q=fetch")
    assert not await _asks(engine, approver, "https://docs.example.org/")
    assert not await _asks(engine, approver, "https://docs.example.org/guide", tool="kb_ingest_url")


@pytest.mark.parametrize(
    "other",
    [
        "https://example.org/guide",  # the parent domain
        "https://evil.docs.example.org/guide",  # a subdomain
        "https://docs.example.org.evil.net/guide",  # a lookalike
        "http://docs.example.org/guide",  # plain HTTP
    ],
)
async def test_it_covers_no_other_site(other):
    approver = _Approver(ALLOW_SESSION_CHOICE)
    engine = _engine(approver)
    await _asks(engine, approver, "https://docs.example.org/guide/intro")

    assert await _asks(engine, approver, other)


@pytest.mark.parametrize(
    "first, second",
    [
        ("http://example.org/a", "http://example.org/b"),
        ("https://203.0.113.7/a", "https://203.0.113.7/b"),
        ("https://[2001:db8::1]/a", "https://[2001:db8::1]/b"),
        ("https://localhost/a", "https://localhost/b"),
        ("https://app.localhost/a", "https://app.localhost/b"),
        ("https://intranet/a", "https://intranet/b"),
    ],
)
async def test_no_site_grant_for_plain_http_addresses_or_local_names(first, second):
    approver = _Approver(ALLOW_SESSION_CHOICE)
    engine = _engine(approver)
    await _asks(engine, approver, first)

    assert not await _asks(engine, approver, first)  # the exact URL is remembered
    assert await _asks(engine, approver, second)


async def test_a_non_default_port_is_part_of_the_site():
    approver = _Approver(ALLOW_SESSION_CHOICE)
    engine = _engine(approver)
    await _asks(engine, approver, "https://docs.example.org:8443/a")

    assert not await _asks(engine, approver, "https://docs.example.org:8443/b")
    assert await _asks(engine, approver, "https://docs.example.org/b")


async def test_allow_once_still_approves_one_url():
    approver = _Approver(ALLOW_ONCE_CHOICE)
    engine = _engine(approver)
    await _asks(engine, approver, "https://docs.example.org/a")

    assert await _asks(engine, approver, "https://docs.example.org/b")


async def test_an_always_approval_covers_the_site_after_a_restart():
    await _asks(_engine(first := _Approver(ALLOW_ALWAYS_CHOICE)), first, "https://docs.example.org/a")
    grants = json.loads((Path.home() / f".{BaseSettings().app_name}" / "project_grants.json").read_text())
    stored = [rule["target"] for entry in grants.values() for rule in entry["permissions"]["allow"]]
    assert stored == ["https://docs.example.org/**"]

    approver = _Approver(ALLOW_ONCE_CHOICE)
    assert not await _asks(_engine(approver), approver, "https://docs.example.org/b")


async def test_the_prompt_shows_the_url_and_the_site():
    approver = _Approver(ALLOW_ONCE_CHOICE)
    await _asks(_engine(approver), approver, "https://docs.example.org/search?q=agents")

    (prompt,) = approver.prompts
    assert "https://docs.example.org/search?q=agents" in prompt
    assert "https://docs.example.org/**" in prompt



@pytest.mark.parametrize(
    "pattern, url, matches",
    [
        ("https://docs.example.org/**", "https://docs.example.org/", True),
        ("https://docs.example.org/**", "https://docs.example.org", True),
        ("https://docs.example.org/guide/**", "https://docs.example.org/guide/", True),
        ("https://docs.example.org/guide/**", "https://docs.example.org/guidebook/", False),
        ("https://docs.example.org/guide", "https://docs.example.org/guide/", False),
    ],
)
def test_a_folder_pattern_covers_its_trailing_slash_form(pattern, url, matches):
    """``/**`` missed the folder's own trailing-slash form, and so the site root."""
    matcher = URLMatcher()

    assert matcher.matches(pattern, matcher.canonicalize_target(url, None)) is matches


async def test_a_deny_rule_now_catches_the_trailing_slash_form(project):
    settings_dir = project / f".{BaseSettings().app_name}"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps({
        "permissions": {
            "deny": [{"capability": "http.read", "target": "https://docs.example.org/private/**"}]
        }
    }))
    engine = _engine(_Approver(ALLOW_ONCE_CHOICE))

    result = await engine.check(
        "web_fetch",
        get_registry().get("web_fetch").capabilities,
        {"url": "https://docs.example.org/private/", "prompt": "summarize"},
    )

    assert result.allowed is False

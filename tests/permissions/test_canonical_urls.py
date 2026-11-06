"""URL rules judge the URL the HTTP client sends, not the one spelled.

The engine compared a URL as written (urllib's hostname, the path as given),
while ``web_fetch`` requests what httpx makes of it: dot segments removed, the
IDNA dots U+3002, U+FF0E and U+FF61 read as ``.``, a Unicode host encoded with
IDNA 2008; and DNS reads ``host.`` as ``host``. So with
``https://docs.example.com/**`` allowed and ``https://docs.example.com/admin/**``
denied, ``/x/../admin/keys`` and ``/%61dmin/keys`` were allowed without a
prompt, and behind an allow-all rule a host deny missed
``https://tracker.example.net./p``. Every URL target and pattern is now
canonicalized as the client sends it (``agentic_cli.urls.canonical_url``), the
prompt shows that form, a site grant names the canonical host, and a URL the
client cannot send is denied as an invalid target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog

import agentic_cli.tools.webfetch_tool  # noqa: F401  (registers web_fetch)
from agentic_cli.config import BaseSettings
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.engine import site_scope
from agentic_cli.workflow.permissions.prompt import ALLOW_ALWAYS_CHOICE, ALLOW_SESSION_CHOICE
from agentic_cli.workflow.permissions.rules import RuleSource


class _Approver:
    def __init__(self, answer: str = ALLOW_SESSION_CHOICE) -> None:
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


def _settings() -> BaseSettings:
    return BaseSettings(workspace_dir=Path.cwd() / "workspace")


def _user_rules(allow: list[str] = (), deny: list[str] = ()) -> None:
    """The user's own ``~/.<app>/settings.json`` permissions."""
    path = Path.home() / f".{_settings().app_name}" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"permissions": {
        "allow": [{"capability": "http.read", "target": t} for t in allow],
        "deny": [{"capability": "http.read", "target": t} for t in deny],
    }}))


def _engine(approver: _Approver) -> PermissionEngine:
    settings = _settings()
    ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name)
    return PermissionEngine(settings=settings, workflow=approver, ctx=ctx)


async def _fetch(engine: PermissionEngine, url: str):
    caps = get_registry().get("web_fetch").capabilities
    return await engine.check("web_fetch", caps, {"url": url, "prompt": "summarize"})


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.example.com/admin/keys",
        "https://docs.example.com/x/../admin/keys",
        "https://docs.example.com/%61dmin/keys",
        "https://docs.example.com/x/%2e%2e/admin/keys",
        "https://docs.example.com./admin/keys",
        "https://docs。example。com/admin/keys",
        "https://DOCS.example.com:443/admin/keys#top",
    ],
)
async def test_a_path_deny_holds_for_every_spelling_of_the_page(url):
    _user_rules(
        allow=["https://docs.example.com/**"],
        deny=["https://docs.example.com/admin/**"],
    )
    approver = _Approver()

    result = await _fetch(_engine(approver), url)

    assert not result.allowed, url
    assert "deny" in result.reason
    assert approver.prompts == []


@pytest.mark.parametrize(
    "url",
    [
        "https://tracker.example.net./p",
        "https://tracker。example。net/p",
        "https://tracker．example．net/p",
        "https://tracker｡example｡net/p",
    ],
)
async def test_a_host_deny_holds_behind_an_allow_all_rule(url):
    _user_rules(allow=["*"], deny=["https://tracker.example.net/**"])
    approver = _Approver()

    result = await _fetch(_engine(approver), url)

    assert not result.allowed, url
    assert approver.prompts == []


async def test_a_deny_written_in_another_spelling_still_names_the_page():
    """A pattern is canonicalized as a target is: a deny written with the
    trailing dot, an escape or a Unicode host denies the page it names."""
    _user_rules(
        allow=["*"],
        deny=[
            "https://tracker.example.net./**",
            "https://docs.example.com/%61dmin/**",
            "https://faß.de/**",
        ],
    )
    engine = _engine(_Approver())

    assert not (await _fetch(engine, "https://tracker.example.net/p")).allowed
    assert not (await _fetch(engine, "https://docs.example.com/admin/k")).allowed
    assert not (await _fetch(engine, "https://xn--fa-hia.de/")).allowed
    assert (await _fetch(engine, "https://docs.example.com/guide")).allowed


async def test_the_prompt_shows_the_url_the_client_sends():
    approver = _Approver()

    await _fetch(_engine(approver), "https://Docs.Example.org./guide/../api/%7Euser")

    (prompt,) = approver.prompts
    assert "http.read → https://docs.example.org/api/~user" in prompt
    assert "all of https://docs.example.org/**" in prompt
    assert "Example.org." not in prompt and "%7E" not in prompt


async def test_a_lookalike_host_is_shown_as_the_domain_it_reaches():
    """``faß.de`` is sent (and resolved, and verified) as ``xn--fa-hia.de``."""
    approver = _Approver()

    await _fetch(_engine(approver), "https://faß.de/")

    (prompt,) = approver.prompts
    assert "https://xn--fa-hia.de/" in prompt


async def test_a_site_grant_names_the_canonical_host():
    approver = _Approver(ALLOW_ALWAYS_CHOICE)
    engine = _engine(approver)
    await _fetch(engine, "https://Docs.Example.org./a")

    grants = [r.target for r in engine.rules if r.source is RuleSource.PROJECT]
    assert grants == ["https://docs.example.org/**"]
    saved = json.loads(
        (Path.home() / f".{_settings().app_name}" / "project_grants.json").read_text()
    )
    assert saved[str(Path.cwd().resolve())]["permissions"]["allow"] == [
        {"capability": "http.read", "target": "https://docs.example.org/**"}
    ]
    # Every spelling of the site is covered, and nothing else.
    assert (await _fetch(engine, "https://docs。example。org/b")).allowed
    assert len(approver.prompts) == 1
    await _fetch(engine, "https://docs.example.org.evil.net/b")
    assert len(approver.prompts) == 2


@pytest.mark.parametrize(
    "url, scope",
    [
        ("https://Docs.Example.org./a", "https://docs.example.org/**"),
        ("https://docs。example。org/a", "https://docs.example.org/**"),
        ("https://faß.de/x", "https://xn--fa-hia.de/**"),
        ("https://localhost./a", None),  # a local name, however it is spelled
        ("https://intranet./a", None),  # a single label
        ("https://i❤.ws/", None),  # not a URL the client can send
    ],
)
def test_site_scope_uses_the_canonical_host(url, scope):
    assert site_scope(url) == scope


@pytest.mark.parametrize("url", ["https://i❤.ws/", "https://ex­ample.com/", "https://example.com:99999/"])
async def test_a_url_the_client_cannot_send_is_denied(url):
    _user_rules(allow=["*"])
    approver = _Approver()

    result = await _fetch(_engine(approver), url)

    assert not result.allowed
    assert result.reason.startswith("invalid target")
    assert approver.prompts == []


async def test_a_rule_that_names_no_url_is_skipped_with_a_warning():
    """A rule the client could never match no longer loads; the rest of the
    file still applies, and the engine still starts."""
    _user_rules(allow=["https://i❤.ws/**"], deny=["https://docs.example.com/admin/**"])

    with structlog.testing.capture_logs() as logs:
        engine = _engine(_Approver())

    assert not (await _fetch(engine, "https://docs.example.com/admin/k")).allowed
    skipped = [e for e in logs if e.get("event") == "permission_rule_skipped"]
    assert [(e["capability"], e["target"]) for e in skipped] == [("http.read", "https://i❤.ws/**")]


async def test_a_saved_grant_that_names_no_url_is_dropped_with_a_warning():
    path = Path.home() / f".{_settings().app_name}" / "project_grants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(Path.cwd().resolve()): {"permissions": {"allow": [
        {"capability": "http.read", "target": "https://i❤.ws/**"},
        {"capability": "http.read", "target": "https://docs.example.org./**"},
    ]}}}))

    with structlog.testing.capture_logs() as logs:
        engine = _engine(_Approver())

    grants = [r.target for r in engine.rules if r.source is RuleSource.PROJECT]
    assert grants == ["https://docs.example.org/**"]
    skipped = [e for e in logs if e.get("event") == "permission_rule_skipped"]
    assert [e["target"] for e in skipped] == ["https://i❤.ws/**"]

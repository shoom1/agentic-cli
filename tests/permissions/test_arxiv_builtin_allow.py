"""The arXiv tools run without a permission prompt.

``search_arxiv``, ``fetch_arxiv_paper`` and ``ingest_arxiv_paper`` can reach
only two fixed endpoints, the arXiv API query URL and ``https://arxiv.org/pdf``
(``download_pdf`` refuses anything else). Since 0.6.1 they declare those
endpoints as ``http.read`` targets, and every call asked unless the user chose
"for this session" or "always"; answering "once" meant a prompt per search,
per paper and per ingestion. Built-in rules now allow exactly those two
endpoints, as they allow the knowledge-base writes. Any other URL, including
other arxiv.org pages through ``web_fetch``, still asks, and a deny rule in
the user's or the project's settings still wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agentic_cli.tools.arxiv_tools  # noqa: F401  (registers the arXiv tools)
import agentic_cli.tools.webfetch_tool  # noqa: F401  (registers web_fetch)
from agentic_cli.config import BaseSettings
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE

PAPER = "2401.00001"
ARXIV_CALLS = [
    ("search_arxiv", {"query": "tool-using agents"}),
    ("fetch_arxiv_paper", {"arxiv_id": PAPER}),
    ("ingest_arxiv_paper", {"arxiv_id": PAPER}),
]


class _Approver:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def request_user_input(self, request) -> str:
        self.asked.append(request.tool_name)
        return ALLOW_ONCE_CHOICE


@pytest.fixture
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


async def _check(engine: PermissionEngine, tool: str, args: dict):
    return await engine.check(tool, get_registry().get(tool).capabilities, args)


async def test_the_arxiv_tools_do_not_ask(project):
    approver = _Approver()
    engine = _engine(approver)

    for tool, args in ARXIV_CALLS:
        assert (await _check(engine, tool, args)).allowed, tool

    assert approver.asked == []


@pytest.mark.parametrize(
    "url",
    [
        f"https://arxiv.org/pdf/{PAPER}",
        f"https://arxiv.org/abs/{PAPER}",
        "https://example.org/arxiv/pdf",
    ],
)
async def test_other_urls_still_ask(project, url):
    approver = _Approver()

    await _check(_engine(approver), "web_fetch", {"url": url, "prompt": "summarize"})

    assert approver.asked == ["web_fetch"]


async def test_a_project_deny_rule_still_wins(project):
    settings_dir = project / f".{BaseSettings().app_name}"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps({
        "permissions": {
            "deny": [{"capability": "http.read", "target": "https://arxiv.org/pdf"}]
        }
    }))

    result = await _check(_engine(_Approver()), "ingest_arxiv_paper", {"arxiv_id": PAPER})

    assert result.allowed is False

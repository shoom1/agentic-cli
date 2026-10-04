"""``kb_list`` lists what was asked for.

It took the ``limit`` newest documents of each knowledge base and only then
applied the ``query`` filter, so a matching document outside the newest 20 was
never listed. It also listed up to ``limit`` documents from the project and
again from the user knowledge base, twice the documented maximum, and an
unknown ``source_type`` was ignored, listing everything as if it had filtered.
Now the filters come first, the two knowledge bases are merged newest first
and cut at ``limit``, and an unknown ``source_type`` is an error.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType
from agentic_cli.tools.knowledge_tools import kb_list
from agentic_cli.workflow.service_registry import set_service_registry

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def kbs(tmp_path):
    project = KnowledgeBaseManager(base_dir=tmp_path / "project", use_mock=True)
    user = KnowledgeBaseManager(base_dir=tmp_path / "user", use_mock=True)
    token = set_service_registry({"kb_manager": project, "user_kb_manager": user})
    try:
        yield {"project": project, "user": user}
    finally:
        token.var.reset(token)


def _add(kb: KnowledgeBaseManager, title: str, minute: int, source=SourceType.USER) -> None:
    doc = kb.ingest_document(content=f"text of {title}", title=title, source_type=source)
    doc.updated_at = _EPOCH + timedelta(minutes=minute)


def _titles(result: dict) -> list[str]:
    return [item["title"] for item in result["documents"]]


@pytest.mark.parametrize("holder", ["project", "user"])
def test_a_match_older_than_the_newest_documents_is_listed(kbs, holder):
    _add(kbs[holder], "Zeppelin mooring", minute=0)
    for n in range(1, 26):
        _add(kbs[holder], f"Note {n}", minute=n)

    result = kb_list(query="zeppelin", limit=20)

    assert _titles(result) == ["Zeppelin mooring"]


def test_the_two_knowledge_bases_together_stay_within_limit(kbs):
    for n in range(15):
        _add(kbs["project"], f"Project {n}", minute=2 * n)
        _add(kbs["user"], f"User {n}", minute=2 * n + 1)

    result = kb_list(limit=20)

    assert result["count"] == 20
    newest_first = [f"{side} {n}" for n in range(14, -1, -1) for side in ("User", "Project")]
    assert _titles(result) == newest_first[:20]


def test_an_unknown_source_type_is_an_error(kbs):
    _add(kbs["project"], "Airships", minute=0)

    result = kb_list(source_type="blog")

    assert result["success"] is False
    assert "blog" in result["error"]
    assert "arxiv" in result["error"]  # names the valid ones


def test_source_type_still_filters(kbs):
    _add(kbs["project"], "Paper", minute=0, source=SourceType.ARXIV)
    _add(kbs["user"], "Page", minute=1, source=SourceType.WEB)

    assert _titles(kb_list(source_type="arxiv")) == ["Paper"]

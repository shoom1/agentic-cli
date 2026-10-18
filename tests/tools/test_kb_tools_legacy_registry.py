"""Registries filled by hand with the 0.6.3 keys still work (until 0.7.0).

The module-level tools find the workflow's ``KnowledgeBase`` under
``knowledge_base``. A registry that holds only ``kb_manager`` (and
``user_kb_manager``), as many tests and some apps build by hand, gets a
``KnowledgeBase`` over those managers on each call. The per-document
sidecar locks live on the managers, so two concurrent first reads still
make one summarizer call.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.tools.knowledge_tools import kb_list, kb_read
from agentic_cli.workflow.service_registry import set_service_registry


class _Slow:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, content, prompt):
        self.calls += 1
        await asyncio.sleep(0.05)
        return "SUMMARY: A lazy summary.\n"


async def test_legacy_registry_keys_share_one_sidecar_lock(tmp_path):
    summarizer = _Slow()
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True, summarizer=summarizer)
    doc = manager.ingest_document(content="body", title="Doc", source_type=SourceType.USER)
    manager.sidecar_path(doc.id).unlink()
    token = set_service_registry({"kb_manager": manager})
    try:
        with pytest.warns(DeprecationWarning, match="0.7.0"):
            first, second = await asyncio.gather(kb_read(doc.id), kb_read(doc.id))
    finally:
        token.var.reset(token)

    assert first["success"] and second["success"]
    assert summarizer.calls == 1


def test_the_legacy_registry_warning_points_at_the_caller(tmp_path):
    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    token = set_service_registry({"kb_manager": manager})
    try:
        with pytest.warns(DeprecationWarning, match="0.7.0") as caught:
            kb_list()
    finally:
        token.var.reset(token)

    assert [w.filename for w in caught if "0.7.0" in str(w.message)] == [__file__]


def test_the_knowledge_base_key_wins_over_the_legacy_keys(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBase

    new = KnowledgeBaseManager(tmp_path / "new", use_mock=True)
    new.ingest_document(content="new", title="From the knowledge base", source_type=SourceType.USER)
    old = KnowledgeBaseManager(tmp_path / "old", use_mock=True)
    old.ingest_document(content="old", title="From the legacy key", source_type=SourceType.USER)
    token = set_service_registry({"knowledge_base": KnowledgeBase({"project": new}), "kb_manager": old})
    try:
        titles = [d["title"] for d in kb_list()["documents"]]
    finally:
        token.var.reset(token)

    assert titles == ["From the knowledge base"]

"""A knowledge base uses the summarizer it is given.

Before, it looked one up in the per-turn service registry, so outside a turn
(the research demo's /kb-backfill, a script, another harness) it silently fell
back to the first ~500 characters of the document.
"""

from __future__ import annotations

from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
from agentic_cli.knowledge_base.models import SourceType
from agentic_cli.workflow.service_registry import LLM_SUMMARIZER, set_service_registry

SIDECAR = "SUMMARY: Injected summary.\nCLAIMS:\n- Injected claim.\n"


class _Summarizer:
    def __init__(self, reply: str = SIDECAR) -> None:
        self.reply = reply
        self.calls = 0

    async def summarize(self, content: str, prompt: str) -> str:
        self.calls += 1
        return self.reply


def _kb(tmp_path, **kwargs) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(base_dir=tmp_path / "kb", use_mock=True, **kwargs)


async def test_a_given_summarizer_is_used_outside_any_turn(tmp_path):
    kb = _kb(tmp_path, summarizer=_Summarizer())

    payload = await kb.generate_sidecar_payload("Body text.", title="T")

    assert payload["summary"] == "Injected summary."
    assert payload["claims"] == ["Injected claim."]


async def test_generate_summary_uses_the_given_summarizer(tmp_path):
    kb = _kb(tmp_path, summarizer=_Summarizer("A short summary."))

    assert await kb.generate_summary("Body text.", title="T") == "A short summary."


async def test_backfill_uses_the_given_summarizer(tmp_path):
    kb = _kb(tmp_path, summarizer=_Summarizer())
    doc = kb.ingest_document(content="Body text.", title="T", source_type=SourceType.USER)
    kb._sidecar_path(doc.id).unlink()

    assert await kb.backfill_sidecars() == 1
    assert "Injected summary." in kb._sidecar_path(doc.id).read_text()


async def test_no_summarizer_means_the_preview_even_inside_a_turn(tmp_path):
    registered = _Summarizer()
    kb = _kb(tmp_path, summarizer=None)
    token = set_service_registry({LLM_SUMMARIZER: registered})
    try:
        payload = await kb.generate_sidecar_payload("Body text.", title="T")
    finally:
        token.var.reset(token)

    assert payload == {"summary": "Body text.", "claims": [], "entities": {}}
    assert registered.calls == 0


async def test_without_the_argument_the_turn_registry_is_still_used(tmp_path):
    """Today's behavior; PR 4 moves it to the deprecated old import path."""
    kb = _kb(tmp_path)
    token = set_service_registry({LLM_SUMMARIZER: _Summarizer()})
    try:
        payload = await kb.generate_sidecar_payload("Body text.", title="T")
    finally:
        token.var.reset(token)

    assert payload["summary"] == "Injected summary."

"""Enriching a document: the prompts, the parser and the one summarizer call.

They moved from ``KnowledgeBaseManager`` to ``agentic_cli.memory.kb.enrich``
so the ingestion pipeline can use them; ``generate_summary`` and
``generate_sidecar_payload`` call them and return what they returned.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory.kb import KnowledgeBaseManager
from agentic_cli.memory.kb import enrich


class _Answer:
    def __init__(self, answer: str = "", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.contents: list[str] = []

    async def summarize(self, content: str, prompt: str) -> str:
        self.contents.append(content)
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer


SIDECAR_ANSWER = (
    "**SUMMARY:**\nFirst paragraph.\n\nSecond paragraph.\n\n"
    "CLAIMS:\n- One claim.\n* Another claim.\n\n"
    "ENTITIES:\nModels: A, B\nDatasets: C\n"
)


def test_the_preview_is_the_first_500_characters():
    assert enrich.preview("") == ""
    assert enrich.preview("short") == "short"
    assert enrich.preview("x" * 501) == "x" * 500 + "..."


async def test_the_sidecar_payload_parses_the_three_sections():
    payload = await enrich.sidecar_payload(_Answer(SIDECAR_ANSWER), "body", title="T")

    assert payload == {
        "summary": "First paragraph.\n\nSecond paragraph.",
        "claims": ["One claim.", "Another claim."],
        "entities": {"Models": ["A", "B"], "Datasets": ["C"]},
    }


async def test_the_sidecar_prompt_carries_the_title_and_the_capped_content():
    summarizer = _Answer(SIDECAR_ANSWER)
    content = "y" * 20_000

    await enrich.sidecar_payload(summarizer, content, title="Paper")

    [prompt] = summarizer.prompts
    assert prompt.startswith("Extract a structured payload from the document below.")
    assert prompt.endswith("Title: Paper\n\nContent:\n---\n" + "y" * 12_000 + "\n---")
    assert summarizer.contents == ["y" * 12_000]


async def test_an_untitled_document_says_so_in_the_prompt():
    summarizer = _Answer(SIDECAR_ANSWER)

    await enrich.sidecar_payload(summarizer, "body")

    assert "Title: (untitled)" in summarizer.prompts[0]


@pytest.mark.parametrize(
    "summarizer",
    [None, _Answer(error=RuntimeError("LLM down")), _Answer("no headers at all"), _Answer("")],
    ids=["none", "raises", "garbage", "empty"],
)
async def test_without_a_usable_answer_the_payload_is_the_preview(summarizer):
    payload = await enrich.sidecar_payload(summarizer, "x" * 600, title="T")

    assert payload == {"summary": "x" * 500 + "...", "claims": [], "entities": {}}


async def test_empty_content_makes_no_call():
    summarizer = _Answer(SIDECAR_ANSWER)

    assert await enrich.sidecar_payload(summarizer, "", title="T") == {"summary": "", "claims": [], "entities": {}}
    assert await enrich.summarize(summarizer, "", title="T") == ""
    assert summarizer.prompts == []


async def test_the_summary_prompt_asks_for_two_or_three_sentences():
    summarizer = _Answer("  A short summary.  ")

    assert await enrich.summarize(summarizer, "body", title="Paper") == "A short summary."
    assert summarizer.prompts[0].startswith("Summarize the following document in 2-3 sentences")


async def test_a_failed_summary_is_the_preview():
    assert await enrich.summarize(_Answer(error=RuntimeError("down")), "body text") == "body text"


async def test_the_manager_returns_what_enrich_returns(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True, summarizer=_Answer(SIDECAR_ANSWER))

    assert await kb.generate_sidecar_payload("body", title="T") == await enrich.sidecar_payload(
        _Answer(SIDECAR_ANSWER), "body", title="T"
    )
    assert await kb.generate_summary("body", title="T") == SIDECAR_ANSWER.strip()


async def test_the_manager_without_a_summarizer_stores_the_preview(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    assert await kb.generate_sidecar_payload("Body text.", title="T") == {
        "summary": "Body text.", "claims": [], "entities": {},
    }
    assert await kb.generate_summary("Body text.") == "Body text."

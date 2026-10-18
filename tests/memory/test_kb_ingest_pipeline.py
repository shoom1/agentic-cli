"""One ingestion pipeline: convert, fill in defaults, enrich, store.

Text, local files, saved pages, fetched URLs and arXiv papers went in
through four code paths in the tools, each with its own defaults. They all
end in ``KnowledgeBase.ingest_text`` or ``ingest_bytes`` now. A value the
caller passed always wins; one summarizer call writes the summary, claims
and entities.
"""

from __future__ import annotations

import sys

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import (
    ConversionFailed,
    EmptyDocument,
    KnowledgeBase,
    KnowledgeBaseManager,
    SourceType,
    UnsupportedDocument,
)
from agentic_cli.memory.kb import convert
from tests.memory.pdf_bytes import pdf_with_text

PAGE = (
    "<html><head><title>Volcano notes</title>"
    '<meta name="author" content="Ada Lovelace">'
    "</head><body><p>Magma rises through the crust.</p></body></html>"
).encode()

SIDECAR = "SUMMARY:\nA synthesis.\n\nCLAIMS:\n- A claim.\n"


class _Summarizer:
    def __init__(self, answer: str = SIDECAR) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def summarize(self, content: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture(autouse=True)
def no_model_no_trafilatura(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))
    monkeypatch.setitem(sys.modules, "trafilatura", None)  # html2text: the page's <title> only


@pytest.fixture
def summarizer() -> _Summarizer:
    return _Summarizer()


@pytest.fixture
def kb(tmp_path, summarizer) -> KnowledgeBase:
    return KnowledgeBase({
        "project": KnowledgeBaseManager(tmp_path / "project", use_mock=True, summarizer=summarizer),
        "user": KnowledgeBaseManager(tmp_path / "user", use_mock=True, summarizer=summarizer),
    })


# -- text --------------------------------------------------------------------


async def test_text_is_stored_in_the_first_scope_with_one_summarizer_call(kb, summarizer):
    found = await kb.ingest_text("Pumice floats.", title="Pumice")

    assert found.scope == "project"
    assert found.document.summary == "A synthesis."
    assert found.document.source_type is SourceType.USER
    assert len(summarizer.prompts) == 1
    sidecar = kb.manager("project").sidecar_path(found.document.id).read_text()
    assert "- A claim." in sidecar


async def test_text_goes_to_the_scope_asked_for(kb):
    found = await kb.ingest_text("Glacier notes.", title="Glacier", scope="user")

    assert found.scope == "user"
    assert kb.manager("user").get_document(found.document.id) is not None


async def test_the_title_defaults_to_the_texts_first_80_characters(kb):
    found = await kb.ingest_text("w" * 100)

    assert found.document.title == "w" * 80 + "..."


async def test_without_a_summarizer_the_summary_is_the_preview(tmp_path):
    kb = KnowledgeBase({"project": KnowledgeBaseManager(tmp_path / "kb", use_mock=True)})

    found = await kb.ingest_text("x" * 600, title="Long")

    assert found.document.summary == "x" * 500 + "..."


async def test_an_answer_without_a_summary_stores_the_preview(tmp_path):
    claims_only = _Summarizer("CLAIMS:\n- one\n- two\n")
    kb = KnowledgeBase({"project": KnowledgeBaseManager(tmp_path / "kb", use_mock=True, summarizer=claims_only)})

    found = await kb.ingest_text("raw body for fallback", title="X")

    assert found.document.summary == "raw body for fallback"


async def test_the_callers_fields_are_stored(kb):
    found = await kb.ingest_text(
        "Text.", title="T", source_type=SourceType.INTERNAL, source_url="https://example.org/t",
        metadata={"tags": ["a"]},
    )

    doc = found.document
    assert (doc.source_type, doc.source_url, doc.metadata) == (SourceType.INTERNAL, "https://example.org/t", {"tags": ["a"]})


async def test_empty_text_is_refused(kb, summarizer):
    with pytest.raises(EmptyDocument):
        await kb.ingest_text("")

    assert summarizer.prompts == []


# -- bytes: defaults -----------------------------------------------------------


async def test_a_local_file_defaults_to_local_and_its_name(kb):
    found = await kb.ingest_bytes(b"# Notes\n\nLava tubes.\n", extension=".md", default_title="notes")

    doc = found.document
    assert doc.title == "notes"
    assert doc.source_type is SourceType.LOCAL
    assert doc.source_url is None
    assert doc.metadata == {"file_size_bytes": len(b"# Notes\n\nLava tubes.\n")}
    assert kb.manager("project").get_file_path(doc.id).read_bytes() == b"# Notes\n\nLava tubes.\n"


async def test_a_page_title_wins_over_the_default_title(kb):
    found = await kb.ingest_bytes(PAGE, extension=".html", charset="utf-8", default_title="page")

    assert found.document.title == "Volcano notes"


async def test_a_fetched_page_defaults_to_web_and_its_url(kb):
    found = await kb.ingest_bytes(
        PAGE, extension=".html", content_type="text/html", charset="utf-8",
        fetched_url="https://example.com/articles/volcanoes", fetched_at="2026-10-01T09:00:00+00:00",
    )

    doc = found.document
    assert doc.source_type is SourceType.WEB
    assert doc.source_url == "https://example.com/articles/volcanoes"
    assert doc.metadata["fetched_at"] == "2026-10-01T09:00:00+00:00"


async def test_a_fetched_page_without_a_title_is_named_by_its_url(kb):
    page = b"<html><body><p>No title here.</p></body></html>"

    named = await kb.ingest_bytes(page, extension=".html", charset="utf-8", fetched_url="https://example.com/a/caldera%20notes")
    host = await kb.ingest_bytes(page, extension=".html", charset="utf-8", fetched_url="https://example.com/")

    assert named.document.title == "caldera notes"
    assert host.document.title == "example.com"


async def test_with_no_title_anywhere_the_title_is_the_texts_start(kb):
    found = await kb.ingest_bytes(b"Plain words about basalt.", extension=".txt")

    assert found.document.title == "Plain words about basalt."


async def test_the_source_url_cites_what_the_caller_says(kb):
    """arXiv cites the abstract page while the bytes come from the PDF URL."""
    found = await kb.ingest_bytes(
        pdf_with_text("Attention replaces recurrence"), extension=".pdf", content_type="application/pdf",
        fetched_url="https://arxiv.org/pdf/1706.03762", source_url="https://arxiv.org/abs/1706.03762",
        source_type=SourceType.ARXIV, title="Attention Is All You Need",
    )

    doc = found.document
    assert (doc.source_url, doc.source_type, doc.title) == (
        "https://arxiv.org/abs/1706.03762", SourceType.ARXIV, "Attention Is All You Need",
    )
    assert doc.content == "Attention replaces recurrence"


async def test_the_callers_metadata_wins_over_the_pages(kb):
    found = await kb.ingest_bytes(
        PAGE, extension=".html", charset="utf-8",
        metadata={"authors": ["Mary Somerville"], "file_size_bytes": 1},
    )

    assert found.document.metadata["authors"] == ["Mary Somerville"]
    assert found.document.metadata["file_size_bytes"] == 1


async def test_the_pages_authors_and_description_fill_the_metadata(tmp_path, monkeypatch, summarizer):
    page = convert.ConvertedDocument(
        text="Main text.", title="Page", authors=("Ada Lovelace",), published="2025-06-01",
        description="How magma rises.", extractor="trafilatura",
    )
    monkeypatch.setattr(convert, "convert_document", lambda *args, **kwargs: page)
    kb = KnowledgeBase({"project": KnowledgeBaseManager(tmp_path / "kb", use_mock=True, summarizer=summarizer)})

    found = await kb.ingest_bytes(b"<html/>", extension=".html", charset="utf-8")

    assert found.document.metadata == {
        "authors": ["Ada Lovelace"], "abstract": "How magma rises.", "published": "2025-06-01",
        "file_size_bytes": len(b"<html/>"),
    }


async def test_fallback_text_is_stored_when_the_bytes_have_none(kb):
    found = await kb.ingest_bytes(
        b"%PDF-1.4 not really", extension=".pdf", fallback_text="We propose the Transformer.", title="Paper",
    )

    assert found.document.content == "We propose the Transformer."
    assert kb.manager("project").get_file_path(found.document.id) is not None


async def test_bytes_with_no_text_are_still_stored(kb):
    found = await kb.ingest_bytes(b"%PDF-1.4 not really", extension=".pdf", default_title="scan")

    assert found.document.content == ""
    assert found.document.title == "scan"


async def test_one_summarizer_call_per_ingest(kb, summarizer):
    await kb.ingest_bytes(PAGE, extension=".html", charset="utf-8")

    assert len(summarizer.prompts) == 1
    assert summarizer.prompts[0].startswith("Extract a structured payload")


# -- bytes: errors -------------------------------------------------------------


async def test_bytes_the_converter_cannot_read_are_refused(kb):
    with pytest.raises(UnsupportedDocument, match="not a PDF or a UTF-8 text file"):
        await kb.ingest_bytes(b"\xff\xfe\x00\x01", extension=".bin")

    assert kb.list_documents() == []


async def test_a_converter_crash_is_a_conversion_failure(kb, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(convert, "convert_document", boom)

    with pytest.raises(ConversionFailed, match="RuntimeError") as caught:
        await kb.ingest_bytes(b"text", extension=".txt")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert kb.list_documents() == []


async def test_no_text_and_no_bytes_is_empty(kb):
    with pytest.raises(EmptyDocument):
        await kb.ingest_bytes(b"", extension=".txt")

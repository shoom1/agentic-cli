"""``kb_ingest_file`` ingests pages ``web_fetch`` saved, and HTML as its main text.

A page reaches the knowledge base in two steps: ``web_fetch`` saves it, then
``kb_ingest_file(saved_path)`` ingests the saved copy without touching the
network. HTML went in as raw markup; it now goes through the ingestion
converter. For a saved page, the metadata beside it supplies the page's URL,
and the page supplies its title, authors and description.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.config import get_settings
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.tools.knowledge_tools import kb_ingest_file
from agentic_cli.tools.webfetch.saved import page_name, save_page, saved_pages_dir
from agentic_cli.workflow.service_registry import set_service_registry

URL = "https://example.com/articles/volcanoes"
ARTICLE = (
    "<html><head><title>Volcano notes</title>"
    '<meta name="author" content="Ada Lovelace; Mary Somerville">'
    '<meta name="description" content="How magma reaches the surface.">'
    '<meta property="article:published_time" content="2025-06-01">'
    "</head><body>"
    "<nav><a href='/'>Home</a> <a href='/about'>About us</a></nav>"
    "<article><h1>Volcanoes</h1><p>"
    + "Magma rises through the crust and erupts as lava. " * 12
    + "</p></article><footer>Copyright Example Press</footer></body></html>"
)


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def kb(tmp_path):
    manager = KnowledgeBaseManager(base_dir=tmp_path / "kb", use_mock=True)
    token = set_service_registry({"kb_manager": manager})
    try:
        yield manager
    finally:
        token.var.reset(token)


def _saved(data=ARTICLE.encode(), content_type="text/html; charset=utf-8",
           charset="utf-8", truncated=False) -> Path:
    return save_page(
        saved_pages_dir(get_settings().app_name), url=URL, final_url=URL,
        content_type=content_type, charset=charset, data=data, truncated=truncated,
    )


async def test_a_saved_page_is_ingested_as_a_web_document(kb):
    result = await kb_ingest_file(path=str(_saved()))

    assert result["success"] is True
    doc = kb.get_document(result["document_id"])
    assert doc.source_url == URL
    assert doc.source_type.value == "web"
    assert "fetched_at" in doc.metadata
    assert "Magma rises through the crust" in doc.content
    assert "<p>" not in doc.content


async def test_a_saved_page_supplies_title_authors_and_abstract(kb):
    pytest.importorskip("trafilatura")
    result = await kb_ingest_file(path=str(_saved()))

    doc = kb.get_document(result["document_id"])
    assert doc.title == "Volcanoes"
    assert doc.metadata["authors"] == ["Ada Lovelace", "Mary Somerville"]
    assert doc.metadata["abstract"] == "How magma reaches the surface."
    assert doc.metadata["published"] == "2025-06-01"
    assert "Home" not in doc.content and "Copyright" not in doc.content


async def test_the_model_arguments_win(kb):
    result = await kb_ingest_file(
        path=str(_saved()), title="My title", source_type="internal",
        source_url="https://example.org/canonical", authors=["Someone"], abstract="Mine",
    )

    doc = kb.get_document(result["document_id"])
    assert doc.title == "My title"
    assert doc.source_type.value == "internal"
    assert doc.source_url == "https://example.org/canonical"
    assert doc.metadata["authors"] == ["Someone"]
    assert doc.metadata["abstract"] == "Mine"


async def test_a_saved_page_without_a_title_is_named_after_its_url(kb, monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    page = b"<html><body><p>No title here, only text about basalt.</p></body></html>"
    result = await kb_ingest_file(path=str(_saved(data=page)))

    assert kb.get_document(result["document_id"]).title == "volcanoes"


async def test_a_saved_page_is_decoded_with_its_charset(kb):
    page = (
        "<html><body><article><p>"
        + "Магма поднимается сквозь кору и извергается как лава. " * 12
        + "</p></article></body></html>"
    ).encode("windows-1251")
    path = _saved(data=page, content_type="text/html; charset=windows-1251", charset="windows-1251")

    result = await kb_ingest_file(path=str(path))

    assert "Магма поднимается" in kb.get_document(result["document_id"]).content


async def test_a_truncated_saved_page_is_reported(kb):
    result = await kb_ingest_file(path=str(_saved(truncated=True)))
    assert result["truncated"] is True


async def test_a_local_html_file_goes_in_as_text(kb, project):
    page = project / "notes.html"
    page.write_text(ARTICLE)

    result = await kb_ingest_file(path=str(page))

    doc = kb.get_document(result["document_id"])
    assert doc.source_type.value == "local"
    assert doc.source_url == str(page.resolve())
    assert "Magma rises through the crust" in doc.content
    assert "<p>" not in doc.content


async def test_metadata_beside_a_file_outside_the_folder_is_ignored(kb, project):
    saved = _saved()
    copy = project / saved.name
    copy.write_bytes(saved.read_bytes())
    meta = saved.with_name(f"{page_name(URL)}.meta.json")
    (project / meta.name).write_text(meta.read_text())

    result = await kb_ingest_file(path=str(copy))

    doc = kb.get_document(result["document_id"])
    assert doc.source_type.value == "local"
    assert doc.source_url == str(copy.resolve())


async def test_a_metadata_file_is_refused(kb):
    saved = _saved()
    result = await kb_ingest_file(path=str(saved.with_name(f"{page_name(URL)}.meta.json")))

    assert result["success"] is False
    assert "metadata" in result["error"]
    assert kb.list_documents() == []


async def test_the_factory_variant_reads_saved_pages(kb):
    from agentic_cli.tools.factories import make_kb_tools

    tools = {t.__name__: t for t in make_kb_tools(kb)}
    result = await tools["kb_ingest_file"](path=str(_saved()))

    assert kb.get_document(result["document_id"]).source_url == URL


async def test_ingesting_a_saved_page_asks_nothing(tmp_path, monkeypatch):
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools.registry import get_registry
    from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine

    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    saved = _saved()

    class _NoPrompt:
        async def request_user_input(self, request):
            raise AssertionError(f"asked: {request.prompt}")

    settings = BaseSettings(workspace_dir=tmp_path / "workspace")
    ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name)
    engine = PermissionEngine(settings=settings, workflow=_NoPrompt(), ctx=ctx)
    result = await engine.check(
        "kb_ingest_file",
        get_registry().get("kb_ingest_file").capabilities,
        {"path": os.path.relpath(saved, Path.cwd())},
    )

    assert result.allowed


async def test_fetch_then_ingest_then_search(kb):
    pytest.importorskip("trafilatura")
    from agentic_cli.tools.webfetch.fetcher import FetchResult
    from agentic_cli.tools.webfetch_tool import web_fetch
    from agentic_cli.workflow.service_registry import LLM_SUMMARIZER

    class _Summarizer:
        async def summarize(self, content, prompt):
            return "A page about volcanoes."

    class _Fetcher:
        async def fetch(self, url, timeout=30):
            return FetchResult(
                success=True, content=ARTICLE, content_type="text/html; charset=utf-8",
                raw=ARTICLE.encode(), charset="utf-8", final_url=url,
            )

    token = set_service_registry({LLM_SUMMARIZER: _Summarizer(), "kb_manager": kb})
    try:
        with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=_Fetcher()):
            fetched = await web_fetch(URL, "What is this page about?")
        ingested = await kb_ingest_file(path=fetched["saved_path"])
    finally:
        token.var.reset(token)

    assert ingested["success"] is True

    def titles(query):
        return [r["document_title"] for r in kb.search(query)["results"]]

    assert titles("magma") == ["Volcanoes"]
    assert titles("copyright") == []

"""``kb_ingest_url`` is deprecated: use ``web_fetch``, then ``kb_ingest_file``.

It keeps working until 0.7.0 through the same converter as ``kb_ingest_file``,
so it no longer stores a page's raw HTML. It saves the page as ``web_fetch``
does, refuses types the knowledge base cannot ingest, and warns on every call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.tools.webfetch.fetcher import FetchResult
from agentic_cli.workflow.service_registry import set_service_registry

URL = "https://example.com/articles/volcanoes"
ARTICLE = (
    "<html><head><title>Volcano notes</title></head><body>"
    "<article><h1>Volcanoes</h1><p>"
    + "Magma rises through the crust and erupts as lava. " * 12
    + "</p></article></body></html>"
)


class _Fetcher:
    def __init__(self, result):
        self.result = result

    async def fetch(self, url, timeout=30):
        return self.result


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


def _page(raw=ARTICLE.encode(), content_type="text/html; charset=utf-8"):
    return FetchResult(
        success=True, content=raw.decode("utf-8", "replace"), content_type=content_type,
        raw=raw, charset="utf-8", final_url=URL,
    )


async def _ingest(result):
    from agentic_cli.tools.knowledge_tools import kb_ingest_url

    with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=_Fetcher(result)):
        return await kb_ingest_url(url=URL)


async def test_a_page_goes_in_as_text_not_html(kb):
    with pytest.warns(DeprecationWarning, match="web_fetch"):
        result = await _ingest(_page())

    doc = kb.get_document(result["document_id"])
    assert "Magma rises through the crust" in doc.content
    assert "<p>" not in doc.content
    assert doc.source_url == URL
    assert doc.source_type.value == "web"


async def test_the_page_is_saved(kb):
    with pytest.warns(DeprecationWarning):
        result = await _ingest(_page())

    assert Path(result["saved_path"]).read_bytes() == ARTICLE.encode()


async def test_each_call_logs_the_deprecation(kb, monkeypatch):
    from agentic_cli.tools import knowledge_tools

    events = []

    class _Log:
        def warning(self, event, **kw):
            events.append(event)

        debug = info = warning

    monkeypatch.setattr(knowledge_tools, "logger", _Log())
    with pytest.warns(DeprecationWarning):
        await _ingest(_page())

    assert "kb_ingest_url_deprecated" in events


async def test_a_type_the_knowledge_base_cannot_ingest_is_refused(kb):
    with pytest.warns(DeprecationWarning):
        result = await _ingest(FetchResult(
            success=True, content="\x89PNG", content_type="image/png", raw=b"\x89PNG", final_url=URL,
        ))

    assert result["success"] is False
    assert "image/png" in result["error"]
    assert kb.list_documents() == []


async def test_a_result_without_the_raw_body_still_ingests(kb):
    with pytest.warns(DeprecationWarning):
        result = await _ingest(FetchResult(success=True, content=ARTICLE, content_type="text/html"))

    assert result["success"] is True
    assert "<p>" not in kb.get_document(result["document_id"]).content


def test_the_description_says_deprecated():
    from agentic_cli.tools.knowledge_tools import kb_ingest_url
    from agentic_cli.tools.registry import get_registry

    assert kb_ingest_url.__doc__.lstrip().startswith("Deprecated")
    assert "kb_ingest_file" in kb_ingest_url.__doc__
    assert get_registry().get("kb_ingest_url").description.startswith("Deprecated")


def test_it_stays_in_the_writer_bundle_until_0_7_0():
    from agentic_cli.tools import KB_WRITER_TOOLS, kb_ingest_url

    assert kb_ingest_url in KB_WRITER_TOOLS


async def test_the_no_content_error_points_to_web_fetch():
    from agentic_cli.tools.knowledge_tools import _finalize_ingest

    result = await _finalize_ingest(
        None, content="", title="", source_type="user", source_url=None,
        meta={}, file_bytes=None, file_extension=".txt",
    )

    assert "kb_ingest_url" not in result["error"]
    assert "web_fetch" in result["error"]

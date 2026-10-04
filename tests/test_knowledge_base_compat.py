"""The old knowledge-base import paths keep working until 0.7.0.

The knowledge base moved to ``agentic_cli.memory.kb``. ``agentic_cli.knowledge_base``
forwards to it and warns once when first imported.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.service_registry import LLM_SUMMARIZER, set_service_registry

OLD_NAMES = [
    ("agentic_cli.knowledge_base", name)
    for name in (
        "KnowledgeBaseManager", "Document", "DocumentChunk", "PaperResult", "SearchResult",
        "SourceType", "WebResult", "EmbeddingService", "VectorStore", "SearchSource",
        "SearchSourceResult", "create_bm25_index",
    )
] + [
    ("agentic_cli.knowledge_base.manager", "KnowledgeBaseManager"),
    ("agentic_cli.knowledge_base.manager", "BackfillAlreadyRunning"),
    ("agentic_cli.knowledge_base.manager", "matches_document_filters"),
    ("agentic_cli.knowledge_base.models", "SourceType"),
    ("agentic_cli.knowledge_base.models", "PaperResult"),
    ("agentic_cli.knowledge_base.embeddings", "EmbeddingService"),
    ("agentic_cli.knowledge_base.embeddings", "resolve_embedding_device"),
    ("agentic_cli.knowledge_base.vector_store", "VectorStore"),
    ("agentic_cli.knowledge_base.bm25_index", "create_bm25_index"),
    ("agentic_cli.knowledge_base.bm25_index", "INDEX_FILES"),
    ("agentic_cli.knowledge_base.concepts", "ConceptStore"),
    ("agentic_cli.knowledge_base.sidecar", "render_sidecar_markdown"),
    ("agentic_cli.knowledge_base.sources", "SearchSource"),
    ("agentic_cli.knowledge_base.sources", "SearchSourceResult"),
    ("agentic_cli.knowledge_base._mocks", "MockEmbeddingService"),
    ("agentic_cli.knowledge_base._mocks", "MockVectorStore"),
]

SAME_OBJECTS = [
    ("agentic_cli.knowledge_base", "agentic_cli.memory.kb", "SourceType"),
    ("agentic_cli.knowledge_base", "agentic_cli.memory", "EmbeddingService"),
    ("agentic_cli.knowledge_base.manager", "agentic_cli.memory.kb", "BackfillAlreadyRunning"),
    ("agentic_cli.knowledge_base.sources", "agentic_cli.tools.search_sources", "SearchSource"),
    ("agentic_cli.knowledge_base._mocks", "agentic_cli.memory", "MockEmbeddingService"),
    ("agentic_cli.knowledge_base._mocks", "agentic_cli.memory.kb", "MockVectorStore"),
]


def _old(module: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(module)


@pytest.mark.parametrize("module, name", OLD_NAMES)
def test_every_old_name_still_imports(module, name):
    assert hasattr(_old(module), name)


@pytest.mark.parametrize("old, new, name", SAME_OBJECTS)
def test_old_names_are_the_moved_objects(old, new, name):
    assert getattr(_old(old), name) is getattr(importlib.import_module(new), name)


@pytest.mark.parametrize("import_statement", [
    "import agentic_cli.knowledge_base",
    "import agentic_cli.knowledge_base.manager",
])
def test_the_first_import_of_the_old_package_warns(import_statement):
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    {import_statement}\n"
        "warnings_list = [w for w in caught if w.category is DeprecationWarning and 'agentic_cli.memory.kb' in str(w.message)]\n"
        "print(len(warnings_list))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "1", f"Expected exactly 1 DeprecationWarning, got: {out.stdout}"


class _Summarizer:
    async def summarize(self, content: str, prompt: str) -> str:
        return "SUMMARY: From the turn.\n"


def _old_class():
    return _old("agentic_cli.knowledge_base").KnowledgeBaseManager


def test_the_old_class_is_a_subclass_of_the_new_one():
    from agentic_cli.memory.kb import KnowledgeBaseManager as New

    assert issubclass(_old_class(), New) and _old_class() is not New


def test_the_old_signature_reads_paths_and_embedding_from_settings(tmp_path):
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", embedding_model="m-legacy")

    kb = _old_class()(settings=settings, use_mock=True)

    assert kb.kb_dir == settings.knowledge_base_dir
    assert kb.get_stats()["embedding_model"] == "m-legacy"


def test_base_dir_overrides_settings_paths(tmp_path):
    """base_dir still wins over settings' paths when both are given (it did
    before the new constructor; a mutation that let settings override
    base_dir passed the whole suite otherwise)."""
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", embedding_model="m-legacy")

    kb = _old_class()(settings=settings, base_dir=tmp_path / "kb", use_mock=True)

    assert kb.kb_dir == tmp_path / "kb"
    assert not settings.knowledge_base_dir.exists()
    assert kb.get_stats()["embedding_model"] == "m-legacy"


def test_the_old_default_directory_without_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    kb = _old_class()(use_mock=True)

    assert kb.kb_dir == tmp_path / ".agentic" / "knowledge_base"


async def test_the_old_class_uses_the_turns_summarizer(tmp_path):
    kb = _old_class()(base_dir=tmp_path / "kb", use_mock=True)

    outside = await kb.generate_sidecar_payload("Body text.", title="T")
    token = set_service_registry({LLM_SUMMARIZER: _Summarizer()})
    try:
        inside = await kb.generate_sidecar_payload("Body text.", title="T")
    finally:
        token.var.reset(token)

    assert outside["summary"] == "Body text."
    assert inside["summary"] == "From the turn."


def test_the_old_class_keeps_extract_text_from_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agentic_cli.tools.pdf_utils.extract_pdf_text",
        lambda source, **kwargs: f"text of {Path(source).name}",
    )

    assert _old_class().extract_text_from_pdf(tmp_path / "paper.pdf") == "text of paper.pdf"


def test_data_written_through_the_old_class_opens_with_the_new_one(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBaseManager as New
    from agentic_cli.memory.kb import SourceType

    old = _old_class()(base_dir=tmp_path / "kb", use_mock=True)
    doc = old.ingest_document(content="Shared text about otters.", title="Otters", source_type=SourceType.USER)

    new = New(tmp_path / "kb", use_mock=True)
    assert new.get_document(doc.id).title == "Otters"
    hits = new.search("otters", top_k=1)["results"]
    assert [hit["document_title"] for hit in hits] == ["Otters"]


async def test_the_old_class_with_an_explicit_none_summarizer_stores_the_preview(tmp_path):
    """Controller ruling: the old path also accepts ``summarizer=`` directly,
    overriding the default turn-registry lookup."""
    kb = _old_class()(base_dir=tmp_path / "kb", use_mock=True, summarizer=None)

    token = set_service_registry({LLM_SUMMARIZER: _Summarizer()})
    try:
        payload = await kb.generate_sidecar_payload("Body text.", title="T")
    finally:
        token.var.reset(token)

    assert payload == {"summary": "Body text.", "claims": [], "entities": {}}

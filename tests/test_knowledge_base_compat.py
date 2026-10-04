"""The old knowledge-base import paths keep working until 0.7.0.

The knowledge base moved to ``agentic_cli.memory.kb``. ``agentic_cli.knowledge_base``
forwards to it and warns once when first imported.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import pytest

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
    ("agentic_cli.knowledge_base", "agentic_cli.memory.kb", "KnowledgeBaseManager"),
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

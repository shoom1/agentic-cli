"""What ``agentic_cli.memory`` and ``agentic_cli.memory.kb`` export in 0.6.3.

0.6.4 keeps the 0.6.3 package API and redesigns it in 0.7.0. New names may
be added. None of these may go, and these classes and methods must stay
call-compatible: the same parameters, in the same order, of the same kind,
with the same defaults. Annotations are not compared.
"""

from __future__ import annotations

import inspect

import pytest

import agentic_cli.memory as memory
import agentic_cli.memory.kb as kb

MEMORY_EXPORTS = {
    "EmbeddingConfig",
    "EmbeddingService",
    "ForgettingPolicy",
    "MemoryItem",
    "MemoryStore",
    "MockEmbeddingService",
    "resolve_embedding_device",
}

KB_EXPORTS = {
    "BackfillAlreadyRunning",
    "ConceptStore",
    "Document",
    "DocumentChunk",
    "KnowledgeBaseManager",
    "MockVectorStore",
    "SearchResult",
    "SourceType",
    "Summarizer",
    "VectorStore",
    "create_bm25_index",
    "matches_document_filters",
}

CLASSES = {
    "KnowledgeBaseManager": kb.KnowledgeBaseManager,
    "MemoryStore": memory.MemoryStore,
    "EmbeddingService": memory.EmbeddingService,
    "EmbeddingConfig": memory.EmbeddingConfig,
}

PROPERTIES = {
    ("KnowledgeBaseManager", "concepts"),
    ("EmbeddingService", "embedding_dim"),
    ("EmbeddingService", "model"),
}

R = "<required>"
O = "<object>"  # a default that is not None, a bool, a number or a string
P, K = "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"

# (class, method): [(parameter, kind, default)]; method "" is the class itself.
SIGNATURES = {
    ("KnowledgeBaseManager", "__init__"): [
        ("self", P, R), ("base_dir", P, R), ("embedding", K, None), ("summarizer", K, None),
        ("use_mock", K, False), ("embedding_service", K, None), ("vector_store", K, None),
    ],
    ("KnowledgeBaseManager", "backfill_sidecars"): [("self", P, R), ("progress_cb", P, None)],
    ("KnowledgeBaseManager", "clear"): [("self", P, R)],
    ("KnowledgeBaseManager", "delete_document"): [("self", P, R), ("doc_id", P, R)],
    ("KnowledgeBaseManager", "find_document"): [("self", P, R), ("id_or_title", P, R)],
    ("KnowledgeBaseManager", "generate_sidecar_payload"): [("self", P, R), ("content", P, R), ("title", P, "")],
    ("KnowledgeBaseManager", "generate_summary"): [("self", P, R), ("content", P, R), ("title", P, "")],
    ("KnowledgeBaseManager", "get_document"): [("self", P, R), ("doc_id", P, R)],
    ("KnowledgeBaseManager", "get_file_path"): [("self", P, R), ("doc_id", P, R)],
    ("KnowledgeBaseManager", "get_or_create_sidecar_lock"): [("self", P, R), ("doc_id", P, R)],
    ("KnowledgeBaseManager", "get_stats"): [("self", P, R)],
    ("KnowledgeBaseManager", "ingest_document"): [
        ("self", P, R), ("content", P, R), ("title", P, R), ("source_type", P, R),
        ("source_url", P, None), ("metadata", P, None), ("file_bytes", P, None),
        ("file_extension", P, ".pdf"), ("chunk_size", P, 512), ("chunk_overlap", P, 50),
        ("summary", P, None), ("sidecar_payload", P, None),
    ],
    ("KnowledgeBaseManager", "list_documents"): [("self", P, R), ("source_type", P, None), ("limit", P, 100)],
    ("KnowledgeBaseManager", "search"): [("self", P, R), ("query", P, R), ("filters", P, None), ("top_k", P, 10)],
    ("KnowledgeBaseManager", "sidecar_path"): [("self", P, R), ("doc_id", P, R)],
    ("KnowledgeBaseManager", "store_file"): [
        ("self", P, R), ("doc_id", P, R), ("file_bytes", P, R), ("extension", P, R),
    ],
    ("KnowledgeBaseManager", "write_sidecar_if_present"): [("self", P, R), ("doc", P, R), ("payload", P, R)],
    ("MemoryStore", "__init__"): [("self", P, R), ("base_dir", P, R), ("embedding_service", P, None)],
    ("MemoryStore", "apply_forgetting"): [("self", P, R), ("policy", P, R)],
    ("MemoryStore", "delete"): [("self", P, R), ("item_id", P, R), ("purge", P, False)],
    ("MemoryStore", "load_all"): [("self", P, R)],
    ("MemoryStore", "search"): [("self", P, R), ("query", P, R), ("limit", P, 10), ("include_archived", P, False)],
    ("MemoryStore", "store"): [("self", P, R), ("content", P, R), ("tags", P, None), ("importance", P, 5)],
    ("MemoryStore", "store_with_similarity_check"): [
        ("self", P, R), ("content", P, R), ("tags", P, None), ("importance", P, 5),
        ("similarity_threshold", P, 0.85),
    ],
    ("MemoryStore", "update"): [("self", P, R), ("item_id", P, R), ("content", P, None), ("tags", P, O)],
    ("EmbeddingService", "__init__"): [
        ("self", P, R), ("model_name", P, "all-MiniLM-L6-v2"), ("batch_size", P, 32), ("device", P, "auto"),
    ],
    ("EmbeddingService", "chunk_document"): [
        ("self", P, R), ("content", P, R), ("chunk_size", P, 512), ("overlap", P, 50),
    ],
    ("EmbeddingService", "embed_batch"): [("self", P, R), ("texts", P, R)],
    ("EmbeddingService", "embed_text"): [("self", P, R), ("text", P, R)],
    ("EmbeddingService", "is_available"): [],
    ("EmbeddingConfig", ""): [("model_name", P, "all-MiniLM-L6-v2"), ("batch_size", P, 32), ("device", P, "auto")],
}


def _shape(fn) -> list[tuple[str, str, object]]:
    shape = []
    for p in inspect.signature(fn).parameters.values():
        if p.default is inspect.Parameter.empty:
            default = R
        elif p.default is None or isinstance(p.default, (bool, int, float, str)):
            default = p.default
        else:
            default = O
        shape.append((p.name, p.kind.name, default))
    return shape


def test_agentic_cli_memory_still_exports_its_0_6_3_names():
    assert MEMORY_EXPORTS <= set(memory.__all__)
    assert all(hasattr(memory, name) for name in MEMORY_EXPORTS)


def test_agentic_cli_memory_kb_still_exports_its_0_6_3_names():
    assert KB_EXPORTS <= set(kb.__all__)
    assert all(hasattr(kb, name) for name in KB_EXPORTS)


@pytest.mark.parametrize(("cls", "method"), sorted(SIGNATURES))
def test_the_signature_is_unchanged(cls, method):
    target = getattr(CLASSES[cls], method) if method else CLASSES[cls]
    assert _shape(target) == SIGNATURES[(cls, method)]


@pytest.mark.parametrize(("cls", "name"), sorted(PROPERTIES))
def test_the_property_is_still_a_property(cls, name):
    assert isinstance(inspect.getattr_static(CLASSES[cls], name), property)

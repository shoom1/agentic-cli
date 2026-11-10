"""What ``agentic_cli.memory`` and ``agentic_cli.memory.kb`` export in 0.7.0.

0.6.4 kept the 0.6.3 package API. 0.7.0 moves the knowledge base onto one
SQLite file and drops ``VectorStore``, ``MockVectorStore``,
``create_bm25_index``, ``matches_document_filters`` and
``KnowledgeBaseManager(vector_store=)``, and adds ``KnowledgeBaseBusy``. 0.7.0
also replaces the memory store: memories are markdown files in two scopes;
``MemoryItem`` and ``ForgettingPolicy`` are gone, ``MemoryStore`` takes the
scope folders. New names may be added. None of these may go, and these classes and methods must
stay call-compatible: the same parameters, in the same order, of the same
kind, with the same defaults. Annotations are not compared.
"""

from __future__ import annotations

import inspect

import pytest

import agentic_cli.memory as memory
import agentic_cli.memory.kb as kb

MEMORY_EXPORTS = {
    "ALL_MEMORY_TOOLS",
    "Candidate",
    "DuplicateMemory",
    "EmbeddingConfig",
    "EmbeddingService",
    "FileInTheWay",
    "InvalidMemory",
    "MAX_CANDIDATES",
    "MEMORY_TYPES",
    "Memory",
    "MemoryBusy",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryStoreFull",
    "MockEmbeddingService",
    "NoProjectScope",
    "Problem",
    "Proposal",
    "SCOPES",
    "UnsafeMemoryStore",
    "Version",
    "extraction_prompt",
    "looks_like_secret",
    "parse_candidates",
    "resolve_embedding_device",
}

KB_EXPORTS = {
    "BackfillAlreadyRunning",
    "ConceptStore",
    "Document",
    "DocumentChunk",
    "KnowledgeBaseBusy",
    "KnowledgeBaseManager",
    "SearchResult",
    "SourceType",
    "Summarizer",
}

CLASSES = {
    "KnowledgeBaseManager": kb.KnowledgeBaseManager,
    "MemoryStore": memory.MemoryStore,
    "EmbeddingService": memory.EmbeddingService,
    "EmbeddingConfig": memory.EmbeddingConfig,
}

PROPERTIES = {
    ("KnowledgeBaseManager", "concepts"),
    ("KnowledgeBaseManager", "summarizer"),
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
        ("use_mock", K, False), ("embedding_service", K, None), ("within", K, None),
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
    ("MemoryStore", "__init__"): [
        ("self", P, R), ("user_dir", P, R), ("project_dir", P, None), ("project_name", K, None),
        ("project_path", K, None), ("screen", K, O), ("now", K, None),
    ],
    ("MemoryStore", "save"): [
        ("self", P, R), ("description", P, R), ("content", P, R), ("type", P, R),
        ("scope", K, "project"), ("source", K, "agent"), ("conversation", K, None),
    ],
    ("MemoryStore", "read"): [("self", P, R), ("memory_id", P, R)],
    ("MemoryStore", "update"): [
        ("self", P, R), ("memory_id", P, R), ("description", K, None), ("content", K, None), ("type", K, None),
    ],
    ("MemoryStore", "delete"): [("self", P, R), ("memory_id", P, R)],
    ("MemoryStore", "memories"): [("self", P, R), ("scope", P, None)],
    ("MemoryStore", "search"): [("self", P, R), ("query", P, R), ("limit", K, 10)],
    ("MemoryStore", "index"): [("self", P, R), ("max_lines", K, 200), ("max_bytes", K, 25_000), ("tools", K, O)],
    ("MemoryStore", "problems"): [("self", P, R)],
    ("MemoryStore", "propose"): [
        ("self", P, R), ("candidate", P, R), ("conversation", K, None), ("outside_content", K, None),
        ("source", K, "extractor"),
    ],
    ("MemoryStore", "proposals"): [("self", P, R), ("scope", P, None)],
    ("MemoryStore", "accept"): [("self", P, R), ("proposal_id", P, R)],
    ("MemoryStore", "discard"): [("self", P, R), ("proposal_id", P, R)],
    ("MemoryStore", "discarded"): [("self", P, R)],
    ("MemoryStore", "history"): [("self", P, R), ("memory_id", P, None), ("days", K, 30)],
    ("MemoryStore", "restore"): [("self", P, R), ("memory_id", P, R), ("saved", P, R)],
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


def test_agentic_cli_memory_exports_its_0_7_0_names():
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

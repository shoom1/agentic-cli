"""Memory for agents: a knowledge base and a memory store.

One package with independent components. It imports nothing from the rest of
agentic-cli, and its components do not import each other
(``tests/memory/test_boundary.py`` checks both):

- ``agentic_cli.memory``: the memory store (markdown files in a user scope
  and a project scope; provisional API), its extraction prompt, the screen
  for text that looks like a credential, and the embedding service the
  knowledge base uses.
- ``agentic_cli.memory.kb``: the knowledge base.
"""

from agentic_cli.memory._core.embeddings import (
    EmbeddingConfig,
    EmbeddingService,
    resolve_embedding_device,
)
from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService
from agentic_cli.memory._core.secrets import looks_like_secret
from agentic_cli.memory.store import (
    ALL_MEMORY_TOOLS,
    INDEX_OPEN,
    MAX_CANDIDATES,
    MEMORY_TYPES,
    SCOPES,
    Candidate,
    DuplicateMemory,
    FileInTheWay,
    InvalidMemory,
    Memory,
    MemoryBusy,
    MemoryStore,
    MemoryStoreError,
    MemoryStoreFull,
    NoProjectScope,
    Problem,
    Proposal,
    UnsafeMemoryStore,
    Version,
    extraction_prompt,
    parse_candidates,
)

__all__ = [
    "ALL_MEMORY_TOOLS",
    "Candidate",
    "DuplicateMemory",
    "EmbeddingConfig",
    "EmbeddingService",
    "FileInTheWay",
    "INDEX_OPEN",
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
]

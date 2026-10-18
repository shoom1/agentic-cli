"""Where agentic-cli's settings become arguments for ``agentic_cli.memory``.

The memory package takes plain arguments (directories, an embedding service, a
summarizer). This module is the one place that reads them from settings, so
the package never sees a settings object.
"""

from __future__ import annotations

import warnings

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.memory import EmbeddingConfig, EmbeddingService, MemoryStore
    from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager


def project_kb_dir(settings: "BaseSettings") -> Path:
    """The project's knowledge base: ``./.{app_name}/knowledge_base``."""
    return Path.cwd() / f".{settings.app_name}" / "knowledge_base"


def embedding_config(settings: "BaseSettings") -> "EmbeddingConfig":
    """The embedding model the settings name."""
    from agentic_cli.memory import EmbeddingConfig

    return EmbeddingConfig(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def build_embedding_service(settings: "BaseSettings") -> "EmbeddingService | None":
    """The embedding service the settings name, or None.

    None in mock mode (``knowledge_base_use_mock``) and without
    sentence-transformers. The model loads on first use, not here. The
    workflow builds one and passes it to both knowledge bases and the memory
    store, so the model loads once per process.
    """
    if settings.knowledge_base_use_mock:
        return None
    from agentic_cli.memory import EmbeddingService

    if not EmbeddingService.is_available():
        return None
    config = embedding_config(settings)
    return EmbeddingService(
        model_name=config.model_name, batch_size=config.batch_size, device=config.device
    )


def build_knowledge_base(
    settings: "BaseSettings",
    base_dir: Path,
    *,
    summarizer: Any = None,
    use_mock: bool | None = None,
    embedding_service: Any = None,
) -> "KnowledgeBaseManager":
    """One knowledge base in ``base_dir``; ``use_mock`` defaults to the setting."""
    from agentic_cli.memory.kb import KnowledgeBaseManager

    if use_mock is None:
        use_mock = settings.knowledge_base_use_mock
    return KnowledgeBaseManager(
        base_dir,
        embedding=embedding_config(settings),
        summarizer=summarizer,
        use_mock=use_mock,
        embedding_service=embedding_service,
    )


_SCOPES = ("project", "user")


def open_knowledge_base(
    settings: "BaseSettings",
    *,
    summarizer: Any,
    embedding_service: Any = None,
    scopes: tuple[str, ...] = _SCOPES,
) -> "KnowledgeBase":
    """The knowledge base over the project and user scopes.

    The project scope is ``./.{app_name}/knowledge_base``, the user scope
    ``knowledge_base_dir``; when both resolve to one directory they are one
    scope. It gets ``summarizer`` only when ``knowledge_base_summarize`` is
    on, and shares ``embedding_service`` when one is given (else it builds
    one, except in mock mode). ``scopes`` opens only some of them, for a
    command that lists one.
    """
    from agentic_cli.memory.kb import KnowledgeBase

    dirs = {"project": project_kb_dir(settings), "user": settings.knowledge_base_dir}
    unknown = [name for name in scopes if name not in dirs]
    if unknown:
        raise ValueError(f"unknown knowledge-base scope(s): {', '.join(unknown)}; scopes: {', '.join(dirs)}")
    if not settings.knowledge_base_summarize:
        summarizer = None
    return KnowledgeBase.open(
        {name: dirs[name] for name in scopes},
        embedding_service=embedding_service,
        embedding=embedding_config(settings),
        summarizer=summarizer,
        use_mock=settings.knowledge_base_use_mock,
    )


def build_knowledge_bases(
    settings: "BaseSettings", *, summarizer: Any, embedding_service: Any = None
) -> tuple["KnowledgeBaseManager", "KnowledgeBaseManager"]:
    """Deprecated: use :func:`open_knowledge_base`. Removed in 0.7.0.

    The project and user knowledge bases (one object if they share a directory).
    """
    warnings.warn(
        "memory_services.build_knowledge_bases is deprecated and will be removed in 0.7.0; "
        "use memory_services.open_knowledge_base",
        DeprecationWarning,
        stacklevel=2,
    )
    kb = open_knowledge_base(settings, summarizer=summarizer, embedding_service=embedding_service)
    return kb.manager("project"), kb.manager("user")


def open_memory_store(settings: "BaseSettings", *, embedding_service: Any = None) -> "MemoryStore":
    """The memory store in ``workspace_dir/memory``.

    It uses ``embedding_service`` when given (the workflow passes the one its
    knowledge bases use); otherwise it builds one with
    :func:`build_embedding_service`. With none (mock mode, or no
    sentence-transformers) it searches by substring. (The mock embedder's
    vectors are hashes of the text: ranking memories by them would return
    arbitrary ones.)
    """
    from agentic_cli.memory import MemoryStore

    if embedding_service is None:
        embedding_service = build_embedding_service(settings)
    return MemoryStore(settings.workspace_dir / "memory", embedding_service=embedding_service)


def build_memory_store(settings: "BaseSettings") -> "MemoryStore":
    """Deprecated: use :func:`open_memory_store`. Removed in 0.7.0."""
    warnings.warn(
        "memory_services.build_memory_store is deprecated and will be removed in 0.7.0; "
        "use memory_services.open_memory_store",
        DeprecationWarning,
        stacklevel=2,
    )
    return open_memory_store(settings)

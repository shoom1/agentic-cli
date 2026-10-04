"""Where agentic-cli's settings become arguments for ``agentic_cli.memory``.

The memory package takes plain arguments (directories, an embedding service, a
summarizer). This module is the one place that reads them from settings, so
the package never sees a settings object.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.memory import EmbeddingConfig, MemoryStore
    from agentic_cli.memory.kb import KnowledgeBaseManager


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


def build_knowledge_base(
    settings: "BaseSettings",
    base_dir: Path,
    *,
    summarizer: Any = None,
    use_mock: bool | None = None,
) -> "KnowledgeBaseManager":
    """One knowledge base in ``base_dir``; ``use_mock`` defaults to the setting."""
    from agentic_cli.memory.kb import KnowledgeBaseManager

    if use_mock is None:
        use_mock = settings.knowledge_base_use_mock
    return KnowledgeBaseManager(
        base_dir, embedding=embedding_config(settings), summarizer=summarizer, use_mock=use_mock
    )


def build_knowledge_bases(
    settings: "BaseSettings", *, summarizer: Any
) -> tuple["KnowledgeBaseManager", "KnowledgeBaseManager"]:
    """The project and user knowledge bases (one object if they share a directory).

    They get ``summarizer`` only when ``knowledge_base_summarize`` is on.
    """
    if not settings.knowledge_base_summarize:
        summarizer = None
    project_dir = project_kb_dir(settings)
    user_dir = settings.knowledge_base_dir
    project = build_knowledge_base(settings, project_dir, summarizer=summarizer)
    if project_dir.resolve() == user_dir.resolve():
        return project, project
    return project, build_knowledge_base(settings, user_dir, summarizer=summarizer)


def build_memory_store(settings: "BaseSettings") -> "MemoryStore":
    """The memory store in ``workspace_dir/memory``.

    With ``knowledge_base_use_mock`` it gets the mock embedder; otherwise the
    real one when sentence-transformers is installed, else none (substring
    search).
    """
    from agentic_cli.memory import MemoryStore

    embedding_service = None
    if settings.knowledge_base_use_mock:
        from agentic_cli.memory import MockEmbeddingService

        embedding_service = MockEmbeddingService()
    else:
        from agentic_cli.memory import EmbeddingService

        config = embedding_config(settings)
        if EmbeddingService.is_available():
            embedding_service = EmbeddingService(
                model_name=config.model_name, batch_size=config.batch_size, device=config.device
            )
    return MemoryStore(settings.workspace_dir / "memory", embedding_service=embedding_service)

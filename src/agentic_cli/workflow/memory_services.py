"""Where agentic-cli's settings become arguments for ``agentic_cli.memory``.

The memory package takes plain arguments (directories, an embedding service, a
summarizer). This module is the one place that reads them from settings, so
the package never sees a settings object.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.memory import EmbeddingConfig, EmbeddingService, MemoryStore
    from agentic_cli.memory.kb import KnowledgeBase


logger = Loggers.workflow()


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
    workflow builds one and passes it to both knowledge bases, so the model
    loads once per process.
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
    scope. A cloned repository can ship ``./.{app_name}`` or its
    ``knowledge_base`` as a symlink, so the project scope opens only when
    both are real folders (else it is left out, with a warning); the user
    scope's folder may be a link (to another disk, say). It gets
    ``summarizer`` only when ``knowledge_base_summarize`` is on, and shares
    ``embedding_service`` when one is given (else it builds one, except in
    mock mode). ``scopes`` opens only some of them, for a command that lists
    one.
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
        within={"project": Path.cwd()},
    )


def project_key(path: Path) -> str:
    """A project scope's folder name: the folder's name (at most 40 of
    ``[A-Za-z0-9._-]``) and 12 hex characters of the SHA-256 of its path."""
    name = re.sub(r"[^A-Za-z0-9._-]", "-", path.name)[:40] or "root"
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}"


def project_memory_dir(settings: "BaseSettings", cwd: Path | None = None) -> Path | None:
    """The project memory scope's folder for the working directory, or None.

    None at ``/``, the home folder or above it (a project there would be all
    of home: the rule the project read allow uses).
    """
    cwd = (cwd or Path.cwd()).resolve()
    home = Path.home().resolve()
    if cwd == Path(cwd.anchor) or home.is_relative_to(cwd):
        return None
    return settings.workspace_dir / "projects" / project_key(cwd) / "memory"


def open_memory_store(settings: "BaseSettings") -> "MemoryStore | None":
    """The memory store: the user scope in ``workspace_dir/memory``, the
    project scope in ``workspace_dir/projects/<key>/memory`` (none at home,
    ``/`` or above home). It searches by keyword and needs no embedding
    service.

    A memory folder that cannot open never stops the app (a link, which is
    refused, or a folder that cannot be written). A project scope that fails
    is left out, with a warning (``memory_project_scope_unavailable``) and
    the user scope still works; a user scope that fails gives None, with a
    warning (``memory_store_unavailable``), and the memory tools answer that
    the store is not available.
    """
    from agentic_cli.memory import MemoryStore, MemoryStoreError

    cwd = Path.cwd().resolve()
    project_dir = project_memory_dir(settings, cwd)
    user_dir = settings.workspace_dir / "memory"
    try:
        return MemoryStore(
            user_dir,
            project_dir,
            project_name=cwd.name if project_dir is not None else None,
            project_path=cwd if project_dir is not None else None,
        )
    except (MemoryStoreError, OSError) as e:
        failure = e
    if project_dir is not None:
        # Which scope failed? The user scope alone says.
        try:
            store = MemoryStore(user_dir)
        except (MemoryStoreError, OSError):
            pass
        else:
            logger.warning(
                "memory_project_scope_unavailable", path=str(project_dir),
                error=f"{type(failure).__name__}: {failure}",
            )
            return store
    logger.warning(
        "memory_store_unavailable", path=str(user_dir), error=f"{type(failure).__name__}: {failure}"
    )
    return None

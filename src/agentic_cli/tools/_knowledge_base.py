"""Finding the knowledge base the knowledge-base tools work on.

The workflow publishes a ``KnowledgeBase`` under ``knowledge_base``. Until
0.7.0 the factories also accept 0.6.3's managers, with a warning.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from agentic_cli.workflow.service_registry import KNOWLEDGE_BASE, get_service

if TYPE_CHECKING:
    from agentic_cli.memory.kb import KnowledgeBase

# 0.6.3's message for a missing knowledge base, kept word for word.
NO_KNOWLEDGE_BASE = "kb manager not available"


def _over(project: Any, user: Any) -> "KnowledgeBase":
    from agentic_cli.memory.kb import KnowledgeBase

    managers = {"project": project}
    if user is not None:
        managers["user"] = user
    return KnowledgeBase(managers)


def current_knowledge_base() -> "KnowledgeBase | None":
    """The workflow's knowledge base, ``None`` when the registry holds none."""
    return get_service(KNOWLEDGE_BASE)


def knowledge_base_argument(
    knowledge_base: Any, user_kb_manager: Any, kb_manager: Any, factory: str
) -> "KnowledgeBase":
    """A factory's knowledge-base argument: a ``KnowledgeBase``, or 0.6.3's managers.

    Managers (positional, or ``kb_manager=``) are wrapped in a
    ``KnowledgeBase`` with a ``DeprecationWarning``.
    """
    from agentic_cli.memory.kb import KnowledgeBase

    if isinstance(knowledge_base, KnowledgeBase):
        if user_kb_manager is not None or kb_manager is not None:
            raise TypeError(f"{factory}() takes a KnowledgeBase or 0.6.3's managers, not both")
        return knowledge_base
    if knowledge_base is not None and kb_manager is not None:
        raise TypeError(f"{factory}() got multiple values for the knowledge base")
    project = kb_manager if kb_manager is not None else knowledge_base
    if project is None:
        raise TypeError(f"{factory}() needs a KnowledgeBase")
    warnings.warn(
        f"{factory}() with KnowledgeBaseManager arguments is deprecated and will be removed "
        "in 0.7.0; pass a KnowledgeBase",
        DeprecationWarning,
        stacklevel=3,
    )
    return _over(project, user_kb_manager)

"""Finding the knowledge base the knowledge-base tools work on.

The workflow publishes a ``KnowledgeBase`` under ``knowledge_base``; the
factories take one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.workflow.service_registry import KNOWLEDGE_BASE, get_service

if TYPE_CHECKING:
    from agentic_cli.memory.kb import KnowledgeBase

# 0.6.3's message for a missing knowledge base, kept word for word.
NO_KNOWLEDGE_BASE = "kb manager not available"


def current_knowledge_base() -> "KnowledgeBase | None":
    """The workflow's knowledge base, ``None`` when the registry holds none."""
    return get_service(KNOWLEDGE_BASE)


def knowledge_base_argument(knowledge_base: Any, factory: str) -> "KnowledgeBase":
    """A factory's knowledge-base argument, which must be a ``KnowledgeBase``."""
    from agentic_cli.memory.kb import KnowledgeBase

    if not isinstance(knowledge_base, KnowledgeBase):
        raise TypeError(f"{factory}() needs a KnowledgeBase, got {type(knowledge_base).__name__}")
    return knowledge_base

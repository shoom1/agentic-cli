"""A service registry for the knowledge-base tools.

The tools read the ``KnowledgeBase`` under ``knowledge_base``; tests build
theirs here from one or two ``KnowledgeBaseManager``s.
"""

from __future__ import annotations

from typing import Any

from agentic_cli.memory.kb import KnowledgeBase


def kb_services(project: Any, user: Any = None, **extra: Any) -> dict[str, Any]:
    """A registry with a ``KnowledgeBase`` over ``project`` (and ``user``), plus ``extra``."""
    managers = {"project": project}
    if user is not None:
        managers["user"] = user
    return {"knowledge_base": KnowledgeBase(managers), **extra}

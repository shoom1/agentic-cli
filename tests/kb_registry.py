"""A service registry for the knowledge-base tools that does not use the 0.6.3 keys.

The tools read the ``KnowledgeBase`` under ``knowledge_base``. A registry
holding only ``kb_manager`` / ``user_kb_manager`` still works until 0.7.0
but warns, so tests that are not about that path build theirs here.
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

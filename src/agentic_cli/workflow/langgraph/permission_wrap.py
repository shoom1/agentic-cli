# src/agentic_cli/workflow/langgraph/permission_wrap.py
"""LangGraph tool wrapper that gates calls via PermissionEngine.

Adapter check order matches ADK's PermissionPlugin:
1. EXEMPT tool → returned unwrapped (no engine call ever).
2. Tool has no capability declaration → wrapper returns deny dict at call time.
3. Engine absent from service registry → wrapper runs the original tool (fallback).
4. Otherwise call engine.check(); return on allow, deny dict on deny.

Replaces LangGraphBuilder._wrap_for_confirmation once Task 25 swaps the
call site.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt
from agentic_cli.workflow.service_registry import get_service

_PERMISSION_ENGINE_KEY = "permission_engine"

logger = Loggers.workflow()


def wrap_tool_for_permission(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``tool`` so every call goes through ``PermissionEngine.check``.

    Exempt tools (``capabilities=EXEMPT``) are returned unmodified.
    Tools that never declared capabilities (``defn is None``) get a wrapper
    that returns a deny error dict, mirroring ADK.
    """
    name = getattr(tool, "__name__", str(tool))
    defn = get_registry().get(name)
    caps = defn.capabilities if defn else None

    if isinstance(caps, _CapabilityExempt):
        return tool

    is_async = asyncio.iscoroutinefunction(tool)

    @functools.wraps(tool)
    async def _guarded(*args: Any, **kwargs: Any) -> Any:
        if not caps:
            logger.warning("permission_undeclared", tool=name)
            return {
                "success": False,
                "error": "Permission denied: tool has no capability declaration",
            }
        engine = get_service(_PERMISSION_ENGINE_KEY)
        if engine is not None:
            result = await engine.check(name, caps, kwargs)
            if not result.allowed:
                return {"success": False, "error": f"Permission denied: {result.reason}"}
        if is_async:
            return await tool(*args, **kwargs)
        return tool(*args, **kwargs)

    return _guarded

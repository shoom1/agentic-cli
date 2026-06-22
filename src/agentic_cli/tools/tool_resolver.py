"""Resolve tool references to callables.

An ``AgentConfig.tools`` entry may be:

- a callable (function tool) or any already-built tool object -> returned as-is;
- a **bare name** (no dot) -> looked up in the framework ``ToolRegistry``
  (e.g. ``"kb_search"``);
- a **dotted path** -> imported (e.g. ``"my_pkg.tools.my_tool"``).

This mirrors ADK's own convention (bare name = built-in/registered, dotted =
import). Resolution must run before manager service-detection and tool
assembly, which key on ``tool.__name__``.
"""

from __future__ import annotations

import difflib
from importlib import import_module
from typing import Any, Callable

from agentic_cli.tools.registry import ToolRegistry, get_registry

_builtins_imported = False


def _ensure_builtin_tools_imported() -> None:
    """Import ``agentic_cli.tools`` once so framework tools self-register.

    The standard tools register via ``@register_tool`` at import time; importing
    the package top-level pulls them all in. Done lazily so callers that only
    use callables or dotted paths don't pay the import cost.
    """
    global _builtins_imported
    if _builtins_imported:
        return
    import_module("agentic_cli.tools")
    _builtins_imported = True


def _import_dotted(path: str) -> Any:
    """Import an object from a dotted path (``module.sub:obj`` style as ``a.b.c``)."""
    module_path, _, obj_name = path.rpartition(".")
    if not module_path or not obj_name:
        raise ValueError(f"Invalid dotted tool path: {path!r}")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Cannot import module {module_path!r} for tool {path!r}: {exc}"
        ) from exc
    try:
        return getattr(module, obj_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_path!r} has no attribute {obj_name!r} (tool {path!r})"
        ) from exc


def _unknown_name_error(name: str, registry: ToolRegistry) -> ValueError:
    """Build a helpful error for an unresolved bare tool name."""
    names = [d.name for d in registry.list_tools()]
    close = difflib.get_close_matches(name, names, n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return ValueError(
        f"Unknown tool name {name!r} (not in the tool registry).{hint} "
        "Use a fully-qualified dotted path for custom tools."
    )


def resolve_tool(
    ref: Callable[..., Any] | str | Any,
    registry: ToolRegistry | None = None,
) -> Any:
    """Resolve a single tool reference to a callable/tool object.

    Args:
        ref: A callable, an already-built tool object, a bare registry name, or
            a dotted import path.
        registry: Registry for bare-name lookups. Defaults to the global
            framework registry (auto-populated with built-in tools on miss).

    Returns:
        The resolved callable or tool object.

    Raises:
        ValueError: If a string ref cannot be resolved.
    """
    # Callables and any already-built tool objects pass through unchanged.
    if not isinstance(ref, str):
        return ref

    name = ref.strip()
    if not name:
        raise ValueError("Empty tool reference.")

    if "." in name:
        return _import_dotted(name)

    # Bare name -> registry lookup.
    reg = registry or get_registry()
    defn = reg.get(name)
    if defn is None and registry is None:
        # Default registry may not have imported the built-ins yet.
        _ensure_builtin_tools_imported()
        reg = get_registry()
        defn = reg.get(name)
    if defn is None:
        raise _unknown_name_error(name, reg)
    return defn.func


def resolve_tools(
    refs: list[Callable[..., Any] | str] | None,
    registry: ToolRegistry | None = None,
) -> list[Any]:
    """Resolve a list of tool references (``None`` -> empty list)."""
    return [resolve_tool(ref, registry) for ref in (refs or [])]

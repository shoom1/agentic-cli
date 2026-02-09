"""Tools module for agentic CLI applications.

Provides function tools for agents including Python execution and knowledge base access.

Tool System:
    The module provides a standardized tool system with:
    - ToolDefinition: Metadata-rich tool definitions
    - ToolError: Standard error class for consistent error handling
    - ToolResult: Standard result wrapper
    - ToolRegistry: Registry for tool management and discovery
    - register_tool: Decorator for easy tool registration
    - requires: Decorator to declare tool's manager requirements

Framework Tools:
    - memory_tools: Working and long-term memory tools
    - planning_tools: Flat markdown plan tools (save_plan, get_plan)
    - hitl_tools: Human-in-the-loop approval and checkpoint tools
    - web_search: Web search with pluggable backends (Tavily, Brave)

For resilience patterns, use tenacity, pybreaker, aiolimiter directly.
"""

import asyncio
import functools
from typing import Any, Callable, Literal, TypeVar

# Type for manager requirements
ManagerRequirement = Literal[
    "memory_manager", "plan_store", "task_store", "paper_store", "approval_manager", "checkpoint_manager", "llm_summarizer"
]

F = TypeVar("F", bound=Callable)


def requires(*managers: ManagerRequirement) -> Callable[[F], F]:
    """Decorator to declare a tool's manager requirements.

    Framework tools use this to declare what managers they need.
    The workflow manager scans tools for this metadata and auto-creates
    the required managers.

    Args:
        *managers: One or more manager requirements.

    Returns:
        Decorator that adds 'requires' attribute to the function.

    Example:
        @requires("memory_manager")
        def save_memory(content: str, tags: list[str] | None = None) -> dict:
            store = get_context_memory_store()
            ...
    """
    def decorator(func: F) -> F:
        func.requires = list(managers)  # type: ignore[attr-defined]
        return func
    return decorator


def require_context(
    context_name: str,
    getter: Callable[..., Any],
    error_message: str | None = None,
) -> Callable[[F], F]:
    """Guard decorator: returns error dict if getter() is None.

    Apply below @requires / @register_tool so it runs first (innermost).
    functools.wraps preserves __dict__ so .requires stays visible.

    Args:
        context_name: Human-readable name for error messages.
        getter: Zero-arg callable returning the context value or None.
        error_message: Custom error message (defaults to "{context_name} not available").
    """
    msg = error_message or f"{context_name} not available"

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if getter() is None:
                    return {"success": False, "error": msg}
                return await func(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if getter() is None:
                    return {"success": False, "error": msg}
                return func(*args, **kwargs)
            return sync_wrapper  # type: ignore[return-value]
    return decorator


from agentic_cli.tools.executor import SafePythonExecutor, MockPythonExecutor, ExecutionTimeoutError
from agentic_cli.tools.shell import shell_executor, is_shell_enabled

# File operation tools - READ (safe)
from agentic_cli.tools.file_read import read_file, diff_compare
from agentic_cli.tools.grep_tool import grep
from agentic_cli.tools.glob_tool import glob, list_dir

# File operation tools - WRITE (caution)
from agentic_cli.tools.file_write import write_file, edit_file

from agentic_cli.tools.knowledge_tools import (
    search_knowledge_base,
    ingest_to_knowledge_base,
)
from agentic_cli.tools.arxiv_tools import (
    search_arxiv,
    fetch_arxiv_paper,
    analyze_arxiv_paper,
)
from agentic_cli.tools.execution_tools import execute_python
from agentic_cli.tools.interaction_tools import ask_clarification
from agentic_cli.tools.search import web_search
from agentic_cli.tools.webfetch_tool import web_fetch
from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    ToolDefinition,
    ToolError,
    ToolResult,
    ToolRegistry,
    ErrorCode,
    get_registry,
    register_tool,
    with_result_wrapper,
)

# Re-export google_search_tool from ADK for convenience
from google.adk.tools import google_search as google_search_tool

__all__ = [
    # Registry classes
    "ToolCategory",
    "PermissionLevel",
    "ToolDefinition",
    "ToolError",
    "ToolResult",
    "ToolRegistry",
    "ErrorCode",
    "get_registry",
    "register_tool",
    "with_result_wrapper",
    # Manager requirements decorator
    "requires",
    "require_context",
    "ManagerRequirement",
    # Executor classes
    "SafePythonExecutor",
    "MockPythonExecutor",
    "ExecutionTimeoutError",
    # Shell executor
    "shell_executor",
    "is_shell_enabled",
    # File operations - READ tools (safe)
    "read_file",
    "diff_compare",
    "grep",
    "glob",
    "list_dir",
    # File operations - WRITE tools (caution)
    "write_file",
    "edit_file",
    # Web search (pluggable backends)
    "web_search",
    # Web fetch (content fetching and summarization)
    "web_fetch",
    # Search (ADK built-in - note: can't mix with function calling)
    "google_search_tool",
    # Standard tool functions (ready to use with agents)
    "search_knowledge_base",
    "ingest_to_knowledge_base",
    "search_arxiv",
    "fetch_arxiv_paper",
    "analyze_arxiv_paper",
    "execute_python",
    "ask_clarification",
    # Framework tool modules (lazy loaded)
    "memory_tools",
    "planning_tools",
    "task_tools",
    "hitl_tools",
    "paper_tools",
]


# Lazy loading for framework tool modules
_lazy_tool_modules = {
    "memory_tools": "agentic_cli.tools.memory_tools",
    "planning_tools": "agentic_cli.tools.planning_tools",
    "task_tools": "agentic_cli.tools.task_tools",
    "hitl_tools": "agentic_cli.tools.hitl_tools",
    "paper_tools": "agentic_cli.tools.paper_tools",
}


def __getattr__(name: str):
    """Lazy import for framework tool modules."""
    if name in _lazy_tool_modules:
        import importlib

        module = importlib.import_module(_lazy_tool_modules[name])
        globals()[name] = module  # Cache for future access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

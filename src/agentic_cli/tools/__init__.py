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
    - planning_tools: Task graph management tools
    - hitl_tools: Human-in-the-loop checkpoint and approval tools
    - web_search: Web search with pluggable backends (Tavily, Brave)

For resilience patterns, use tenacity, pybreaker, aiolimiter directly.
"""

from typing import Callable, Literal, TypeVar

# Type for manager requirements
ManagerRequirement = Literal[
    "memory_manager", "task_graph", "approval_manager", "checkpoint_manager", "llm_summarizer"
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
        def remember_context(key: str, value: str) -> dict:
            manager = get_context_memory_manager()
            ...
    """
    def decorator(func: F) -> F:
        func.requires = list(managers)  # type: ignore[attr-defined]
        return func
    return decorator

from agentic_cli.tools.executor import SafePythonExecutor, MockPythonExecutor
from agentic_cli.tools.shell import shell_executor
from agentic_cli.tools.file_ops import file_manager, diff_compare
from agentic_cli.tools.standard import (
    search_knowledge_base,
    ingest_to_knowledge_base,
    search_arxiv,
    fetch_arxiv_paper,
    analyze_arxiv_paper,
    execute_python,
    ask_clarification,
)
from agentic_cli.tools.search import web_search
from agentic_cli.tools.webfetch_tool import web_fetch
from agentic_cli.tools.registry import (
    ToolCategory,
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
    "ManagerRequirement",
    # Executor classes
    "SafePythonExecutor",
    "MockPythonExecutor",
    # Shell executor
    "shell_executor",
    # File operations
    "file_manager",
    "diff_compare",
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
    "hitl_tools",
]


# Lazy loading for framework tool modules
_lazy_tool_modules = {
    "memory_tools": "agentic_cli.tools.memory_tools",
    "planning_tools": "agentic_cli.tools.planning_tools",
    "hitl_tools": "agentic_cli.tools.hitl_tools",
}


def __getattr__(name: str):
    """Lazy import for framework tool modules."""
    if name in _lazy_tool_modules:
        import importlib

        module = importlib.import_module(_lazy_tool_modules[name])
        globals()[name] = module  # Cache for future access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Tools module for agentic CLI applications.

Provides function tools for agents including Python execution and knowledge base access.

Tool System:
    The module provides a standardized tool system with:
    - ToolDefinition: Metadata-rich tool definitions
    - ToolRegistry: Registry for tool management and discovery
    - register_tool: Decorator for easy tool registration

Framework Tools:
    - memory_tools: Working and long-term memory tools
    - hitl_tools: Human-in-the-loop approval tools
    - web_search: Web search with pluggable backends (Tavily, Brave)

For resilience patterns, use tenacity, pybreaker, aiolimiter directly.
"""

from agentic_cli.tools.executor import SafePythonExecutor, MockPythonExecutor, ExecutionTimeoutError
from agentic_cli.tools.shell import shell_executor, is_shell_enabled

# File operation tools - READ (safe)
from agentic_cli.tools.file_read import read_file, diff_compare
from agentic_cli.tools.grep_tool import grep
from agentic_cli.tools.glob_tool import glob, list_dir

# File operation tools - WRITE (caution)
from agentic_cli.tools.file_write import write_file, edit_file

from agentic_cli.tools.knowledge_tools import (
    kb_search,
    kb_ingest,
    kb_read,
    kb_list,
    kb_write_concept,
    kb_search_concepts,
)

KB_READER_TOOLS = [kb_search, kb_read, kb_list, kb_search_concepts]
KB_WRITER_TOOLS = [*KB_READER_TOOLS, kb_ingest, kb_write_concept]
from agentic_cli.tools.arxiv_tools import (
    search_arxiv,
    fetch_arxiv_paper,
)
from agentic_cli.tools.execution_tools import execute_python
from agentic_cli.tools.interaction_tools import ask_clarification
from agentic_cli.tools.search import web_search
from agentic_cli.tools.webfetch_tool import web_fetch
from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    ToolDefinition,
    ToolRegistry,
    get_registry,
    register_tool,
)

# Re-export google_search_tool from ADK for convenience
from google.adk.tools import google_search as google_search_tool

__all__ = [
    # Registry classes
    "ToolCategory",
    "PermissionLevel",
    "ToolDefinition",
    "ToolRegistry",
    "get_registry",
    "register_tool",
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
    "kb_search",
    "kb_ingest",
    "kb_read",
    "kb_list",
    "kb_write_concept",
    "kb_search_concepts",
    "KB_READER_TOOLS",
    "KB_WRITER_TOOLS",
    "search_arxiv",
    "fetch_arxiv_paper",
    "execute_python",
    "ask_clarification",
    # Framework tool modules (lazy loaded)
    "memory_tools",
    "hitl_tools",
    "sandbox_tools",
    "reflection_tools",
]


# Lazy loading for framework tool modules
_lazy_tool_modules = {
    "memory_tools": "agentic_cli.tools.memory_tools",
    "hitl_tools": "agentic_cli.tools.hitl_tools",
    "sandbox_tools": "agentic_cli.tools.sandbox",
    "reflection_tools": "agentic_cli.tools.reflection_tools",
}


def __getattr__(name: str):
    """Lazy import for framework tool modules."""
    if name in _lazy_tool_modules:
        import importlib

        module = importlib.import_module(_lazy_tool_modules[name])
        globals()[name] = module  # Cache for future access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

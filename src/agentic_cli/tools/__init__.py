"""Tools module for agentic CLI applications.

Provides function tools for agents including Python execution and knowledge base access.

Tool System:
    The module provides a standardized tool system with:
    - ToolDefinition: Metadata-rich tool definitions
    - ToolRegistry: Registry for tool management and discovery
    - register_tool: Decorator for easy tool registration
    - declare_tool: Declare a tool implemented only by backend-native variants

Framework Tools:
    - memory_tools: Working and long-term memory tools
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
    kb_ingest_text,
    kb_ingest_file,
    kb_ingest_url,
    kb_read,
    kb_list,
    kb_write_concept,
    kb_search_concepts,
)

KB_READER_TOOLS = [kb_search, kb_read, kb_list, kb_search_concepts]
KB_WRITER_TOOLS = [
    *KB_READER_TOOLS,
    kb_ingest_text,
    kb_ingest_file,
    kb_ingest_url,
    kb_write_concept,
]
from agentic_cli.tools.arxiv_tools import (
    search_arxiv,
    fetch_arxiv_paper,
)
from agentic_cli.tools.execution_tools import execute_python
from agentic_cli.tools.document import compile_document
from agentic_cli.tools.interaction_tools import ask_clarification

# Long-running job tools (generic observe-only management). Typed long-running
# tools that *start* work (e.g. run_shell_job) are application-provided — see
# examples/jobs_demo.py — because they choose what runs and how.
from agentic_cli.tools.jobs import (
    job_status,
    job_result,
    job_logs,
    job_cancel,
    job_list,
)

# Minimal companion to a typed long-running tool: job_status alone returns
# state + stdout tail + result-when-finished, keeping the agent's tool surface
# small (tool-selection quality drops past ~15-20 tools). Apps that want the
# LLM to also enumerate/cancel jobs can use JOB_MANAGEMENT_TOOLS instead — but
# listing/cancelling is usually a human job via the /jobs command.
JOB_TOOLS = [job_status]
JOB_MANAGEMENT_TOOLS = [job_status, job_result, job_logs, job_cancel, job_list]
from agentic_cli.tools.search import web_search
from agentic_cli.tools.webfetch_tool import web_fetch
from agentic_cli.tools.registry import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
    get_registry,
    declare_tool,
    register_tool,
)

__all__ = [
    # Registry classes
    "ToolCategory",
    "ToolDefinition",
    "ToolRegistry",
    "get_registry",
    "declare_tool",
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
    # Search (deprecated ADK re-export - removed in 0.7.0; use web_search)
    "google_search_tool",
    # Standard tool functions (ready to use with agents)
    "kb_search",
    "kb_ingest_text",
    "kb_ingest_file",
    "kb_ingest_url",
    "kb_read",
    "kb_list",
    "kb_write_concept",
    "kb_search_concepts",
    "KB_READER_TOOLS",
    "KB_WRITER_TOOLS",
    "search_arxiv",
    "fetch_arxiv_paper",
    "execute_python",
    "compile_document",
    "ask_clarification",
    # Long-running jobs (observe-only; typed starters are app-provided)
    "job_status",
    "job_result",
    "job_logs",
    "job_cancel",
    "job_list",
    "JOB_TOOLS",
    "JOB_MANAGEMENT_TOOLS",
    # Framework tool modules (lazy loaded)
    "memory_tools",
    "sandbox_tools",
]


# Lazy loading for framework tool modules
_lazy_tool_modules = {
    "memory_tools": "agentic_cli.tools.memory_tools",
    "sandbox_tools": "agentic_cli.tools.sandbox",
}

# Deprecated (removed in 0.7.0): a bare re-export of ADK's GoogleSearchTool
# singleton, never an Agentic CLI integration. Resolved lazily so importing this
# package does not warn, and so the deprecation is charged to the code that
# actually uses the name.
_GOOGLE_SEARCH_TOOL_DEPRECATION = (
    "agentic_cli.tools.google_search_tool is deprecated and will be removed in "
    "0.7.0. It only re-exports ADK's native GoogleSearchTool singleton, so it is "
    "backend-, model- and UI-specific rather than an end-to-end Agentic CLI "
    "integration. Use agentic_cli.tools.web_search, the supported "
    "framework-level alternative. Applications that intentionally want native "
    "ADK Google Search should import the class directly from ADK "
    "(from google.adk.tools.google_search_tool import GoogleSearchTool), "
    "instantiate and configure it as ADK's documentation describes, and are "
    "then responsible for ADK's model/tool constraints, grounding metadata, "
    "citations and rendering Search Suggestions when returned."
)


def __getattr__(name: str):
    """Lazy import for framework tool modules and deprecated re-exports."""
    if name in _lazy_tool_modules:
        import importlib

        module = importlib.import_module(_lazy_tool_modules[name])
        globals()[name] = module  # Cache for future access
        return module
    if name == "google_search_tool":
        import warnings

        from google.adk.tools import google_search

        warnings.warn(
            _GOOGLE_SEARCH_TOOL_DEPRECATION, DeprecationWarning, stacklevel=2
        )
        globals()[name] = google_search  # Cache: one import warns once
        return google_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

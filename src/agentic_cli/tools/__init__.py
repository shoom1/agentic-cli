"""Tools module for agentic CLI applications.

Provides function tools for agents including Python execution and knowledge base access.

Tool System:
    The module provides a standardized tool system with:
    - ToolDefinition: Metadata-rich tool definitions
    - ToolError: Standard error class for consistent error handling
    - ToolResult: Standard result wrapper
    - ToolRegistry: Registry for tool management and discovery
    - register_tool: Decorator for easy tool registration

For web search, use google_search_tool directly from google.adk.tools.
For resilience patterns, use tenacity, pybreaker, aiolimiter directly.
"""

from agentic_cli.tools.executor import SafePythonExecutor, MockPythonExecutor
from agentic_cli.tools.standard import (
    search_knowledge_base,
    ingest_to_knowledge_base,
    search_arxiv,
    execute_python,
    ask_clarification,
)
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
    # Executor classes
    "SafePythonExecutor",
    "MockPythonExecutor",
    # Search (ADK built-in)
    "google_search_tool",
    # Standard tool functions (ready to use with agents)
    "search_knowledge_base",
    "ingest_to_knowledge_base",
    "search_arxiv",
    "execute_python",
    "ask_clarification",
]

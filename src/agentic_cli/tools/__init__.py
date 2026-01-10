"""Tools module for agentic CLI applications.

Provides function tools for agents including web search and Python execution.

Tool System:
    The module provides a standardized tool system with:
    - ToolDefinition: Metadata-rich tool definitions
    - ToolError: Standard error class for consistent error handling
    - ToolResult: Standard result wrapper
    - ToolRegistry: Registry for tool management and discovery
    - register_tool: Decorator for easy tool registration
"""

from agentic_cli.tools.executor import SafePythonExecutor, MockPythonExecutor
from agentic_cli.tools.search import WebSearchClient, MockWebSearchClient
from agentic_cli.tools.standard import (
    search_knowledge_base,
    ingest_to_knowledge_base,
    search_arxiv,
    web_search,
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
from agentic_cli.tools.resilience import (
    retry,
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    RetryConfig,
    with_timeout,
)

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
    # Resilience patterns
    "retry",
    "CircuitBreaker",
    "CircuitState",
    "RateLimiter",
    "RetryConfig",
    "with_timeout",
    # Executor classes
    "SafePythonExecutor",
    "MockPythonExecutor",
    # Search classes
    "WebSearchClient",
    "MockWebSearchClient",
    # Standard tool functions (ready to use with agents)
    "search_knowledge_base",
    "ingest_to_knowledge_base",
    "search_arxiv",
    "web_search",
    "execute_python",
    "ask_clarification",
]

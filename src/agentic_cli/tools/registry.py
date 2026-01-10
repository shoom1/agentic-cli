"""Tool registry for standardized tool management.

Provides:
- ToolDefinition: Metadata-rich tool definition
- ToolError: Standard error class for tool failures
- ToolResult: Standard result wrapper
- ToolRegistry: Registry for tool discovery and management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, ParamSpec
from functools import wraps
import inspect
import time


class ToolCategory(Enum):
    """Categories for organizing tools."""

    SEARCH = "search"
    KNOWLEDGE = "knowledge"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    FILE = "file"
    ANALYSIS = "analysis"
    OTHER = "other"


@dataclass
class ToolDefinition:
    """Metadata-rich tool definition.

    Provides comprehensive metadata for tools beyond what's inferred
    from function signatures and docstrings.

    Attributes:
        name: Tool name (defaults to function name)
        description: Human-readable description
        category: Tool category for organization
        requires_api_key: API key type required (if any)
        is_async: Whether the tool is async
        timeout_seconds: Suggested timeout for this tool
        rate_limit: Maximum calls per minute (0 = unlimited)
        func: The actual tool function
    """

    name: str
    description: str
    func: Callable[..., Any]
    category: ToolCategory = ToolCategory.OTHER
    requires_api_key: str | None = None
    is_async: bool = False
    timeout_seconds: int = 30
    rate_limit: int = 0  # 0 = unlimited
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Infer is_async from function."""
        if inspect.iscoroutinefunction(self.func):
            self.is_async = True


class ToolError(Exception):
    """Standard error for tool failures.

    Provides structured error information that can be used by agents
    to understand and potentially recover from failures.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        recoverable: Whether the error might be recoverable
        details: Additional error details
        tool_name: Name of the tool that failed
    """

    def __init__(
        self,
        message: str,
        error_code: str = "TOOL_ERROR",
        recoverable: bool = False,
        details: dict[str, Any] | None = None,
        tool_name: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.details = details or {}
        self.tool_name = tool_name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": False,
            "error": {
                "message": self.message,
                "code": self.error_code,
                "recoverable": self.recoverable,
                "details": self.details,
                "tool_name": self.tool_name,
            },
        }


# Common error codes
class ErrorCode:
    """Standard error codes for tool failures."""

    # Input errors
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED = "MISSING_REQUIRED"
    VALIDATION_FAILED = "VALIDATION_FAILED"

    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # External service errors
    API_ERROR = "API_ERROR"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

    # Internal errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


@dataclass
class ToolResult:
    """Standard result wrapper for tool outputs.

    Provides a consistent structure for tool results that includes
    success/failure status, data, and execution metadata.
    """

    success: bool
    data: Any = None
    error: ToolError | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "success": self.success,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }

        if self.success:
            result["data"] = self.data
        else:
            result["error"] = self.error.to_dict()["error"] if self.error else None

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def ok(cls, data: Any, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: ToolError, **metadata) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


class ToolRegistry:
    """Registry for managing and discovering tools.

    Provides:
    - Tool registration with metadata
    - Tool lookup by name or category
    - Tool list generation for agents
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory = ToolCategory.OTHER,
        requires_api_key: str | None = None,
        timeout_seconds: int = 30,
        rate_limit: int = 0,
        **metadata,
    ) -> Callable[..., Any]:
        """Register a tool function.

        Can be used as a decorator:
            @registry.register(category=ToolCategory.SEARCH)
            def my_tool(query: str) -> dict:
                ...

        Or called directly:
            registry.register(my_tool, category=ToolCategory.SEARCH)
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (f.__doc__ or "").split("\n")[0].strip()

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=f,
                category=category,
                requires_api_key=requires_api_key,
                timeout_seconds=timeout_seconds,
                rate_limit=rate_limit,
                metadata=metadata,
            )

            self._tools[tool_name] = definition
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_function(self, name: str) -> Callable[..., Any] | None:
        """Get the tool function by name."""
        definition = self._tools.get(name)
        return definition.func if definition else None

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """List tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_functions(self) -> list[Callable[..., Any]]:
        """Get all tool functions (for passing to agents)."""
        return [t.func for t in self._tools.values()]

    def get_functions_by_category(
        self, category: ToolCategory
    ) -> list[Callable[..., Any]]:
        """Get tool functions by category."""
        return [t.func for t in self._tools.values() if t.category == category]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry instance
_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the default tool registry."""
    return _default_registry


def register_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.OTHER,
    requires_api_key: str | None = None,
    timeout_seconds: int = 30,
    rate_limit: int = 0,
    **metadata,
) -> Callable[..., Any]:
    """Register a tool with the default registry.

    Decorator for registering tools:
        @register_tool(category=ToolCategory.SEARCH)
        def search_web(query: str) -> dict:
            '''Search the web for information.'''
            ...
    """
    return _default_registry.register(
        func,
        name=name,
        description=description,
        category=category,
        requires_api_key=requires_api_key,
        timeout_seconds=timeout_seconds,
        rate_limit=rate_limit,
        **metadata,
    )


P = ParamSpec("P")
T = TypeVar("T")


def with_result_wrapper(func: Callable[P, T]) -> Callable[P, dict[str, Any]]:
    """Decorator to wrap tool output in ToolResult and convert to dict.

    Catches exceptions and converts them to ToolError format.
    Also records execution time.

    Example:
        @with_result_wrapper
        def my_tool(query: str) -> dict:
            return {"results": [...]}

        # Returns: {"success": True, "data": {"results": [...]}, "execution_time_ms": 10.5}
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            return ToolResult.ok(result, execution_time_ms=execution_time).to_dict()
        except ToolError as e:
            execution_time = (time.time() - start_time) * 1000
            e.tool_name = e.tool_name or func.__name__
            return ToolResult.fail(e, execution_time_ms=execution_time).to_dict()
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error = ToolError(
                message=str(e),
                error_code=ErrorCode.INTERNAL_ERROR,
                recoverable=False,
                tool_name=func.__name__,
            )
            return ToolResult.fail(error, execution_time_ms=execution_time).to_dict()

    return wrapper

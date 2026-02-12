"""Tool registry for standardized tool management.

Provides:
- ToolDefinition: Metadata-rich tool definition
- ToolRegistry: Registry for tool discovery and management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import inspect


class ToolCategory(Enum):
    """Categories for organizing tools.

    Categories are organized by primary function:
    - READ: Read-only file operations (safe)
    - WRITE: File modification operations (require caution)
    - NETWORK: Network and web operations
    - EXECUTION: Shell and code execution
    - PLANNING: Task and workflow management
    - MEMORY: State and context management
    - KNOWLEDGE: External knowledge access
    - INTERACTION: Human-in-the-loop operations
    """

    # File operations (split by safety)
    READ = "read"  # read_file, grep, glob, diff
    WRITE = "write"  # write_file, edit_file

    # Network operations
    NETWORK = "network"  # web_search, web_fetch, api calls

    # Execution (potentially dangerous)
    EXECUTION = "execution"  # shell, python executor

    # Planning and task management
    PLANNING = "planning"  # save_plan, get_plan

    # Memory and state
    MEMORY = "memory"  # remember, recall, search_memory

    # External knowledge
    KNOWLEDGE = "knowledge"  # arxiv, knowledge_base

    # Human interaction
    INTERACTION = "interaction"  # ask_clarification, request_approval

    OTHER = "other"


class PermissionLevel(Enum):
    """Permission levels for tool safety classification.

    Used to determine whether user confirmation is needed before tool execution.
    """

    SAFE = "safe"  # No confirmation needed (read operations)
    CAUTION = "caution"  # May need allowlisting (write operations)
    DANGEROUS = "dangerous"  # Always requires confirmation (delete, shell)


@dataclass
class ToolDefinition:
    """Metadata-rich tool definition.

    Provides comprehensive metadata for tools beyond what's inferred
    from function signatures and docstrings.

    Attributes:
        name: Tool name (defaults to function name)
        description: Human-readable description
        category: Tool category for organization (READ, WRITE, NETWORK, etc.)
        permission_level: Safety classification (SAFE, CAUTION, DANGEROUS)
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
    permission_level: PermissionLevel = PermissionLevel.SAFE
    requires_api_key: str | None = None
    is_async: bool = False
    timeout_seconds: int = 30
    rate_limit: int = 0  # 0 = unlimited
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Infer is_async from function."""
        if inspect.iscoroutinefunction(self.func):
            self.is_async = True


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
        permission_level: PermissionLevel = PermissionLevel.SAFE,
        requires_api_key: str | None = None,
        timeout_seconds: int = 30,
        rate_limit: int = 0,
        **metadata,
    ) -> Callable[..., Any]:
        """Register a tool function.

        Can be used as a decorator:
            @registry.register(category=ToolCategory.READ, permission_level=PermissionLevel.SAFE)
            def my_tool(query: str) -> dict:
                ...

        Or called directly:
            registry.register(my_tool, category=ToolCategory.READ)
        """

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (f.__doc__ or "").split("\n")[0].strip()

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=f,
                category=category,
                permission_level=permission_level,
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

    def list_by_permission(self, permission: PermissionLevel) -> list[ToolDefinition]:
        """List tools by permission level."""
        return [t for t in self._tools.values() if t.permission_level == permission]

    def get_safe_tools(self) -> list[ToolDefinition]:
        """Get all tools with SAFE permission level."""
        return self.list_by_permission(PermissionLevel.SAFE)

    def get_dangerous_tools(self) -> list[ToolDefinition]:
        """Get all tools with DANGEROUS permission level."""
        return self.list_by_permission(PermissionLevel.DANGEROUS)

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
    permission_level: PermissionLevel = PermissionLevel.SAFE,
    requires_api_key: str | None = None,
    timeout_seconds: int = 30,
    rate_limit: int = 0,
    **metadata,
) -> Callable[..., Any]:
    """Register a tool with the default registry.

    Decorator for registering tools:
        @register_tool(category=ToolCategory.READ, permission_level=PermissionLevel.SAFE)
        def read_file(path: str) -> dict:
            '''Read file contents.'''
            ...
    """
    return _default_registry.register(
        func,
        name=name,
        description=description,
        category=category,
        permission_level=permission_level,
        requires_api_key=requires_api_key,
        timeout_seconds=timeout_seconds,
        rate_limit=rate_limit,
        **metadata,
    )



"""Tool registry for standardized tool management.

Provides:
- ToolDefinition: Metadata-rich tool definition
- ToolRegistry: Registry for tool discovery and management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import inspect

from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    CapabilitiesSpec,
    EXEMPT,
)
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt


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
    INTERACTION = "interaction"  # ask_clarification

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

    Attributes:
        name: Tool name (defaults to function name)
        description: Human-readable description
        category: Tool category for organization (READ, WRITE, NETWORK, etc.)
        permission_level: Safety classification (SAFE, CAUTION, DANGEROUS)
        is_async: Whether the tool is async
        func: The actual tool function
    """

    name: str
    description: str
    func: Callable[..., Any]
    capabilities: CapabilitiesSpec = field(default_factory=lambda: EXEMPT)
    category: ToolCategory = ToolCategory.OTHER
    permission_level: PermissionLevel = PermissionLevel.SAFE
    is_async: bool = False

    def __post_init__(self):
        """Infer is_async from function."""
        if inspect.iscoroutinefunction(self.func):
            self.is_async = True


def _validate_capabilities(caps: Any, tool_name: str) -> CapabilitiesSpec:
    """Validate and return a capabilities value for a tool registration.

    - ``EXEMPT`` (or any ``_CapabilityExempt`` instance) passes through unchanged.
    - A non-empty ``list`` of ``Capability`` instances is returned as-is.
    - An empty list raises ``ValueError`` (use ``EXEMPT`` to opt out explicitly).
    - Any other type raises ``TypeError``.
    """
    if isinstance(caps, _CapabilityExempt):
        return caps
    if isinstance(caps, list):
        if not caps:
            raise ValueError(
                f"Tool {tool_name!r}: capabilities=[] is not allowed. "
                "Use capabilities=EXEMPT to opt out explicitly."
            )
        for item in caps:
            if not isinstance(item, Capability):
                raise TypeError(
                    f"Tool {tool_name!r}: capabilities list items must be "
                    f"Capability instances, got {type(item)!r}."
                )
        return caps
    raise TypeError(
        f"Tool {tool_name!r}: capabilities must be EXEMPT or a list of Capability "
        f"instances, got {type(caps)!r}."
    )


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
        capabilities: CapabilitiesSpec = EXEMPT,
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
            validated_caps = _validate_capabilities(capabilities, tool_name)

            definition = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=f,
                capabilities=validated_caps,
                category=category,
                permission_level=permission_level,
            )

            self._tools[tool_name] = definition
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """List tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_functions(self) -> list[Callable[..., Any]]:
        """Get all tool functions (for passing to agents)."""
        return [t.func for t in self._tools.values()]

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
    capabilities: CapabilitiesSpec = EXEMPT,
) -> Callable[..., Any]:
    """Register a tool with the default registry.

    DANGEROUS tools are no longer wrapped at registration time. Instead,
    confirmation is handled at the framework level by ADK ConfirmationPlugin
    and LangGraph's _wrap_for_confirmation wrapper.
    """

    def _outer(f: Callable[..., Any]) -> Callable[..., Any]:
        return _default_registry.register(
            f,
            name=name,
            description=description,
            category=category,
            permission_level=permission_level,
            capabilities=capabilities,
        )

    if func is not None:
        return _outer(func)
    return _outer



"""Python execution tools for agentic workflows.

Provides sandboxed Python code execution via SafePythonExecutor.
"""

from typing import Any

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)


@register_tool(
    category=ToolCategory.EXECUTION,
    permission_level=PermissionLevel.DANGEROUS,
    description="Execute Python code in a sandboxed subprocess with restricted imports. Use this for calculations, data processing, or prototyping. Only whitelisted modules (math, numpy, pandas, json, etc.) are available.",
)
def execute_python(
    code: str,
    context: str = "",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute Python code safely.

    Args:
        code: Python code to execute
        context: Optional JSON string of variables to inject (e.g. '{"x": 5, "name": "test"}')
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    import json as _json
    from agentic_cli.tools.executor import SafePythonExecutor

    parsed_context = None
    if context:
        try:
            parsed_context = _json.loads(context)
        except _json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON in context: {context}"}

    settings = get_settings()
    executor = SafePythonExecutor(
        default_timeout=settings.python_executor_timeout,
        max_memory_mb=settings.python_executor_max_memory_mb,
    )
    return executor.execute(code, context=parsed_context, timeout_seconds=timeout_seconds)

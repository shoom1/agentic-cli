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
from agentic_cli.workflow.permissions import Capability


@register_tool(
    category=ToolCategory.EXECUTION,
    permission_level=PermissionLevel.DANGEROUS,
    capabilities=[Capability("python.exec")],
    description="Stateless Python scratchpad for quick calculations, formula checks, and mathematical reasoning. Each call starts fresh — no state persists. Only whitelisted modules (math, numpy, pandas, json, etc.) are available. Use sandbox_execute instead for stateful work.",
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

    # Build OS sandbox policy from settings
    os_sandbox_policy = None
    if getattr(settings, "os_sandbox_enabled", False):
        from agentic_cli.tools.shell.os_sandbox import OSSandboxPolicy

        os_sandbox_policy = OSSandboxPolicy(
            enabled=True,
            writable_paths=getattr(settings, "os_sandbox_writable_paths", []),
            allow_network=getattr(settings, "os_sandbox_allow_network", False),
        )

    executor = SafePythonExecutor(
        default_timeout=settings.python_executor_timeout,
        max_memory_mb=settings.python_executor_max_memory_mb,
        os_sandbox_policy=os_sandbox_policy,
    )
    return executor.execute(code, context=parsed_context, timeout_seconds=timeout_seconds)

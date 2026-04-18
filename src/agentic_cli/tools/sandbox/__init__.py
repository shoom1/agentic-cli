"""Sandbox execution tools for stateful, multi-turn code execution.

Provides sandbox_execute tool backed by Jupyter kernels,
enabling persistent state across calls (variables, imports, DataFrames).
"""

from typing import Any

from agentic_cli.tools.registry import register_tool, ToolCategory, PermissionLevel
from agentic_cli.workflow.service_registry import require_service, SANDBOX_MANAGER
from agentic_cli.workflow.permissions import Capability


@register_tool(
    category=ToolCategory.EXECUTION,
    permission_level=PermissionLevel.DANGEROUS,
    capabilities=[Capability("python.exec")],
    description=(
        "Execute Python code in a stateful sandbox session. "
        "State (variables, imports) persists across calls within the same session. "
        "The sandbox shares the workspace filesystem — code can read/write files directly. "
        "Network access and package installation are blocked — use web_fetch/web_search for HTTP. "
        "Use for data analysis, prototyping, and producing work output. "
        "Use execute_python instead for quick stateless calculations."
    ),
)
def sandbox_execute(
    code: str,
    session_id: str = "default",
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Execute Python code in a stateful sandbox.

    Args:
        code: Python code to execute.
        session_id: Session identifier for state persistence (default: "default").
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        Dictionary with execution results.
    """
    manager = require_service(SANDBOX_MANAGER)
    if isinstance(manager, dict):
        return manager
    result = manager.execute(
        code=code,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "result": result.result,
        "artifacts": result.artifacts,
        "execution_time": round(result.execution_time, 3),
        "error": result.error,
    }

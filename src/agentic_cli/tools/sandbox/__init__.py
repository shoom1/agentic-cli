"""Sandbox execution tools for stateful, multi-turn code execution.

Provides sandbox_execute tool backed by Jupyter kernels,
enabling persistent state across calls (variables, imports, DataFrames).
"""

from typing import Any

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import register_tool, ToolCategory
from agentic_cli.workflow.service_registry import require_service, SANDBOX_MANAGER
from agentic_cli.workflow.permissions import Capability


@register_tool(
    category=ToolCategory.EXECUTION,
    # Distinct from execute_python's ``python.exec`` on purpose: this kernel is
    # unsandboxed and stateful, so an "Allow always" for the stateless scratchpad
    # must NOT silently authorize it. A deliberate ``python.*`` grant still covers
    # both.
    capabilities=[Capability("python.exec.stateful")],
    description=(
        "Execute Python code in a stateful session. "
        "State (variables, imports) persists across calls within the same session. "
        "Isolation depends on sandbox_backend: 'jupyter_docker' runs in a "
        "network-isolated container (no network egress, resource-capped); "
        "'jupyter_local' runs with host privileges and shared filesystem. "
        "Disabled unless explicitly enabled. Use for data analysis, prototyping, "
        "and producing work output. Use execute_python for quick stateless calculations."
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
    # Opt-in (see sandbox_execute_enabled). Gate before touching the service so a
    # disabled deployment fails fast with a message accurate for the selected
    # backend: jupyter_local is host-privileged; jupyter_docker is isolated.
    settings = get_settings()
    if not getattr(settings, "sandbox_execute_enabled", False):
        backend = getattr(settings, "sandbox_backend", "jupyter_local")
        if backend == "jupyter_docker":
            detail = ("The 'jupyter_docker' backend runs it in a network-isolated, "
                      "resource-capped container.")
        else:
            detail = (f"The '{backend}' backend runs Python with host privileges and "
                      "no OS sandbox (use 'jupyter_docker' for isolation).")
        return {
            "success": False,
            "error": (f"sandbox_execute is not enabled. {detail} "
                      "Enable sandbox_execute_enabled in settings to use it."),
        }

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

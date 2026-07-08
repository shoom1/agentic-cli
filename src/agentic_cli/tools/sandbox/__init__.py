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
    capabilities=[Capability("python.exec.stateful"), Capability("filesystem.read", target_arg="inputs")],
    description=(
        "Execute Python code in a stateful session. "
        "State (variables, imports) persists across calls within the same session. "
        "Isolation depends on stateful_executor_backend: 'docker' runs in a "
        "network-isolated container (no network egress; memory/CPU/PID-capped, "
        "though disk is not); 'local' runs with host privileges and shared filesystem. "
        "Disabled unless explicitly enabled. Use for data analysis, prototyping, "
        "and producing work output. Use execute_python for quick stateless calculations. "
        "Each `inputs` file is copied to `inputs/<filename>` inside the session before "
        "the code runs; load it by that relative path (e.g. open('inputs/data.csv')). "
        "Write scratch/intermediate files to the working directory; write FINAL deliverables "
        "(figures, tables) to `outputs/` — those persist and are shared with other agents."
    ),
)
def sandbox_execute(
    code: str,
    session_id: str = "default",
    timeout_seconds: int = 120,
    inputs: list[str] | None = None,
) -> dict[str, Any]:
    """Execute Python code in a stateful sandbox.

    Args:
        code: Python code to execute.
        session_id: Session identifier for state persistence (default: "default").
        timeout_seconds: Maximum execution time in seconds.
        inputs: Optional list of host file paths to stage into
            inputs/<basename> inside the session before execution.

    Returns:
        Dictionary with execution results.
    """
    # Opt-in gate. Gate before touching the service so a disabled deployment
    # fails fast. NOTE: the workflow binds the factory tool (tools/factories.py),
    # which gates identically — this gate covers the module-level tool used
    # outside the workflow.
    from agentic_cli.tools.sandbox.manager import sandbox_disabled_reason

    settings = get_settings()
    if getattr(settings, "stateful_executor_backend", "none") == "none":
        return {"success": False, "error": sandbox_disabled_reason(settings)}

    manager = require_service(SANDBOX_MANAGER)
    if isinstance(manager, dict):
        return manager
    result = manager.execute(
        code=code,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
        inputs=inputs,
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

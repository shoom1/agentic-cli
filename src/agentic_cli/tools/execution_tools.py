"""Python execution tools for agentic workflows.

``execute_python`` runs code through ``SafePythonExecutor``: inside the OS
sandbox by default, refused when there is none, and with the user's
privileges only when the settings turn the sandbox off or let it fall back
(``tools/exec_policy.py``).
"""

from typing import Any

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
)
from agentic_cli.workflow.permissions import Capability


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("python.exec")],
    description=(
        "Stateless Python scratchpad for quick calculations, formula checks, and "
        "mathematical reasoning. Each call starts fresh — no state persists. The "
        "code runs inside the OS sandbox, where numpy, scipy, sklearn, sympy, "
        "pandas and matplotlib can be imported besides math, statistics, "
        "cmath, collections, itertools, json, re, datetime, decimal, fractions "
        "and random; with no sandbox it is refused, unless the user turned the "
        "sandbox off: then it runs with the user's privileges, only those "
        "standard modules import, and the user approves each call. Use "
        "sandbox_execute instead for stateful work."
    ),
)
def execute_python(
    code: str,
    context: str = "",
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run Python code in a fresh subprocess, inside the OS sandbox by default.

    Args:
        code: Python code to execute
        context: Optional JSON string of variables to inject (e.g. '{"x": 5, "name": "test"}')
        timeout_seconds: Maximum execution time in seconds. Omit to use the
            configured default (``python_executor_timeout``).

    Returns:
        Dictionary with execution results
    """
    import json as _json
    from agentic_cli.tools.exec_policy import python_sandbox_policy, sandbox_opt_out
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
        # None when the settings turn the sandbox off: the code runs as the
        # user, which the permission engine approves call by call.
        os_sandbox_policy=python_sandbox_policy(settings),
        opt_out=sandbox_opt_out(settings),
    )
    return executor.execute(code, context=parsed_context, timeout_seconds=timeout_seconds)

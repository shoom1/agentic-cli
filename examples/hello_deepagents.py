"""Hello Deep Agents - a simple example using the Deep Agents orchestrator.

This example runs the same assistant as ``hello_agent.py`` on the Deep Agents
backend (``DeepAgentsWorkflowManager``) instead of Google ADK.

Run with: python examples/hello_deepagents.py

Requires the deepagents extra:
    pip install 'agentic-cli[deepagents]'
"""

import ast
import asyncio
import math
import operator
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pydantic_settings import SettingsConfigDict
from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.tools import ToolCategory, register_tool
from agentic_cli.workflow import AgentConfig
from agentic_cli.workflow.permissions import EXEMPT

# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """Settings for the Hello Deep Agents example.

    API keys are read from standard environment variables (GOOGLE_API_KEY,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY) or from
    ``~/.hello_deepagents/.env``.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".hello_deepagents" / ".env"),
        extra="ignore",
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("app_name", "hello_deepagents")
        kwargs.setdefault("workspace_dir", Path.home() / ".hello_deepagents")
        # Use Deep Agents as the orchestrator
        kwargs.setdefault("orchestrator", "deepagents")
        super().__init__(**kwargs)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# Tools
# =============================================================================
#
# Every tool call passes the permission engine, which denies a tool the
# registry did not issue. These three touch nothing, so they are EXEMPT.

@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def get_current_time() -> dict:
    """Get the current date and time."""
    now = datetime.now()
    return {"success": True, "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> float:
    """Evaluate arithmetic: numbers, operators, and the names above, nothing else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")  # 9**9**9 would never finish
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression (e.g., "2 + 2" or "sqrt(16)")."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, TypeError, ArithmeticError) as e:
        return {"success": False, "expression": expression, "error": str(e)}
    return {"success": True, "expression": expression, "result": result}


@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def echo(message: str) -> dict:
    """Echo back a message."""
    return {"success": True, "echoed": message, "length": len(message)}


# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIGS = [
    AgentConfig(
        name="assistant",
        prompt="You are a friendly assistant with tools: get_current_time, calculate, echo.",
        tools=[get_current_time, calculate, echo],
        description="Friendly assistant with utility tools",
    ),
]


# =============================================================================
# Application
# =============================================================================

def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("Hello Deep Agents\n\n", style="bold green")
    text.append("Orchestrator: Deep Agents\n", style="dim")
    text.append("Tools: get_current_time, calculate, echo\n", style="dim")
    text.append("Type /help for commands", style="dim")
    return AppInfo(
        name="Hello Deep Agents",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="green"),
        echo_thinking=False,
    )


if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=_create_app_info(),
        agent_configs=AGENT_CONFIGS,
        settings=get_settings(),
    )
    asyncio.run(app.run())

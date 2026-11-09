"""Hello LangGraph - A simple example using LangGraphWorkflowManager.

This example demonstrates using the LangGraph orchestration backend
instead of Google ADK for agent workflows.

The LangGraph backend is deprecated and will be removed in agentic-cli 0.8.0,
with this example; hello_deepagents.py runs the same assistant on Deep Agents.

Run with: python examples/hello_langgraph.py

Requires LangGraph dependencies:
    pip install agentic-cli[langgraph]
    pip install langchain-anthropic  # or langchain-google-genai
"""

import ast
import asyncio
import math
import operator
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.tools import ToolCategory, register_tool
from agentic_cli.workflow import AgentConfig
from agentic_cli.workflow.permissions import EXEMPT


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """Settings for the Hello LangGraph example.

    API keys are read from standard environment variables (GOOGLE_API_KEY,
    ANTHROPIC_API_KEY) while app-specific settings use the HELLO_LG_ prefix.
    """
    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".hello_langgraph" / ".env"),
        extra="ignore",
    )
    app_name: str = Field(default="hello_langgraph")
    workspace_dir: Path = Field(default=Path.home() / ".hello_langgraph")
    # Use LangGraph as the orchestrator
    orchestrator: str = Field(default="langgraph")


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
_MAX_BITS = 10_000  # about 3,000 digits: quick to compute, and to print


def _round(number, ndigits=None):
    """``round``, with few digits: an int's ``round(1, -10**8)`` computes 10**10**8."""
    if ndigits is not None and abs(ndigits) > 100:
        raise ValueError("ndigits too large")
    return round(number, ndigits)


_FUNCTIONS = {
    "abs": abs, "round": _round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _sized(value: float) -> float:
    """``value``, unless it is an int too large to work with."""
    if isinstance(value, int) and value.bit_length() > _MAX_BITS:
        raise ValueError("number too large")
    return value


def _evaluate(node: ast.AST) -> float:
    """Evaluate arithmetic: numbers, operators, and the names above, nothing else.

    It runs in the tool's thread and holds the GIL, so every step is bounded:
    no int grows past ``_MAX_BITS``, and a power that would is refused before
    it is computed.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("exponent too large")  # 9**9**9 would never finish
            if (
                isinstance(left, int) and isinstance(right, int)
                and left.bit_length() * right > _MAX_BITS
            ):
                raise ValueError("number too large")  # nor (((10**100)**100)**100)**100
        return _sized(_OPERATORS[type(node.op)](left, right))
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
        return _sized(_FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args)))
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


@register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT)
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression (e.g., "2 + 2" or "sqrt(16)")."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
    except (
        SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError, MemoryError
    ) as e:
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
    text.append("Hello LangGraph\n\n", style="bold magenta")
    text.append("Orchestrator: LangGraph\n", style="dim")
    text.append("Tools: get_current_time, calculate, echo\n", style="dim")
    text.append("Type /help for commands", style="dim")
    return AppInfo(
        name="Hello LangGraph",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="magenta"),
    )


if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=_create_app_info(),
        agent_configs=AGENT_CONFIGS,
        settings=get_settings(),
    )
    asyncio.run(app.run())

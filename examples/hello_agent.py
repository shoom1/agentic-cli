"""Hello Agent - A simple example showcasing the agentic-cli package.

Run with: python examples/hello_agent.py
"""

import asyncio
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.workflow import AgentConfig


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """Settings for the Hello Agent example.

    API keys are read from standard environment variables (GOOGLE_API_KEY,
    ANTHROPIC_API_KEY) while app-specific settings can be in the .env file.
    """
    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".hello_agent" / ".env"),
        extra="ignore",
    )
    app_name: str = Field(default="hello_agent")
    workspace_dir: Path = Field(default=Path.home() / ".hello_agent")


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# Tools
# =============================================================================

def get_current_time() -> dict:
    """Get the current date and time."""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression (e.g., "2 + 2" or "sqrt(16)")."""
    import math
    allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum,
               "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
               "tan": math.tan, "pi": math.pi, "e": math.e}
    try:
        return {"expression": expression, "result": eval(expression, {"__builtins__": {}}, allowed)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def echo(message: str) -> dict:
    """Echo back a message."""
    return {"echoed": message, "length": len(message)}


# =============================================================================
# Agent & App
# =============================================================================

AGENT_CONFIGS = [
    AgentConfig(
        name="assistant",
        prompt="You are a friendly assistant with tools: get_current_time, calculate, echo.",
        tools=[get_current_time, calculate, echo],
        description="Friendly assistant with utility tools",
    ),
]


def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("Hello Agent\n\n", style="bold cyan")
    text.append("Tools: get_current_time, calculate, echo\n", style="dim")
    text.append("Type /help for commands", style="dim")
    return AppInfo(
        name="Hello Agent",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="cyan"),
        echo_thinking=False,
    )


if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=_create_app_info(),
        agent_configs=AGENT_CONFIGS,
        settings=get_settings(),
    )
    asyncio.run(app.run())

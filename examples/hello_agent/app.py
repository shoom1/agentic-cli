"""Hello Agent - A simple example showcasing the agentic-cli package.

This single file contains everything needed:
- Settings configuration
- Tool definitions
- Agent configuration
- CLI application
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.config import set_settings
from agentic_cli.workflow import AgentConfig, WorkflowManager

__version__ = "0.1.0"


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """Hello Agent settings."""

    model_config = SettingsConfigDict(
        env_prefix="HELLO_",
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
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
    """
    import math

    allowed = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def echo(message: str) -> dict:
    """Echo back a message.

    Args:
        message: The message to echo back
    """
    return {"echoed": message, "length": len(message)}


# =============================================================================
# Agent Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a friendly assistant with access to utility tools.

Available tools:
- get_current_time: Get the current date and time
- calculate: Evaluate math expressions
- echo: Echo back messages

Be helpful and concise in your responses."""


AGENT_CONFIGS = [
    AgentConfig(
        name="assistant",
        prompt=SYSTEM_PROMPT,
        tools=[get_current_time, calculate, echo],
        description="Friendly assistant with utility tools",
    ),
]


def create_workflow_manager(settings: Settings) -> WorkflowManager:
    """Create the workflow manager."""
    set_settings(settings)
    return WorkflowManager(agent_configs=AGENT_CONFIGS, settings=settings)


# =============================================================================
# CLI Application
# =============================================================================

def create_welcome_message() -> Panel:
    """Create the welcome message panel."""
    text = Text()
    text.append("Hello Agent", style="bold cyan")
    text.append(f" v{__version__}\n\n", style="dim")
    text.append("A simple assistant with utility tools.\n\n", style="white")
    text.append("Available tools:\n", style="yellow")
    text.append("  - get_current_time: Get date and time\n", style="dim")
    text.append("  - calculate: Math expressions\n", style="dim")
    text.append("  - echo: Echo messages\n\n", style="dim")
    text.append("Type ", style="white")
    text.append("/help", style="bold green")
    text.append(" for commands, or just start chatting!", style="white")

    return Panel(text, border_style="cyan", padding=(1, 2))


class HelloAgentApp(BaseCLIApp):
    """Hello Agent CLI application."""

    def get_app_info(self) -> AppInfo:
        return AppInfo(
            name="Hello Agent",
            version=__version__,
            welcome_message=create_welcome_message,
            echo_thinking=False,
        )

    def get_settings(self) -> Settings:
        return get_settings()

    def create_workflow_manager(self) -> WorkflowManager:
        return create_workflow_manager(settings=self._settings)

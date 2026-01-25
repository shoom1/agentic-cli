"""Hello LangGraph - A simple example using LangGraphWorkflowManager.

This example demonstrates using the LangGraph orchestration backend
instead of Google ADK for agent workflows.

Run with: python examples/hello_langgraph.py

Requires LangGraph dependencies:
    pip install agentic-cli[langgraph]
    pip install langchain-anthropic  # or langchain-google-genai
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
from agentic_cli.config import set_settings
from agentic_cli.workflow import AgentConfig
from agentic_cli.workflow.langgraph_manager import LangGraphWorkflowManager


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
# Tools (using LangChain tool format)
# =============================================================================

def get_current_time() -> dict:
    """Get the current date and time."""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression (e.g., "2 + 2" or "sqrt(16)")."""
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
    """Echo back a message."""
    return {"echoed": message, "length": len(message)}


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

class HelloLangGraphApp(BaseCLIApp):
    """Example CLI app using LangGraph for orchestration."""

    def get_app_info(self) -> AppInfo:
        text = Text()
        text.append("Hello LangGraph\n\n", style="bold magenta")
        text.append("Orchestrator: LangGraph\n", style="dim")
        text.append("Tools: get_current_time, calculate, echo\n", style="dim")
        text.append("Type /help for commands", style="dim")
        return AppInfo(
            name="Hello LangGraph",
            version="0.1.0",
            welcome_message=lambda: Panel(text, border_style="magenta"),
            echo_thinking=False,
        )

    def get_settings(self) -> Settings:
        return get_settings()

    def create_workflow_manager(self) -> LangGraphWorkflowManager:
        set_settings(self._settings)
        return LangGraphWorkflowManager(
            agent_configs=AGENT_CONFIGS,
            settings=self._settings,
            checkpointer="memory",  # Use in-memory checkpointing
        )


if __name__ == "__main__":
    asyncio.run(HelloLangGraphApp().run())

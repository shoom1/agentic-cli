"""YAML Agents - agents defined in a YAML file instead of in Python.

``agents.yaml`` declares a coordinator and a researcher, their prompts in
``prompts/``; ``load_agents_from_yaml`` turns it into the same ``AgentConfig``s
the other examples write by hand. The researcher's tool, ``word_count``, is a
registered name: ``tools.py`` registers it, and this module imports that
before it builds the agents, since a name is looked up when the manager is built.

Run with: python examples/yaml_agents/app.py

Ask it how many words a text has: the coordinator delegates the counting to the
researcher, which calls ``word_count``.

It runs on Google ADK. To run it on Deep Agents instead, install the extra
(``pip install 'agentic-cli[deepagents]'``) and set
``AGENTIC_ORCHESTRATOR=deepagents``: the YAML is the same.
"""

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import SettingsConfigDict
from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp, BaseSettings
from agentic_cli.cli import AppInfo
from agentic_cli.workflow import load_agents_from_yaml

# tools.py sits beside this file. Run as a script, Python puts this folder on
# sys.path; loaded by path (a test), it is not, so add it: either way `tools`
# imports, once, and registers word_count before the agents are built.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import tools  # noqa: E402,F401


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """Settings for the YAML Agents example.

    API keys are read from standard environment variables (GOOGLE_API_KEY,
    ANTHROPIC_API_KEY, OPENAI_API_KEY) or from ``~/.yaml_agents/.env``.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path.home() / ".yaml_agents" / ".env"),
        extra="ignore",
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("app_name", "yaml_agents")
        kwargs.setdefault("workspace_dir", Path.home() / ".yaml_agents")
        super().__init__(**kwargs)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIGS = load_agents_from_yaml(_HERE / "agents.yaml")


# =============================================================================
# Application
# =============================================================================

def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("YAML Agents\n\n", style="bold green")
    text.append("Agents: coordinator, researcher (from agents.yaml)\n", style="dim")
    text.append("Tools: word_count\n", style="dim")
    text.append("Try: how many words are in 'the quick brown fox'?\n", style="dim")
    text.append("Type /help for commands", style="dim")
    return AppInfo(
        name="YAML Agents",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="green"),
    )


if __name__ == "__main__":
    app = BaseCLIApp(
        app_info=_create_app_info(),
        agent_configs=AGENT_CONFIGS,
        settings=get_settings(),
    )
    asyncio.run(app.run())

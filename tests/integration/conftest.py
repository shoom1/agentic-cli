"""Shared fixtures for integration tests.

Provides:
- mock_settings: BaseSettings with test API key + temp workspace
- simple_agent_config: Single AgentConfig with a few real tools
- multi_agent_config: Coordinator + sub-agent configs
- collect_events: Async helper that calls process() and returns events list
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch

import pytest

from agentic_cli.config import BaseSettings, set_settings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent


def _load_live_test_env() -> None:
    """Load real API keys from a dotenv file for ``@pytest.mark.llm`` live tests.

    Keys are read from a file at runtime — never hardcoded. Search order:

    1. ``$AGENTIC_TEST_ENV_FILE`` if set,
    2. research_demo's own key file (``~/.research_demo/.env``).

    The first existing file wins. Variables already in the environment are
    never overridden (a key exported in the shell takes precedence), and this
    is a no-op when no file is found. It runs at import time so the
    module-level ``skipif`` gates in ``test_live.py`` /
    ``test_research_scenarios.py`` observe the keys during collection.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("AGENTIC_TEST_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.home() / ".research_demo" / ".env")

    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        break


_load_live_test_env()


@pytest.fixture
def integration_workspace(tmp_path: Path) -> Path:
    """Temporary workspace directory for integration tests."""
    workspace = tmp_path / "integration_workspace"
    workspace.mkdir(parents=True)
    return workspace


@pytest.fixture
def mock_settings(integration_workspace: Path) -> BaseSettings:
    """BaseSettings with test API key and temp workspace.

    Uses a fake Google API key so model resolution picks Gemini.
    """
    with patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-google-api-key"},
        clear=False,
    ):
        settings = BaseSettings(workspace_dir=integration_workspace)
        set_settings(settings)
        yield settings


@pytest.fixture
def mock_settings_anthropic(integration_workspace: Path) -> BaseSettings:
    """BaseSettings with test Anthropic API key and temp workspace."""
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-anthropic-api-key"},
        clear=False,
    ):
        settings = BaseSettings(workspace_dir=integration_workspace)
        set_settings(settings)
        yield settings


@pytest.fixture
def simple_agent_config() -> list[AgentConfig]:
    """Single agent with a few lightweight tools."""
    return [
        AgentConfig(
            name="assistant",
            prompt="You are a helpful assistant.",
            tools=[],
            description="General assistant",
        ),
    ]


@pytest.fixture
def multi_agent_config() -> list[AgentConfig]:
    """Coordinator + sub-agent configs."""
    return [
        AgentConfig(
            name="coordinator",
            prompt="You coordinate sub-agents.",
            tools=[],
            sub_agents=["researcher"],
            description="Coordinator agent",
        ),
        AgentConfig(
            name="researcher",
            prompt="You research topics.",
            tools=[],
            description="Research sub-agent",
        ),
    ]


async def collect_events(
    workflow,
    message: str,
    user_id: str = "test-user",
    session_id: str | None = None,
) -> list[WorkflowEvent]:
    """Call workflow.process() and collect all events into a list.

    Args:
        workflow: A workflow manager instance.
        message: User message to process.
        user_id: User identifier.
        session_id: Optional session identifier.

    Returns:
        List of all WorkflowEvent objects yielded by process().
    """
    events: list[WorkflowEvent] = []
    async for event in workflow.process(
        message=message,
        user_id=user_id,
        session_id=session_id,
    ):
        events.append(event)
    return events

"""Live LLM integration tests.

These tests hit real LLM APIs with automated prompts. They are marked with
@pytest.mark.llm and skipped by default. Run them explicitly with:

    conda run -n agenticcli python -m pytest tests/integration/test_live.py -v -m llm

Requires GOOGLE_API_KEY or ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings, set_settings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType

from tests.integration.conftest import collect_events
from tests.integration.helpers import (
    assert_no_errors,
    find_events,
    find_tool_calls,
)


# Skip all tests in this module if no real API key is available
_has_google_key = bool(os.environ.get("GOOGLE_API_KEY"))
_has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_any_key = _has_google_key or _has_anthropic_key

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not _has_any_key, reason="No LLM API key in environment"),
]


@pytest.fixture
def live_settings(tmp_path: Path) -> BaseSettings:
    """Settings using real API keys from the environment."""
    workspace = tmp_path / "live_workspace"
    workspace.mkdir(parents=True)
    settings = BaseSettings(workspace_dir=workspace)
    set_settings(settings)
    return settings


@pytest.fixture
def live_simple_agent_config() -> list[AgentConfig]:
    """Simple agent config for live tests (no tools)."""
    return [
        AgentConfig(
            name="assistant",
            prompt="You are a helpful assistant. Respond concisely.",
            tools=[],
            description="General assistant",
        ),
    ]


class TestLiveTextResponse:
    """Verify that the real LLM produces a text response."""

    @pytest.mark.llm
    async def test_real_text_response(self, live_settings, live_simple_agent_config):
        """Real LLM: ask a simple question, verify we get text back."""
        from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

        manager = GoogleADKWorkflowManager(
            agent_configs=live_simple_agent_config,
            settings=live_settings,
        )

        events = await collect_events(manager, "What is 2 + 2? Answer with just the number.")

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) > 0

        # The response should contain "4"
        full_text = "".join(e.content for e in text_events)
        assert "4" in full_text

        await manager.cleanup()


class TestLiveMultiTurn:
    """Verify multi-turn conversation works with real LLM."""

    @pytest.mark.llm
    async def test_multi_turn_conversation(self, live_settings, live_simple_agent_config):
        """Real LLM: send two messages in sequence, verify coherent responses."""
        from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

        manager = GoogleADKWorkflowManager(
            agent_configs=live_simple_agent_config,
            settings=live_settings,
        )

        # First message
        events1 = await collect_events(
            manager,
            "My name is TestBot. Remember that.",
            session_id="multi-turn-test",
        )
        text1 = find_events(events1, EventType.TEXT)
        assert len(text1) > 0

        # Second message referencing the first
        events2 = await collect_events(
            manager,
            "What is my name?",
            session_id="multi-turn-test",
        )
        text2 = find_events(events2, EventType.TEXT)
        assert len(text2) > 0

        full_text = "".join(e.content for e in text2)
        assert "testbot" in full_text.lower()

        await manager.cleanup()

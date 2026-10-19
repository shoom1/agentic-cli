"""Real ADK agent construction from declarative ``AgentConfig``s.

These tests build actual ``LlmAgent`` objects by initializing a
``GoogleADKWorkflowManager`` rather than pre-seeding manager internals, so a
config the framework documents must really construct.

Regression: ``description`` defaulted to ``""`` on ``AgentConfig`` but was
converted to ``None`` on the way into ``LlmAgent``. ADK types the field as
``str``, so the documented minimal config (README quick-start: name + prompt +
tools) raised ``ValidationError`` during initialization.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.conftest import MockContext  # noqa: E402


async def _root_agent(configs: list[AgentConfig]):
    """The root agent a manager over ``configs`` builds when it initializes."""
    with MockContext(google_api_key="test-key", session_store="memory") as ctx:
        mgr = GoogleADKWorkflowManager(agent_configs=configs, settings=ctx.settings)
        await mgr.initialize_services()
        try:
            return mgr._root_agent
        finally:
            await mgr.cleanup()


async def test_minimal_config_builds_a_real_adk_agent():
    """The documented minimal config (no description) must construct."""
    root = await _root_agent([AgentConfig(name="assistant", prompt="You are helpful.")])

    assert root.name == "assistant"
    # ADK types description as ``str``; the empty default must survive as "".
    assert root.description == ""


async def test_explicit_description_is_preserved():
    root = await _root_agent([AgentConfig(name="a", prompt="p", description="Does a thing")])

    assert root.description == "Does a thing"


async def test_coordinator_with_sub_agents_builds():
    """A coordinator + leaf pair (README example) constructs end to end."""
    root = await _root_agent([
        AgentConfig(name="coordinator", prompt="Route.", sub_agents=["worker"]),
        AgentConfig(name="worker", prompt="Work."),
    ])

    assert root.name == "coordinator"
    assert [a.name for a in root.sub_agents] == ["worker"]
    assert root.description == ""

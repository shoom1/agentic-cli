"""ADK: the app's whole plugin chain, in order, whoever built its agents.

The order is the backend's core invariant. ADK runs a callback of each
plugin in turn and stops at the first that returns a value, so a plugin
placed after one that answers never runs for that call; an agent's own
callbacks run after every plugin, and only when none answered. The chain
is listed in one place (``_plugin_chain``, whose head is ``_init_plugins``),
and these tests pin it whole, so a move anywhere fails one of them:

- ``main_agent`` first: it records the running agent, which the summarizer
  reads;
- ``call_latency`` before any plugin that could replace an answer: it marks
  each call's answer with how long the call took;
- ``permission`` before every other tool callback: a denial is the call's
  answer;
- ``tool_error`` answers a call that raised (and one to an unknown tool);
- ``llm_logging`` records each call as the model answers it;
- ``foreign_reasoning`` drops the thoughts another provider wrote, before
  the request is measured;
- ``context_cache`` makes one Gemini cache attempt a turn;
- ``interrupted_calls`` before ``tool_result_cap``: it records every
  finished result, and the cap returns its cut, which ends ADK's after-tool
  callbacks;
- ``roster``, for a main agent that continues its sub-agents, before
  ``memory_index``, which puts the index ahead of it; both before
  ``context_overflow`` and ``fit_guard``, which keep them;
- ``fit_guard`` last but the raw log and the clock: it measures the request
  every other plugin edited;
- ``llm_logging_sent``: it logs the request as sent;
- ``call_sent`` last: each call's clock starts as its request goes out.

A native ADK YAML app gets the same chain: only the building of its agents
differs (``workflow/adk/build.py``). Each sub-agent type's runner has a chain
of its own in the same order (``_agent_plugin_chain``): no memory index, no roster, and
after the raw log (which records an answer as the model gave it)
``run_control``, where the run's limits are kept at its model calls, then
``subagent_outcome``, where a sub-agent's refused or cut answer is said so;
both change what they see in place, so every plugin after them still runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.memory_tools import search_memory  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.agents import SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.workflow.adk.fakes import dead_base_urls  # noqa: E402
from tests.workflow.test_adk_context_cache import _native_yaml  # noqa: E402

MODEL = "claude-sonnet-4-6"


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    dead_base_urls(monkeypatch)
    return tmp_path


def _settings(*, extras: bool) -> BaseSettings:
    """With ``extras``: the raw LLM log and prompt caching on (Claude caches)."""
    return BaseSettings(
        session_store="memory", default_model=MODEL,
        prompt_caching_enabled=extras, raw_llm_logging=extras,
    )


async def _chain(manager: GoogleADKWorkflowManager) -> list[str]:
    try:
        await manager.initialize_services()
        return [plugin.name for plugin in manager._runner.app.plugins]
    finally:
        await manager.cleanup()


async def test_a_config_built_app_with_every_option(project):
    """The raw log, a caching model and a main agent with a memory tool."""
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Help.", tools=[search_memory])],
        settings=_settings(extras=True),
    )

    assert await _chain(manager) == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "llm_logging",
        "foreign_reasoning",
        "context_cache",
        "interrupted_calls",
        "tool_result_cap",
        "memory_index",
        "context_overflow",
        "fit_guard",
        "llm_logging_sent",
        "call_sent",
    ]


async def test_a_plain_config_built_app(project):
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Help.")],
        settings=_settings(extras=False),
    )

    assert await _chain(manager) == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "foreign_reasoning",
        "interrupted_calls",
        "tool_result_cap",
        "context_overflow",
        "fit_guard",
        "call_sent",
    ]


async def test_a_delegating_app(project):
    """A tree with a sub-agent: the conversation's chain is the app's; the
    sub-agent type's own runner's keeps a run's limits (``run_control``) and
    passes its answers through ``subagent_outcome``."""
    manager = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"],
                        tools=[search_memory]),
            AgentConfig(name="helper", prompt="Help.", description="Helps.",
                        tools=[search_memory]),
        ],
        settings=_settings(extras=True),
    )

    try:
        await manager.initialize_services()
        chain = [plugin.name for plugin in manager._runner.app.plugins]
        helpers = [plugin.name for plugin in manager._agent_runners["helper"].app.plugins]
        shared = manager._llm_logging_plugin
        logs = [p for p in manager._agent_runners["helper"].app.plugins if p.name == "llm_logging"]
    finally:
        await manager.cleanup()

    assert chain == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "llm_logging",
        "foreign_reasoning",
        "context_cache",
        "interrupted_calls",
        "tool_result_cap",
        "memory_index",
        "context_overflow",
        "fit_guard",
        "llm_logging_sent",
        "call_sent",
    ]
    assert helpers == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "llm_logging",
        "run_control",
        "subagent_outcome",
        "foreign_reasoning",
        "context_cache",
        "interrupted_calls",
        "tool_result_cap",
        "context_overflow",
        "fit_guard",
        "llm_logging_sent",
        "call_sent",
    ]
    # One raw log for every runner, whose buffer the manager drains.
    assert logs == [shared]


async def test_a_main_agent_that_continues_its_sub_agents_gets_the_roster(project):
    """With the policy's ``follow_ups`` the main agent has ``message_agent``:
    its chain carries the roster, before the memory index; a sub-agent type's
    has none."""
    manager = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"],
                        tools=[search_memory]),
            AgentConfig(name="helper", prompt="Help.", description="Helps."),
        ],
        settings=_settings(extras=False),
        subagents=SubAgentPolicy(follow_ups=True),
    )

    try:
        await manager.initialize_services()
        chain = [plugin.name for plugin in manager._runner.app.plugins]
        helpers = [plugin.name for plugin in manager._agent_runners["helper"].app.plugins]
    finally:
        await manager.cleanup()

    assert chain == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "foreign_reasoning",
        "interrupted_calls",
        "tool_result_cap",
        "roster",
        "memory_index",
        "context_overflow",
        "fit_guard",
        "call_sent",
    ]
    assert "roster" not in helpers


async def test_a_native_adk_yaml_app_gets_the_same_chain(project):
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=_settings(extras=False),
        adk_config_path=_native_yaml(project, MODEL, helper=True),
    )

    assert await _chain(manager) == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "foreign_reasoning",
        "interrupted_calls",
        "tool_result_cap",
        "context_overflow",
        "fit_guard",
        "call_sent",
    ]


async def test_a_native_adk_yaml_app_with_every_option(project):
    """The raw log, a caching model and a root whose YAML names a memory tool."""
    path = _native_yaml(project, MODEL, helper=True)
    with open(path, "a", encoding="utf-8") as yaml:
        yaml.write("tools:\n  - name: agentic_cli.tools.memory_tools.search_memory\n")
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=_settings(extras=True), adk_config_path=path,
    )

    assert await _chain(manager) == [
        "main_agent",
        "call_latency",
        "permission",
        "tool_error",
        "llm_logging",
        "foreign_reasoning",
        "context_cache",
        "interrupted_calls",
        "tool_result_cap",
        "memory_index",
        "context_overflow",
        "fit_guard",
        "llm_logging_sent",
        "call_sent",
    ]

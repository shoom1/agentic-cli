"""ADK: the app's whole plugin chain, in order, for each kind of app.

The order is the backend's core invariant. ADK runs a callback of each
plugin in turn and stops at the first that returns a value, so a plugin
placed after one that answers never runs for that call. The chain is built
in two places (``_init_plugins`` gives its head, ``_do_initialize`` the
rest), and these tests pin it whole, so a move anywhere fails one of them:

- ``main_agent`` first: it records the running agent, which the summarizer
  reads;
- ``permission`` before every other tool callback: a denial is the call's
  answer;
- ``tool_error`` answers a call that raised (and one to an unknown tool);
- ``task_progress`` before the cap: it reads the task list after every call
  and returns nothing;
- ``llm_logging`` records each call as the model answers it;
- ``foreign_reasoning`` drops the thoughts another provider wrote, before
  the request is measured;
- ``context_cache`` makes one Gemini cache attempt a turn;
- ``interrupted_calls`` before ``tool_result_cap``: it records every
  finished result, and the cap returns its cut, which ends ADK's after-tool
  callbacks;
- ``memory_index`` before ``context_overflow`` and ``fit_guard``, which keep
  it;
- ``fit_guard`` last but the raw log: it measures the request every other
  plugin edited;
- ``llm_logging_sent`` last: it logs the request as sent.

A native ADK YAML app gets the head alone (and the context cache): its
agents' own after-tool callbacks would be skipped by the cap, and its
agents' context is ADK's own.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.memory_tools import search_memory  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.workflow.test_adk_context_cache import _native_yaml  # noqa: E402

MODEL = "claude-sonnet-4-6"


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
        "permission",
        "tool_error",
        "task_progress",
        "llm_logging",
        "foreign_reasoning",
        "context_cache",
        "interrupted_calls",
        "tool_result_cap",
        "memory_index",
        "context_overflow",
        "fit_guard",
        "llm_logging_sent",
    ]


async def test_a_plain_config_built_app(project):
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Help.")],
        settings=_settings(extras=False),
    )

    assert await _chain(manager) == [
        "main_agent",
        "permission",
        "tool_error",
        "task_progress",
        "foreign_reasoning",
        "interrupted_calls",
        "tool_result_cap",
        "context_overflow",
        "fit_guard",
    ]


async def test_a_native_adk_yaml_app(project):
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=_settings(extras=False),
        adk_config_path=_native_yaml(project, MODEL, helper=True),
    )

    assert await _chain(manager) == ["permission", "tool_error", "task_progress"]


async def test_a_native_adk_yaml_app_with_the_raw_log_and_caching(project):
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=_settings(extras=True),
        adk_config_path=_native_yaml(project, MODEL, helper=True),
    )

    assert await _chain(manager) == [
        "permission", "tool_error", "task_progress", "llm_logging", "context_cache",
    ]

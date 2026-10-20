"""LIVE: the Deep Agents backend's own behaviour against real providers.

Marked ``llm``; each case skips without the keys it needs (the integration
``conftest`` loads keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly:

    python -m pytest tests/integration/test_live_deepagents.py -v -m llm

- Delegation: a coordinator hands a calculation to its sub-agent through
  ``task`` and answers with its result.
- Claude's prompt cache: the second call of a conversation reads the system
  prompt back from the cache (``AnthropicPromptCachingMiddleware``).
- A Gemini to Claude switch that keeps the session: Claude answers from the
  conversation Gemini held, and is sent none of Gemini's thoughts.

Override the models with LIVE_DEEPAGENTS_MODEL (delegation; default
``google:gemini-2.5-flash``), LIVE_DEEPAGENTS_CLAUDE_MODEL (default
``anthropic:claude-sonnet-4-5``) and LIVE_DEEPAGENTS_GEMINI_MODEL (default
``google:gemini-2.5-flash``).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ModelCatalog  # noqa: E402
from agentic_cli.workflow import model_services  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.integration.conftest import collect_events  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    assert_no_errors,
    find_events,
    find_tool_calls,
    find_tool_results,
)

pytestmark = pytest.mark.llm

MODEL = os.getenv("LIVE_DEEPAGENTS_MODEL", "google:gemini-2.5-flash")
CLAUDE = os.getenv("LIVE_DEEPAGENTS_CLAUDE_MODEL", "anthropic:claude-sonnet-4-5")
GEMINI = os.getenv("LIVE_DEEPAGENTS_GEMINI_MODEL", "google:gemini-2.5-flash")


def _needs(*variables: str):
    missing = [v for v in variables if not os.environ.get(v)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(missing)} not set")


def _key_for(model: str) -> str:
    return {"google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[model.partition(":")[0]]


@pytest.fixture
def listed(monkeypatch):
    """Real model listings, as on a first start, written to no cache file."""

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)


def _settings(tmp_path, **values) -> BaseSettings:
    return BaseSettings(workspace_dir=tmp_path, session_store="memory", **values)


def _text(events) -> str:
    return "".join(event.content or "" for event in find_events(events, EventType.TEXT))


@_needs(_key_for(MODEL))
async def test_delegation_and_its_answer(listed, tmp_path):
    manager = DeepAgentsWorkflowManager(
        agent_configs=[
            AgentConfig(
                name="coordinator",
                prompt=(
                    "You never calculate yourself: hand every calculation to the "
                    "calculator sub-agent with the task tool, then give the user its "
                    "result."
                ),
                sub_agents=["calculator"],
            ),
            AgentConfig(
                name="calculator",
                prompt="You multiply numbers. Reply with the product only.",
                description="Multiplies numbers and replies with the product.",
            ),
        ],
        settings=_settings(tmp_path, thinking_effort="low"),
        model=MODEL,
    )
    try:
        events = await collect_events(manager, "What is 6 times 7? Reply with the number only.")
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    (delegation, *_) = find_tool_calls(events, "task")
    assert delegation.metadata["tool_args"]["subagent_type"] == "calculator"
    assert find_tool_results(events, "task"), "the task call got no result"
    assert {e.metadata["agent"] for e in events} == {"coordinator", "calculator"}
    assert "42" in _text(events)


#: Long enough to clear Claude's minimum cacheable prompt (1,024 tokens on
#: Sonnet), with room to spare.
_HANDBOOK = " ".join(
    f"Rule {n}: answer briefly, state facts plainly, and never invent a source."
    for n in range(1, 160)
)


@_needs("ANTHROPIC_API_KEY")
async def test_claude_reads_its_prompt_from_the_cache_on_the_second_call(listed, tmp_path):
    manager = DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(name="assistant", prompt=f"You are concise. {_HANDBOOK}")],
        settings=_settings(tmp_path, thinking_effort="none"),
        model=CLAUDE,
    )
    try:
        first = await collect_events(manager, "Say hello.", session_id="cache")
        second = await collect_events(manager, "Say goodbye.", session_id="cache")
    finally:
        await manager.cleanup()

    assert_no_errors(first + second)
    (written,) = find_events(first, EventType.LLM_USAGE)
    (read,) = find_events(second, EventType.LLM_USAGE)
    assert (written.metadata.get("cache_creation_tokens") or 0) > 0, written.metadata
    assert (read.metadata.get("cached_tokens") or 0) > 0, read.metadata


@_needs("GOOGLE_API_KEY", "ANTHROPIC_API_KEY")
async def test_a_switch_from_gemini_to_claude_keeps_the_session(listed, tmp_path):
    manager = DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(name="assistant", prompt="You are a concise assistant.")],
        settings=_settings(tmp_path, thinking_effort="low"),
        model=GEMINI,
    )
    try:
        told = await collect_events(
            manager, "The project we are discussing is named Quillfeather. Reply OK.",
            session_id="switch",
        )
        await manager.reinitialize(model=CLAUDE, preserve_sessions=True)
        asked = await collect_events(
            manager, "What is the project's name? Reply with the name only.",
            session_id="switch",
        )
    finally:
        await manager.cleanup()

    assert_no_errors(told + asked)
    assert [e.metadata["model"] for e in find_events(asked, EventType.LLM_USAGE)] == [CLAUDE]
    assert "quillfeather" in _text(asked).lower()

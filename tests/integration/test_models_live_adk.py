"""LIVE: every provider on the ADK backend, through the model factory (real API calls).

Marked ``llm``; each case skips without its provider's key (the integration
``conftest`` loads keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly:

    python -m pytest tests/integration/test_models_live_adk.py -v -m llm

Each case runs one agent on the provider's model with thinking on and asks it
to call a tool (``save_plan``, the state tool the ADK manager gives every
agent) before it answers, so the model class, its client, its thinking
parameter and a tool-call round trip all meet the real API. The catalog lists
each provider for real, as on a first start, and caches nothing.

Override a model with LIVE_ADK_GEMINI_MODEL, LIVE_ADK_CLAUDE_ADAPTIVE_MODEL,
LIVE_ADK_CLAUDE_BUDGET_MODEL, LIVE_ADK_GPT_MODEL or LIVE_ADK_OPENROUTER_MODEL
(a ``provider:model`` reference). The vLLM case runs only when
LIVE_VLLM_BASE_URL names a server (its model: LIVE_VLLM_MODEL, a Qwen3-style
model whose chat template switches thinking).

A last case runs two Claude turns in one session with prompt caching on (the
default): the second reads the prompt the first cached. Its model:
LIVE_ADK_CLAUDE_CACHE_MODEL.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ModelCatalog  # noqa: E402
from agentic_cli.workflow import model_services  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.integration.conftest import collect_events  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    assert_no_errors,
    find_events,
    find_tool_calls,
)

pytestmark = pytest.mark.llm

PROMPT = (
    'First call save_plan with the plan "1. Multiply 6 by 7". '
    "Then answer: what is 6 times 7? Reply with the number only."
)
VLLM_ENDPOINT = "live-vllm"


def _case(case_id: str, model: str, needs: str, *, shows_thinking: bool):
    """One provider: its model, the variable it needs, and whether its
    thinking text reliably shows (adaptive Claude and GPT may answer an easy
    question without visible reasoning)."""
    return pytest.param(
        model,
        shows_thinking,
        id=case_id,
        marks=pytest.mark.skipif(not os.environ.get(needs), reason=f"{needs} is not set"),
    )


CASES = [
    _case(
        "gemini",
        os.getenv("LIVE_ADK_GEMINI_MODEL", "google:gemini-2.5-flash"),
        "GOOGLE_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "claude-adaptive",
        os.getenv("LIVE_ADK_CLAUDE_ADAPTIVE_MODEL", "anthropic:claude-sonnet-4-6"),
        "ANTHROPIC_API_KEY",
        shows_thinking=False,
    ),
    _case(
        "claude-budget",
        os.getenv("LIVE_ADK_CLAUDE_BUDGET_MODEL", "anthropic:claude-sonnet-4-5"),
        "ANTHROPIC_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "gpt",
        os.getenv("LIVE_ADK_GPT_MODEL", "openai:gpt-5-mini"),
        "OPENAI_API_KEY",
        shows_thinking=False,
    ),
    _case(
        "openrouter",
        os.getenv("LIVE_ADK_OPENROUTER_MODEL", "openrouter:qwen/qwen3-235b-a22b"),
        "OPENROUTER_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "vllm",
        f"{VLLM_ENDPOINT}:{os.getenv('LIVE_VLLM_MODEL', 'Qwen/Qwen3-32B')}",
        "LIVE_VLLM_BASE_URL",
        shows_thinking=True,
    ),
]


@pytest.fixture
def listed(monkeypatch):
    """Real model listings, as on a first start, written to no cache file.

    (``tests/conftest.py`` otherwise gives every manager an unlisted catalog.)
    """

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)


def _endpoints(model: str) -> dict:
    """The live vLLM server, when the case runs on it."""
    if not model.startswith(f"{VLLM_ENDPOINT}:"):
        return {}
    served = model.partition(":")[2]
    return {
        VLLM_ENDPOINT: {
            "base_url": os.environ["LIVE_VLLM_BASE_URL"],
            "models": {served: {"thinking": "chat_template"}},
        }
    }


@pytest.mark.parametrize("model, shows_thinking", CASES)
async def test_a_tool_call_with_thinking(model, shows_thinking, listed, tmp_path):
    settings = BaseSettings(
        workspace_dir=tmp_path,
        session_store="memory",
        thinking_effort="low",
        model_endpoints=_endpoints(model),
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="assistant", prompt="You are a careful assistant.")],
        settings=settings,
        model=model,
    )
    try:
        events = await collect_events(manager, PROMPT)
    finally:
        await manager.cleanup()

    assert_no_errors(events)
    assert find_tool_calls(events, "save_plan"), "the model did not call the tool"
    answer = "".join(event.content or "" for event in find_events(events, EventType.TEXT))
    assert "42" in answer
    if shows_thinking:
        assert find_events(events, EventType.THINKING), "no thinking text arrived"


#: A system prompt every Claude model caches: about 8000 tokens, where the
#: largest minimum cacheable prompt is 4096 (Opus 4.5 and 4.6, Haiku 4.5).
REFERENCE = "\n".join(
    f"Reference note {n}: item {n} weighs {n * 7 % 97} grams and is filed on shelf {n % 13}."
    for n in range(1, 401)
)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
async def test_a_second_claude_turn_reads_the_cache(listed, tmp_path):
    """The first turn writes the tools, the system instruction and the
    conversation to Claude's cache; the second, in the same session, reads
    them back, and its usage reports the tokens it read."""
    settings = BaseSettings(workspace_dir=tmp_path, session_store="memory", thinking_effort="none")
    manager = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(
                name="assistant",
                prompt=f"You are a careful assistant. Keep these notes at hand.\n\n{REFERENCE}",
            )
        ],
        settings=settings,
        model=os.getenv("LIVE_ADK_CLAUDE_CACHE_MODEL", "anthropic:claude-sonnet-4-5"),
    )
    try:
        first = await collect_events(manager, "Reply with the word ready.", session_id="cached")
        assert manager._runner.app.context_cache_config is not None, "the app does not cache"
        second = await collect_events(manager, "Reply with the word done.", session_id="cached")
    finally:
        await manager.cleanup()

    assert_no_errors(first + second)
    cached = [
        event.metadata.get("cached_tokens") or 0
        for event in find_events(second, EventType.LLM_USAGE)
    ]
    assert cached, "the second turn reported no usage"
    assert max(cached) > 0, f"the second turn read nothing from the cache: {cached}"

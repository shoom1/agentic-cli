"""LIVE: every provider on the Deep Agents backend, through the LangChain adapter (real API calls).

Marked ``llm``; each case skips without its provider's key (the integration
``conftest`` loads keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly:

    python -m pytest tests/integration/test_models_live_deepagents.py -v -m llm

Each case runs one agent on the provider's model with thinking on and asks it
to call a tool (``save_plan``, the state tool every agent gets) before it
answers. That way the chat model class, its client, its thinking parameter
and a tool-call round trip all meet the real API through the whole Deep
Agents stack (the permission gate, summarization, prompt caching on Claude),
including the reasoning each provider needs back with the tool result
(Claude's signed thinking, GPT's encrypted reasoning, OpenRouter's
``reasoning_details``). The catalog lists each provider for real, as on a
first start, and caches nothing.

Override a model with LIVE_DEEPAGENTS_GEMINI_MODEL,
LIVE_DEEPAGENTS_CLAUDE_ADAPTIVE_MODEL, LIVE_DEEPAGENTS_CLAUDE_BUDGET_MODEL,
LIVE_DEEPAGENTS_GPT_MODEL or LIVE_DEEPAGENTS_OPENROUTER_MODEL (a
``provider:model`` reference). The vLLM case runs only when
LIVE_VLLM_BASE_URL names a server (its model: LIVE_VLLM_MODEL, a Qwen3-style
model whose chat template switches thinking).

The Synthetic case runs when SYNTHETIC_API_KEY is set: Synthetic
(synthetic.new) is an OpenAI-compatible service that needs a key, defined as
a ``model_providers`` entry, and its model takes a reasoning effort (its
model: LIVE_SYNTHETIC_MODEL, an id from Synthetic's listing).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("langchain_openrouter")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.conftest import patch_catalogs  # noqa: E402
from tests.integration.conftest import collect_events  # noqa: E402
from tests.models.fakes import openrouter_spec  # noqa: E402
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
#: Synthetic (synthetic.new): an OpenAI-compatible service that needs a key.
SYNTHETIC = "synthetic"
SYNTHETIC_PROVIDER = {
    "base_url": "https://api.synthetic.new/openai/v1",
    "api_key_env": "SYNTHETIC_API_KEY",
}


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
        os.getenv("LIVE_DEEPAGENTS_GEMINI_MODEL", "google:gemini-2.5-flash"),
        "GOOGLE_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "claude-adaptive",
        os.getenv("LIVE_DEEPAGENTS_CLAUDE_ADAPTIVE_MODEL", "anthropic:claude-opus-4-7"),
        "ANTHROPIC_API_KEY",
        shows_thinking=False,
    ),
    _case(
        "claude-budget",
        os.getenv("LIVE_DEEPAGENTS_CLAUDE_BUDGET_MODEL", "anthropic:claude-haiku-4-5"),
        "ANTHROPIC_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "gpt",
        os.getenv("LIVE_DEEPAGENTS_GPT_MODEL", "openai:gpt-5-mini"),
        "OPENAI_API_KEY",
        shows_thinking=False,
    ),
    _case(
        "openrouter",
        os.getenv("LIVE_DEEPAGENTS_OPENROUTER_MODEL", "openrouter:qwen/qwen3.7-plus"),
        "OPENROUTER_API_KEY",
        shows_thinking=True,
    ),
    _case(
        "synthetic",
        f"{SYNTHETIC}:{os.getenv('LIVE_SYNTHETIC_MODEL', 'hf:deepseek-ai/DeepSeek-V4.1-Flash-1017')}",
        "SYNTHETIC_API_KEY",
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

    patch_catalogs(monkeypatch)


def _providers(model: str) -> dict:
    """OpenRouter (its key in OPENROUTER_API_KEY), and Synthetic or the live
    vLLM server when the case runs on it."""
    providers = {"openrouter": openrouter_spec()}
    if model.startswith(f"{SYNTHETIC}:"):
        providers[SYNTHETIC] = SYNTHETIC_PROVIDER
    if model.startswith(f"{VLLM_ENDPOINT}:"):
        providers[VLLM_ENDPOINT] = {"base_url": os.environ["LIVE_VLLM_BASE_URL"]}
    return providers


def _entries(model: str) -> dict:
    """What the case says about its model on Synthetic (a reasoning effort)
    or on the vLLM server (thinking switched by its chat template)."""
    if model.startswith(f"{SYNTHETIC}:"):
        return {model: {"thinking": "effort"}}
    if model.startswith(f"{VLLM_ENDPOINT}:"):
        return {model: {"thinking": "chat_template"}}
    return {}


@pytest.mark.parametrize("model, shows_thinking", CASES)
async def test_a_tool_call_with_thinking(model, shows_thinking, listed, tmp_path):
    settings = BaseSettings(
        workspace_dir=tmp_path,
        session_store="memory",
        thinking_effort="low",
        model_providers=_providers(model),
        models=_entries(model),
    )
    manager = DeepAgentsWorkflowManager(
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
    assert {event.metadata["agent"] for event in events} == {"assistant"}

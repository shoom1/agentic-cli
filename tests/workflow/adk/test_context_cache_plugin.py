"""``ContextCachePlugin``: at most one Gemini cache try an agent's turn, and
what it keeps let go of as each run ends (``workflow/adk/context_cache_plugin.py``).

A sub-agent run is an invocation of its type's runner, and two runs of one
type share that runner's chain, this plugin included: a run that ends, well
or failing, lets go of its own invocation's state and of no other's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.models.cache_metadata import CacheMetadata  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402

from agentic_cli.workflow.adk.context_cache_plugin import ContextCachePlugin  # noqa: E402


def _call(invocation: str, agent: str = "researcher") -> SimpleNamespace:
    return SimpleNamespace(invocation_id=invocation, agent_name=agent)


def _cacheable() -> LlmRequest:
    """A request that could make a cache: ADK's metadata from an earlier call."""
    return LlmRequest(
        cache_config=ContextCacheConfig(),
        cache_metadata=CacheMetadata(fingerprint="f", contents_count=1),
    )


async def _spend(plugin: ContextCachePlugin, invocation: str) -> None:
    """A call of ``invocation`` that could have made a cache and made none."""
    await plugin.before_model_callback(callback_context=_call(invocation), llm_request=_cacheable())
    await plugin.after_model_callback(
        callback_context=_call(invocation),
        llm_response=LlmResponse(cache_metadata=CacheMetadata(fingerprint="f", contents_count=1)),
    )


async def test_after_a_call_made_no_cache_the_turn_tries_no_more():
    plugin = ContextCachePlugin()
    await _spend(plugin, "inv-1")

    again = _cacheable()
    await plugin.before_model_callback(callback_context=_call("inv-1"), llm_request=again)

    assert again.cache_config is None


@pytest.mark.parametrize("ending", ["after_run_callback", "on_run_error_callback"])
async def test_a_run_lets_go_of_what_it_kept_as_it_ends_and_of_nothing_else(ending):
    plugin = ContextCachePlugin()
    await _spend(plugin, "inv-1")
    await _spend(plugin, "inv-2")
    for invocation in ("inv-1", "inv-2"):
        # A call still out: it could make a cache.
        await plugin.before_model_callback(
            callback_context=_call(invocation, "writer"), llm_request=_cacheable()
        )

    context = SimpleNamespace(invocation_id="inv-1")
    if ending == "after_run_callback":
        await plugin.after_run_callback(invocation_context=context)
    else:
        await plugin.on_run_error_callback(invocation_context=context, error=RuntimeError("529"))

    assert plugin._spent == {("inv-2", "researcher")}
    assert set(plugin._could_cache) == {("inv-2", "writer")}

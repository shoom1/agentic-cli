"""Tests for DirectAnthropicLlm (Seam B) and claude-string registration.

Covers: effort → output_config injection, the extra-param client wrapper
(merge + per-call precedence + delegation), and that registering the subclass
makes ``LLMRegistry`` resolve ``claude-*`` strings to it instead of Vertex Claude.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("anthropic")

from google.adk.models.anthropic_llm import AnthropicLlm, Claude  # noqa: E402
from google.adk.models.registry import LLMRegistry  # noqa: E402
import google.adk.models.registry as _registry_mod  # noqa: E402

from agentic_cli.workflow.adk.anthropic_llm import (  # noqa: E402
    DirectAnthropicLlm,
    _ExtraParamClient,
    register_direct_anthropic,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return "RESULT"


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()
        self.base_url = "https://example.test"  # for delegation test


# ---------------------------------------------------------------------------
# effort → output_config
# ---------------------------------------------------------------------------


class TestExtraCreateParams:
    def test_effort_maps_to_output_config(self):
        llm = DirectAnthropicLlm(model="claude-opus-4-8", effort="high")
        assert llm._extra_create_params() == {"output_config": {"effort": "high"}}

    def test_no_effort_no_extra(self):
        llm = DirectAnthropicLlm(model="claude-opus-4-8")
        assert llm._extra_create_params() == {}

    def test_extra_params_plus_effort_merge(self):
        llm = DirectAnthropicLlm(
            model="claude-opus-4-8", effort="max",
            extra_params={"service_tier": "auto"},
        )
        params = llm._extra_create_params()
        assert params["service_tier"] == "auto"
        assert params["output_config"] == {"effort": "max"}

    def test_explicit_output_config_effort_preserved(self):
        llm = DirectAnthropicLlm(
            model="claude-opus-4-8", effort="low",
            extra_params={"output_config": {"effort": "max"}},
        )
        # setdefault: an explicit output_config.effort wins over the field
        assert llm._extra_create_params()["output_config"] == {"effort": "max"}


# ---------------------------------------------------------------------------
# Client wrapper: merge, per-call precedence, delegation
# ---------------------------------------------------------------------------


class TestExtraParamClient:
    async def test_merges_extra_into_create(self):
        fake = _FakeClient()
        wrapped = _ExtraParamClient(fake, {"output_config": {"effort": "high"}})
        out = await wrapped.messages.create(model="claude-opus-4-8", messages=[])
        assert out == "RESULT"
        call = fake.messages.calls[0]
        assert call["output_config"] == {"effort": "high"}
        assert call["model"] == "claude-opus-4-8"
        assert call["messages"] == []

    async def test_per_call_kwarg_wins_over_extra(self):
        fake = _FakeClient()
        wrapped = _ExtraParamClient(fake, {"service_tier": "auto"})
        await wrapped.messages.create(model="m", service_tier="batch")
        assert fake.messages.calls[0]["service_tier"] == "batch"

    def test_delegates_unknown_attributes(self):
        fake = _FakeClient()
        wrapped = _ExtraParamClient(fake, {})
        assert wrapped.base_url == "https://example.test"


# ---------------------------------------------------------------------------
# _anthropic_client wiring (no real network / API key)
# ---------------------------------------------------------------------------


class TestAnthropicClientWiring:
    def test_no_extra_returns_plain_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        llm = DirectAnthropicLlm(model="claude-opus-4-8")  # no effort
        from anthropic import AsyncAnthropic

        assert isinstance(llm._anthropic_client, AsyncAnthropic)

    def test_effort_returns_wrapped_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        llm = DirectAnthropicLlm(model="claude-opus-4-8", effort="high")
        assert isinstance(llm._anthropic_client, _ExtraParamClient)


# ---------------------------------------------------------------------------
# Registry: claude strings → DirectAnthropicLlm
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_registry():
    """Snapshot and restore the global LLM registry + resolve cache."""
    snapshot = dict(_registry_mod._llm_registry_dict)
    LLMRegistry.resolve.cache_clear()
    try:
        yield
    finally:
        _registry_mod._llm_registry_dict.clear()
        _registry_mod._llm_registry_dict.update(snapshot)
        LLMRegistry.resolve.cache_clear()


class TestRegistration:
    def test_register_overrides_vertex_claude(self, _isolated_registry):
        # Self-contained: force ADK's default (Vertex Claude), confirm it, then
        # confirm our registration overrides it for the same string.
        LLMRegistry.register(Claude)
        LLMRegistry.resolve.cache_clear()
        assert LLMRegistry.resolve("claude-sonnet-4-5") is Claude

        register_direct_anthropic()
        assert LLMRegistry.resolve("claude-sonnet-4-5") is DirectAnthropicLlm

    def test_register_resolves_3x_and_4x(self, _isolated_registry):
        register_direct_anthropic()
        assert LLMRegistry.resolve("claude-opus-4-8") is DirectAnthropicLlm
        assert LLMRegistry.resolve("claude-3-5-haiku") is DirectAnthropicLlm

    def test_catch_all_covers_newer_families(self, _isolated_registry):
        register_direct_anthropic()
        # claude-fable-5 isn't matched by ADK's claude-.*-4.* patterns at all.
        assert LLMRegistry.resolve("claude-fable-5") is DirectAnthropicLlm

    def test_new_llm_constructs_instance_with_defaults(self, _isolated_registry):
        register_direct_anthropic()
        inst = LLMRegistry.new_llm("claude-opus-4-8")
        assert isinstance(inst, DirectAnthropicLlm)
        assert isinstance(inst, AnthropicLlm)  # still an AnthropicLlm
        assert inst.model == "claude-opus-4-8"
        assert inst.max_tokens == 8192  # class default — per-agent override needs an instance
        assert inst.effort is None

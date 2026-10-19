"""resolve_model: one agent's model, thinking, max_tokens and sampling settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentic_cli.models import (
    DeclaredModel,
    EndpointSpec,
    ModelRecord,
    ModelRef,
    ModelResolutionError,
    NativeThinking,
    ProviderKind,
    ResolvedModel,
    RoleMap,
    ThinkingMechanism,
    ThinkingRule,
    resolve_model,
)
from tests.models.fakes import FakeCatalog, endpoints, table_records

G, A, O = ProviderKind.GOOGLE, ProviderKind.ANTHROPIC, ProviderKind.OPENAI
M = ThinkingMechanism

ALL_KEYS = {"google": "k-g", "anthropic": "k-a", "openai": "k-o", "openrouter": "k-r"}
LAB = EndpointSpec(base_url="http://gpu-box:8000/v1")

OPENROUTER_REASONING = ModelRecord(
    provider="openrouter",
    kind=ProviderKind.OPENROUTER,
    model="qwen/qwen3-235b-a22b",
    thinking=ThinkingRule(
        mechanism=M.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
    ),
)
LAB_EFFORT = ModelRecord(
    provider="lab", kind=ProviderKind.OPENAI_COMPATIBLE, model="gpt-oss-120b",
    thinking=ThinkingRule(mechanism=M.EFFORT, levels=("low", "medium", "high")),
)
LAB_CHAT_TEMPLATE = ModelRecord(
    provider="lab", kind=ProviderKind.OPENAI_COMPATIBLE, model="Qwen/Qwen3-32B",
    thinking=ThinkingRule(mechanism=M.CHAT_TEMPLATE),
)
CATALOG = FakeCatalog(
    [OPENROUTER_REASONING, LAB_EFFORT, LAB_CHAT_TEMPLATE],
    kinds={"lab": ProviderKind.OPENAI_COMPATIBLE},
)


@dataclass
class Thinking:
    mode: str
    budget_tokens: int | None = None


@dataclass
class Generation:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    thinking: Thinking | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve(spec: str | None, *, mode: str | None = None, budget: int | None = None,
             generation: Generation | None = None, effort: str = "medium",
             agent: str | None = "writer", roles: RoleMap | None = None,
             catalog: FakeCatalog = CATALOG, keys: dict | None = None) -> ResolvedModel:
    if mode is not None:
        generation = generation or Generation()
        generation.thinking = Thinking(mode, budget)
    return resolve_model(
        agent=agent,
        spec=spec,
        generation=generation,
        thinking_effort=effort,
        roles=roles or RoleMap(default_model=None, model_roles={}),
        catalog=catalog,
        endpoints=endpoints(specs={"lab": LAB}, **(ALL_KEYS if keys is None else keys)),
    )


def _thinking(spec: str, mode: str, budget: int | None = None) -> NativeThinking:
    return _resolve(spec, mode=mode, budget=budget).thinking


# --- the mapping table, one case per row and level ---------------------------

ON, OFF = True, False


@pytest.mark.parametrize(
    ("spec", "mode", "budget", "expected"),
    [
        # Gemini 2.5 (budget): none = 0, or the minimum if it cannot turn off.
        ("gemini-2.5-flash", "none", None, (M.GEMINI_BUDGET, OFF, 0, None)),
        ("gemini-2.5-flash", "low", None, (M.GEMINI_BUDGET, ON, 4096, None)),
        ("gemini-2.5-flash", "medium", None, (M.GEMINI_BUDGET, ON, 12288, None)),
        ("gemini-2.5-flash", "high", None, (M.GEMINI_BUDGET, ON, 24576, None)),
        ("gemini-2.5-flash", "budget", 5000, (M.GEMINI_BUDGET, ON, 5000, None)),
        ("gemini-2.5-flash", "budget", 99999, (M.GEMINI_BUDGET, ON, 24576, None)),
        ("gemini-2.5-flash", "budget", 0, (M.GEMINI_BUDGET, OFF, 0, None)),
        ("gemini-2.5-flash-lite", "budget", 100, (M.GEMINI_BUDGET, ON, 512, None)),
        ("gemini-2.5-pro", "none", None, (M.GEMINI_BUDGET, ON, 128, None)),
        ("gemini-2.5-pro", "budget", 0, (M.GEMINI_BUDGET, ON, 128, None)),
        ("gemini-2.5-pro", "budget", 40000, (M.GEMINI_BUDGET, ON, 32768, None)),
        ("gemini-2.5-pro", "budget", None, (M.GEMINI_BUDGET, ON, 12288, None)),
        # Gemini 3+ (level): none = the lowest level; 3 Pro: medium becomes high.
        ("gemini-3-pro-preview", "none", None, (M.GEMINI_LEVEL, ON, None, "low")),
        ("gemini-3-pro-preview", "low", None, (M.GEMINI_LEVEL, ON, None, "low")),
        ("gemini-3-pro-preview", "medium", None, (M.GEMINI_LEVEL, ON, None, "high")),
        ("gemini-3-pro-preview", "high", None, (M.GEMINI_LEVEL, ON, None, "high")),
        ("gemini-3.1-pro-preview", "medium", None, (M.GEMINI_LEVEL, ON, None, "medium")),
        ("gemini-3.5-flash", "none", None, (M.GEMINI_LEVEL, ON, None, "minimal")),
        ("gemini-3.5-flash", "medium", None, (M.GEMINI_LEVEL, ON, None, "medium")),
        ("gemini-3.5-flash", "minimal", None, (M.GEMINI_LEVEL, ON, None, "minimal")),
        # Claude adaptive: disabled, or effort low if it cannot disable.
        ("claude-sonnet-4-6", "none", None, (M.ANTHROPIC_ADAPTIVE, OFF, None, None)),
        ("claude-sonnet-4-6", "low", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "low")),
        ("claude-sonnet-4-6", "medium", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "medium")),
        ("claude-sonnet-4-6", "high", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "high")),
        ("claude-sonnet-4-6", "max", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "max")),
        ("claude-opus-5-5", "none", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "low")),
        ("claude-opus-5-5", "xhigh", None, (M.ANTHROPIC_ADAPTIVE, ON, None, "xhigh")),
        # Claude budget (up to 4.5): disabled, 4096 / 10000 / 32000, N at least 1024.
        ("claude-sonnet-4-5", "none", None, (M.ANTHROPIC_BUDGET, OFF, 0, None)),
        ("claude-sonnet-4-5", "low", None, (M.ANTHROPIC_BUDGET, ON, 4096, None)),
        ("claude-sonnet-4-5", "medium", None, (M.ANTHROPIC_BUDGET, ON, 10000, None)),
        ("claude-sonnet-4-5", "high", None, (M.ANTHROPIC_BUDGET, ON, 32000, None)),
        ("claude-sonnet-4-5", "budget", 500, (M.ANTHROPIC_BUDGET, ON, 1024, None)),
        ("claude-sonnet-4-5", "budget", 20000, (M.ANTHROPIC_BUDGET, ON, 20000, None)),
        ("claude-sonnet-4-5", "budget", 90000, (M.ANTHROPIC_BUDGET, ON, 63999, None)),
        # OpenAI / OpenRouter reasoning, endpoint effort: none = the lowest effort.
        ("gpt-5.5", "none", None, (M.EFFORT, OFF, None, "none")),
        ("gpt-5.5", "low", None, (M.EFFORT, ON, None, "low")),
        ("gpt-5.5", "medium", None, (M.EFFORT, ON, None, "medium")),
        ("gpt-5.5", "high", None, (M.EFFORT, ON, None, "high")),
        ("gpt-5.5", "xhigh", None, (M.EFFORT, ON, None, "xhigh")),
        ("gpt-5", "none", None, (M.EFFORT, ON, None, "minimal")),
        ("o4-mini", "none", None, (M.EFFORT, ON, None, "low")),
        ("openrouter:qwen/qwen3-235b-a22b", "none", None, (M.EFFORT, OFF, None, "none")),
        ("openrouter:qwen/qwen3-235b-a22b", "high", None, (M.EFFORT, ON, None, "high")),
        ("lab:gpt-oss-120b", "none", None, (M.EFFORT, ON, None, "low")),
        ("lab:gpt-oss-120b", "medium", None, (M.EFFORT, ON, None, "medium")),
        # Endpoint chat_template: enable_thinking false / true.
        ("lab:Qwen/Qwen3-32B", "none", None, (M.CHAT_TEMPLATE, OFF, None, None)),
        ("lab:Qwen/Qwen3-32B", "low", None, (M.CHAT_TEMPLATE, ON, None, None)),
        ("lab:Qwen/Qwen3-32B", "high", None, (M.CHAT_TEMPLATE, ON, None, None)),
    ],
)
def test_the_mapping_table(spec, mode, budget, expected):
    thinking = _thinking(spec, mode, budget)

    assert (thinking.mechanism, thinking.enabled, thinking.budget, thinking.level) == expected


@pytest.mark.parametrize(
    ("spec", "description", "allowed"),
    [
        ("claude-opus-5-5", "adaptive thinking", "none, low, medium, high, xhigh or max"),
        ("claude-sonnet-4-6", "adaptive thinking", "none, low, medium, high or max"),
        ("gemini-3-pro-preview", "thinking levels", "none, low, medium or high"),
        ("gemini-3.5-flash", "thinking levels", "none, minimal, low, medium or high"),
        ("gpt-5.5", "reasoning effort", "none, low, medium, high or xhigh"),
        ("lab:Qwen/Qwen3-32B", "chat-template thinking", "none, low, medium or high"),
    ],
)
def test_budget_on_a_model_without_budget_thinking_is_an_error(spec, description, allowed):
    with pytest.raises(ModelResolutionError) as exc:
        _resolve(spec, mode="budget", budget=8000)

    ref = str(_resolve(spec, mode="none").ref)
    assert str(exc.value) == (
        f"Agent 'writer': thinking mode 'budget' is not available on {ref} "
        f"({description}); use {allowed}."
    )


def test_the_specs_example_message():
    with pytest.raises(ModelResolutionError) as exc:
        _resolve("anthropic:claude-opus-5-5", mode="budget", budget=8000)

    assert str(exc.value) == (
        "Agent 'writer': thinking mode 'budget' is not available on anthropic:claude-opus-5-5 "
        "(adaptive thinking); use none, low, medium, high, xhigh or max."
    )


@pytest.mark.parametrize(
    ("spec", "mode"),
    [
        ("claude-sonnet-4-6", "xhigh"),  # 4.6's ladder stops at high, max
        ("gpt-5.5", "max"),  # 5.5's ladder stops at xhigh
        ("gemini-3-pro-preview", "minimal"),
        ("gemini-2.5-flash", "xhigh"),  # budget models have no ladder
        ("claude-sonnet-4-5", "max"),
        ("lab:Qwen/Qwen3-32B", "xhigh"),
        ("gpt-5.5", "turbo"),
    ],
)
def test_a_native_level_off_the_ladder_is_an_error(spec, mode):
    with pytest.raises(ModelResolutionError, match=f"thinking mode '{mode}' is not available"):
        _resolve(spec, mode=mode)


def test_a_native_level_on_the_ladder_is_accepted():
    assert _thinking("claude-opus-4-7", "xhigh").level == "xhigh"
    assert _thinking("gpt-5.6", "max").level == "max"


def test_a_generic_level_the_ladder_lacks_is_an_error():
    record = ModelRecord(
        provider="lab", kind=ProviderKind.OPENAI_COMPATIBLE, model="odd",
        thinking=ThinkingRule(mechanism=M.EFFORT, levels=("low", "high")),
    )

    with pytest.raises(ModelResolutionError, match="use none, low or high"):
        _resolve("lab:odd", mode="medium", catalog=FakeCatalog([record]))


@pytest.mark.parametrize("mode", ["none", "high", "budget", "xhigh"])
def test_a_model_without_a_knob_ignores_thinking(mode):
    assert _thinking("gpt-4.1", mode, 4000) is None
    assert _thinking("openrouter:meta-llama/llama-4-maverick", mode, 4000) is None


@pytest.mark.parametrize(
    ("spec", "mode", "label"),
    [
        ("claude-sonnet-5-5", "high", "adaptive, effort high"),
        ("claude-sonnet-4-6", "none", "off"),
        ("claude-sonnet-4-5", "high", "budget 32000"),
        ("gemini-2.5-flash", "medium", "budget 12288"),
        ("gemini-3.5-flash", "high", "level high"),
        ("gpt-5.5", "low", "effort low"),
        ("lab:Qwen/Qwen3-32B", "high", "thinking on"),
    ],
)
def test_the_label(spec, mode, label):
    assert _thinking(spec, mode).label == label


class TestThinkingIntent:
    def test_the_global_effort_applies_without_an_agent_setting(self):
        assert _resolve("gpt-5.5", effort="high").thinking.level == "high"
        assert _resolve("gpt-5.5", generation=Generation(), effort="low").thinking.level == "low"

    def test_the_agent_setting_wins(self):
        assert _resolve("gpt-5.5", mode="xhigh", effort="low").thinking.level == "xhigh"


# --- max_tokens --------------------------------------------------------------


class TestMaxTokens:
    def test_the_agent_value(self):
        assert _resolve("gemini-2.5-flash", generation=Generation(max_tokens=4000)).max_tokens == 4000

    def test_no_default_outside_anthropic(self):
        assert _resolve("gemini-2.5-flash").max_tokens is None
        assert _resolve("gpt-5.5").max_tokens is None

    def test_the_anthropic_default(self):
        assert _resolve("claude-sonnet-4-6", mode="high").max_tokens == 8192
        assert _resolve("claude-sonnet-4-5", mode="none").max_tokens == 8192

    def test_clamped_to_max_output(self):
        assert _resolve("gemini-2.5-flash", generation=Generation(max_tokens=100000)).max_tokens == 65536

    def test_budget_claude_with_a_small_max_tokens(self):
        """The ADK 2 regression: max_tokens must stay above the thinking budget."""
        resolved = _resolve("claude-sonnet-4-5", generation=Generation(max_tokens=2000), mode="high")

        assert resolved.thinking.budget == 32000
        assert resolved.max_tokens == 40192

    def test_budget_claude_raises_the_default_too(self):
        assert _resolve("claude-sonnet-4-5", mode="high").max_tokens == 40192

    def test_budget_claude_keeps_a_larger_max_tokens(self):
        assert _resolve("claude-sonnet-4-5", mode="high", generation=Generation(max_tokens=50000)).max_tokens == 50000

    def test_a_large_budget_stays_below_max_tokens(self):
        resolved = _resolve("claude-sonnet-4-5", mode="budget", budget=60000)

        assert resolved.thinking.budget == 60000
        assert resolved.max_tokens == 64000
        assert resolved.max_tokens > resolved.thinking.budget


# --- sampling ----------------------------------------------------------------

SAMPLING = dict(temperature=0.3, top_p=0.9, top_k=40)


class TestSampling:
    def test_kept_when_the_model_takes_it(self):
        resolved = _resolve("gemini-2.5-flash", generation=Generation(**SAMPLING), mode="high")

        assert (resolved.temperature, resolved.top_p, resolved.top_k) == (0.3, 0.9, 40)
        assert resolved.dropped == ()

    def test_dropped_while_claude_thinks(self):
        resolved = _resolve("claude-sonnet-4-6", generation=Generation(**SAMPLING), mode="high")

        assert (resolved.temperature, resolved.top_p, resolved.top_k) == (None, None, None)
        assert resolved.dropped == ("temperature", "top_p", "top_k")

    def test_kept_while_claude_does_not_think(self):
        resolved = _resolve("claude-sonnet-4-6", generation=Generation(temperature=0.3), mode="none")

        assert resolved.temperature == 0.3
        assert resolved.dropped == ()

    def test_dropped_when_claude_cannot_stop_thinking(self):
        resolved = _resolve("claude-opus-4-6", generation=Generation(temperature=0.3), mode="none")
        assert resolved.temperature == 0.3  # 4.6 can disable

        resolved = _resolve("claude-opus-5-5", generation=Generation(temperature=0.3), mode="none")
        assert resolved.temperature is None
        assert resolved.dropped == ("temperature",)

    @pytest.mark.parametrize("spec", ["claude-fable-5", "gpt-5.5", "o3"])
    def test_always_dropped_on_a_sampling_free_model(self, spec):
        resolved = _resolve(spec, generation=Generation(temperature=0.3, top_p=0.5), mode="none")

        assert (resolved.temperature, resolved.top_p) == (None, None)
        assert resolved.dropped == ("temperature", "top_p")

    def test_stop_sequences_and_extra_pass_through(self):
        extra = {"seed": 7, "response_mime_type": "application/json"}

        resolved = _resolve(
            "claude-fable-5", generation=Generation(stop_sequences=["END"], extra=extra), mode="high"
        )

        assert resolved.stop_sequences == ("END",)
        assert dict(resolved.extra) == extra


# --- the reference, the endpoint and the errors ------------------------------


def test_the_resolved_model_carries_the_record_and_endpoint():
    resolved = _resolve("anthropic:claude-sonnet-4-5")

    assert resolved.agent == "writer"
    assert resolved.ref == ModelRef("anthropic", "claude-sonnet-4-5")
    assert resolved.record.ref == resolved.ref
    assert resolved.endpoint.name == "anthropic"
    assert resolved.endpoint.api_key == "k-a"


def test_no_spec_means_the_default_role():
    roles = RoleMap(default_model="google:gemini-2.5-flash", model_roles={})

    assert _resolve(None, roles=roles).ref == ModelRef("google", "gemini-2.5-flash")


def test_a_role_spec_resolves_through_the_role_map():
    roles = RoleMap(default_model=None, model_roles={"fast": "lab:Qwen/Qwen3-32B"})

    resolved = _resolve("fast", roles=roles)

    assert resolved.ref == ModelRef("lab", "Qwen/Qwen3-32B")
    assert resolved.endpoint.kind is ProviderKind.OPENAI_COMPATIBLE


def test_an_unknown_endpoint_is_an_error_naming_agent_and_endpoint():
    with pytest.raises(ModelResolutionError, match=r"^Agent 'writer': .*'gpu'.*model_endpoints"):
        _resolve("gpu:llama-3")


def test_a_provider_without_a_key_is_an_error_naming_the_variable():
    with pytest.raises(ModelResolutionError, match="OPENAI_API_KEY") as exc:
        _resolve("gpt-5.5", keys={"google": "k"})

    assert "Agent 'writer'" in str(exc.value)


def test_an_unknown_id_on_an_authoritative_provider_is_an_error():
    catalog = FakeCatalog(table_records(G, "gemini-2.5-flash"), authoritative=["google"])

    with pytest.raises(ModelResolutionError, match=r"^Agent 'writer': .*gemini-9-ultra.*gemini-2\.5-flash"):
        _resolve("gemini-9-ultra", catalog=catalog)


def test_a_malformed_spec_is_a_resolution_error():
    with pytest.raises(ModelResolutionError, match="^Agent 'writer': .*'anthropic:'"):
        _resolve("anthropic:")


def test_the_default_model_is_named_without_an_agent():
    with pytest.raises(ModelResolutionError, match="^Default model: "):
        _resolve("claude-opus-5-5", mode="budget", agent=None)

"""The ADK model factory: a resolved model as the model, config and planner an ADK agent takes.

Each case resolves an agent's model the way the manager does
(``resolve_model``, against the shipped table), builds it with
``build_adk_model`` and reads what ADK will send: the instance and its
client, the agent's ``generate_content_config`` and the planner's thinking
config, and where it matters the request ADK's own model class builds,
against a fake provider API.
"""

from __future__ import annotations

import warnings

import httpx
import pytest

pytest.importorskip("google.adk")

import google.adk.models.registry as adk_registry  # noqa: E402
from anthropic import AsyncAnthropic  # noqa: E402
from anthropic._constants import DEFAULT_TIMEOUT as ANTHROPIC_DEFAULT_TIMEOUT  # noqa: E402
from google.adk.integrations.openai import (  # noqa: E402
    OpenAIGenerateContentConfig,
    OpenAIResponsesLlm,
)
from google.adk.models import Gemini  # noqa: E402
from google.adk.models.anthropic_llm import (  # noqa: E402
    AnthropicGenerateContentConfig,
    AnthropicLlm,
    Claude,
)
from google.adk.models.registry import LLMRegistry  # noqa: E402
from google.genai import types  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from agentic_cli.models import (  # noqa: E402
    DeclaredModel,
    EndpointSpec,
    ModelCatalog,
    ModelRecord,
    ProviderKind,
    ResolvedModel,
    RoleMap,
    ThinkingMechanism,
    ThinkingRule,
    resolve_model,
)
from agentic_cli.workflow.adk.anthropic_llm import (  # noqa: E402
    DirectAnthropicLlm,
    register_direct_anthropic,
)
from agentic_cli.workflow.adk.model_factory import AdkModel, build_adk_model  # noqa: E402
from agentic_cli.workflow.adk.openai_compatible import OpenAICompatibleLlm  # noqa: E402
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings  # noqa: E402
from tests.models.fakes import FakeCatalog, endpoints  # noqa: E402
from tests.workflow.adk.fakes import (  # noqa: E402
    FakeAPI,
    anthropic_message,
    chat_completion,
    generate,
    openai_response,
    run_agent,
    serve,
)

KEYS = {
    "google": "g-test-key",
    "anthropic": "a-test-key",
    "openai": "o-test-key",
    "openrouter": "r-test-key",
}
Level = types.ThinkingLevel


@pytest.fixture
def settings(mock_context):
    return mock_context.settings.model_copy(
        update={
            "retry_max_attempts": 4,
            "retry_initial_delay": 1.5,
            "retry_backoff_factor": 3.0,
            "anthropic_request_timeout": 321.0,
        }
    )


#: An OpenRouter model whose listing says it reasons (see ModelCatalog).
OPENROUTER_REASONER = ModelRecord(
    provider="openrouter",
    kind=ProviderKind.OPENROUTER,
    model="qwen/qwen3-235b-a22b",
    thinking=ThinkingRule(
        ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
    ),
    reasoning_text=True,
)
#: A user's vLLM server, its key in LOCAL_KEY.
LOCAL = {
    "local": EndpointSpec(
        base_url="http://localhost:8000/v1",
        api_key_env="LOCAL_KEY",
        models={
            "Qwen/Qwen3-32B": DeclaredModel(thinking="chat_template"),
            "reasoner": DeclaredModel(thinking="effort", effort_levels=["low", "medium", "high"]),
            "plain": DeclaredModel(),
        },
    ),
    "keyless": EndpointSpec(base_url="http://localhost:9000/v1", models={"plain": DeclaredModel()}),
}


def resolved(
    spec: str,
    *,
    effort: str = "none",
    thinking: str | None = None,
    budget: int | None = None,
    **generation,
) -> ResolvedModel:
    """``spec`` resolved for an agent, as the manager resolves it.

    ``effort`` is the global ``thinking_effort``; ``thinking`` (with
    ``budget``) the agent's own mode; ``generation`` its other model settings.
    The catalog is the shipped table, plus ``OPENROUTER_REASONER`` and the
    models ``LOCAL`` declares.
    """
    if thinking is not None:
        generation["thinking"] = ThinkingSettings(mode=thinking, budget_tokens=budget)
    every = endpoints(**KEYS, specs=LOCAL, env={"LOCAL_KEY": "l-test-key"})
    declared = ModelCatalog(endpoints=every)  # never loaded: declarations and the table
    catalog = FakeCatalog([
        OPENROUTER_REASONER,
        *declared.records("local"),
        *declared.records("keyless"),
    ])
    return resolve_model(
        agent="writer",
        spec=spec,
        generation=ModelSettings(**generation) if generation else None,
        thinking_effort=effort,
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=catalog,
        endpoints=every,
    )


def build(spec: str, settings, **kwargs) -> AdkModel:
    return build_adk_model(resolved(spec, **kwargs), settings=settings, app_name="factory-test")


def thinking_config(built: AdkModel) -> types.ThinkingConfig:
    assert built.planner is not None
    return built.planner.thinking_config


# --- what every provider shares -----------------------------------------------------


class TestGenerationSettings:
    def test_the_agent_s_settings_go_into_the_config(self, settings):
        config = build(
            "gemini-2.5-flash", settings,
            temperature=0.3, top_p=0.9, top_k=40, max_tokens=1000, stop_sequences=["END"],
        ).generate_content_config

        assert config.temperature == 0.3
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_output_tokens == 1000
        assert config.stop_sequences == ["END"]

    def test_unset_settings_stay_unset(self, settings):
        config = build("gemini-2.5-flash", settings).generate_content_config

        assert (config.temperature, config.top_p, config.top_k) == (None, None, None)
        assert config.max_output_tokens is None
        assert config.stop_sequences is None

    def test_extra_keys_the_config_takes_pass_through_and_the_rest_are_dropped(self, settings):
        config = build(
            "gemini-2.5-flash", settings, extra={"seed": 7, "definitely_not_a_field": 1}
        ).generate_content_config

        assert config.seed == 7
        assert "definitely_not_a_field" not in config.model_dump()

    @pytest.mark.parametrize(
        "spec, effort",
        [("claude-sonnet-4-5", "high"), ("claude-opus-4-7", "none"), ("gpt-5.5", "none")],
    )
    def test_dropped_sampling_never_reaches_the_config(self, settings, spec, effort):
        """Claude while it thinks, Claude 4.6+ and GPT reasoning models reject
        sampling settings; resolution dropped them and the config never has them."""
        model = resolved(spec, effort=effort, temperature=0.3, top_p=0.9, top_k=40)
        config = build_adk_model(model, settings=settings, app_name="t").generate_content_config

        assert model.dropped == ("temperature", "top_p", "top_k")
        assert (config.temperature, config.top_p, config.top_k) == (None, None, None)


# --- Google -------------------------------------------------------------------------


class TestGemini:
    def test_a_gemini_model_with_its_own_key(self, settings):
        built = build("google:gemini-2.5-flash", settings)

        assert type(built.model) is Gemini
        assert built.model.model == "gemini-2.5-flash"
        assert built.model.client_kwargs == {"api_key": "g-test-key"}
        assert type(built.generate_content_config) is types.GenerateContentConfig

    def test_retries_and_a_timeout_ride_in_the_config(self, settings):
        http = build("gemini-2.5-flash", settings).generate_content_config.http_options

        assert http.retry_options == types.HttpRetryOptions(
            initial_delay=1.5, attempts=4, exp_base=3.0, http_status_codes=[500, 502, 503, 504]
        )
        assert 429 not in http.retry_options.http_status_codes  # a rate limit fails the turn
        assert http.timeout == 600_000  # milliseconds: google-genai's own default is none

    @pytest.mark.parametrize(
        "model, effort, budget",
        [
            ("gemini-2.5-flash", "low", 4096),
            ("gemini-2.5-flash", "medium", 12288),
            ("gemini-2.5-flash", "high", 24576),
            ("gemini-2.5-pro", "high", 24576),
        ],
    )
    def test_gemini_2_5_thinks_on_a_budget(self, settings, model, effort, budget):
        config = thinking_config(build(model, settings, effort=effort))

        assert config.thinking_budget == budget
        assert config.thinking_level is None  # Gemini 2.5 answers a level with HTTP 400
        assert config.include_thoughts is True

    def test_none_turns_gemini_2_5_off_where_the_model_allows_it(self, settings):
        """0.6 sent no thinking config for ``none``: Flash and Pro kept thinking
        at their own default."""
        flash = thinking_config(build("gemini-2.5-flash", settings, effort="none"))
        pro = thinking_config(build("gemini-2.5-pro", settings, effort="none"))

        assert (flash.thinking_budget, flash.include_thoughts) == (0, False)
        assert pro.thinking_budget == 128  # Pro cannot turn thinking off: its minimum

    def test_an_explicit_budget(self, settings):
        config = thinking_config(
            build("gemini-2.5-flash", settings, thinking="budget", budget=5000)
        )

        assert (config.thinking_budget, config.thinking_level) == (5000, None)

    @pytest.mark.parametrize(
        "model, effort, level",
        [
            ("gemini-3-flash-preview", "none", Level.MINIMAL),  # cannot turn off: lowest
            ("gemini-3-flash-preview", "low", Level.LOW),
            ("gemini-3-flash-preview", "medium", Level.MEDIUM),
            ("gemini-3-flash-preview", "high", Level.HIGH),
            ("gemini-3-pro-preview", "none", Level.LOW),
            ("gemini-3-pro-preview", "low", Level.LOW),
            ("gemini-3-pro-preview", "medium", Level.HIGH),  # 3 Pro has no medium
            ("gemini-3-pro-preview", "high", Level.HIGH),
        ],
    )
    def test_gemini_3_thinks_in_levels(self, settings, model, effort, level):
        config = thinking_config(build(model, settings, effort=effort))

        assert config.thinking_level == level
        assert config.thinking_budget is None
        assert config.include_thoughts is True

    def test_a_native_level(self, settings):
        config = thinking_config(build("gemini-3-flash-preview", settings, thinking="minimal"))

        assert config.thinking_level == Level.MINIMAL

    @pytest.mark.parametrize(
        "model, high", [("gemini-2.5-flash", 24576), ("claude-sonnet-4-5", 32000)]
    )
    def test_the_agent_s_mode_wins_over_the_global_effort(self, settings, model, high):
        on = thinking_config(build(model, settings, effort="none", thinking="high"))
        off = thinking_config(build(model, settings, effort="high", thinking="none"))

        assert (on.thinking_budget, off.thinking_budget) == (high, 0)

    def test_the_model_decides_budget_or_level(self, settings):
        gemini_2_5 = thinking_config(build("gemini-2.5-flash", settings, effort="medium"))
        gemini_3 = thinking_config(build("gemini-3-flash-preview", settings, effort="medium"))

        assert (gemini_2_5.thinking_budget, gemini_2_5.thinking_level) == (12288, None)
        assert (gemini_3.thinking_budget, gemini_3.thinking_level) == (None, Level.MEDIUM)

    def test_a_model_without_a_thinking_knob_has_no_planner(self, settings):
        assert build("gemini-2.0-flash", settings, effort="high").planner is None


# --- Anthropic ----------------------------------------------------------------------


class TestClaude:
    def test_claude_runs_on_the_direct_api_with_its_own_client(self, settings):
        built = build("anthropic:claude-sonnet-4-5", settings)

        assert type(built.model) is AnthropicLlm  # not Claude, which is Vertex AI
        assert built.model.model == "claude-sonnet-4-5"
        client = built.model.client
        assert isinstance(client, AsyncAnthropic)
        assert client.api_key == "a-test-key"
        assert client.timeout == 321.0  # anthropic_request_timeout
        assert client.max_retries == 4  # retry_max_attempts
        assert type(built.generate_content_config) is AnthropicGenerateContentConfig
        # ADK's Claude ignores http_options: the client retries.
        assert built.generate_content_config.http_options is None

    def test_the_client_s_timeout_lets_a_large_max_tokens_run_unstreamed(self, settings):
        """The Anthropic SDK refuses an unstreamed request whose max_tokens could
        take over ten minutes, unless the client has its own timeout; ADK sends
        an agent's requests unstreamed."""
        built = build("claude-sonnet-4-5", settings, effort="high")

        assert built.model.max_tokens > 21_333
        assert built.model.client.timeout != ANTHROPIC_DEFAULT_TIMEOUT

    @pytest.mark.parametrize(
        "effort, budget", [("none", 0), ("low", 4096), ("medium", 10000), ("high", 32000)]
    )
    def test_claude_up_to_4_5_thinks_on_a_budget(self, settings, effort, budget):
        built = build("claude-sonnet-4-5", settings, effort=effort)

        assert thinking_config(built).thinking_budget == budget  # 0: disabled
        assert thinking_config(built).thinking_level is None
        assert built.generate_content_config.effort is None  # 4.5 rejects effort

    def test_an_explicit_budget_is_at_least_1024(self, settings):
        def budget(tokens):
            built = build("claude-sonnet-4-5", settings, thinking="budget", budget=tokens)
            return thinking_config(built).thinking_budget

        assert (budget(20000), budget(200)) == (20000, 1024)

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_claude_4_6_and_later_think_adaptively_at_an_effort(self, settings, effort):
        built = build("claude-opus-4-7", settings, effort=effort)

        assert thinking_config(built).thinking_budget == -1  # ADK's adaptive thinking
        assert thinking_config(built).thinking_level is None
        assert built.generate_content_config.effort == effort

    def test_a_native_effort_level(self, settings):
        built = build("claude-opus-4-7", settings, thinking="xhigh")

        assert built.generate_content_config.effort == "xhigh"

    def test_none_turns_adaptive_claude_off(self, settings):
        built = build("claude-opus-4-7", settings, effort="none")

        assert thinking_config(built).thinking_budget == 0
        assert built.generate_content_config.effort is None

    def test_claude_that_cannot_stop_thinking_runs_at_its_lowest_effort(self, settings):
        built = build("claude-opus-5-5", settings, effort="none")

        assert thinking_config(built).thinking_budget == -1
        assert built.generate_content_config.effort == "low"

    @pytest.mark.parametrize(
        "model, budget",
        [
            ("claude-opus-4-8", -1),
            ("claude-opus-4-7", -1),
            ("claude-opus-4-6", -1),
            ("claude-sonnet-4-6", -1),
            ("claude-fable-5", -1),
            ("claude-opus-4-5", 32000),
            ("claude-sonnet-4-5", 32000),
            ("claude-haiku-4-5", 32000),
            ("claude-sonnet-4", 32000),
            ("claude-3-5-sonnet", 32000),
            # A dated id is its version, not its date: Claude 4.0, on a budget.
            ("claude-opus-4-20250514", 32000),
            ("claude-sonnet-4-20250514", 32000),
            ("claude-opus-4-1-20250805", 32000),
            ("claude-haiku-4-5-20251001", 32000),
        ],
    )
    def test_the_version_decides_adaptive_or_budget_without_a_listing(
        self, settings, model, budget
    ):
        assert thinking_config(build(model, settings, effort="high")).thinking_budget == budget

    @pytest.mark.parametrize(
        "spec, kwargs, max_tokens",
        [
            ("claude-sonnet-4-5", {}, 8192),  # the Anthropic default
            ("claude-sonnet-4-5", {"max_tokens": 20000}, 20000),
            ("claude-opus-4-7", {"effort": "high"}, 8192),  # adaptive: no budget to fit
            ("claude-opus-4-7", {"effort": "high", "max_tokens": 20000}, 20000),
            ("claude-sonnet-4-5", {"effort": "high"}, 40192),  # the budget plus 8192
            ("claude-sonnet-4-5", {"effort": "high", "max_tokens": 10000}, 40192),
        ],
    )
    def test_one_max_tokens_on_the_instance_and_in_the_config(
        self, settings, spec, kwargs, max_tokens
    ):
        built = build(spec, settings, **kwargs)

        assert built.model.max_tokens == max_tokens
        assert built.generate_content_config.max_output_tokens == max_tokens

    async def test_budget_claude_with_a_small_max_tokens_leaves_room_for_the_budget(
        self, settings
    ):
        """The ADK 2 regression, pinned at ADK's own request building.

        ADK 2.12's ``AnthropicLlm`` sends the config's ``max_output_tokens`` in
        preference to the instance's ``max_tokens``. 0.6 put the agent's raw
        ``max_tokens`` (2000) there, below the 32000-token thinking budget, and
        Anthropic answered HTTP 400.
        """
        built = build("claude-sonnet-4-5", settings, thinking="high", max_tokens=2000)
        api = FakeAPI(anthropic_message())
        serve(built.model, api)

        await run_agent(built)

        body = api.bodies[0]
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert body["max_tokens"] == 40192
        assert body["max_tokens"] > body["thinking"]["budget_tokens"]

    async def test_the_effort_and_adaptive_thinking_reach_the_request(self, settings):
        built = build("claude-opus-4-7", settings, effort="high")
        api = FakeAPI(anthropic_message(model="claude-opus-4-7"))
        serve(built.model, api)

        await run_agent(built)

        body = api.bodies[0]
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"] == {"effort": "high"}
        assert api.requests[0].headers["x-api-key"] == "a-test-key"

    async def test_adk_has_no_sampling_warning_to_print(self, settings):
        """ADK warns when it drops sampling settings for a thinking Claude; it
        never sees any, so nothing prints over the UI."""
        built = build("claude-opus-4-7", settings, effort="high", temperature=0.3, top_p=0.9)
        api = FakeAPI(anthropic_message(model="claude-opus-4-7"))
        serve(built.model, api)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Sampling parameters")
            await generate(built)

        assert "temperature" not in api.bodies[0]
        assert "top_p" not in api.bodies[0]

    def test_sampling_is_kept_while_budget_claude_does_not_think(self, settings):
        config = build(
            "claude-sonnet-4-5", settings, effort="none", temperature=0.3
        ).generate_content_config

        assert config.temperature == 0.3


@pytest.fixture
def isolated_registry():
    """Snapshot ADK's model registry and restore it afterwards."""
    snapshot = dict(adk_registry._llm_registry_dict)
    LLMRegistry.resolve.cache_clear()
    yield
    adk_registry._llm_registry_dict.clear()
    adk_registry._llm_registry_dict.update(snapshot)
    LLMRegistry.resolve.cache_clear()


class TestNativeAdkConfig:
    """A native ADK agent YAML names its model as a string, resolved by ADK's registry."""

    def test_a_bare_claude_id_runs_on_the_direct_api_not_vertex(self, isolated_registry):
        LLMRegistry.register(Claude)  # ADK's own mapping: Claude on Vertex AI
        LLMRegistry.resolve.cache_clear()
        assert LLMRegistry.resolve("claude-sonnet-4-5") is Claude

        register_direct_anthropic()

        for model in ("claude-sonnet-4-5", "claude-3-5-haiku", "claude-opus-4-8", "claude-fable-5"):
            assert LLMRegistry.resolve(model) is DirectAnthropicLlm, model
        llm = LLMRegistry.new_llm("claude-opus-4-8")
        assert isinstance(llm, AnthropicLlm)
        assert not isinstance(llm, Claude)
        assert (llm.model, llm.max_tokens) == ("claude-opus-4-8", 8192)
        assert llm.client is None  # ADK's own client, keyed by ANTHROPIC_API_KEY


# --- OpenAI -------------------------------------------------------------------------


class TestGpt:
    def test_gpt_runs_on_the_responses_api_with_its_own_client(self, settings):
        built = build("openai:gpt-5.5", settings)

        assert type(built.model) is OpenAIResponsesLlm
        assert built.model.model == "gpt-5.5"
        assert built.model.store is False
        client = built.model.client
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "o-test-key"
        assert str(client.base_url) == "https://api.openai.com/v1/"
        assert client.max_retries == 4
        assert client.timeout == httpx.Timeout(600.0, connect=5.0)
        assert type(built.generate_content_config) is OpenAIGenerateContentConfig
        assert built.generate_content_config.http_options is None
        assert built.planner is None  # the effort rides in the config

    @pytest.mark.parametrize(
        "spec, mode, effort",
        [
            ("gpt-5.5", "low", "low"),
            ("gpt-5.5", "high", "high"),
            ("gpt-5.5", "xhigh", "xhigh"),  # a native level
            ("gpt-5.6", "max", "max"),
            ("gpt-5", "none", "minimal"),  # none: the lowest effort on the ladder
            ("o3", "none", "low"),
        ],
    )
    def test_the_effort_goes_in_the_openai_config(self, settings, spec, mode, effort):
        built = build(spec, settings, thinking=mode)

        assert built.generate_content_config.effort == effort
        assert built.model.reasoning is None

    def test_an_effort_of_none_is_the_model_s_reasoning_default(self, settings):
        """ADK's config cannot say ``none``; its model-wide ``reasoning`` can."""
        built = build("gpt-5.5", settings, effort="none")

        assert built.generate_content_config.effort is None
        assert built.model.reasoning == {"effort": "none"}

    @pytest.mark.parametrize(
        "effort, reasoning",
        [("high", {"effort": "high", "summary": "concise"}), ("none", {"effort": "none"})],
    )
    async def test_the_request(self, settings, effort, reasoning):
        built = build("gpt-5.5", settings, effort=effort, max_tokens=1000)
        api = FakeAPI(openai_response())
        serve(built.model, api)

        responses = await generate(built)

        body = api.bodies[0]
        assert body["model"] == "gpt-5.5"
        assert body["reasoning"] == reasoning
        assert body["store"] is False
        assert body["max_output_tokens"] == 1000
        assert api.requests[0].headers["authorization"] == "Bearer o-test-key"
        assert responses[-1].content.parts[0].text == "ok"

    def test_a_model_without_effort_keeps_its_sampling(self, settings):
        built = build("gpt-4.1", settings, effort="high", temperature=0.2)

        assert built.generate_content_config.effort is None
        assert built.model.reasoning is None
        assert built.generate_content_config.temperature == 0.2


# --- OpenRouter and the user's endpoints ----------------------------------------------


class TestOpenRouter:
    def test_an_openrouter_model_on_the_openai_compatible_class(self, settings):
        built = build("openrouter:qwen/qwen3-235b-a22b", settings)

        assert type(built.model) is OpenAICompatibleLlm
        assert built.model.model == "qwen/qwen3-235b-a22b"
        client = built.model.client
        assert str(client.base_url) == "https://openrouter.ai/api/v1/"
        assert client.api_key == "r-test-key"
        assert client.max_retries == 4
        assert client.timeout == httpx.Timeout(600.0, connect=5.0)
        # OpenRouter's attribution headers name the app.
        assert client.default_headers["HTTP-Referer"] == "factory-test"
        assert client.default_headers["X-Title"] == "factory-test"
        assert type(built.generate_content_config) is types.GenerateContentConfig
        assert built.planner is None  # the reasoning parameter rides in extra_body

    @pytest.mark.parametrize("effort", ["none", "low", "high"])
    def test_the_effort_is_openrouter_s_reasoning_parameter(self, settings, effort):
        built = build("openrouter:qwen/qwen3-235b-a22b", settings, effort=effort)

        assert built.model.extra_body == {"reasoning": {"effort": effort}}

    def test_a_model_that_does_not_reason_gets_no_reasoning_parameter(self, settings):
        built = build("openrouter:mistralai/mistral-small", settings, effort="high")

        assert built.model.extra_body == {}

    def test_max_tokens_is_left_to_the_server_unless_set(self, settings):
        unset = build("openrouter:qwen/qwen3-235b-a22b", settings)
        given = build("openrouter:qwen/qwen3-235b-a22b", settings, max_tokens=1000)

        assert (unset.model.max_tokens, unset.generate_content_config.max_output_tokens) == (
            None, None,
        )
        assert (given.model.max_tokens, given.generate_content_config.max_output_tokens) == (
            1000, 1000,
        )

    async def test_the_request_names_the_app_and_carries_the_effort(self, settings):
        built = build("openrouter:qwen/qwen3-235b-a22b", settings, effort="high")
        api = FakeAPI(chat_completion())
        serve(built.model, api)

        await generate(built)

        sent = api.requests[0]
        assert sent.url == "https://openrouter.ai/api/v1/chat/completions"
        assert sent.headers["authorization"] == "Bearer r-test-key"
        assert sent.headers["http-referer"] == "factory-test"
        assert sent.headers["x-title"] == "factory-test"
        assert api.bodies[0]["reasoning"] == {"effort": "high"}


class TestEndpoints:
    def test_an_endpoint_model_on_the_openai_compatible_class(self, settings):
        built = build("local:plain", settings)

        assert type(built.model) is OpenAICompatibleLlm
        assert built.model.model == "plain"
        client = built.model.client
        assert str(client.base_url) == "http://localhost:8000/v1/"
        assert client.api_key == "l-test-key"  # from the variable api_key_env names
        assert "X-Title" not in client.default_headers
        assert built.model.extra_body == {}

    @pytest.mark.parametrize("effort, enabled", [("none", False), ("low", True), ("high", True)])
    def test_chat_template_thinking_is_switched_in_the_template(self, settings, effort, enabled):
        built = build("local:Qwen/Qwen3-32B", settings, effort=effort)

        assert built.model.extra_body == {"chat_template_kwargs": {"enable_thinking": enabled}}

    @pytest.mark.parametrize("effort, sent", [("none", "low"), ("medium", "medium")])
    def test_a_declared_effort_is_sent_as_reasoning_effort(self, settings, effort, sent):
        built = build("local:reasoner", settings, effort=effort)

        assert built.model.extra_body == {"reasoning_effort": sent}

    def test_a_keyless_endpoint_sends_a_placeholder_never_the_openai_key(
        self, settings, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "o-env-key")

        built = build("keyless:plain", settings)

        assert built.model.client.api_key == "EMPTY"

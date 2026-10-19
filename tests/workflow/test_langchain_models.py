"""The LangChain model adapter: a resolved model as a LangChain chat model.

Each case resolves an agent's model the way the manager does
(``resolve_model``, against the shipped table) and builds it with
``build_chat_model``. The tests read the instance (its class, key, timeout,
retries, thinking settings and ``profile``) and, where it matters, the
request the class itself sends, served by a fake provider API: every client
the class builds is routed to the fake, keeping the key, base URL, headers,
timeout and retries the adapter gave it. Nothing reaches the network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_google_genai")
pytest.importorskip("langchain_openrouter")

import anthropic  # noqa: E402
import openai  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_openrouter import ChatOpenRouter  # noqa: E402

from agentic_cli.models import (  # noqa: E402
    OPENROUTER_BASE_URL,
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
from agentic_cli.workflow import langchain_models  # noqa: E402
from agentic_cli.workflow.langchain_models import (  # noqa: E402
    OpenAICompatibleChat,
    build_chat_model,
    profile_for,
)
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings  # noqa: E402
from tests.models.fakes import FakeCatalog, endpoints  # noqa: E402
from tests.workflow.adk.fakes import (  # noqa: E402
    FakeAPI,
    anthropic_message,
    chat_completion,
    openai_response,
)
from tests.workflow.adk.test_openai_compatible import (  # noqa: E402
    TOOL_CALL_MESSAGE,
    Server,
    tool_call_deltas,
)
from tests.workflow.langchain_fakes import (  # noqa: E402
    gemini_response,
    openrouter_completion,
    route,
    route_google,
    route_openrouter,
)

APP = "chat-test"
KEYS = {
    "google": "g-test-key",
    "anthropic": "a-test-key",
    "openai": "o-test-key",
    "openrouter": "r-test-key",
}


@pytest.fixture
def settings(mock_context):
    return mock_context.settings.model_copy(
        update={"retry_max_attempts": 4, "anthropic_request_timeout": 321.0}
    )


#: An OpenRouter model whose listing says it reasons (see ModelCatalog).
OPENROUTER_REASONER = ModelRecord(
    provider="openrouter",
    kind=ProviderKind.OPENROUTER,
    model="qwen/qwen3-235b-a22b",
    context_window=131072,
    max_output=32768,
    thinking=ThinkingRule(
        ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
    ),
    reasoning_text=True,
)
OPENROUTER_PLAIN = ModelRecord(
    provider="openrouter", kind=ProviderKind.OPENROUTER, model="meta-llama/llama-3.3-70b"
)
#: A user's vLLM server, its key in LOCAL_KEY.
LOCAL = {
    "local": EndpointSpec(
        base_url="http://localhost:8000/v1",
        api_key_env="LOCAL_KEY",
        models={
            "Qwen/Qwen3-32B": DeclaredModel(
                context_window=32768, max_output=8192, thinking="chat_template"
            ),
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
    The catalog is the shipped table, plus the OpenRouter records above and
    the models ``LOCAL`` declares.
    """
    if thinking is not None:
        generation["thinking"] = ThinkingSettings(mode=thinking, budget_tokens=budget)
    every = endpoints(**KEYS, specs=LOCAL, env={"LOCAL_KEY": "l-test-key"})
    declared = ModelCatalog(endpoints=every)  # never loaded: declarations and the table
    catalog = FakeCatalog([
        OPENROUTER_REASONER,
        OPENROUTER_PLAIN,
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


def build(spec: str, settings, **kwargs):
    return build_chat_model(resolved(spec, **kwargs), settings=settings, app_name=APP)


def get_plan() -> str:
    """Get the current plan."""
    return "the plan"


# --- every provider: class, profile, and no string-built models -----------------------

SPECS = [
    "gemini-2.5-flash",
    "claude-opus-4-7",
    "openai:gpt-5.5",
    "openrouter:qwen/qwen3-235b-a22b",
    "local:Qwen/Qwen3-32B",
]
CLASSES = [
    ChatGoogleGenerativeAI, ChatAnthropic, ChatOpenAI, ChatOpenRouter, OpenAICompatibleChat
]


class TestEveryProvider:
    @pytest.mark.parametrize("spec, cls", zip(SPECS, CLASSES))
    def test_each_provider_has_its_own_class(self, settings, monkeypatch, spec, cls):
        def no_strings(*args, **kwargs):
            raise AssertionError("a model is built from its class, never from a string")

        monkeypatch.setattr("langchain.chat_models.init_chat_model", no_strings)

        chat = build(spec, settings)

        assert type(chat) is cls

    @pytest.mark.parametrize("spec", SPECS)
    def test_every_instance_carries_its_record_s_profile(self, settings, spec):
        model = resolved(spec)

        chat = build_chat_model(model, settings=settings, app_name=APP)

        assert chat.profile == profile_for(model.record)


class TestProfile:
    def test_the_profile_comes_from_the_record(self):
        record = resolved("claude-opus-4-7").record

        assert profile_for(record) == {
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "tool_calling": True,
            "image_inputs": True,
            "reasoning_output": True,
            "reasoning_effort_levels": ["low", "medium", "high", "xhigh", "max"],
        }

    def test_an_endpoint_model_s_declared_window(self):
        """A small vLLM window reaches the profile, so Deep Agents summarizes in time."""
        profile = profile_for(resolved("local:Qwen/Qwen3-32B").record)

        assert profile["max_input_tokens"] == 32768
        assert profile["max_output_tokens"] == 8192
        assert profile["reasoning_output"] is True
        assert "reasoning_effort_levels" not in profile  # thinking on or off, no levels

    def test_unknown_numbers_are_left_out(self):
        profile = profile_for(resolved("local:plain").record)

        assert "max_input_tokens" not in profile
        assert "max_output_tokens" not in profile
        assert profile["tool_calling"] is True

    def test_a_model_without_a_thinking_knob(self):
        profile = profile_for(resolved("openai:gpt-4.1").record)

        assert profile["reasoning_output"] is False
        assert "reasoning_effort_levels" not in profile

    def test_a_budget_model_has_no_effort_levels(self):
        profile = profile_for(resolved("claude-sonnet-4-5").record)

        assert profile["reasoning_output"] is True
        assert "reasoning_effort_levels" not in profile


class TestGenerationSettings:
    def test_the_agent_s_settings_reach_the_instance(self, settings):
        chat = build(
            "openai:gpt-4.1", settings,
            temperature=0.3, top_p=0.9, max_tokens=1234, stop_sequences=["END"],
        )

        assert (chat.temperature, chat.top_p, chat.max_tokens) == (0.3, 0.9, 1234)
        assert chat.stop == ["END"]

    def test_unset_settings_stay_unset(self, settings):
        """Not the class's own defaults (ChatGoogleGenerativeAI's is 0.7)."""
        chat = build("gemini-2.5-flash", settings)

        assert chat.temperature is None
        assert chat.max_output_tokens is None

    def test_dropped_sampling_never_reaches_the_instance(self, settings):
        chat = build("claude-opus-4-7", settings, effort="high", temperature=0.3)

        assert chat.temperature is None

    def test_extra_keys_the_class_takes_pass_through_and_the_rest_are_dropped(
        self, settings, monkeypatch
    ):
        warned = []
        monkeypatch.setattr(
            langchain_models.logger, "warning", lambda *a, **k: warned.append((a, k))
        )

        chat = build("openai:gpt-4.1", settings, extra={"seed": 7, "bogus": 1})

        assert chat.seed == 7
        assert [(a[0], k["key"]) for a, k in warned] == [("model_settings_extra_ignored", "bogus")]

    @pytest.mark.parametrize("key", ["base_url", "api_key", "openai_api_base", "default_headers"])
    def test_extra_never_redirects_the_request_or_its_key(self, settings, key):
        chat = build("openai:gpt-4.1", settings, extra={key: "https://elsewhere.test"})

        assert chat.openai_api_base is None
        assert chat.openai_api_key.get_secret_value() == KEYS["openai"]
        assert chat.default_headers is None


# --- Google ----------------------------------------------------------------------------


class TestGemini:
    def test_gemini_with_its_own_key_timeout_and_retries(self, settings):
        chat = build("gemini-2.5-flash", settings)

        assert chat.model == "gemini-2.5-flash"
        assert chat.google_api_key.get_secret_value() == KEYS["google"]
        assert chat.timeout == 600.0
        assert chat.max_retries == 4

    @pytest.mark.parametrize("effort, budget", [("low", 4096), ("medium", 12288), ("high", 24576)])
    def test_gemini_2_5_thinks_on_a_budget(self, settings, effort, budget):
        chat = build("gemini-2.5-flash", settings, effort=effort)

        assert (chat.thinking_budget, chat.include_thoughts) == (budget, True)
        assert chat.reasoning_effort is None

    def test_none_turns_gemini_2_5_off(self, settings):
        chat = build("gemini-2.5-flash", settings, effort="none")

        assert (chat.thinking_budget, chat.include_thoughts) == (0, False)

    @pytest.mark.parametrize(
        "model, effort, level",
        [
            ("gemini-3-flash-preview", "medium", "medium"),
            ("gemini-3-flash-preview", "none", "minimal"),
            ("gemini-3-pro-preview", "medium", "high"),
            ("gemini-3-pro-preview", "none", "low"),
        ],
    )
    def test_gemini_3_thinks_in_levels(self, settings, model, effort, level):
        chat = build(model, settings, effort=effort)

        assert (chat.reasoning_effort, chat.include_thoughts) == (level, True)
        assert chat.thinking_budget is None

    async def test_the_request(self, settings, monkeypatch):
        api = FakeAPI(gemini_response({"text": "Hmm.", "thought": True}, {"text": "Hi"}))
        route_google(monkeypatch, api)
        chat = build("gemini-2.5-flash", settings, effort="high", max_tokens=1000)

        answer = await chat.ainvoke([HumanMessage("Hello")])

        (request,) = api.requests
        assert request.headers["x-goog-api-key"] == KEYS["google"]
        assert request.headers["x-server-timeout"] == "600"
        config = api.bodies[0]["generationConfig"]
        assert config["maxOutputTokens"] == 1000
        assert config["thinkingConfig"] == {"include_thoughts": True, "thinking_budget": 24576}
        assert answer.content_blocks[0] == {"type": "reasoning", "reasoning": "Hmm."}


# --- Anthropic -------------------------------------------------------------------------


class TestClaude:
    def test_claude_with_its_own_key_timeout_and_retries(self, settings):
        chat = build("claude-sonnet-4-5", settings)

        assert chat.model == "claude-sonnet-4-5"
        assert chat.anthropic_api_key.get_secret_value() == KEYS["anthropic"]
        assert chat.default_request_timeout == 321.0  # anthropic_request_timeout
        assert chat.max_retries == 4
        assert chat.max_tokens == 8192  # the Anthropic default

    @pytest.mark.parametrize("effort, budget", [("low", 4096), ("medium", 10000), ("high", 32000)])
    def test_claude_up_to_4_5_thinks_on_a_budget(self, settings, effort, budget):
        chat = build("claude-sonnet-4-5", settings, effort=effort)

        assert chat.thinking == {"type": "enabled", "budget_tokens": budget}
        assert chat.reasoning_effort is None

    def test_none_turns_budget_claude_off(self, settings):
        chat = build("claude-sonnet-4-5", settings, effort="none")

        assert chat.thinking == {"type": "disabled"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_claude_4_6_and_later_think_adaptively_at_an_effort(self, settings, effort):
        chat = build("claude-opus-4-7", settings, effort=effort)

        assert chat.thinking == {"type": "adaptive", "display": "summarized"}
        assert chat.reasoning_effort == effort

    def test_a_native_effort_level(self, settings):
        chat = build("claude-opus-4-7", settings, thinking="xhigh")

        assert chat.reasoning_effort == "xhigh"

    def test_none_turns_adaptive_claude_off(self, settings):
        chat = build("claude-opus-4-7", settings, effort="none")

        assert chat.thinking == {"type": "disabled"}
        assert chat.reasoning_effort is None

    def test_claude_that_cannot_stop_thinking_runs_at_its_lowest_effort(self, settings):
        chat = build("claude-opus-5-5", settings, effort="none")

        assert chat.thinking == {"type": "adaptive", "display": "summarized"}
        assert chat.reasoning_effort == "low"

    async def test_adaptive_thinking_and_the_effort_reach_the_request(self, settings, monkeypatch):
        """Opus 4.7 rejects a thinking budget: it is sent adaptive thinking and an effort."""
        api = FakeAPI(anthropic_message(model="claude-opus-4-7"))
        route(monkeypatch, anthropic, "AsyncClient", api)
        chat = build("claude-opus-4-7", settings, effort="high")

        await chat.ainvoke([HumanMessage("Hello")])

        (body,) = api.bodies
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"] == {"effort": "high"}
        assert api.requests[0].headers["x-api-key"] == KEYS["anthropic"]

    async def test_budget_claude_with_a_small_max_tokens_leaves_room_for_the_budget(
        self, settings, monkeypatch
    ):
        """The same rule as on ADK: a per-agent max_tokens below the budget is raised."""
        api = FakeAPI(anthropic_message())
        route(monkeypatch, anthropic, "AsyncClient", api)
        chat = build("claude-sonnet-4-5", settings, effort="high", max_tokens=2000)

        await chat.ainvoke([HumanMessage("Hello")])

        (body,) = api.bodies
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert body["max_tokens"] == 40192
        assert body["max_tokens"] > body["thinking"]["budget_tokens"]

    def test_sampling_is_kept_while_budget_claude_does_not_think(self, settings):
        chat = build("claude-sonnet-4-5", settings, effort="none", temperature=0.3, top_k=40)

        assert (chat.temperature, chat.top_k) == (0.3, 40)


# --- OpenAI ----------------------------------------------------------------------------


class TestGpt:
    def test_gpt_runs_on_the_responses_api_with_its_own_client(self, settings):
        chat = build("openai:gpt-5.5", settings, effort="high")

        assert chat.model_name == "gpt-5.5"
        assert chat.openai_api_key.get_secret_value() == KEYS["openai"]
        assert chat.openai_api_base is None  # the SDK's own host
        assert (chat.use_responses_api, chat.store) == (True, False)
        assert chat.include == ["reasoning.encrypted_content"]
        assert chat.request_timeout == httpx.Timeout(600.0, connect=5.0)
        assert chat.max_retries == 4

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_the_effort_with_a_reasoning_summary(self, settings, effort):
        chat = build("openai:gpt-5.5", settings, effort=effort)

        assert chat.reasoning == {"effort": effort, "summary": "auto"}

    def test_an_effort_of_none_asks_for_no_summary(self, settings):
        chat = build("openai:gpt-5.5", settings, effort="none")

        assert chat.reasoning == {"effort": "none"}

    def test_a_model_without_effort_has_no_reasoning_parameters(self, settings):
        """Encrypted reasoning is requested only from a reasoning model."""
        chat = build("openai:gpt-4.1", settings, effort="high", temperature=0.3)

        assert chat.reasoning is None
        assert chat.include is None
        assert chat.temperature == 0.3

    async def test_the_request(self, settings, monkeypatch):
        api = FakeAPI(openai_response("Hi"))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        chat = build("openai:gpt-5.5", settings, effort="high", max_tokens=500)

        await chat.ainvoke([HumanMessage("Hello")])

        (request,) = api.requests
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == f"Bearer {KEYS['openai']}"
        body = api.bodies[0]
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["reasoning"] == {"effort": "high", "summary": "auto"}
        assert body["max_output_tokens"] == 500


# --- OpenRouter ------------------------------------------------------------------------


class TestOpenRouter:
    def test_an_openrouter_model_on_chat_openrouter(self, settings):
        chat = build("openrouter:qwen/qwen3-235b-a22b", settings)

        assert chat.model_name == "qwen/qwen3-235b-a22b"
        assert chat.openrouter_api_key.get_secret_value() == KEYS["openrouter"]
        assert chat.openrouter_api_base == OPENROUTER_BASE_URL
        assert (chat.app_url, chat.app_title) == (APP, APP)
        assert chat.request_timeout == 600_000  # milliseconds
        assert chat.max_retries == 4

    @pytest.mark.parametrize("effort, sent", [("none", "none"), ("high", "high")])
    def test_the_effort_is_openrouter_s_reasoning_parameter(self, settings, effort, sent):
        chat = build("openrouter:qwen/qwen3-235b-a22b", settings, effort=effort)

        assert chat.reasoning == {"effort": sent}

    def test_a_model_that_does_not_reason_gets_no_reasoning_parameter(self, settings):
        chat = build("openrouter:meta-llama/llama-3.3-70b", settings, effort="high")

        assert chat.reasoning is None

    def test_the_environment_cannot_redirect_it(self, settings, monkeypatch):
        """ChatOpenRouter reads its host and attribution from the environment
        when they are not given."""
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://elsewhere.test/v1")
        monkeypatch.setenv("OPENROUTER_APP_TITLE", "Someone else")

        chat = build("openrouter:qwen/qwen3-235b-a22b", settings)

        assert chat.openrouter_api_base == OPENROUTER_BASE_URL
        assert chat.app_title == APP

    async def test_the_request_names_the_app_and_carries_the_effort(self, settings, monkeypatch):
        api = FakeAPI(openrouter_completion({"role": "assistant", "content": "Hi"}))
        route_openrouter(monkeypatch, api)
        chat = build("openrouter:qwen/qwen3-235b-a22b", settings, effort="high")

        await chat.ainvoke([HumanMessage("Hello")])

        (request,) = api.requests
        assert request.url == f"{OPENROUTER_BASE_URL}/chat/completions"
        assert request.headers["authorization"] == f"Bearer {KEYS['openrouter']}"
        assert (request.headers["http-referer"], request.headers["x-title"]) == (APP, APP)
        assert api.bodies[0]["reasoning"] == {"effort": "high"}

    async def test_reasoning_details_go_back_across_a_tool_call(self, settings, monkeypatch):
        """Claude and Gemini through OpenRouter need their signed reasoning back
        with the tool result."""
        details = [{
            "type": "reasoning.text", "text": "I need the plan.", "signature": "sig-1",
            "format": "anthropic-claude-v1", "index": 0,
        }]
        answers = [
            openrouter_completion({
                "role": "assistant", "content": "", "reasoning": "I need the plan.",
                "reasoning_details": details,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_plan", "arguments": "{}"},
                }],
            }, finish_reason="tool_calls"),
            openrouter_completion({"role": "assistant", "content": "Done."}),
        ]
        api = FakeAPI(lambda request: answers.pop(0))
        route_openrouter(monkeypatch, api)
        chat = build("openrouter:qwen/qwen3-235b-a22b", settings, effort="high").bind_tools(
            [get_plan]
        )

        first = await chat.ainvoke([HumanMessage("What is the plan?")])
        await chat.ainvoke(
            [HumanMessage("What is the plan?"), first, ToolMessage("the plan", tool_call_id="call_1")]
        )

        assert first.content_blocks[0] == {"type": "reasoning", "reasoning": "I need the plan."}
        replayed = api.bodies[1]["messages"][1]
        assert replayed["role"] == "assistant"
        assert replayed["reasoning_details"] == details
        assert replayed["tool_calls"][0]["id"] == "call_1"


# --- the user's endpoints --------------------------------------------------------------


class TestEndpoints:
    def test_an_endpoint_model_on_the_compatible_class(self, settings):
        chat = build("local:plain", settings)

        assert chat.model_name == "plain"
        assert chat.openai_api_base == "http://localhost:8000/v1"
        assert chat.openai_api_key.get_secret_value() == "l-test-key"  # from LOCAL_KEY
        assert (chat.use_responses_api, chat.stream_usage) == (False, True)
        assert chat.request_timeout == httpx.Timeout(600.0, connect=5.0)
        assert chat.max_retries == 4
        assert chat.extra_body is None

    def test_a_keyless_endpoint_sends_a_placeholder_never_the_openai_key(
        self, settings, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "o-env-key")

        chat = build("keyless:plain", settings)

        assert chat.openai_api_key.get_secret_value() == "EMPTY"

    @pytest.mark.parametrize("effort, enabled", [("none", False), ("high", True)])
    def test_chat_template_thinking_is_switched_in_the_template(self, settings, effort, enabled):
        chat = build("local:Qwen/Qwen3-32B", settings, effort=effort)

        assert chat.extra_body == {"chat_template_kwargs": {"enable_thinking": enabled}}

    @pytest.mark.parametrize("effort, sent", [("none", "low"), ("medium", "medium")])
    def test_a_declared_effort_is_sent_as_reasoning_effort(self, settings, effort, sent):
        chat = build("local:reasoner", settings, effort=effort)

        assert chat.extra_body == {"reasoning_effort": sent}

    async def test_the_request(self, settings, monkeypatch):
        """The limit goes as ``max_tokens``, which every compatible server knows."""
        api = FakeAPI(chat_completion("Hi"))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        chat = build("local:Qwen/Qwen3-32B", settings, effort="high", max_tokens=500)

        await chat.ainvoke([HumanMessage("Hello")])

        (request,) = api.requests
        assert request.url == "http://localhost:8000/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer l-test-key"
        body = api.bodies[0]
        assert body["max_tokens"] == 500
        assert "max_completion_tokens" not in body
        assert body["chat_template_kwargs"] == {"enable_thinking": True}


def compatible_chat(server: Server, monkeypatch, settings) -> Any:
    route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(server._handle))
    return build("local:Qwen/Qwen3-32B", settings, effort="high")


class TestCompatibleReasoning:
    """The server's ``reasoning`` (or older ``reasoning_content``) becomes a
    standard reasoning block, streamed and final."""

    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    async def test_a_final_message_s_reasoning(self, settings, monkeypatch, field):
        server = Server(message={"role": "assistant", field: "Let me think.", "content": "Hi"})
        chat = compatible_chat(server, monkeypatch, settings)

        answer = await chat.ainvoke([HumanMessage("Hello")])

        assert answer.content_blocks == [
            {"type": "reasoning", "reasoning": "Let me think."},
            {"type": "text", "text": "Hi"},
        ]

    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    async def test_streamed_reasoning(self, settings, monkeypatch, field):
        server = Server(deltas=[
            {"role": "assistant", field: "Let me "},
            {field: "think."},
            {"content": "Hi"},
        ])
        chat = compatible_chat(server, monkeypatch, settings)

        chunks = [chunk async for chunk in chat.astream([HumanMessage("Hello")])]
        answer = sum(chunks[1:], chunks[0])

        assert answer.content_blocks == [
            {"type": "reasoning", "reasoning": "Let me think."},
            {"type": "text", "text": "Hi"},
        ]

    async def test_a_streamed_tool_call_after_reasoning(self, settings, monkeypatch):
        server = Server(
            deltas=[{"role": "assistant", "reasoning": "I need the weather."}, *tool_call_deltas()],
            finish_reason="tool_calls",
        )
        chat = compatible_chat(server, monkeypatch, settings)

        chunks = [chunk async for chunk in chat.astream([HumanMessage("Weather?")])]
        answer = sum(chunks[1:], chunks[0])

        assert answer.content_blocks[0] == {"type": "reasoning", "reasoning": "I need the weather."}
        (call,) = answer.tool_calls
        assert call["name"] == TOOL_CALL_MESSAGE["tool_calls"][0]["function"]["name"]

    async def test_without_reasoning_the_answer_is_chat_openai_s_own(self, settings, monkeypatch):
        server = Server(message={"role": "assistant", "content": "Hi"})
        chat = compatible_chat(server, monkeypatch, settings)

        answer = await chat.ainvoke([HumanMessage("Hello")])

        assert answer.content == "Hi"
        assert "reasoning_content" not in answer.additional_kwargs


# --- one set of wire details for both backends -----------------------------------------


class TestOneWireForBothBackends:
    """ADK and LangChain send an OpenAI-compatible model the same parameters,
    key, headers and timeouts (``workflow/model_wire.py``)."""

    @pytest.mark.parametrize(
        "spec", ["local:Qwen/Qwen3-32B", "local:reasoner", "keyless:plain"]
    )
    def test_an_endpoint(self, settings, spec):
        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk.model_factory import build_adk_model

        model = resolved(spec, effort="high")
        adk = build_adk_model(model, settings=settings, app_name=APP).model
        chat = build_chat_model(model, settings=settings, app_name=APP)

        assert (chat.extra_body or {}) == adk.extra_body
        assert chat.openai_api_key.get_secret_value() == adk.client.api_key
        assert chat.request_timeout == adk.client.timeout

    def test_openrouter(self, settings):
        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk.model_factory import build_adk_model

        model = resolved("openrouter:qwen/qwen3-235b-a22b", effort="high")
        adk = build_adk_model(model, settings=settings, app_name=APP).model
        chat = build_chat_model(model, settings=settings, app_name=APP)

        assert chat.reasoning == adk.extra_body["reasoning"]
        headers = adk.client.default_headers
        assert (chat.app_url, chat.app_title) == (headers["HTTP-Referer"], headers["X-Title"])
        assert chat.request_timeout == adk.client.timeout.read * 1000

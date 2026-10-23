"""Context management and prompt caching on Deep Agents.

Three middlewares shape each model request, outermost first:

- ``ForeignReasoningMiddleware`` leaves other providers' reasoning out of the
  request (the checkpoint keeps it);
- Deep Agents' ``SummarizationMiddleware``, with each agent's thresholds
  from ``workflow/context.py``: summarize at the smaller of 85% of the
  model's input budget (``B = 0.95 * window - max_tokens``) and the 200,000
  ceiling, keep the newest 12% of that; an unknown window counts as 128,000.
  Its archive stores nothing, so the summary names no file and its "not
  recoverable" warning becomes a debug log. The summary call reports
  ``CONTEXT_TRIMMED`` and its own ``LLM_USAGE`` (``purpose="summarization"``);
- ``ContextOverflowMiddleware`` raises a provider's overflow that
  ``workflow/context.py`` recognizes as LangChain's ``ContextOverflowError``,
  so Deep Agents summarizes and retries on Gemini's wording too; an overflow
  that survives summarizing ends the turn with one clear ``ERROR``;
- ``AnthropicPromptCachingMiddleware`` on Claude, with ``prompt_caching_enabled``:
  the system prompt, the last tool and the conversation, for 5 minutes.

Requests go to ``FakeAPI``: a small window comes from a user's endpoint model
that declares one. Nothing reaches a provider.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import warnings
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

pytest.importorskip("deepagents")

from deepagents import SubAgentMiddleware  # noqa: E402
from deepagents.backends.protocol import BackendProtocol  # noqa: E402
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware  # noqa: E402
from deepagents.middleware.summarization import SummarizationMiddleware  # noqa: E402
from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware  # noqa: E402
from langchain_core.exceptions import ContextOverflowError  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ModelCatalog, ProviderKind, RoleMap, resolve_model  # noqa: E402
from agentic_cli.models.endpoints import EndpointSpec  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import (  # noqa: E402
    OVERFLOW_ERROR_CODE,
    ContextBudgetError,
    context_budget,
    overflow_message,
)
from agentic_cli.workflow.deepagents import agent_builder, summarization  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    ActiveToolCallMiddleware,
    ContextOverflowMiddleware,
    ForeignReasoningMiddleware,
    PermissionMiddleware,
    ToolCallErrorMiddleware,
    ToolResultCapMiddleware,
)
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import (  # noqa: E402
    NoArchiveBackend,
    ReportingSummarizationMiddleware,
    build_summarization,
    install_history_filters,
    summarization_thresholds,
)
from agentic_cli.workflow import live_events  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.models.fakes import endpoints as fake_endpoints  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, chat_completion, openai_response  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai  # noqa: E402
from tests.workflow.langchain_fakes import (  # noqa: E402
    anthropic_reply,
    gemini_reply,
    route,
    route_google,
)

#: A user's server whose model declares a 20,000-token window and no output
#: limit: B = 19,000, summarize at 16,150 tokens, keep the newest 1,938.
SMALL = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "models": {"small": {"context_window": 20_000}},
    }
}
MARK = {"type": "ephemeral", "ttl": "5m"}
USER_TURN = "word " * 2_400  # 12,000 characters, about 3,000 tokens


@register_tool(capabilities=EXEMPT)
def huge_report() -> str:
    """Returns far more text than the model's window holds (an exempt test tool)."""
    return "x" * 100_000


@register_tool(capabilities=EXEMPT)
def bulky_report() -> str:
    """Returns 56,000 characters, about 14,000 tokens; the model reads the
    50,000 the tool result cap keeps (an exempt test tool)."""
    return "y " * 28_000


@register_tool(capabilities=EXEMPT)
def cache_probe(x: str) -> dict:
    """Does nothing (an exempt test tool, so a request has a tool)."""
    return {"success": True}


# --- models and settings ------------------------------------------------------------------


#: Fake keys, set through the environment so a manager's key export is undone.
KEYS = {
    "GOOGLE_API_KEY": "g-test-key",
    "ANTHROPIC_API_KEY": "a-test-key",
    "OPENAI_API_KEY": "o-test-key",
}


def _settings(**values) -> BaseSettings:
    return BaseSettings(session_store="memory", **values)


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, and every provider's fake key."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for variable, key in KEYS.items():
        monkeypatch.setenv(variable, key)
    return tmp_path


def resolved(spec: str, agent: str = "a", *, effort: str = "none"):
    """``spec`` resolved for ``agent`` as a manager would, without listings."""
    endpoints = fake_endpoints(
        specs={"local": EndpointSpec(**SMALL["local"])},
        anthropic="a-test-key", google="g-test-key", openai="o-test-key",
    )
    return resolve_model(
        agent=agent, spec=spec, generation=None, thinking_effort=effort,
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(endpoints=endpoints), endpoints=endpoints,
    )


def windowed(model, window: int | None):
    record = dataclasses.replace(model.record, context_window=window)
    return dataclasses.replace(model, record=record)


def context(**values) -> SimpleNamespace:
    """The settings the agents are built from, at their defaults unless given."""
    return SimpleNamespace(**{
        "app_name": "agentic_cli",
        "context_summarize_at": 0.85,
        "context_max_tokens": 200_000,
        "prompt_caching_enabled": True,
        "tool_result_max_chars": 50_000,
        **values,
    })


def thresholds_of(model, settings) -> tuple:
    """``context_budget``'s trigger and keep, as Deep Agents takes them."""
    budget = context_budget(model, settings)
    return ("tokens", budget.trigger), ("tokens", budget.keep)


#: The spec's example: Claude Sonnet 4.5 (200,000 tokens) with a 32,000-token
#: thinking budget, so max_tokens is 40,192 and B is 149,808.
SONNET_THINKING = "anthropic:claude-sonnet-4-5"
#: Claude Opus 4.7: a 1,000,000-token window.
OPUS = "anthropic:claude-opus-4-7"


class TestThresholds:
    def test_the_specs_example(self):
        sonnet = resolved(SONNET_THINKING, effort="high")
        assert (sonnet.record.context_window, sonnet.max_tokens) == (200_000, 40_192)

        # 85% of B, and 12% of that kept.
        assert summarization_thresholds(sonnet, context()) == (
            ("tokens", 127_336), ("tokens", 15_280)
        )

    def test_they_are_the_context_budgets(self):
        for model in (resolved(SONNET_THINKING, effort="high"), resolved(OPUS),
                      resolved("google:gemini-2.5-flash")):
            for settings in (context(), context(context_summarize_at=0.5),
                             context(context_max_tokens=0)):
                assert summarization_thresholds(model, settings) == thresholds_of(
                    model, settings
                )

    def test_a_1m_model_summarizes_at_the_ceiling(self):
        assert summarization_thresholds(resolved(OPUS), context()) == (
            ("tokens", 200_000), ("tokens", 24_000)
        )

    def test_an_unknown_window_counts_as_128k(self):
        # Not Deep Agents' fixed 170,000 tokens / 6 messages, which never
        # fire on a 32k model.
        unknown = windowed(resolved(SONNET_THINKING), None)

        assert summarization_thresholds(unknown, context()) == (
            ("tokens", 96_396), ("tokens", 11_567)
        )
        assert summarization_thresholds(unknown, context()) == thresholds_of(
            unknown, context()
        )

    def test_a_budget_too_small_to_work_raises(self):
        small_window = windowed(resolved(SONNET_THINKING), 4_000)

        with pytest.raises(ContextBudgetError, match="'a'"):
            summarization_thresholds(small_window, context())


class TestNoArchive:
    async def test_it_stores_nothing(self):
        backend = NoArchiveBackend()

        assert isinstance(backend, BackendProtocol)
        for write in (backend.write("/h.md", "x"), await backend.awrite("/h.md", "x")):
            assert (write.error, write.path) == ("archive disabled", None)
        for edit in (backend.edit("/h.md", "a", "b"), await backend.aedit("/h.md", "a", "b")):
            assert (edit.error, edit.path) == ("archive disabled", None)
        for upload in (backend.upload_files([("/m.png", b"x")]),
                       await backend.aupload_files([("/m.png", b"x")])):
            assert [(u.path, u.error) for u in upload] == [("/m.png", "archive disabled")]
        for download in (backend.download_files(["/h.md"]),
                         await backend.adownload_files(["/h.md"])):
            assert [(d.path, d.content) for d in download] == [("/h.md", None)]
            assert download[0].error is not None


class TestBuildSummarization:
    def test_deep_agents_middleware_on_the_agents_model_with_our_thresholds(self):
        model = ScriptedChat()
        middleware = build_summarization(model, resolved(SONNET_THINKING, effort="high"),
                                         context())

        assert isinstance(middleware, SummarizationMiddleware)
        assert middleware.model is model
        assert isinstance(middleware._backend, NoArchiveBackend)
        assert middleware._lc_helper.trigger == ("tokens", 127_336)
        assert middleware._lc_helper.keep == ("tokens", 15_280)
        # Older history is summarized whole, as create_deep_agent does, and no
        # tool arguments are cut.
        assert middleware._lc_helper.trim_tokens_to_summarize is None
        assert middleware._truncate_args_trigger is None
        # Nothing is archived, so the prompt says nothing of archived media.
        assert "conversation_history" not in middleware._lc_helper.summary_prompt

    def test_an_unknown_window_gets_the_128k_floor(self):
        middleware = build_summarization(
            ScriptedChat(), windowed(resolved(SONNET_THINKING), None), context()
        )

        assert middleware._lc_helper.trigger == ("tokens", 96_396)
        assert middleware._lc_helper.keep == ("tokens", 11_567)


# --- the stack ----------------------------------------------------------------------------


@pytest.fixture
def created(monkeypatch) -> dict[str, dict]:
    """Every ``create_agent`` call the builder makes, by agent name."""
    calls: dict[str, dict] = {}
    real = agent_builder.create_agent

    def spy(*args, **kwargs):
        calls[kwargs["name"]] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_builder, "create_agent", spy)
    return calls


def _build(configs, models, settings):
    build_agent_tree(
        configs,
        chat_models={config.name: ScriptedChat(label=config.name) for config in configs},
        resolved={config.name: models[config.name] for config in configs},
        tools={},
        prompts={config.name: "p" for config in configs},
        settings=settings,
        checkpointer=None,
    )


TEAM = [
    AgentConfig(name="lead", prompt="p", sub_agents=["helper"]),
    AgentConfig(name="helper", prompt="p", description="Helps."),
]


class TestTheStack:
    def test_every_stack_in_order(self, created, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        claude = resolved(SONNET_THINKING, "lead")
        gemini = resolved("google:gemini-2.5-flash", "helper")

        _build(TEAM, {"lead": claude, "helper": gemini}, context(tool_result_max_chars=30_000))

        lead = created["lead"]["middleware"]
        assert [type(m) for m in lead] == [
            PatchToolCallsMiddleware, SubAgentMiddleware, ForeignReasoningMiddleware,
            ReportingSummarizationMiddleware, ContextOverflowMiddleware,
            AnthropicPromptCachingMiddleware,
            PermissionMiddleware, ToolResultCapMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]
        # The cap is inside the gate (a denial is short) and outside the error
        # middleware (an error result is capped too), on the settings.
        assert (lead[7].max_chars, lead[7].folder) == (
            30_000, tmp_path / ".agentic_cli" / "tool-output"
        )
        assert issubclass(ReportingSummarizationMiddleware, SummarizationMiddleware)
        assert lead[2].kind is ProviderKind.ANTHROPIC
        assert lead[3].agent == "lead"
        assert (lead[5].ttl, lead[5].unsupported_model_behavior) == ("5m", "ignore")
        assert [type(m) for m in created["helper"]["middleware"]] == [
            PatchToolCallsMiddleware, ForeignReasoningMiddleware,
            ReportingSummarizationMiddleware, ContextOverflowMiddleware,
            PermissionMiddleware, ToolResultCapMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]
        assert created["helper"]["middleware"][1].kind is ProviderKind.GOOGLE
        assert created["helper"]["middleware"][2].agent == "helper"

    def test_every_agent_summarizes_on_its_own_models_budget(self, created):
        # The coordinator on a 1M-token Claude (the ceiling), the helper on a
        # model whose window is unknown (the 128,000-token floor).
        claude = resolved(OPUS, "lead")
        unknown = windowed(resolved("google:gemini-2.5-flash", "helper"), None)
        settings = context(context_summarize_at=0.7)

        _build(TEAM, {"lead": claude, "helper": unknown}, settings)

        lead = created["lead"]["middleware"][3]._lc_helper
        helper = created["helper"]["middleware"][2]._lc_helper
        assert (lead.trigger, lead.keep) == thresholds_of(claude, settings)
        assert (helper.trigger, helper.keep) == thresholds_of(unknown, settings)
        assert lead.trigger == ("tokens", 200_000)
        assert context_budget(unknown, settings).window_known is False

    def test_no_prompt_caching_when_it_is_off(self, created):
        settings = context(prompt_caching_enabled=False)
        claude = resolved(SONNET_THINKING, "lead")

        _build([AgentConfig(name="lead", prompt="p")], {"lead": claude}, settings)

        assert AnthropicPromptCachingMiddleware not in [
            type(m) for m in created["lead"]["middleware"]
        ]


class TestForeignReasoning:
    async def test_only_the_request_is_filtered(self):
        gemini_message = AIMessage(
            content=[{"type": "thinking", "thinking": "Gemini thought."},
                     {"type": "text", "text": "Gemini answer."}],
            response_metadata={"model_provider": "google_genai"},
        )
        history = [HumanMessage("hi"), gemini_message]
        sent = []

        async def handler(request):
            sent.append(request.messages)
            return "response"

        request = ModelRequest(model=ScriptedChat(), messages=history, state={"messages": history})
        middleware = ForeignReasoningMiddleware(ProviderKind.ANTHROPIC)

        assert await middleware.awrap_model_call(request, handler) == "response"
        assert sent[0][1].content == [{"type": "text", "text": "Gemini answer."}]
        assert gemini_message.content[0]["type"] == "thinking"  # the state keeps it
        assert middleware.wrap_model_call(request, lambda r: r.messages)[1].content == (
            [{"type": "text", "text": "Gemini answer."}]
        )


def _gemini_overflow() -> genai_errors.ClientError:
    """Gemini's overflow, as google-genai raises it: an int ``code``, no ``status_code``."""
    return genai_errors.ClientError(400, {"error": {
        "code": 400,
        "message": "The input token count (1200000) exceeds the maximum number of "
                   "tokens allowed (1048576).",
        "status": "INVALID_ARGUMENT",
    }})


def _anthropic_error(error_class, status: int, kind: str, message: str):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_class(
        message,
        response=httpx.Response(status, request=request),
        body={"type": "error", "error": {"type": kind, "message": message}},
    )


def _claude_too_large() -> anthropic.RequestTooLargeError:
    """Claude's request-size limit: a 413 whose text names no context."""
    return _anthropic_error(
        anthropic.RequestTooLargeError, 413, "request_too_large",
        "Request exceeds the maximum size",
    )


def _claude_bad_request() -> anthropic.BadRequestError:
    """A 400 that is not an overflow."""
    return _anthropic_error(
        anthropic.BadRequestError, 400, "invalid_request_error",
        "temperature: must be at most 1",
    )


class TestContextOverflowMiddleware:
    @staticmethod
    def _request() -> ModelRequest:
        history = [HumanMessage("hi")]
        return ModelRequest(model=ScriptedChat(), messages=history, state={"messages": history})

    @pytest.mark.parametrize("error", [_gemini_overflow(), _claude_too_large()])
    async def test_a_recognized_overflow_is_raised_as_langchains(self, error):
        async def handler(request):
            raise error

        def sync_handler(request):
            raise error

        middleware = ContextOverflowMiddleware()
        with pytest.raises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(self._request(), handler)
        with pytest.raises(ContextOverflowError) as caught_sync:
            middleware.wrap_model_call(self._request(), sync_handler)

        for raised in (caught.value, caught_sync.value):
            assert raised.__cause__ is error
            assert str(raised) == str(error)

    @pytest.mark.parametrize("error", [
        _claude_bad_request(), ValueError("prompt is too long"),
        ContextOverflowError("already LangChain's"),
    ])
    async def test_anything_else_passes_unchanged(self, error):
        async def handler(request):
            raise error

        with pytest.raises(type(error)) as caught:
            await ContextOverflowMiddleware().awrap_model_call(self._request(), handler)

        assert caught.value is error

    @pytest.mark.parametrize("error", [
        _claude_bad_request(), ValueError("prompt is too long"),
        ContextOverflowError("already LangChain's"),
    ])
    def test_anything_else_passes_the_sync_path_unchanged(self, error):
        def handler(request):
            raise error

        with pytest.raises(type(error)) as caught:
            ContextOverflowMiddleware().wrap_model_call(self._request(), handler)

        assert caught.value is error

    async def test_an_answer_passes_through(self):
        async def handler(request):
            return "response"

        middleware = ContextOverflowMiddleware()
        assert await middleware.awrap_model_call(self._request(), handler) == "response"
        assert middleware.wrap_model_call(self._request(), lambda r: "response") == "response"


# --- whole turns ----------------------------------------------------------------------------


def _manager(settings, configs=None) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs or [AgentConfig(name="solo", prompt="Answer.")],
        settings=settings,
    )


def _agent(name: str, prompt: str, **fields) -> AgentConfig:
    """An agent without the state tools: no schemas count toward the window,
    so the token arithmetic below is the messages' own."""
    return AgentConfig(name=name, prompt=prompt, include_state_tools=False, **fields)


async def turn(manager, text: str, session: str = "s") -> list:
    return [event async for event in manager.process(text, "u", session_id=session)]


def _call(name: str, args: dict, call_id: str) -> dict:
    """A chat completion that calls one tool."""
    message = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }
    return {**chat_completion(), "choices": [
        {"index": 0, "message": message, "finish_reason": "tool_calls"}
    ]}


def _is_summary(messages: list[dict]) -> bool:
    return len(messages) == 1 and "<messages>" in str(messages[0]["content"])


class EndpointAPI:
    """The user's server: each agent answers from its own script, by system prompt.

    A summary request (one user message holding the summary prompt) is
    answered with a summary naming how many messages it was given.
    """

    def __init__(self, **scripts: list[dict]) -> None:
        self.scripts = scripts
        self.api = FakeAPI(self._answer)

    def _answer(self, request) -> dict:
        messages = json.loads(request.content)["messages"]
        if _is_summary(messages):
            count = str(messages[0]["content"]).count('<message type="')
            return chat_completion(f"A summary of {count} messages.")
        return self.scripts[messages[0]["content"]].pop(0)

    def requests(self, system: str | None = None) -> list[list[dict]]:
        """The message lists sent, of one agent's (by system prompt) or the summaries."""
        sent = [json.loads(r.content)["messages"] for r in self.api.requests]
        if system is None:
            return [m for m in sent if _is_summary(m)]
        return [m for m in sent if not _is_summary(m) and m[0]["content"] == system]


@pytest.fixture
def small(hermetic, monkeypatch):
    """Settings on the small-window endpoint model."""
    return _settings(model_endpoints=SMALL, default_model="local:small")


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


class TestSummarizationInATurn:
    async def test_a_long_conversation_is_summarized_once(self, small, monkeypatch, caplog):
        server = EndpointAPI(**{"Answer.": [chat_completion(f"Answer {n}.") for n in range(1, 8)]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        recorder = _Recorder()
        monkeypatch.setattr(summarization, "logger", recorder)
        caplog.set_level(logging.DEBUG, logger="deepagents")
        manager = _manager(small, [_agent("solo", "Answer.")])
        try:
            with warnings.catch_warnings(record=True) as caught:
                turns = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in range(1, 8)]
            recent = await manager.recent_messages("s", 50, user_id="u")
        finally:
            await manager.cleanup()

        # One summary, of the first five exchanges; the sixth turn's request
        # is the summary and the newest message, and names no archive file.
        (summary_request,) = server.requests()
        assert "Turn 1:" in summary_request[0]["content"]
        assert "Turn 6:" not in summary_request[0]["content"]
        sixth = server.requests("Answer.")[5]
        assert [m["role"] for m in sixth] == ["system", "user", "user"]
        assert sixth[1]["content"] == (
            "Here is a summary of the conversation to date:\n\nA summary of 10 messages."
        )
        assert sixth[2]["content"].startswith("Turn 6:")
        assert "conversation_history" not in json.dumps(server.api.bodies)
        # The next turn still starts from the summary, then the kept tail.
        seventh = server.requests("Answer.")[6]
        assert [m["content"][:7] for m in seventh[1:]] == [
            "Here is", "Turn 6:", "Answer ", "Turn 7:"
        ]

        # The summary call reports as it starts and when it is done, before
        # the answer it made room for.
        assert [e.type for e in turns[5]] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED,
            EventType.TEXT, EventType.LLM_USAGE,
        ]
        summarizing, summary_usage, trimmed, _, answer_usage = turns[5]
        assert summarizing.content == "Summarizing the earlier conversation…"
        assert summarizing.metadata["agent"] == "solo"
        assert summary_usage.metadata["purpose"] == "summarization"
        assert summary_usage.metadata["agent"] == "solo"
        assert summary_usage.metadata["model"] == "local:small"
        assert "purpose" not in answer_usage.metadata
        assert (trimmed.metadata["source"], trimmed.metadata["agent"]) == ("summarization", "solo")
        assert (trimmed.metadata["messages_before"], trimmed.metadata["messages_after"]) == (11, 2)
        assert trimmed.metadata["messages_summarized"] == 10
        # Six questions of about 3,000 tokens passed the 16,150-token trigger;
        # the summary and the sixth question are what is left.
        assert trimmed.metadata["tokens_before"] > 16_150
        assert 3_000 < trimmed.metadata["tokens_after"] < 3_200
        others = turns[:5] + turns[6:]
        assert all(EventType.CONTEXT_TRIMMED not in [e.type for e in t] for t in others)
        assert all(EventType.CONTEXT_SUMMARIZING not in [e.type for e in t] for t in others)

        # The checkpoint keeps every message.
        assert len(recent) == 14

        # Deep Agents' "not recoverable" warning and log are one debug event.
        assert not [w for w in caught if "conversation history" in str(w.message).lower()]
        assert not [r for r in caplog.records if "offload" in r.getMessage().lower()]
        assert [e for _, e, _ in recorder.calls] == ["deepagents_history_not_archived"]
        assert recorder.calls[0][0] == "debug"

    async def test_a_second_summary_condenses_the_first_and_what_followed(
        self, small, monkeypatch
    ):
        server = EndpointAPI(**{"Answer.": [chat_completion(f"Answer {n}.") for n in range(1, 12)]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(small, [_agent("solo", "Answer.")])
        try:
            turns = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in range(1, 12)]
        finally:
            await manager.cleanup()

        trimmed = [e for t in turns for e in t if e.type is EventType.CONTEXT_TRIMMED]
        assert [(e.metadata["messages_before"], e.metadata["messages_after"],
                 e.metadata["messages_summarized"]) for e in trimmed] == [
            (11, 2, 10),  # turn 6: five exchanges, the sixth question kept
            (12, 2, 11),  # turn 11: the first summary and the ten messages after it
        ]
        second = server.requests()[1][0]["content"]
        assert "A summary of 10 messages." in second
        assert "Turn 6:" in second and "Turn 11:" not in second

    async def test_a_model_too_small_to_work_on_fails_initialization(
        self, hermetic, monkeypatch
    ):
        # A 4,000-token window leaves a 3,800-token budget, under 16,000.
        tiny = {"local": {**SMALL["local"], "models": {"small": {"context_window": 4_000}}}}
        server = EndpointAPI()
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(
            _settings(model_endpoints=tiny, default_model="local:small"),
            [_agent("solo", "Answer.")],
        )
        try:
            with pytest.raises(ContextBudgetError, match="'solo'") as caught:
                await manager.initialize_services()
        finally:
            await manager.cleanup()

        assert "local:small" in str(caught.value)
        assert not manager._initialized
        assert server.api.requests == []

    async def test_a_tool_result_too_big_for_the_window_ends_the_turn(self, hermetic, monkeypatch):
        # With the tool result cap out of the way: at its default the result
        # is cut and fits (the test after this one).
        server = EndpointAPI(**{"Answer.": [_call("huge_report", {}, "c1")]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        uncut = _settings(
            model_endpoints=SMALL, default_model="local:small", tool_result_max_chars=1_000_000
        )
        manager = _manager(uncut, [AgentConfig(name="solo", prompt="Answer.", tools=[huge_report])])
        try:
            events = await turn(manager, "Report.")
        finally:
            await manager.cleanup()

        # Deep Agents' "Context remains above the input budget" reaches the
        # user as the one clear sentence ADK gives.
        kinds = [e.type for e in events]
        assert kinds[-1] is EventType.ERROR
        assert kinds.count(EventType.ERROR) == 1
        assert EventType.TEXT not in kinds
        assert events[-1].metadata["error_code"] == OVERFLOW_ERROR_CODE
        assert events[-1].content == overflow_message("local:small")

    async def test_at_the_default_limit_that_result_is_cut_and_the_turn_answers(
        self, small, monkeypatch
    ):
        # 100,000 characters cut to 50,000, about 12,500 tokens: the
        # 19,000-token budget holds them.
        server = EndpointAPI(**{"Answer.": [_call("huge_report", {}, "c1"),
                                            chat_completion("Reported.")]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(small, [AgentConfig(name="solo", prompt="Answer.", tools=[huge_report])])
        try:
            events = await turn(manager, "Report.")
        finally:
            await manager.cleanup()

        assert [e.content for e in events if e.type is EventType.TEXT] == ["Reported."]
        assert EventType.ERROR not in [e.type for e in events]
        tool_message = server.requests("Answer.")[-1][-1]
        assert tool_message["role"] == "tool"
        assert "characters cut]" in tool_message["content"]
        assert len(tool_message["content"]) <= 50_000

    async def test_a_summary_stays_with_the_agent_that_wrote_it(self, small, monkeypatch):
        """The coordinator's summary never reaches a sub-agent's brief, and
        the sub-agent's own summary never reaches the coordinator."""
        lead = [chat_completion(f"Answer {n}.") for n in range(1, 7)] + [
            _call("task", {"description": "Brief.", "subagent_type": "worker"}, "t1"),
            chat_completion("Done."),
        ]
        # The worker's second report puts it over its trigger: it summarizes.
        worker = [_call("bulky_report", {}, "b1"), _call("bulky_report", {}, "b2"),
                  chat_completion("Worked.")]
        server = EndpointAPI(**{"Lead.": lead, "Work.": worker})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(small, [
            _agent("lead", "Lead.", sub_agents=["worker"]),
            _agent("worker", "Work.", description="Works.", tools=[bulky_report]),
        ])
        ref = manager.session_ref("s", "u")
        try:
            earlier = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in range(1, 7)]
            summarized = (await manager._root.aget_state(run_config(ref))).values
            delegating = await turn(manager, "Delegate.")
            state = (await manager._root.aget_state(run_config(ref))).values
        finally:
            await manager.cleanup()

        # The coordinator summarized before it delegated...
        assert [e.metadata["agent"] for t in earlier for e in t
                if e.type is EventType.CONTEXT_TRIMMED] == ["lead"]
        own = summarized["_summarization_event"]
        # ...yet the worker's brief is the brief alone,
        first, *_, last = server.requests("Work.")
        assert first == [{"role": "system", "content": "Work."},
                         {"role": "user", "content": "Brief."}]
        # and the worker's summary, of its brief and first report, stays its own.
        assert last[1]["content"].endswith("A summary of 3 messages.")
        assert state["_summarization_event"]["cutoff_index"] == own["cutoff_index"]
        assert state["_summarization_event"]["summary_message"].text == own["summary_message"].text
        trimmed = [e for e in delegating if e.type is EventType.CONTEXT_TRIMMED]
        assert [e.metadata["agent"] for e in trimmed] == ["worker"]
        summaries = [e for e in delegating if e.metadata.get("purpose") == "summarization"]
        assert [e.metadata["agent"] for e in summaries] == ["worker"]


class TestOverflowInATurn:
    """A provider refuses a request as too long, on scripted models.

    Two earlier turns of about 3,000 tokens each sit under the 16,150-token
    trigger; the third request is refused, which Deep Agents answers by
    summarizing the earlier turns and retrying once, on the agent's own model.
    """

    @staticmethod
    async def _three_turns(settings, monkeypatch, script) -> tuple[list, ScriptedChat]:
        models = ScriptedModels(solo=script)
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = _manager(settings, [_agent("solo", "Answer.")])
        try:
            turns = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in (1, 2, 3)]
        finally:
            await manager.cleanup()
        return turns, models.chats["solo"]

    @pytest.mark.parametrize(
        "overflow", [_gemini_overflow(), _claude_too_large()], ids=["gemini", "claude-413"]
    )
    async def test_an_overflow_is_summarized_and_retried(self, small, monkeypatch, overflow):
        turns, chat = await self._three_turns(small, monkeypatch, [
            ai("Answer 1."), ai("Answer 2."), overflow, ai("A summary."), ai("Answer 3."),
        ])

        third = turns[2]
        assert [e.content for e in third if e.type is EventType.TEXT] == ["Answer 3."]
        assert EventType.ERROR not in [e.type for e in third]
        trimmed = [e for e in third if e.type is EventType.CONTEXT_TRIMMED]
        assert [e.metadata["messages_summarized"] for e in trimmed] == [4]
        # The refused request, the summary of the first two exchanges, then
        # the retry: the summary and the third question.
        assert len(chat.requests) == 5
        assert "Turn 1:" in chat.conversation(3)[0].text
        retry = chat.conversation(4)
        assert [m.text[:14] for m in retry] == ["Here is a summ", "Turn 3: word w"]
        assert retry[0].text.endswith("A summary.")

    async def test_an_overflow_that_survives_summarizing_is_one_clear_error(
        self, small, monkeypatch
    ):
        turns, chat = await self._three_turns(small, monkeypatch, [
            ai("Answer 1."), ai("Answer 2."), _gemini_overflow(), ai("A summary."),
            _gemini_overflow(),
        ])

        third = turns[2]
        errors = [e for e in third if e.type is EventType.ERROR]
        assert [(e.content, e.metadata["error_code"]) for e in errors] == [
            (overflow_message("local:small"), OVERFLOW_ERROR_CODE)
        ]
        assert errors[0].metadata["agent"] == "solo"
        assert EventType.TEXT not in [e.type for e in third]
        assert len(chat.requests) == 5  # one retry, no more

    async def test_any_other_error_is_unchanged(self, small, monkeypatch):
        error = _claude_bad_request()
        turns, chat = await self._three_turns(
            small, monkeypatch, [ai("Answer 1."), ai("Answer 2."), error]
        )

        errors = [e for e in turns[2] if e.type is EventType.ERROR]
        assert [(e.content, e.metadata["error_code"]) for e in errors] == [
            (str(error), "BadRequestError")
        ]
        assert len(chat.requests) == 3  # nothing summarized, nothing retried


class GatedChat(ScriptedChat):
    """Holds call ``gate_at`` (1-based) until released, as a slow model would.

    A cancel while it is held records nothing, so the script is not used up.
    """

    gate_at: int = 0
    _started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _held: bool = PrivateAttr(default=False)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if len(self.requests) + 1 == self.gate_at and not self._held:
            self._held = True
            self._started.set()
            await self._release.wait()
        return self._answer(messages)


def pumps() -> list[asyncio.Task]:
    """The live-events pump tasks still running."""
    return [t for t in asyncio.all_tasks() if t.get_name() == live_events.PUMP_TASK]


class TestCancelledTurns:
    """Ctrl+C while a model call runs, the summary call included."""

    @staticmethod
    def _manager_on(chat: GatedChat, settings, monkeypatch) -> DeepAgentsWorkflowManager:
        models = ScriptedModels()
        models.chats["solo"] = chat
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        return _manager(settings, [_agent("solo", "Answer.")])

    async def test_ctrl_c_during_a_model_call_leaves_no_pump_and_frees_the_turn(
        self, small, monkeypatch
    ):
        chat = GatedChat(label="solo", script=[ai("Answer 1.")], gate_at=1)
        manager = self._manager_on(chat, small, monkeypatch)
        try:
            consumer = asyncio.create_task(turn(manager, "First."))
            await asyncio.wait_for(chat._started.wait(), 5)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(consumer, 5)
            left = pumps()
            # The turn lock is free: the next turn runs at once.
            again = await asyncio.wait_for(turn(manager, "Again."), 5)
        finally:
            chat._release.set()
            await manager.cleanup()

        assert left == []
        assert [e.content for e in again if e.type is EventType.TEXT] == ["Answer 1."]

    async def test_ctrl_c_during_a_summary_writes_nothing_and_the_next_turn_summarizes_again(
        self, small, monkeypatch
    ):
        # Five turns under the trigger; the sixth summarizes (call 6) first.
        script = [ai(f"Answer {n}.") for n in range(1, 6)]
        script += [ai("A summary."), ai("Answer 6.")]
        chat = GatedChat(label="solo", script=script, gate_at=6)
        manager = self._manager_on(chat, small, monkeypatch)
        sixth: list = []

        async def consume() -> None:
            async for event in manager.process(f"Turn 6: {USER_TURN}", "u", session_id="s"):
                sixth.append(event)

        try:
            for n in range(1, 6):
                await turn(manager, f"Turn {n}: {USER_TURN}")
            consumer = asyncio.create_task(consume())
            await asyncio.wait_for(chat._started.wait(), 5)
            await asyncio.sleep(0.05)
            # The line is out while the summary call still runs.
            while_summarizing = [e.type for e in sixth]
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(consumer, 5)
            left = pumps()
            state = (await manager._root.aget_state(
                run_config(manager.session_ref("s", "u"))
            )).values
            chat._release.set()
            again = await asyncio.wait_for(turn(manager, f"Turn 6: {USER_TURN}"), 5)
        finally:
            chat._release.set()
            await manager.cleanup()

        assert while_summarizing == [EventType.CONTEXT_SUMMARIZING]
        assert left == []
        assert state.get("_summarization_event") is None  # nothing written
        assert [e.type for e in again][:3] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED,
        ]
        assert [e.content for e in again if e.type is EventType.TEXT] == ["Answer 6."]


class TestSettingsChange:
    async def test_a_new_trigger_rebuilds_the_agents_and_keeps_the_conversation(
        self, hermetic, monkeypatch
    ):
        from tests.cli.test_settings_apply_rollback import _app

        models = ScriptedModels(solo=[ai("Answer 1."), ai("Answer 2.")])
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        triggers = []
        build = agent_builder.build_summarization

        def recorded(model, resolved, settings):
            middleware = build(model, resolved, settings)
            triggers.append(middleware._lc_helper.trigger)
            return middleware

        monkeypatch.setattr(agent_builder, "build_summarization", recorded)
        settings = _settings(
            model_endpoints=SMALL, default_model="local:small", orchestrator="deepagents"
        )
        manager = _manager(settings, [_agent("solo", "Answer.")])
        try:
            await turn(manager, "First.")
            app = _app(settings, manager)
            assert await app.apply_settings({"context_summarize_at": "0.5"}) is True
            await turn(manager, "Second.")
        finally:
            await manager.cleanup()

        # 85%, then 50%, of the 19,000-token budget.
        assert triggers == [("tokens", 16_150), ("tokens", 9_500)]
        assert [m.text for m in models.chats["solo"].conversation(1)] == [
            "First.", "Answer 1.", "Second."
        ]


class TestHistoryFilters:
    def test_nothing_else_from_deep_agents_is_silenced(self, caplog):
        install_history_filters()
        caplog.set_level(logging.WARNING, logger="deepagents")
        source = logging.getLogger("deepagents.middleware.summarization")

        source.warning("Malformed _summarization_event (missing keys): %s", "x")
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("Conversation history was offloaded, but 2 media blocks were not")

        assert [r.getMessage() for r in caplog.records] == [
            "Malformed _summarization_event (missing keys): x"
        ]
        assert len(caught) == 1

    def test_installing_twice_adds_nothing(self):
        install_history_filters()
        source = logging.getLogger("deepagents.middleware.summarization")
        filters, warning_filters = list(source.filters), list(warnings.filters)

        install_history_filters()

        assert source.filters == filters
        assert warnings.filters == warning_filters


# --- prompt caching --------------------------------------------------------------------------


class TestPromptCaching:
    async def test_claude_requests_mark_the_prompt_the_tools_and_the_conversation(
        self, hermetic, monkeypatch
    ):
        api = FakeAPI(anthropic_reply({"type": "text", "text": "Hi."}))
        route(monkeypatch, anthropic, "AsyncClient", api)
        manager = _manager(_settings(default_model="claude-sonnet-4-5"), [
            AgentConfig(name="solo", prompt="Answer.", tools=[cache_probe]),
        ])
        try:
            await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        (body,) = api.bodies
        assert body["system"][-1]["cache_control"] == MARK
        assert body["tools"][-1]["cache_control"] == MARK
        assert body["cache_control"] == MARK

    async def test_no_marks_when_prompt_caching_is_off(self, hermetic, monkeypatch):
        api = FakeAPI(anthropic_reply({"type": "text", "text": "Hi."}))
        route(monkeypatch, anthropic, "AsyncClient", api)
        manager = _manager(_settings(default_model="claude-sonnet-4-5",
                                     prompt_caching_enabled=False))
        try:
            await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        assert "cache_control" not in json.dumps(api.bodies)

    async def test_gemini_requests_carry_no_marks(self, hermetic, monkeypatch):
        api = FakeAPI(gemini_reply({"text": "Hi."}))
        route_google(monkeypatch, api)
        manager = _manager(_settings(default_model="gemini-2.5-flash"))
        try:
            await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        sent = json.dumps(api.bodies)
        assert "cache_control" not in sent and "cachedContent" not in sent

    async def test_openai_requests_carry_no_marks(self, hermetic, monkeypatch):
        api = FakeAPI(openai_response("Hi."))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        manager = _manager(_settings(default_model="openai:gpt-5.5"))
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        assert [e.content for e in events if e.type is EventType.TEXT] == ["Hi."]
        assert "cache_control" not in json.dumps(api.bodies)


# --- foreign reasoning, through whole turns ---------------------------------------------------


def _gemini_thinking():
    return gemini_reply(
        {"text": "Gemini thought.", "thought": True},
        {"text": "Gemini answer.", "thoughtSignature": "R0VNSU5JX1NJRw=="},
    )


def _claude_thinking():
    return anthropic_reply(
        {"type": "thinking", "thinking": "Claude thought.", "signature": "claude-sig"},
        {"type": "text", "text": "Claude answer."},
        model="claude-opus-4-7",
    )


def _assistant_blocks(body: dict) -> list:
    return [
        block
        for message in body["messages"]
        if message["role"] == "assistant" and isinstance(message["content"], list)
        for block in message["content"]
    ]


class TestAcrossProviders:
    async def test_a_switch_from_gemini_sends_claude_no_gemini_thoughts(
        self, hermetic, monkeypatch
    ):
        route_google(monkeypatch, FakeAPI(_gemini_thinking()))
        claude = FakeAPI(anthropic_reply({"type": "text", "text": "Claude answer."}))
        route(monkeypatch, anthropic, "AsyncClient", claude)
        manager = _manager(_settings(default_model="gemini-3-flash-preview"))
        try:
            await turn(manager, "Hello")
            await manager.reinitialize(model="anthropic:claude-sonnet-4-5", preserve_sessions=True)
            await turn(manager, "And now?")
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        sent = claude.bodies[0]
        assert "Gemini thought" not in json.dumps(sent)
        assert [b["type"] for b in _assistant_blocks(sent)] == ["text"]
        assert "Gemini answer." in [m["content"] for m in recent]

    async def test_claude_still_gets_its_own_thinking_back(self, hermetic, monkeypatch):
        claude = FakeAPI(_claude_thinking())
        route(monkeypatch, anthropic, "AsyncClient", claude)
        manager = _manager(_settings(default_model="claude-opus-4-7"))
        try:
            await turn(manager, "one")
            await turn(manager, "two")
        finally:
            await manager.cleanup()

        (thinking,) = [b for b in _assistant_blocks(claude.bodies[1]) if b["type"] == "thinking"]
        assert (thinking["thinking"], thinking["signature"]) == ("Claude thought.", "claude-sig")

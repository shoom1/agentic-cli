"""Context management and prompt caching on Deep Agents.

Three middlewares shape each model request, outermost first:

- ``ForeignReasoningMiddleware`` leaves other providers' reasoning out of the
  request (the checkpoint keeps it);
- Deep Agents' ``SummarizationMiddleware``, with thresholds from the model's
  input budget (``B = 0.95 * window - max_tokens``: summarize at 85% of B,
  keep the newest 10%), or from ``context_window_*`` when that is on (the
  trigger capped at the automatic one). Its archive stores nothing, so the
  summary names no file and its "not recoverable" warning becomes a debug
  log. The summary call reports ``CONTEXT_TRIMMED`` and its own
  ``LLM_USAGE`` (``purpose="summarization"``);
- ``AnthropicPromptCachingMiddleware`` on Claude, with ``prompt_caching_enabled``:
  the system prompt, the last tool and the conversation, for 5 minutes.

Requests go to ``FakeAPI``: a small window comes from a user's endpoint model
that declares one. Nothing reaches a provider.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from types import SimpleNamespace

import anthropic
import openai
import pytest

pytest.importorskip("deepagents")

from deepagents import SubAgentMiddleware  # noqa: E402
from deepagents.backends.protocol import BackendProtocol  # noqa: E402
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware  # noqa: E402
from deepagents.middleware.summarization import SummarizationMiddleware  # noqa: E402
from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ModelCatalog, ProviderKind, RoleMap, resolve_model  # noqa: E402
from agentic_cli.models.endpoints import EndpointSpec  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import agent_builder, summarization  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    ActiveToolCallMiddleware,
    ForeignReasoningMiddleware,
    PermissionMiddleware,
    ToolCallErrorMiddleware,
)
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import (  # noqa: E402
    NoArchiveBackend,
    ReportingSummarizationMiddleware,
    build_summarization,
    install_history_filters,
    summarization_thresholds,
)
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from tests.models.fakes import endpoints as fake_endpoints  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, chat_completion, openai_response  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat  # noqa: E402
from tests.workflow.langchain_fakes import (  # noqa: E402
    anthropic_reply,
    gemini_reply,
    route,
    route_google,
)

#: A user's server whose model declares a 4,000-token window and no output
#: limit: B = 3,800, summarize at 3,230 tokens, keep the newest 380.
SMALL = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "models": {"small": {"context_window": 4000}},
    }
}
MARK = {"type": "ephemeral", "ttl": "5m"}
USER_TURN = "word " * 480  # 2,400 characters, about 600 tokens


@register_tool(capabilities=EXEMPT)
def huge_report() -> str:
    """Returns far more text than the model's window holds (an exempt test tool)."""
    return "x" * 20_000


@register_tool(capabilities=EXEMPT)
def bulky_report() -> str:
    """Returns about 2,800 tokens of text (an exempt test tool)."""
    return "y " * 5_600


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


def context(enabled: bool, trigger: int = 100_000, target: int = 80_000) -> SimpleNamespace:
    return SimpleNamespace(
        context_window_enabled=enabled,
        context_window_trigger_tokens=trigger,
        context_window_target_tokens=target,
        prompt_caching_enabled=True,
    )


#: The spec's example: Claude Sonnet 4.5 (200,000 tokens) with a 32,000-token
#: thinking budget, so max_tokens is 40,192 and B is 149,808.
SONNET_THINKING = "anthropic:claude-sonnet-4-5"


class TestThresholds:
    def test_the_specs_example(self):
        sonnet = resolved(SONNET_THINKING, effort="high")
        assert (sonnet.record.context_window, sonnet.max_tokens) == (200_000, 40_192)

        assert summarization_thresholds(sonnet, context(False)) == (
            ("tokens", 127_336), ("tokens", 14_980)
        )

    def test_without_an_output_allowance_the_budget_is_95_percent(self):
        sonnet = dataclasses.replace(resolved(SONNET_THINKING), max_tokens=None)

        assert summarization_thresholds(sonnet, context(False)) == (
            ("tokens", 161_500), ("tokens", 19_000)
        )

    def test_an_unknown_window_leaves_deep_agents_defaults(self):
        unknown = windowed(resolved(SONNET_THINKING), None)

        assert summarization_thresholds(unknown, context(False)) is None

    def test_an_unknown_window_takes_the_settings_as_they_are(self):
        # A user's endpoint model with no known window: the two counts are the
        # only thresholds anyone gave, and there is no automatic one to cap at.
        unknown = windowed(resolved(SONNET_THINKING), None)

        assert summarization_thresholds(unknown, context(True, 24_000, 4_000)) == (
            ("tokens", 24_000), ("tokens", 4_000)
        )

    def test_context_window_settings_pick_the_thresholds(self):
        sonnet = resolved(SONNET_THINKING, effort="high")

        assert summarization_thresholds(sonnet, context(True, 100_000, 80_000)) == (
            ("tokens", 100_000), ("tokens", 80_000)
        )

    def test_the_trigger_is_capped_at_the_automatic_one(self):
        sonnet = resolved(SONNET_THINKING, effort="high")

        # The kept tail shrinks with it: 20,000 · 127,336 / 500,000.
        assert summarization_thresholds(sonnet, context(True, 500_000, 20_000)) == (
            ("tokens", 127_336), ("tokens", 5_093)
        )

    def test_a_capped_trigger_keeps_the_tail_below_it(self):
        # A 4,000-token window: B = 3,800, the automatic trigger 3,230. The
        # default 100,000 / 80,000 would keep more than the whole budget.
        small_window = dataclasses.replace(
            windowed(resolved(SONNET_THINKING), 4_000), max_tokens=None
        )

        assert summarization_thresholds(small_window, context(True)) == (
            ("tokens", 3_230), ("tokens", 2_584)
        )

    def test_off_the_settings_are_ignored(self):
        sonnet = resolved(SONNET_THINKING, effort="high")

        assert summarization_thresholds(sonnet, context(False, 1_000, 500)) == (
            ("tokens", 127_336), ("tokens", 14_980)
        )


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
                                         context(False))

        assert isinstance(middleware, SummarizationMiddleware)
        assert middleware.model is model
        assert isinstance(middleware._backend, NoArchiveBackend)
        assert middleware._lc_helper.trigger == ("tokens", 127_336)
        assert middleware._lc_helper.keep == ("tokens", 14_980)
        # Older history is summarized whole, as create_deep_agent does, and no
        # tool arguments are cut.
        assert middleware._lc_helper.trim_tokens_to_summarize is None
        assert middleware._truncate_args_trigger is None
        # Nothing is archived, so the prompt says nothing of archived media.
        assert "conversation_history" not in middleware._lc_helper.summary_prompt

    def test_an_unknown_window_gets_deep_agents_fixed_defaults(self):
        middleware = build_summarization(
            ScriptedChat(), windowed(resolved(SONNET_THINKING), None), context(False)
        )

        assert middleware._lc_helper.trigger == ("tokens", 170_000)
        assert middleware._lc_helper.keep == ("messages", 6)


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
    def test_every_stack_in_order(self, created):
        claude = resolved(SONNET_THINKING, "lead")
        gemini = resolved("google:gemini-2.5-flash", "helper")

        _build(TEAM, {"lead": claude, "helper": gemini}, context(False))

        lead = created["lead"]["middleware"]
        assert [type(m) for m in lead] == [
            PatchToolCallsMiddleware, SubAgentMiddleware, ForeignReasoningMiddleware,
            ReportingSummarizationMiddleware, AnthropicPromptCachingMiddleware,
            PermissionMiddleware, ToolCallErrorMiddleware, ActiveToolCallMiddleware,
        ]
        assert issubclass(ReportingSummarizationMiddleware, SummarizationMiddleware)
        assert lead[2].kind is ProviderKind.ANTHROPIC
        assert lead[3].agent == "lead"
        assert (lead[4].ttl, lead[4].unsupported_model_behavior) == ("5m", "ignore")
        assert [type(m) for m in created["helper"]["middleware"]] == [
            PatchToolCallsMiddleware, ForeignReasoningMiddleware,
            ReportingSummarizationMiddleware, PermissionMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]
        assert created["helper"]["middleware"][1].kind is ProviderKind.GOOGLE
        assert created["helper"]["middleware"][2].agent == "helper"

    def test_no_prompt_caching_when_it_is_off(self, created):
        settings = context(False)
        settings.prompt_caching_enabled = False
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

        # The summary call reports, before the answer it made room for.
        assert [e.type for e in turns[5]] == [
            EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED, EventType.TEXT, EventType.LLM_USAGE
        ]
        summary_usage, trimmed, _, answer_usage = turns[5]
        assert summary_usage.metadata["purpose"] == "summarization"
        assert summary_usage.metadata["agent"] == "solo"
        assert summary_usage.metadata["model"] == "local:small"
        assert "purpose" not in answer_usage.metadata
        assert (trimmed.metadata["source"], trimmed.metadata["agent"]) == ("summarization", "solo")
        assert (trimmed.metadata["messages_before"], trimmed.metadata["messages_after"]) == (11, 2)
        assert trimmed.metadata["messages_summarized"] == 10
        others = turns[:5] + turns[6:]
        assert all(EventType.CONTEXT_TRIMMED not in [e.type for e in t] for t in others)

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

    async def test_with_context_window_settings_a_small_model_keeps_answering(
        self, hermetic, monkeypatch
    ):
        # The default 100,000 / 80,000 on the 4,000-token model: the trigger is
        # capped, and so must the tail be, or nothing can ever be summarized.
        settings = _settings(
            model_endpoints=SMALL, default_model="local:small", context_window_enabled=True
        )
        server = EndpointAPI(**{"Answer.": [chat_completion(f"Answer {n}.") for n in range(1, 12)]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(settings, [_agent("solo", "Answer.")])
        try:
            turns = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in range(1, 12)]
        finally:
            await manager.cleanup()

        assert [e.content for t in turns for e in t if e.type is EventType.ERROR] == []
        assert [[e.content for e in t if e.type is EventType.TEXT] for t in turns] == [
            [f"Answer {n}."] for n in range(1, 12)
        ]
        assert any(e.type is EventType.CONTEXT_TRIMMED for t in turns for e in t)

    async def test_a_tool_result_too_big_for_the_window_ends_the_turn(self, small, monkeypatch):
        server = EndpointAPI(**{"Answer.": [_call("huge_report", {}, "c1")]})
        route(monkeypatch, openai, "AsyncOpenAI", server.api)
        manager = _manager(small, [AgentConfig(name="solo", prompt="Answer.", tools=[huge_report])])
        try:
            events = await turn(manager, "Report.")
        finally:
            await manager.cleanup()

        kinds = [e.type for e in events]
        assert kinds[-1] is EventType.ERROR
        assert kinds.count(EventType.ERROR) == 1
        assert EventType.TEXT not in kinds
        assert events[-1].metadata["error_code"] == "ContextOverflowError"
        assert "input budget" in events[-1].content

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

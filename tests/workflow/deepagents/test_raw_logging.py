"""Raw LLM logging on Deep Agents (``raw_llm_logging``).

With the setting on, each turn's run config carries ``RunCallback`` with the
shared ``LLMLogWriter``, and LangChain hands the callback every model call in
the run: each agent's and sub-agent's, the summary call, and a call a tool
makes (``generate_simple``). Each call becomes an ``LLM_REQUEST`` and an
``LLM_RESPONSE``, attributed to its agent and purpose, in the stream (before
the stream item of the node that made it) and in
``~/.{app_name}/logs/llm_events.jsonl``, redacted. Usage stays the stream's:
logging adds no ``LLM_USAGE``. A log that cannot be written never fails the
turn, and with the setting off nothing is recorded.

Every request goes to ``FakeAPI``; nothing reaches a provider.
"""

from __future__ import annotations

import json
import logging

import anthropic
import openai
import pytest

pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.service_registry import WORKFLOW, get_service  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, chat_completion  # noqa: E402
from tests.workflow.langchain_fakes import anthropic_reply, route  # noqa: E402

#: A user's server whose model declares a 20,000-token window (as in
#: ``test_context.py``): a sub-agent's second bulky report makes it summarize.
SMALL = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "models": {"small": {"context_window": 20_000}},
    }
}
KEYS = {"ANTHROPIC_API_KEY": "a-test-key", "OPENAI_API_KEY": "o-test-key"}
DIGEST_PROMPT = "Digest the notes."
LOGGED = (EventType.LLM_REQUEST, EventType.LLM_RESPONSE)


@register_tool(capabilities=EXEMPT)
def rl_report() -> str:
    """Returns about 14,000 tokens of text (an exempt test tool)."""
    return "y " * 28_000


@register_tool(capabilities=EXEMPT)
async def rl_digest() -> dict:
    """Digests the notes with a model call of its own (an exempt test tool)."""
    digest = await get_service(WORKFLOW).generate_simple(DIGEST_PROMPT)
    return {"success": True, "digest": digest}


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


class Endpoint:
    """The user's server: each caller answers from its own script.

    A request belongs to the script named by its first message (an agent's
    system prompt, or ``generate_simple``'s prompt); a summary request is
    answered with a summary.
    """

    def __init__(self, **scripts: list) -> None:
        self.scripts = {key: list(script) for key, script in scripts.items()}
        self.api = FakeAPI(self._answer)

    def _answer(self, request):
        messages = json.loads(request.content)["messages"]
        if len(messages) == 1 and "<messages>" in str(messages[0]["content"]):
            return chat_completion("A summary.")
        return self.scripts[messages[0]["content"]].pop(0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A hermetic cwd and HOME (where the log lives), and fake keys."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable, key in KEYS.items():
        monkeypatch.setenv(variable, key)
    return tmp_path / "home"


def _settings(**values) -> BaseSettings:
    values.setdefault("model_endpoints", SMALL)
    values.setdefault("default_model", "local:small")
    return BaseSettings(session_store="memory", **values)


def _log_file(home, settings) -> object:
    return home / f".{settings.app_name}" / "logs" / "llm_events.jsonl"


def _records(home, settings) -> list[dict]:
    path = _log_file(home, settings)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def of(events, *kinds: EventType) -> list:
    return [event for event in events if event.type in kinds]


TEAM = [
    AgentConfig(name="lead", prompt="Lead.", sub_agents=["worker"],
                include_state_tools=False),
    AgentConfig(name="worker", prompt="Work.", description="Works.",
                tools=[rl_digest, rl_report], include_state_tools=False),
]


def _delegating_server(monkeypatch, turns: int = 1) -> Endpoint:
    """The server for ``turns`` delegating turns, each on a fresh manager."""
    server = Endpoint(**{
        "Lead.": [
            _call("task", {"description": "Report.", "subagent_type": "worker"}, "t1"),
            chat_completion("Done."),
        ] * turns,
        # The second report puts the worker over its trigger: it summarizes.
        "Work.": [
            _call("rl_digest", {}, "d1"),
            _call("rl_report", {}, "r1"),
            _call("rl_report", {}, "r2"),
            chat_completion("Worked."),
        ] * turns,
        DIGEST_PROMPT: [chat_completion("A digest.")] * turns,
    })
    route(monkeypatch, openai, "AsyncOpenAI", server.api)
    return server


async def _delegating_turn(settings) -> list:
    manager = DeepAgentsWorkflowManager(agent_configs=TEAM, settings=settings)
    try:
        return await turn(manager, "Go.")
    finally:
        await manager.cleanup()


class TestEveryCall:
    async def test_every_model_call_of_a_delegating_turn_is_recorded(self, home, monkeypatch):
        settings = _settings(raw_llm_logging=True)
        server = _delegating_server(monkeypatch)

        events = await _delegating_turn(settings)

        assert of(events, EventType.ERROR) == []
        requests = of(events, EventType.LLM_REQUEST)
        assert [(e.metadata["agent"], e.metadata["purpose"]) for e in requests] == [
            ("lead", "agent"),
            ("worker", "agent"),
            ("worker", "tool"),  # rl_digest's generate_simple
            ("worker", "agent"),
            ("worker", "agent"),
            ("worker", "summarization"),
            ("worker", "agent"),
            ("lead", "agent"),
        ]
        assert len(requests) == len(server.api.requests)  # every call, no other
        # Each call's response follows its request, with the same attribution.
        logged = of(events, *LOGGED)
        assert [e.type for e in logged] == [*LOGGED] * len(requests)
        for request, response in zip(logged[::2], logged[1::2]):
            assert response.metadata["run_id"] == request.metadata["run_id"]
            assert (response.metadata["agent"], response.metadata["purpose"]) == (
                request.metadata["agent"], request.metadata["purpose"]
            )
        # The same records, in the same order, in the log file.
        records = [r for r in _records(home, settings)
                   if r["type"] in ("llm_request", "llm_response")]
        assert [(r["type"], r["metadata"]["agent"], r["metadata"]["purpose"])
                for r in records] == [
            (e.type.value, e.metadata["agent"], e.metadata["purpose"]) for e in logged
        ]

    async def test_a_record_holds_what_was_sent_and_what_came_back(self, home, monkeypatch):
        settings = _settings(raw_llm_logging=True)
        _delegating_server(monkeypatch)

        events = await _delegating_turn(settings)

        logged = of(events, *LOGGED)
        request, response = logged[2], logged[3]  # the worker's first call
        assert request.metadata["model"] == "small"
        assert request.metadata["system_instruction"] == "Work."
        assert request.metadata["messages"] == [{"role": "user", "content": "Report."}]
        assert request.metadata["tools"] == ["rl_digest", "rl_report"]
        assert request.metadata["config"]["model_name"] == "small"
        assert "tools" not in request.metadata["config"]
        assert response.metadata["finish_reason"] == "tool_calls"
        assert response.metadata["model_version"] == "qwen/qwen3-235b-a22b"
        assert response.metadata["usage"]["input_tokens"] == 3
        assert response.metadata["latency_ms"] >= 0
        assert response.metadata["author"] == "worker"
        (call,) = [b for b in response.metadata["raw_parts"] if b["type"] == "tool_call"]
        assert (call["name"], call["id"]) == ("rl_digest", "d1")
        # A tool's own call: its prompt, and the agent whose tool made it.
        digest_request, digest_response = logged[4], logged[5]
        assert digest_request.metadata["messages"] == [
            {"role": "user", "content": DIGEST_PROMPT}
        ]
        assert "system_instruction" not in digest_request.metadata
        assert digest_response.content == "LLM Response: A digest."
        # Each answer's call is streamed before the node output that holds it.
        for index, event in enumerate(events):
            if event.type is EventType.TEXT:
                response = of(events[:index], EventType.LLM_RESPONSE)[-1]
                assert (response.metadata["agent"], response.metadata["purpose"]) == (
                    event.metadata["agent"], "agent"
                )
                assert response.content == f"LLM Response: {event.content}"

    async def test_logging_adds_no_usage_and_off_records_nothing(self, home, monkeypatch):
        _delegating_server(monkeypatch, turns=2)
        on = await _delegating_turn(_settings(raw_llm_logging=True))
        log = _log_file(home, _settings())
        log.unlink()

        off = await _delegating_turn(_settings(raw_llm_logging=False))

        def usage(events):
            return [(e.metadata["agent"], e.metadata.get("purpose"), e.metadata["model"])
                    for e in of(events, EventType.LLM_USAGE)]

        assert usage(on) == usage(off)
        # The agents' six calls and the summary; a tool's own call has none.
        assert len(usage(off)) == 7
        assert of(off, *LOGGED) == []
        assert not log.exists()
        assert [e.type for e in on if e.type not in LOGGED] == [e.type for e in off]

    async def test_the_setting_is_read_each_turn(self, home, monkeypatch):
        settings = _settings(raw_llm_logging=False)
        api = FakeAPI(chat_completion("Hi."))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.")], settings=settings
        )
        try:
            first = await turn(manager, "One.")
            settings.raw_llm_logging = True
            second = await turn(manager, "Two.")
        finally:
            await manager.cleanup()

        assert of(first, *LOGGED) == []
        assert [e.type for e in of(second, *LOGGED)] == [*LOGGED]


class TestWhatIsSent:
    async def test_a_claude_request_records_its_cache_marks(self, home, monkeypatch):
        settings = BaseSettings(session_store="memory", default_model="anthropic:claude-opus-4-7",
                                raw_llm_logging=True)
        route(monkeypatch, anthropic, "AsyncClient", FakeAPI(
            anthropic_reply({"type": "text", "text": "Hi."}, model="claude-opus-4-7")
        ))
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.", tools=[rl_report],
                                       include_state_tools=False)],
            settings=settings,
        )
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        (request,) = of(events, EventType.LLM_REQUEST)
        assert request.metadata["model"] == "claude-opus-4-7"
        assert request.metadata["system_instruction"] == "Answer."
        assert request.metadata["tools"] == ["rl_report"]
        assert request.metadata["config"]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
        (response,) = of(events, EventType.LLM_RESPONSE)
        assert (response.metadata["finish_reason"], response.metadata["model_version"]) == (
            "end_turn", "claude-opus-4-7"
        )

    async def test_secrets_are_redacted_in_the_file(self, home, monkeypatch):
        secret = "sk-" + "a1b2c3d4" * 4
        settings = _settings(raw_llm_logging=True)
        route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(chat_completion("Noted.")))
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.")], settings=settings
        )
        try:
            await turn(manager, f"My key is {secret}.")
        finally:
            await manager.cleanup()

        text = _log_file(home, settings).read_text()
        assert secret not in text
        assert "My key is [REDACTED]." in text


class TestFailures:
    async def test_a_failed_call_is_recorded_with_its_error(self, home, monkeypatch):
        import httpx

        settings = _settings(raw_llm_logging=True)
        route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(
            httpx.Response(400, json={"error": {"message": "the request was refused"}})
        ))
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.")], settings=settings
        )
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        request, response, error = events
        assert (request.type, response.type, error.type) == (*LOGGED, EventType.ERROR)
        # What the chat model raised, as the turn's ERROR names it.
        assert response.metadata["error_code"] == error.metadata["error_code"]
        assert "the request was refused" in response.metadata["error_message"]
        assert (response.metadata["agent"], response.metadata["purpose"]) == ("solo", "agent")
        assert response.metadata["run_id"] == request.metadata["run_id"]

    @pytest.mark.parametrize("unwritable", ["directory", "file"])
    async def test_an_unwritable_log_does_not_fail_the_turn(
        self, home, monkeypatch, caplog, unwritable
    ):
        caplog.set_level(logging.WARNING, logger="langchain_core")
        settings = _settings(raw_llm_logging=True)
        if unwritable == "directory":  # it cannot be made
            (home / f".{settings.app_name}").write_text("not a directory")
        else:  # it is made, but every append fails
            _log_file(home, settings).mkdir(parents=True)
        route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(chat_completion("Hi.")))
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Answer.")], settings=settings
        )
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        assert [e.type for e in events] == [
            *LOGGED, EventType.TEXT, EventType.LLM_USAGE
        ]
        # Nor does anything print over the UI (LangChain reports a failing
        # callback as a warning).
        assert [r.getMessage() for r in caplog.records] == []

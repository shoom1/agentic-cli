"""The Deep Agents workflow manager (``workflow/deepagents/manager.py``).

``DeepAgentsWorkflowManager`` is the ``deepagents`` orchestrator. Initializing
it validates the agent graph, opens the checkpointer ``session_store`` names,
and builds every agent on its own chat model (``build_chat_model``, from the
agent's resolved model), with its prompt resolved once and its tools
converted. A turn streams the root agent on the thread of the turn's
``SessionRef`` and yields each node's output as ``WorkflowEvent``s through
``on_event``; a turn that fails is one ``ERROR`` and leaves the user's message
in the conversation. The session hooks answer from the checkpointer, and
``generate_simple`` runs the default model outside the agents.

Agent models are scripted (``fakes.ScriptedModels`` patched over
``build_chat_model``); ``generate_simple``'s requests go to ``FakeAPI``.
Nothing reaches a provider.
"""

from __future__ import annotations

import anthropic
import pytest

pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings, get_settings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow import factory  # noqa: E402
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import thread_id  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.service_registry import WORKFLOW, get_service  # noqa: E402
from agentic_cli.workflow.sessions import get_active_turn  # noqa: E402
from agentic_cli.workflow.settings import OrchestratorType  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task  # noqa: E402

SOLO = [AgentConfig(name="solo", prompt="Answer.")]


@register_tool(capabilities=EXEMPT)
def turn_probe() -> dict:
    """Reports what a tool sees of the turn running it (an exempt test tool)."""
    turn = get_active_turn()
    return {
        "success": True,
        "user": turn.user_id if turn else None,
        "session": turn.session_id if turn else None,
        "workflow": type(get_service(WORKFLOW)).__name__,
    }


class _Recorder:
    """Stands in for the module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def ctx(hermetic):
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    """Every agent built from now on runs on its scripted model."""
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _manager(settings, configs=SOLO, **kwargs) -> DeepAgentsWorkflowManager:
    kwargs.setdefault("model", "gemini-2.5-flash")
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings, **kwargs)


@pytest.fixture
async def cleanup():
    """Clean up every manager a test hands it."""
    managers: list[DeepAgentsWorkflowManager] = []
    yield managers.append
    for manager in managers:
        await manager.cleanup()


async def turn(manager, text: str = "Hello.", *, user: str = "u", session: str = "s") -> list:
    return [event async for event in manager.process(text, user, session_id=session)]


def of(events, kind: EventType) -> list[WorkflowEvent]:
    return [event for event in events if event.type is kind]


class TestRegistration:
    def test_the_orchestrator_setting_accepts_deepagents(self, hermetic):
        settings = BaseSettings(orchestrator="deepagents")
        assert settings.orchestrator is OrchestratorType.DEEPAGENTS
        assert BaseSettings().orchestrator is OrchestratorType.ADK  # still the default

    def test_the_factory_builds_it(self, ctx):
        ctx.settings.orchestrator = OrchestratorType.DEEPAGENTS

        manager = factory.create_workflow_manager_from_settings(SOLO, ctx.settings)

        assert type(manager) is DeepAgentsWorkflowManager
        assert manager.backend_type == "deepagents"

    def test_the_package_exports_it(self):
        import agentic_cli.workflow.deepagents as backend

        assert backend.DeepAgentsWorkflowManager is DeepAgentsWorkflowManager
        assert backend.__all__ == ["DeepAgentsWorkflowManager"]

    def test_it_has_durable_sessions(self, ctx):
        assert _manager(ctx.settings).supports_sessions is True

    def test_its_state_tools_are_the_langgraph_library_ones(self, ctx):
        from agentic_cli.tools.langgraph import state_tools

        assert _manager(ctx.settings)._get_state_tools() == [
            state_tools.save_plan, state_tools.get_plan,
            state_tools.save_tasks, state_tools.get_tasks,
        ]


class TestInitialization:
    async def test_each_agent_runs_on_its_own_resolved_model(self, ctx, monkeypatch, cleanup):
        models = scripted(
            monkeypatch,
            coordinator=[ai(calls=[task("writer", "Write it.")]), ai("Written.")],
            writer=[ai("A draft.")],
        )
        configs = [
            AgentConfig(name="coordinator", prompt="Lead.", sub_agents=["writer"]),
            AgentConfig(name="writer", prompt="Write.", description="Writes.",
                        model="gemini-2.5-pro"),
        ]
        manager = _manager(ctx.settings, configs)
        cleanup(manager)

        events = await turn(manager)

        assert {r.agent: str(r.ref) for r in models.built} == {
            "coordinator": "google:gemini-2.5-flash",
            "writer": "google:gemini-2.5-pro",
        }
        assert dict(manager.agent_models) == {r.agent: r for r in models.built}
        assert models.chats["writer"].conversation(0)[0].text == "Write it."
        assert [e.metadata["model"] for e in of(events, EventType.LLM_USAGE)] == [
            "google:gemini-2.5-flash", "google:gemini-2.5-pro", "google:gemini-2.5-flash"
        ]

    async def test_a_prompt_is_resolved_once_under_the_managers_settings(
        self, ctx, monkeypatch, cleanup
    ):
        models = scripted(monkeypatch, solo=[ai("One."), ai("Two.")])
        seen = []

        def prompt() -> str:
            seen.append(get_settings())
            return "You answer."

        manager = _manager(ctx.settings, [AgentConfig(name="solo", prompt=prompt)])
        cleanup(manager)

        await turn(manager)
        await turn(manager)

        assert seen == [ctx.settings]
        assert models.chats["solo"].system_prompt(1) == "You answer."

    async def test_tools_are_the_assembled_ones_converted(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[ai("Ok.")])
        configs = [AgentConfig(name="solo", prompt="p", tools=[turn_probe],
                               include_state_tools=True)]
        manager = _manager(ctx.settings, configs)
        cleanup(manager)

        await turn(manager)

        assert models.chats["solo"].tool_names(0) == [
            "turn_probe", "save_plan", "get_plan", "save_tasks", "get_tasks"
        ]

    async def test_an_invalid_graph_fails_before_anything_is_opened(self, ctx, monkeypatch):
        opened = []
        monkeypatch.setattr(
            manager_module, "open_checkpointer", lambda *a, **k: opened.append(a)
        )
        two_parents = [
            AgentConfig(name="a", prompt="p", sub_agents=["c"]),
            AgentConfig(name="b", prompt="p", sub_agents=["c"]),
            AgentConfig(name="c", prompt="p"),
        ]
        manager = _manager(ctx.settings, two_parents)

        with pytest.raises(AgentGraphError, match="deepagents backend requires a tree"):
            await manager.initialize_services()

        assert opened == []
        assert manager.is_initialized is False


class TestTurns:
    async def test_a_turn_streams_its_events_by_agent(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hello there.")])
        manager = _manager(ctx.settings)
        cleanup(manager)

        events = await turn(manager, session="s1")

        assert [e.type for e in events] == [EventType.TEXT, EventType.LLM_USAGE]
        assert events[0].content == "Hello there."
        assert {e.metadata["agent"] for e in events} == {"solo"}
        assert events[0].metadata["session_id"] == "s1"

    async def test_every_event_passes_the_event_hook(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hidden.")])
        seen = []

        def hook(event: WorkflowEvent) -> WorkflowEvent | None:
            seen.append(event.type)
            return None if event.type is EventType.TEXT else event

        manager = _manager(ctx.settings, on_event=hook)
        cleanup(manager)

        events = await turn(manager)

        assert seen == [EventType.TEXT, EventType.LLM_USAGE]
        assert [e.type for e in events] == [EventType.LLM_USAGE]

    async def test_a_tool_sees_the_turn_and_the_services(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai(calls=[("turn_probe", {})]), ai("Done.")])
        manager = _manager(ctx.settings, [AgentConfig(name="solo", prompt="p",
                                                      tools=[turn_probe])])
        cleanup(manager)

        events = await turn(manager, user="alice", session="s7")

        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["result"] == {
            "success": True, "user": "alice", "session": "s7",
            "workflow": "DeepAgentsWorkflowManager",
        }

    async def test_the_conversation_is_the_session_refs(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[ai("One."), ai("Two."), ai("Three.")])
        manager = _manager(ctx.settings)
        cleanup(manager)

        await turn(manager, "Alice here.", user="alice", session="s")
        await turn(manager, "Bob here.", user="bob", session="s")
        await turn(manager, "Alice again.", user="alice", session="s")

        chat = models.chats["solo"]
        assert [m.text for m in chat.conversation(1)] == ["Bob here."]
        assert [m.text for m in chat.conversation(2)] == ["Alice here.", "One.", "Alice again."]
        saved = await manager._checkpointer.aget_tuple({"configurable": {
            "thread_id": thread_id(manager.session_ref("s", "bob")), "checkpoint_ns": "",
        }})
        assert saved is not None

    async def test_a_failed_turn_is_one_error_and_the_next_continues(
        self, ctx, monkeypatch, cleanup
    ):
        models = scripted(
            monkeypatch, solo=[RuntimeError("provider said 400"), ai("Recovered.")]
        )
        recorder = _Recorder()
        monkeypatch.setattr(manager_module, "logger", recorder)
        manager = _manager(ctx.settings)
        cleanup(manager)

        failed = await turn(manager, "First.")
        answered = await turn(manager, "Second.")

        assert [e.type for e in failed] == [EventType.ERROR]
        assert failed[0].content == "provider said 400"
        assert failed[0].metadata["error_code"] == "RuntimeError"
        assert failed[0].metadata["recoverable"] is False
        assert failed[0].metadata["agent"] == "solo"
        assert [e for _, e, _ in recorder.calls].count("deepagents_turn_failed") == 1
        # The failed turn's message stays; the next turn continues after it.
        assert [m.text for m in models.chats["solo"].conversation(1)] == ["First.", "Second."]
        assert [e.content for e in of(answered, EventType.TEXT)] == ["Recovered."]

    async def test_pending_notices_open_the_next_turn_once(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("One."), ai("Two.")])
        manager = _manager(ctx.settings)
        cleanup(manager)
        await manager.initialize_services()
        notice = WorkflowEvent.error("Server x is unavailable.", error_code="mcp_unavailable",
                                     recoverable=True)
        manager._pending_notices.append(notice)

        first = await turn(manager)
        second = await turn(manager)

        assert first[0] is notice
        assert of(second, EventType.ERROR) == []


class TestSessionHooks:
    async def test_before_initialization_there_is_nothing(self, ctx):
        manager = _manager(ctx.settings)

        assert await manager.session_exists("s") is False
        assert await manager.list_sessions() == []
        assert await manager.delete_session("s") is False
        assert await manager.recent_messages("s") == []

    async def test_they_default_to_the_default_user(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hi.")])
        manager = _manager(ctx.settings)
        cleanup(manager)
        await turn(manager, "Hello.", user=ctx.settings.default_user, session="s")

        assert await manager.session_exists("s") is True
        assert [e["session_id"] for e in await manager.list_sessions()] == ["s"]
        assert await manager.recent_messages("s") == [
            {"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}
        ]
        assert await manager.delete_session("s") is True
        assert await manager.session_exists("s") is False
        assert await manager.delete_session("s") is False

    async def test_an_explicit_user_is_never_the_default(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hi.")])
        manager = _manager(ctx.settings)
        cleanup(manager)
        await turn(manager, user="alice", session="s")

        assert await manager.session_exists("s") is False
        assert await manager.list_sessions() == []
        assert await manager.recent_messages("s") == []
        assert await manager.session_exists("s", user_id="alice") is True
        assert [e["session_id"] for e in await manager.list_sessions(user_id="alice")] == ["s"]
        assert len(await manager.recent_messages("s", user_id="alice")) == 2

    async def test_load_session_adopts_the_id_and_reports_a_resume(
        self, ctx, monkeypatch, cleanup
    ):
        scripted(monkeypatch, solo=[ai("Hi.")])
        manager = _manager(ctx.settings)
        cleanup(manager)
        await turn(manager, user=ctx.settings.default_user, session="old")

        assert await manager.load_session("old") is True
        assert manager.session_id == "old"
        assert await manager.load_session("new") is False
        assert manager.session_id == "new"
        assert (await manager.save_session())["session_id"] == "new"

    async def test_a_session_survives_a_fresh_manager_on_sqlite(self, hermetic, monkeypatch):
        scripted(monkeypatch, solo=[ai("Remembered.")])
        with MockContext(google_api_key="test-key", session_store="sqlite") as context:
            first = _manager(context.settings)
            try:
                await turn(first, "Remember this.", user="u", session="s")
            finally:
                await first.cleanup()

            second = _manager(context.settings)
            try:
                await second.initialize_services()
                assert await second.session_exists("s", user_id="u") is True
                assert [m["content"] for m in await second.recent_messages(
                    "s", user_id="u"
                )] == ["Remember this.", "Remembered."]
            finally:
                await second.cleanup()


class TestGenerateSimple:
    """Summaries run on the default model, through the LangChain adapter, without thinking."""

    async def test_claude(self, hermetic, monkeypatch):
        from tests.workflow import langchain_fakes

        api = FakeAPI(anthropic_message({"type": "text", "text": "A summary."}))
        langchain_fakes.route(monkeypatch, anthropic, "AsyncClient", api)
        with MockContext(anthropic_api_key="a-test-key", session_store="memory",
                         thinking_effort="high") as context:
            manager = _manager(context.settings, model="claude-sonnet-4-5")
            try:
                summary = await manager.generate_simple("Summarize this.", max_tokens=123)
            finally:
                await manager.cleanup()

        assert summary == "A summary."
        (body,) = api.bodies
        assert (body["model"], body["max_tokens"]) == ("claude-sonnet-4-5", 123)
        assert body["thinking"] == {"type": "disabled"}
        assert body["messages"] == [{"role": "user", "content": "Summarize this."}]

    async def test_gemini_and_its_thoughts_are_not_the_summary(self, hermetic, monkeypatch):
        from tests.workflow import langchain_fakes

        api = FakeAPI(langchain_fakes.gemini_reply(
            {"text": "Thinking it over.", "thought": True}, {"text": "A summary."}
        ))
        langchain_fakes.route_google(monkeypatch, api)
        with MockContext(google_api_key="g-test-key", session_store="memory",
                         thinking_effort="high") as context:
            manager = _manager(context.settings)
            try:
                summary = await manager.generate_simple("Summarize this.", max_tokens=123)
            finally:
                await manager.cleanup()

        assert summary == "A summary."
        (body,) = api.bodies
        assert body["generationConfig"]["maxOutputTokens"] == 123
        assert body["generationConfig"]["thinkingConfig"]["thinking_budget"] == 0

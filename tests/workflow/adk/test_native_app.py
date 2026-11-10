"""ADK: a native ADK YAML app (``adk_config_path``) is built from its YAML and run as any app.

The YAML only decides how the agents are built. It is loaded when the
manager validates its agents, before any model, service or store: a root
that is not an ``LlmAgent``, or a task-mode agent anywhere in the tree, is
refused there with the agent named. Agent configs passed with it are
ignored, with a warning.

Every provider base URL points at a dead local port, and both Anthropic
client bindings (the framework's and the one ADK's own ``AnthropicLlm``
builds for a bare ``claude-*`` model) are routed to a fake API.
"""

from __future__ import annotations

import json
import textwrap

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk import anthropic_llm  # noqa: E402
from agentic_cli.workflow.adk import manager as adk_manager  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.compaction import (  # noqa: E402
    AgentCliSummarizer,
    NoAfterTurnCompactionRunner,
)
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.adk import native_agents  # noqa: E402
from tests.workflow.adk.fakes import (  # noqa: E402
    FakeAPI,
    anthropic_message,
    dead_base_urls,
    route_anthropic,
)
from tests.workflow.cap_probes import CUT  # noqa: E402

MODEL = "claude-sonnet-4-6"


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    dead_base_urls(monkeypatch)
    return tmp_path


class _LogRecorder:
    """Stands in for the manager's logger and records every call."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, *args, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    recorder = _LogRecorder()
    monkeypatch.setattr(adk_manager, "logger", recorder)
    return recorder.events


def settings(**overrides) -> BaseSettings:
    return BaseSettings(**{"session_store": "memory", "default_model": MODEL, **overrides})


def write(project, name: str, text: str, extra: str = "") -> str:
    """``text`` (dedented) and then ``extra`` (as it is) in ``name``."""
    path = project / name
    path.write_text(textwrap.dedent(text) + extra, encoding="utf-8")
    return str(path)


def root_yaml(project, *, model: str = MODEL, extra: str = "") -> str:
    return write(project, "root_agent.yaml", f"""
        agent_class: LlmAgent
        name: root
        model: {model}
        instruction: You are root.
    """, extra)


def native(path: str, configs=(), **overrides) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=list(configs), settings=settings(**overrides), adk_config_path=path
    )


class TestTheYAMLIsLoadedWhenItIsValidated:
    async def test_a_root_that_is_not_an_llm_agent_is_refused(self, project):
        write(project, "helper.yaml", """
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: You help.
        """)
        path = write(project, "root_agent.yaml", """
            agent_class: SequentialAgent
            name: pipeline
            sub_agents:
              - config_path: helper.yaml
        """)
        manager = native(path)

        with pytest.raises(AgentGraphError, match="root agent 'pipeline' is a SequentialAgent"):
            await manager.initialize_services()

        # Refused before anything was allocated.
        assert not manager.is_initialized
        assert manager._session_service is None
        assert manager.catalog is None

    @pytest.mark.parametrize("nested", [False, True], ids=["a-sub-agent", "deeper"])
    async def test_a_task_mode_agent_is_refused_by_name(self, project, nested):
        task = "sub_agents:\n  - code: tests.workflow.adk.native_agents.asker\n"
        if nested:
            write(project, "helper.yaml", """
                agent_class: LlmAgent
                name: helper
                model: claude-sonnet-4-6
                instruction: You help.
            """, task)
            path = root_yaml(project, extra="sub_agents:\n  - config_path: helper.yaml\n")
        else:
            path = root_yaml(project, extra=task)
        manager = native(path)

        with pytest.raises(AgentGraphError) as raised:
            await manager.initialize_services()

        assert str(raised.value) == (
            "agent 'asker' is in task mode, which is not supported; use single_turn or chat"
        )
        assert manager._session_service is None

    async def test_its_root_is_the_main_agent(self, project):
        manager = native(root_yaml(project))
        await manager.initialize_services()
        try:
            assert manager.root_agent_name == "root"
        finally:
            await manager.cleanup()

    async def test_agent_configs_passed_with_it_are_ignored_with_a_warning(
        self, project, logged
    ):
        manager = native(root_yaml(project), configs=[AgentConfig(name="a", prompt="Help.")])

        assert manager.agent_configs == []
        (warning,) = [e for e in logged if e["event"] == "adk_native_configs_ignored"]
        assert (warning["log_level"], warning["agents"]) == ("warning", ["a"])

    async def test_only_a_native_app_points_bare_claude_models_at_anthropics_api(
        self, project, monkeypatch
    ):
        calls: list[str] = []
        monkeypatch.setattr(
            anthropic_llm, "ensure_direct_anthropic", lambda: calls.append("registered")
        )
        built = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Help.")], settings=settings()
        )
        await built.initialize_services()
        await built.cleanup()
        assert calls == []

        manager = native(root_yaml(project))
        assert calls == []
        await manager.initialize_services()
        await manager.cleanup()
        assert calls == ["registered"]


class TestItsAgentsRunOnTheirOwnModels:
    """Each agent's model is the one ADK resolves for it (its YAML's, else its
    parent's), resolved through the catalog for its context budget, as an
    agent config naming it would be. An agent whose model does not resolve
    gets no budget, and the fit guard and the overflow handler leave it
    alone."""

    async def test_each_agent_gets_the_budget_of_the_model_adk_runs_it_on(
        self, project, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        write(project, "helper.yaml", """
            agent_class: LlmAgent
            name: helper
            instruction: You help.
        """)
        path = root_yaml(project, model="gemini-2.5-flash", extra=(
            "sub_agents:\n  - config_path: helper.yaml\n"
            "  - code: tests.workflow.adk.native_agents.chatter\n"
        ))
        manager = native(path)
        await manager.initialize_services()
        try:
            models = {name: str(model.ref) for name, model in manager.agent_models.items()}
            budgets = dict(manager._budgets)
            main = manager.context_budget()
        finally:
            await manager.cleanup()

        # The helper names no model: ADK runs it on its parent's.
        assert models == {
            "root": "google:gemini-2.5-flash",
            "helper": "google:gemini-2.5-flash",
            "chatter": "anthropic:claude-sonnet-4-6",
        }
        assert set(budgets) == {"root", "helper", "chatter"}
        assert main == budgets["root"] and main.window == 1_048_576

    async def test_an_agent_whose_model_does_not_resolve_gets_no_budget(self, project, logged):
        write(project, "helper.yaml", """
            agent_class: LlmAgent
            name: helper
            model: local-model
            instruction: You help.
        """)
        manager = native(root_yaml(project, extra="sub_agents:\n  - config_path: helper.yaml\n"))
        await manager.initialize_services()
        try:
            assert set(manager.agent_models) == {"root"}
            assert set(manager._budgets) == {"root"}
        finally:
            await manager.cleanup()

        (warning,) = [e for e in logged if e["event"] == "adk_agent_model_unresolved"]
        assert (warning["log_level"], warning["agent"]) == ("warning", "helper")

    async def test_a_registered_tool_it_names_gets_its_service(self, project):
        path = root_yaml(project, extra=(
            "tools:\n  - name: agentic_cli.tools.memory_tools.search_memory\n"
        ))
        manager = native(path)
        await manager.initialize_services()
        try:
            assert "memory_store" in manager.required_managers
            assert manager.services.get("memory_store") is not None
        finally:
            await manager.cleanup()


class TestWhatTheRunReadsOfTheTree:
    """What the run needs of a tree is read from the tree, whoever built it."""

    async def test_the_agents_that_reply_and_those_with_their_own_after_tool_callbacks(
        self, project
    ):
        from agentic_cli.workflow.adk.build import (
            load_native_root,
            own_after_tool_agents,
            reply_agents,
        )

        write(project, "helper.yaml", """
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: You help.
            after_tool_callbacks:
              - name: tests.workflow.adk.native_agents.after_tool
        """)
        root = load_native_root(root_yaml(project, extra=(
            "sub_agents:\n  - config_path: helper.yaml\n"
            "  - code: tests.workflow.adk.native_agents.solver\n"
        )))

        # The helper is chat-mode (ADK's default for a sub-agent): it takes
        # the conversation. The solver is a single-turn tool of the root's.
        assert reply_agents(root) == {"root", "helper"}
        assert own_after_tool_agents(root) == {"helper"}

    async def test_an_app_built_from_configs_has_its_root_reply_alone(self, project):
        from agentic_cli.workflow.adk.build import own_after_tool_agents, reply_agents

        manager = GoogleADKWorkflowManager(
            agent_configs=[
                AgentConfig(name="lead", prompt="Lead.", sub_agents=["aide"]),
                AgentConfig(name="aide", prompt="Aid.", description="Aids."),
            ],
            settings=settings(),
        )
        await manager.initialize_services()
        try:
            root = manager._root_agent
            assert (reply_agents(root), own_after_tool_agents(root)) == ({"lead"}, set())
        finally:
            await manager.cleanup()



class Claude(FakeAPI):
    """Claude for each agent, told apart by its system prompt; it records
    which agent each request was for."""

    PROMPTS = {"You are root.": "root", "You help.": "helper"}

    def __init__(self, **scripts) -> None:
        super().__init__(self._answer)
        self.scripts = {name: list(script) for name, script in scripts.items()}
        self.agents: list[str] = []

    def _answer(self, request):
        system = json.dumps(json.loads(request.content).get("system"))
        agent = next(name for prompt, name in self.PROMPTS.items() if prompt in system)
        self.agents.append(agent)
        return self.scripts[agent].pop(0)

    def sent_to(self, agent: str) -> list[dict]:
        """The bodies of the requests made for ``agent``, in order."""
        return [body for body, who in zip(self.bodies, self.agents) if who == agent]


def says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text}, model=MODEL)


def calls(name: str, call_id: str, **args) -> dict:
    return anthropic_message(
        {"type": "tool_use", "id": call_id, "name": name, "input": args}, model=MODEL
    )


def helper_yaml(project, extra: str = "") -> None:
    write(project, "helper.yaml", """
        agent_class: LlmAgent
        name: helper
        model: claude-sonnet-4-6
        description: Helps.
        instruction: You help.
    """, extra)


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def results_sent(body: dict) -> list[str]:
    """The tool results a Claude request carries, as text."""
    contents = [
        block.get("content")
        for message in body["messages"] if isinstance(message["content"], list)
        for block in message["content"] if block.get("type") == "tool_result"
    ]
    return [content if isinstance(content, str) else json.dumps(content) for content in contents]


class TestItRunsAsAnyApp:
    """The run is the one an app built from configs gets: its whole plugin
    chain (pinned by ``test_plugin_chain.py``), the main agent summarized,
    the tool thread pool, calls left open answered, and a job told in a
    notice."""

    async def test_its_main_agent_is_summarized_at_its_trigger(self, project):
        manager = native(root_yaml(project))
        await manager.initialize_services()
        try:
            runner, supports = manager._runner, manager.supports_compact
            config = runner.app.events_compaction_config
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        assert type(runner) is NoAfterTurnCompactionRunner
        assert isinstance(config.summarizer, AgentCliSummarizer)
        assert (config.summarizer.root, config.token_threshold) == ("root", trigger)
        assert supports is True

    async def test_a_main_agent_without_a_budget_is_not_summarized(self, project):
        manager = native(root_yaml(project, model="local-model"))
        await manager.initialize_services()
        try:
            config = manager._runner.app.events_compaction_config
            supports = manager.supports_compact
            with pytest.raises(NotImplementedError, match="no context budget"):
                await manager.compact(manager.session_ref("s", "u"))
        finally:
            await manager.cleanup()

        assert config is None
        assert supports is False

    async def test_each_agents_usage_names_its_own_model(self, project, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        manager = native(root_yaml(project, model="gemini-2.5-flash"))
        await manager.initialize_services()
        try:
            models = dict(manager._event_processor.models)
        finally:
            await manager.cleanup()

        assert models == {"root": "google:gemini-2.5-flash"}

    async def test_an_agent_it_transfers_to_replies_to_the_user(self, project, monkeypatch):
        helper_yaml(project)
        claude = Claude(
            root=[calls("transfer_to_agent", "toolu_t", agent_name="helper")],
            helper=[says("Helper here."), says("Still helper.")],
        )
        route_anthropic(monkeypatch, claude)
        manager = native(root_yaml(project, extra="sub_agents:\n  - config_path: helper.yaml\n"))
        await manager.initialize_services()
        try:
            replying = manager._event_processor.reply_agents
            first = await turn(manager, "Ask the helper.")
            second = await turn(manager, "Again.")
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert replying == {"root", "helper"}
        # ADK starts the next turn at the agent the root transferred to.
        assert claude.agents == ["root", "helper", "helper"]
        for events, text in ((first, "Helper here."), (second, "Still helper.")):
            (reply,) = [e for e in events if e.type is EventType.TEXT]
            assert (reply.content, reply.metadata.get("subagent")) == (text, None)
            assert all("purpose" not in e.metadata for e in events
                       if e.type is EventType.LLM_USAGE)
        assert [m["content"] for m in recent if m["role"] == "assistant"] == [
            "Helper here.", "Still helper.",
        ]

    async def test_the_cap_leaves_out_an_agent_with_its_own_after_tool_callbacks(
        self, project, monkeypatch
    ):
        """ADK runs an agent's own after-tool callbacks only when no plugin
        answered: the cap leaves that agent's results whole, and cuts the
        others'."""
        helper_yaml(project, "tools:\n  - name: tests.workflow.adk.native_agents.native_app_report\n")
        claude = Claude(
            root=[calls("native_app_report", "toolu_r1"), says("Read it."),
                  calls("transfer_to_agent", "toolu_t", agent_name="helper")],
            helper=[calls("native_app_report", "toolu_h1"), says("Read it too.")],
        )
        route_anthropic(monkeypatch, claude)
        native_agents.SEEN.clear()
        manager = native(root_yaml(project, extra=(
            "tools:\n  - name: tests.workflow.adk.native_agents.native_app_report\n"
            "after_tool_callbacks:\n  - name: tests.workflow.adk.native_agents.after_tool\n"
            "sub_agents:\n  - config_path: helper.yaml\n"
        )), tool_result_max_chars=20_000)
        try:
            await turn(manager, "Report.")
            await turn(manager, "Ask the helper.")
        finally:
            await manager.cleanup()

        assert [seen for seen in native_agents.SEEN if seen[1] == "native_app_report"] == [
            ("after_tool", "native_app_report", 50_000)
        ]
        (root_result,) = results_sent(claude.sent_to("root")[1])
        assert root_result == "x" * 50_000
        helper_results = results_sent(claude.sent_to("helper")[1])
        assert CUT.search(helper_results[-1])

    async def test_its_synchronous_tools_run_in_the_tool_thread_pool(self, project):
        manager = native(root_yaml(project))
        passed: dict = {}

        async def run_async(**kwargs):
            passed.update(kwargs)
            return
            yield  # an async generator that yields no event

        try:
            await manager.initialize_services()
            manager._runner.run_async = run_async
            await turn(manager, "Go.")
        finally:
            await manager.cleanup()

        assert passed["run_config"].tool_thread_pool_config.max_workers == 8

    async def test_a_job_is_told_in_a_notice_and_needs_no_call(self, project):
        manager = native(root_yaml(project))
        try:
            assert manager._answers_job_calls is False
        finally:
            await manager.cleanup()

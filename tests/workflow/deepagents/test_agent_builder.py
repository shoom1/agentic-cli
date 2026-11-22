"""The agents (``workflow/deepagents/agent_builder.py``).

Every ``AgentConfig`` becomes one ``create_agent`` graph on its own chat
model, each built on its own. A coordinator delegates through the
framework's ``task`` tool (``task_tool.build_task_tool``), which lists each
sub-agent by its description, takes the brief's fields and is bound to the
framework's declaration; its calls go to the manager's delegations, which
run the sub-agent on its own graph (here ``fakes.tree_delegations``, as the
manager's hook does). A sub-agent sees only its task message, shares
``plan`` and ``tasks``, and its last text is the result its parent reads in
the envelope. Every agent is compiled on the manager's checkpointer (a
sub-agent's runs are on a thread of its instance's own); a tree built without
one compiles its sub-agents with none. Every stack has the
permission gate and, inside it, the tool-error middleware.

The models are scripted (``fakes.ScriptedChat``); nothing reaches a provider.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agentic_cli.tools.langgraph.state_tools import (  # noqa: E402
    get_plan,
    save_plan,
    save_tasks,
)
from agentic_cli.tools.registry import (  # noqa: E402
    ToolDefinition,
    get_registry,
    identify_tool,
    register_tool,
)
from agentic_cli.workflow.agents import brief as brief_module  # noqa: E402
from agentic_cli.workflow.agents import SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.agents.brief import (  # noqa: E402
    Brief,
    follow_up_description,
    follow_up_parameters,
    task_message,
)
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.deepagents import agent_builder  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    ActiveToolCallMiddleware,
    PermissionMiddleware,
    RunControlMiddleware,
    SubAgentOutcomeMiddleware,
    ToolCallErrorMiddleware,
)
from agentic_cli.workflow.deepagents.state import AgentCliState  # noqa: E402
from agentic_cli.workflow.deepagents.tools import to_langchain_tools  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.capabilities import Capability  # noqa: E402
from tests.workflow.deepagents.fakes import (  # noqa: E402
    ScriptedChat,
    ai,
    task,
    tree_delegations,
)


@register_tool(capabilities=EXEMPT)
def builder_flaky(x: str) -> dict:
    """Always raises (an exempt test tool)."""
    raise RuntimeError("flaky broke")


class _Recorder:
    """Stands in for the module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def created(monkeypatch) -> dict[str, dict]:
    """Every ``create_agent`` call the builder makes, by agent name, in order."""
    calls: dict[str, dict] = {}
    real = agent_builder.create_agent

    def spy(*args, **kwargs):
        calls[kwargs["name"]] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_builder, "create_agent", spy)
    return calls


def _tree(configs, models, tools=None, checkpointer=None, policy=None, **options):
    graphs: dict = {}
    return build_agent_tree(
        configs,
        chat_models=models,
        resolved={},
        tools=tools or {},
        prompts={config.name: f"You are {config.name}." for config in configs},
        settings=None,
        checkpointer=checkpointer,
        delegations=tree_delegations(configs, graphs, policy=policy),
        graphs=graphs,
        **options,
    )


def _task_tool(created: dict, agent: str) -> StructuredTool:
    """The ``task`` tool the builder gave ``agent``."""
    (found,) = [tool for tool in created[agent]["tools"] if tool.name == "task"]
    return found


def _read(message: ToolMessage) -> dict:
    """The envelope a ``task`` call's answer holds."""
    return json.loads(message.text)


async def _turn(root, message: str = "Go.", thread: str = "t") -> dict:
    return await root.ainvoke(
        {"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": thread}}
    )


COORDINATOR = AgentConfig(
    name="coordinator", prompt="p", sub_agents=["researcher"], description="Leads."
)
RESEARCHER = AgentConfig(name="researcher", prompt="p", description="Finds facts.")


def _shape(messages) -> list[tuple[str, str]]:
    """Each message's type and text."""
    return [(m.type, m.text) for m in messages]


def _tool_message(messages, call_id: str) -> ToolMessage:
    return next(
        m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id
    )


class TestTheTree:
    async def test_a_coordinator_delegates_and_answers(self, created):
        coordinator = ScriptedChat(label="coordinator", script=[
            ai(calls=[task("researcher", "Find X.", "c1")]),
            ai("X is 42, says the researcher."),
        ])
        researcher = ScriptedChat(label="researcher", script=[ai("X is 42.")])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            checkpointer=InMemorySaver(),
        )
        state = await _turn(root, "What is X?")

        assert list(created) == ["coordinator", "researcher"]  # each on its own
        assert [type(m) for m in state["messages"]] == [
            HumanMessage, AIMessage, ToolMessage, AIMessage
        ]
        assert _read(state["messages"][2])["result"] == "X is 42."
        assert state["messages"][-1].text == "X is 42, says the researcher."

    async def test_the_root_is_the_agent_nobody_delegates_to(self, created):
        coordinator = ScriptedChat(script=[ai("Hello.")])
        researcher = ScriptedChat(script=[])

        root = _tree(
            [RESEARCHER, COORDINATOR],  # declared sub-agent first
            {"coordinator": coordinator, "researcher": researcher},
            checkpointer=InMemorySaver(),
        )
        await _turn(root)

        assert created["coordinator"]["checkpointer"] is not None
        assert len(coordinator.requests) == 1 and researcher.requests == []

    async def test_a_free_type_listed_first_is_no_root(self, created):
        helper = AgentConfig(name="helper", prompt="p", spawnable_by=frozenset({"agent"}))
        coordinator = ScriptedChat(script=[ai("Hello.")])
        free = ScriptedChat(script=[])

        root = _tree(
            [helper, COORDINATOR, RESEARCHER],
            {"helper": free, "coordinator": coordinator, "researcher": ScriptedChat()},
            checkpointer=InMemorySaver(),
        )
        await _turn(root)

        assert len(coordinator.requests) == 1 and free.requests == []
        # Built as a sub-agent is: a run's control in its stack.
        assert RunControlMiddleware in [type(m) for m in created["helper"]["middleware"]]

    async def test_the_root_given_is_the_root(self, created):
        coordinator = ScriptedChat(script=[ai("Hello.")])
        other = ScriptedChat(script=[])
        alone = AgentConfig(name="alone", prompt="p")

        root = _tree(
            [alone, COORDINATOR, RESEARCHER],
            {"alone": other, "coordinator": coordinator, "researcher": ScriptedChat()},
            checkpointer=InMemorySaver(), root="coordinator",
        )
        await _turn(root)

        assert len(coordinator.requests) == 1 and other.requests == []
        assert RunControlMiddleware not in [
            type(m) for m in created["coordinator"]["middleware"]
        ]

    async def test_a_single_agent_is_its_own_root(self, created):
        solo = ScriptedChat(script=[ai("Hi.")])
        saver = InMemorySaver()

        root = _tree([AgentConfig(name="solo", prompt="p")], {"solo": solo}, checkpointer=saver)
        state = await _turn(root)

        assert state["messages"][-1].text == "Hi."
        assert root.checkpointer is saver
        assert [type(m) for m in created["solo"]["middleware"]] == [
            PatchToolCallsMiddleware, PermissionMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]

    async def test_each_agent_runs_on_its_own_model_and_prompt(self, created):
        coordinator = ScriptedChat(label="coordinator", script=[
            ai(calls=[task("researcher", "Find X.")]), ai("Done."),
        ])
        researcher = ScriptedChat(label="researcher", script=[ai("X is 42.")])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            checkpointer=InMemorySaver(),
        )
        await _turn(root)

        assert created["coordinator"]["model"] is coordinator
        assert created["researcher"]["model"] is researcher
        assert len(coordinator.requests) == 2 and len(researcher.requests) == 1
        assert coordinator.system_prompt(0) == "You are coordinator."
        assert researcher.system_prompt(0) == "You are researcher."

    async def test_every_agent_is_compiled_on_the_checkpointer(self, created):
        saver = InMemorySaver()
        coordinator = ScriptedChat(script=[ai(calls=[task("researcher", "Find X.")]), ai("Done.")])
        researcher = ScriptedChat(script=[ai("X is 42.")])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            checkpointer=saver,
        )
        await _turn(root)

        assert created["coordinator"]["checkpointer"] is saver
        assert created["researcher"]["checkpointer"] is saver
        assert root.checkpointer is saver
        # The sub-agent's run is on a thread of its own (its instance's), never
        # a namespace under its caller's.
        saved = {(c.config["configurable"]["thread_id"], c.config["configurable"]["checkpoint_ns"])
                 async for c in saver.alist(None)}
        assert saved == {("t", ""), ("instance:researcher-1", "")}

    def test_without_one_a_sub_agent_never_takes_a_callers(self, created):
        _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
        )

        assert created["coordinator"]["checkpointer"] is None
        assert created["researcher"]["checkpointer"] is False


class TestDelegation:
    def test_task_lists_each_sub_agent_by_its_description(self, created):
        coordinator = AgentConfig(
            name="coordinator", prompt="p", sub_agents=["researcher", "writer"]
        )
        writer = AgentConfig(name="writer", prompt="p", description="Writes drafts.")
        models = {name: ScriptedChat() for name in ("coordinator", "researcher", "writer")}

        _tree([coordinator, RESEARCHER, writer], models)

        task_tool = _task_tool(created, "coordinator")
        lines = task_tool.description.splitlines()
        assert "- researcher: Finds facts." in lines
        assert "- writer: Writes drafts." in lines
        assert not any(line.startswith("- general-purpose") for line in lines)
        assert task_tool.args["agent_type"]["enum"] == ["researcher", "writer"]
        assert not any(tool.name == "task" for tool in created["writer"]["tools"])

    async def test_the_coordinator_is_offered_task_and_nothing_of_deep_agents_else(self):
        coordinator = ScriptedChat(script=[ai("Hi.")])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": ScriptedChat()},
            tools={"coordinator": to_langchain_tools([save_plan])},
            checkpointer=InMemorySaver(),
        )
        await _turn(root)

        assert sorted(coordinator.tool_names()) == ["save_plan", "task"]
        schema = coordinator.tool_schema("task")
        assert set(schema["parameters"]["properties"]) == {
            "agent_type", "title", "objective", "deliverable", "constraints", "known",
            "done_when",
        }
        assert schema["parameters"]["required"] == ["agent_type", "title", "objective"]

    def test_an_empty_description_falls_back_and_warns(self, created, monkeypatch):
        log = _Recorder()
        monkeypatch.setattr(brief_module, "logger", log)
        coordinator = AgentConfig(name="coordinator", prompt="p", sub_agents=["helper"])
        helper = AgentConfig(name="helper", prompt="p", description="  ")

        _tree([coordinator, helper], {"coordinator": ScriptedChat(), "helper": ScriptedChat()})

        assert "- helper: The helper agent." in (
            _task_tool(created, "coordinator").description.splitlines()
        )
        assert ("warning", "subagent_without_description", {"agent": "helper"}) in log.calls

    def test_the_task_tool_is_bound_to_its_declaration(self, created):
        _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
        )

        assert identify_tool(_task_tool(created, "coordinator")) is get_registry().get("task")

    def test_without_the_delegations_no_agent_delegates(self, created):
        build_agent_tree(
            [COORDINATOR, RESEARCHER],
            chat_models={"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
            resolved={}, tools={}, prompts={"coordinator": "c", "researcher": "r"},
            settings=None, checkpointer=None,
        )

        assert created["coordinator"]["tools"] == []

    def test_every_agents_graph_is_kept_by_name(self):
        graphs: dict = {}
        root = build_agent_tree(
            [COORDINATOR, RESEARCHER],
            chat_models={"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
            resolved={}, tools={}, prompts={"coordinator": "c", "researcher": "r"},
            settings=None, checkpointer=None, graphs=graphs,
        )

        assert sorted(graphs) == ["coordinator", "researcher"]
        assert graphs["coordinator"] is root

    def test_a_coordinator_tool_named_task_is_refused(self):
        def task(description: str) -> dict:  # noqa: F811 - the point of the test
            """An application's own delegation."""
            return {"success": True}

        with pytest.raises(AgentGraphError, match="'coordinator'.*'task'"):
            _tree(
                [COORDINATOR, RESEARCHER],
                {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
                tools={"coordinator": to_langchain_tools([task])},
            )

    async def test_nested_delegation(self, created):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["lead"], description="c"),
            AgentConfig(name="lead", prompt="p", sub_agents=["worker"], description="Leads work."),
            AgentConfig(name="worker", prompt="p", description="Does work."),
        ]
        models = {
            "coordinator": ScriptedChat(label="coordinator", script=[
                ai(calls=[task("lead", "Lead it.", "c1")]), ai("All done."),
            ]),
            "lead": ScriptedChat(label="lead", script=[
                ai(calls=[task("worker", "Do it.", "l1")]), ai("Lead done."),
            ]),
            "worker": ScriptedChat(label="worker", script=[ai("Worker done.")]),
        }

        root = _tree(configs, models, checkpointer=InMemorySaver())
        state = await _turn(root)

        assert list(created) == ["coordinator", "lead", "worker"]
        assert _read(_tool_message(state["messages"], "c1"))["result"] == "Lead done."
        assert _read(_tool_message(models["lead"].conversation(1), "l1"))["result"] == (
            "Worker done."
        )
        assert _shape(models["worker"].conversation(0)) == [
            ("human", task_message(Brief(title="Do it.", objective="Do it.")))
        ]
        assert state["messages"][-1].text == "All done."


class TestWhatASubAgentSees:
    async def test_only_the_brief_and_the_mark_reach_it_and_its_last_text_comes_back(self):
        coordinator = ScriptedChat(label="coordinator", script=[
            ai(calls=[("save_plan", {"content": "- [ ] find X"})]),
            ai(calls=[task("researcher", "Find X. Report the number.", "c1")]),
            ai("Done."),
        ])
        researcher = ScriptedChat(label="researcher", script=[
            ai("Looking.", calls=[("get_plan", {}, "r1")]),
            ai("X is 42.", calls=[
                ("save_tasks", {"tasks": [{"description": "find X", "status": "completed"}]}),
            ]),
            ai(""),  # an empty last answer: the text before it is the result
        ])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            tools={
                "coordinator": to_langchain_tools([save_plan]),
                "researcher": to_langchain_tools([get_plan, save_tasks]),
            },
            checkpointer=InMemorySaver(),
        )
        state = await _turn(root, "A long conversation the researcher must not see.")

        # Its input is exactly one HumanMessage: the task message.
        assert _shape(researcher.conversation(0)) == [("human", task_message(Brief(
            title="Find X. Report the number.", objective="Find X. Report the number."
        )))]
        # It saw none of the coordinator's plan ...
        plan = json.loads(_tool_message(researcher.conversation(1), "r1").text)
        assert "find X" not in json.dumps(plan)
        # ... and its task list stays its own: the coordinator's state has none.
        assert not state.get("tasks")
        assert state["plan"] == "- [ ] find X"
        # Its last non-empty text is the result in the coordinator's envelope.
        assert _read(_tool_message(state["messages"], "c1"))["result"] == "X is 42."
        # Its intermediate messages stay out of the coordinator's conversation.
        assert not any("Looking." in m.text for m in state["messages"])

    async def test_parallel_sub_agents_each_keep_their_own_tasks(self):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["a", "b"]),
            AgentConfig(name="a", prompt="p", description="Does A."),
            AgentConfig(name="b", prompt="p", description="Does B."),
        ]

        def worker(label: str) -> ScriptedChat:
            return ScriptedChat(label=label, script=[
                ai(calls=[("save_tasks", {"tasks": [{"description": label}]})]),
                ai(f"{label} done."),
            ])

        models = {
            "coordinator": ScriptedChat(label="coordinator", script=[
                ai(calls=[task("a", "Do A.", "ca"), task("b", "Do B.", "cb")]),
                ai("Both done."),
            ]),
            "a": worker("a"),
            "b": worker("b"),
        }
        tools = {name: to_langchain_tools([save_tasks]) for name in ("a", "b")}

        root = _tree(configs, models, tools=tools, checkpointer=InMemorySaver())
        state = await _turn(root)

        assert _read(_tool_message(state["messages"], "ca"))["result"] == "a done."
        assert _read(_tool_message(state["messages"], "cb"))["result"] == "b done."
        assert not state.get("tasks")
        assert state["messages"][-1].text == "Both done."


class TestEveryStack:
    def test_the_gate_the_error_and_the_active_call_middleware_in_order(self, created):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["lead"]),
            AgentConfig(name="lead", prompt="p", sub_agents=["worker"], description="l"),
            AgentConfig(name="worker", prompt="p", description="w"),
        ]

        _tree(configs, {config.name: ScriptedChat() for config in configs})

        assert [type(m) for m in created["coordinator"]["middleware"]] == [
            PatchToolCallsMiddleware, PermissionMiddleware, ToolCallErrorMiddleware,
            ActiveToolCallMiddleware,
        ]
        # A sub-agent's model side has SubAgentOutcomeMiddleware and, given
        # the delegations, RunControlMiddleware; the root's neither.
        assert [type(m) for m in created["lead"]["middleware"]] == [
            PatchToolCallsMiddleware, SubAgentOutcomeMiddleware, RunControlMiddleware,
            PermissionMiddleware, ToolCallErrorMiddleware, ActiveToolCallMiddleware,
        ]
        assert [type(m) for m in created["worker"]["middleware"]] == [
            PatchToolCallsMiddleware, SubAgentOutcomeMiddleware, RunControlMiddleware,
            PermissionMiddleware, ToolCallErrorMiddleware, ActiveToolCallMiddleware,
        ]
        for name, kwargs in created.items():
            (gate,) = [m for m in kwargs["middleware"] if isinstance(m, PermissionMiddleware)]
            assert gate.agent_name == name
            assert kwargs["state_schema"] is AgentCliState

    async def test_every_gate_resolves_bound_objects_with_the_given_resolver(
        self, monkeypatch
    ):
        """A manager's own bindings (its MCP tools) reach every agent's gate."""

        async def remote_op(q: str) -> str:
            return "remote"

        remote = StructuredTool.from_function(coroutine=remote_op, description="d")
        mcp = ToolDefinition(name="remote_op", description="d", func=None,
                             capabilities=[Capability("mcp", target="remote_op")])
        asked: list[object] = []

        def bound(obj: object) -> ToolDefinition | None:
            asked.append(obj)
            return mcp if obj is remote else identify_tool(obj)

        researcher = ScriptedChat(label="researcher", script=[
            ai(calls=[("remote_op", {"q": "x"}, "r1")]), ai("Remote."),
        ])
        coordinator = ScriptedChat(label="coordinator", script=[
            ai(calls=[task("researcher", "Ask.", "c1")]), ai("Done."),
        ])
        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            tools={"researcher": [remote]},
            checkpointer=InMemorySaver(),
            bound=bound,
        )
        # No engine and permissions off: a resolved tool runs, an unresolved
        # one is still denied.
        monkeypatch.setattr(
            "agentic_cli.config.get_settings",
            lambda: SimpleNamespace(permissions_enabled=False),
        )

        await _turn(root)

        assert _tool_message(researcher.conversation(1), "r1").text == "remote"
        assert remote in asked  # the sub-agent's gate asked it
        assert any(identify_tool(obj) is get_registry().get("task") for obj in asked)

    def test_by_default_the_gate_asks_the_framework_registry(self, created):
        _tree([RESEARCHER], {"researcher": ScriptedChat()})

        (gate,) = [m for m in created["researcher"]["middleware"]
                   if isinstance(m, PermissionMiddleware)]
        assert gate._bound is identify_tool

    async def test_a_sub_agent_is_gated_and_survives_a_raising_tool(self):
        def scratch(path: str) -> dict:
            """Not registered: the gate denies it."""
            raise AssertionError("an unregistered tool ran")

        researcher = ScriptedChat(label="researcher", script=[
            ai(calls=[("scratch", {"path": "/x"}, "r1"), ("builder_flaky", {"x": "1"}, "r2")]),
            ai("Could not."),
        ])
        coordinator = ScriptedChat(label="coordinator", script=[
            ai(calls=[task("researcher", "Try.", "c1")]), ai("Done."),
        ])

        root = _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": coordinator, "researcher": researcher},
            tools={"researcher": to_langchain_tools([scratch, builder_flaky])},
            checkpointer=InMemorySaver(),
        )
        state = await _turn(root)

        denied = _tool_message(researcher.conversation(1), "r1")
        assert denied.status == "error"
        assert json.loads(denied.text)["error"].startswith(
            "Permission denied: tool is not registered"
        )
        raised = _tool_message(researcher.conversation(1), "r2")
        assert json.loads(raised.text) == {
            "success": False,
            "error": "builder_flaky failed: RuntimeError: flaky broke",
        }
        assert _read(_tool_message(state["messages"], "c1"))["result"] == "Could not."


class TestTheParentsTools:
    """With the policy's ``follow_ups`` a delegating agent also gets
    ``message_agent``, ``stop_agent`` and ``check_agents``, each bound to its
    declaration; by default ``task`` alone."""

    NAMES = ["task", "message_agent", "stop_agent", "check_agents"]

    def test_by_default_task_alone(self, created):
        _tree([COORDINATOR, RESEARCHER],
              {"coordinator": ScriptedChat(), "researcher": ScriptedChat()})

        assert [tool.name for tool in created["coordinator"]["tools"]] == ["task"]

    def test_with_follow_ups_every_delegating_agent_gets_them(self, created):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["lead"], description="c"),
            AgentConfig(name="lead", prompt="p", sub_agents=["worker"], description="Leads."),
            AgentConfig(name="worker", prompt="p", description="Works."),
        ]
        _tree(configs, {name: ScriptedChat() for name in ("coordinator", "lead", "worker")},
              policy=SubAgentPolicy(follow_ups=True))

        assert [tool.name for tool in created["coordinator"]["tools"]] == self.NAMES
        assert [tool.name for tool in created["lead"]["tools"]] == self.NAMES
        assert created["worker"]["tools"] == []
        for tool in created["coordinator"]["tools"][1:]:
            assert identify_tool(tool) is get_registry().get(tool.name)
            assert (tool.description, tool.args_schema) == (
                follow_up_description(tool.name), follow_up_parameters(tool.name)
            )

    def test_an_own_tool_of_one_of_their_names_is_refused(self):
        def stop_agent(agent: str) -> dict:
            """An application's own."""
            return {"success": True}

        with pytest.raises(AgentGraphError) as exc:
            _tree(
                [COORDINATOR, RESEARCHER],
                {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
                tools={"coordinator": to_langchain_tools([stop_agent])},
                policy=SubAgentPolicy(follow_ups=True),
            )

        assert str(exc.value) == (
            "Agent 'coordinator' continues its sub-agents through tools named message_agent, "
            "stop_agent and check_agents, so its own tools cannot include 'stop_agent'; "
            "rename that tool."
        )

    def test_the_error_names_the_first_clash_in_the_order_of_their_names(self):
        """On both backends, whatever the order of the agent's own tools."""
        def check_agents(agent: str = "") -> dict:
            """An application's own."""
            return {"success": True}

        def message_agent(agent: str = "") -> dict:
            """An application's own."""
            return {"success": True}

        with pytest.raises(AgentGraphError, match="cannot include 'message_agent'"):
            _tree(
                [COORDINATOR, RESEARCHER],
                {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
                tools={"coordinator": to_langchain_tools([check_agents, message_agent])},
                policy=SubAgentPolicy(follow_ups=True),
            )

    def test_without_follow_ups_an_own_tool_of_one_of_their_names_is_its_own(self, created):
        def stop_agent(agent: str) -> dict:
            """An application's own."""
            return {"success": True}

        _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
            tools={"coordinator": to_langchain_tools([stop_agent])},
        )

        assert [tool.name for tool in created["coordinator"]["tools"]] == ["stop_agent", "task"]

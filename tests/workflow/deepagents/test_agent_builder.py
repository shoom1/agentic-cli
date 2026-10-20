"""The agent tree (``workflow/deepagents/agent_builder.py``).

Every ``AgentConfig`` becomes one ``create_agent`` graph on its own chat
model, built bottom-up so a coordinator is built after its sub-agents. A
coordinator delegates through Deep Agents' ``task`` tool, which lists each
sub-agent by its description and is bound to the framework's declaration. A
sub-agent sees only the brief, shares ``plan`` and ``tasks``, and answers with
its last text. Only the root has the checkpointer. Every stack has the
permission gate and, inside it, the tool-error middleware.

The models are scripted (``fakes.ScriptedChat``); nothing reaches a provider.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from deepagents import SubAgentMiddleware  # noqa: E402
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
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.deepagents import agent_builder  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    PermissionMiddleware,
    ToolCallErrorMiddleware,
)
from agentic_cli.workflow.deepagents.state import AgentCliState  # noqa: E402
from agentic_cli.workflow.deepagents.tools import to_langchain_tools  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.capabilities import Capability  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ai, task  # noqa: E402


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


def _tree(configs, models, tools=None, checkpointer=None):
    return build_agent_tree(
        configs,
        chat_models=models,
        resolved={},
        tools=tools or {},
        prompts={config.name: f"You are {config.name}." for config in configs},
        settings=None,
        checkpointer=checkpointer,
    )


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

        assert list(created) == ["researcher", "coordinator"]  # bottom-up
        assert [type(m) for m in state["messages"]] == [
            HumanMessage, AIMessage, ToolMessage, AIMessage
        ]
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

    async def test_a_single_agent_is_its_own_root(self, created):
        solo = ScriptedChat(script=[ai("Hi.")])
        saver = InMemorySaver()

        root = _tree([AgentConfig(name="solo", prompt="p")], {"solo": solo}, checkpointer=saver)
        state = await _turn(root)

        assert state["messages"][-1].text == "Hi."
        assert root.checkpointer is saver
        assert [type(m) for m in created["solo"]["middleware"]] == [
            PatchToolCallsMiddleware, PermissionMiddleware, ToolCallErrorMiddleware
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

    async def test_only_the_root_has_the_checkpointer(self, created):
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
        assert created["researcher"]["checkpointer"] is None
        assert root.checkpointer is saver
        # The sub-agent inherits it: its checkpoints sit in the same thread,
        # under its task's namespace.
        namespaces = {c.config["configurable"]["checkpoint_ns"] async for c in saver.alist(None)}
        assert "" in namespaces
        assert any(ns.startswith("tools:") for ns in namespaces)


class TestDelegation:
    def test_task_lists_each_sub_agent_by_its_description(self, created):
        coordinator = AgentConfig(
            name="coordinator", prompt="p", sub_agents=["researcher", "writer"]
        )
        writer = AgentConfig(name="writer", prompt="p", description="Writes drafts.")
        models = {name: ScriptedChat() for name in ("coordinator", "researcher", "writer")}

        _tree([coordinator, RESEARCHER, writer], models)

        (delegation,) = [
            m for m in created["coordinator"]["middleware"] if isinstance(m, SubAgentMiddleware)
        ]
        (task_tool,) = delegation.tools
        lines = task_tool.description.splitlines()
        assert "- researcher: Finds facts." in lines
        assert "- writer: Writes drafts." in lines
        assert not any(line.startswith("- general-purpose") for line in lines)
        assert delegation.subagent_names == {"researcher", "writer"}

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
        assert set(schema["parameters"]["properties"]) == {"description", "subagent_type"}

    def test_an_empty_description_falls_back_and_warns(self, created, monkeypatch):
        log = _Recorder()
        monkeypatch.setattr(agent_builder, "logger", log)
        coordinator = AgentConfig(name="coordinator", prompt="p", sub_agents=["helper"])
        helper = AgentConfig(name="helper", prompt="p", description="  ")

        _tree([coordinator, helper], {"coordinator": ScriptedChat(), "helper": ScriptedChat()})

        (delegation,) = [
            m for m in created["coordinator"]["middleware"] if isinstance(m, SubAgentMiddleware)
        ]
        assert "- helper: The helper agent." in delegation.tools[0].description.splitlines()
        assert (
            "warning", "deepagents_subagent_without_description", {"agent": "helper"}
        ) in log.calls

    def test_the_task_tool_is_bound_to_its_declaration(self, created):
        _tree(
            [COORDINATOR, RESEARCHER],
            {"coordinator": ScriptedChat(), "researcher": ScriptedChat()},
        )

        (delegation,) = [
            m for m in created["coordinator"]["middleware"] if isinstance(m, SubAgentMiddleware)
        ]
        assert identify_tool(delegation.tools[0]) is get_registry().get("task")

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

        assert list(created) == ["worker", "lead", "coordinator"]
        assert _tool_message(state["messages"], "c1").text == "Lead done."
        assert _tool_message(models["lead"].conversation(1), "l1").text == "Worker done."
        assert _shape(models["worker"].conversation(0)) == [("human", "Do it.")]
        assert state["messages"][-1].text == "All done."


class TestWhatASubAgentSees:
    async def test_only_the_brief_shared_state_and_its_last_text(self):
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

        # Its input is exactly one HumanMessage: the brief.
        assert _shape(researcher.conversation(0)) == [("human", "Find X. Report the number.")]
        # It saw the coordinator's plan ...
        plan = json.loads(_tool_message(researcher.conversation(1), "r1").text)
        assert plan == {"success": True, "content": "- [ ] find X"}
        # ... and its task list flowed back to the coordinator.
        assert [t["description"] for t in state["tasks"]] == ["find X"]
        assert state["plan"] == "- [ ] find X"
        # Its last non-empty text is the coordinator's tool result.
        assert _tool_message(state["messages"], "c1").text == "X is 42."
        # Its intermediate messages stay out of the coordinator's conversation.
        assert not any("Looking." in m.text for m in state["messages"])

    async def test_parallel_sub_agents_may_both_update_tasks(self):
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

        assert _tool_message(state["messages"], "ca").text == "a done."
        assert _tool_message(state["messages"], "cb").text == "b done."
        assert [t["description"] for t in state["tasks"]] in (["a"], ["b"])
        assert state["messages"][-1].text == "Both done."


class TestEveryStack:
    def test_the_gate_and_the_error_middleware_in_order(self, created):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["lead"]),
            AgentConfig(name="lead", prompt="p", sub_agents=["worker"], description="l"),
            AgentConfig(name="worker", prompt="p", description="w"),
        ]

        _tree(configs, {config.name: ScriptedChat() for config in configs})

        delegating = [
            PatchToolCallsMiddleware, SubAgentMiddleware,
            PermissionMiddleware, ToolCallErrorMiddleware,
        ]
        assert [type(m) for m in created["coordinator"]["middleware"]] == delegating
        assert [type(m) for m in created["lead"]["middleware"]] == delegating
        assert [type(m) for m in created["worker"]["middleware"]] == [
            PatchToolCallsMiddleware, PermissionMiddleware, ToolCallErrorMiddleware
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
        root = build_agent_tree(
            [COORDINATOR, RESEARCHER],
            chat_models={"coordinator": coordinator, "researcher": researcher},
            resolved={},
            tools={"researcher": [remote]},
            prompts={"coordinator": "c", "researcher": "r"},
            settings=None,
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
        assert _tool_message(state["messages"], "c1").text == "Could not."

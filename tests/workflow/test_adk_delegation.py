"""ADK delegates to a sub-agent and gets its result back, in the same turn.

Every agent built from an ``AgentConfig`` is built on its own: no ADK
``sub_agents``, so none is offered ``transfer_to_agent``. A delegating agent
is offered one tool, the framework's ``task`` (``adk/task_tool.py``): its
call carries a brief (``agent_type``, ``title``, ``objective``, ...), the
sub-agent runs on a runner of its own, on a child session of the
conversation's (``<app>--agents``, ``<conversation>/<instance>``), its first
message the task message (``agents/brief.py::task_message``), and the call's
result is the run's envelope, as JSON text (``agents/envelope.py``). The root
then answers the user, and the next turn starts at the root again. A
sub-agent with ``inherit_context`` False (the default) sees only its task
message; one with ``inherit_context=True`` sees the conversation first, and
its task message ends with ``INHERITED_CONTEXT_NOTE``.

The framework builds the ``task`` tool, so it binds it to the exempt
delegation declaration (``tools/_core/delegation.py``) as it is made: an
application's tool of that name, or a look-alike, is denied. A delegating
agent's own tool named ``task`` fails the build. The sub-agent's own tool
calls go through the same gate.

A sub-agent's long-running tool is an ordinary call there: only the root's
are wrapped. A job a sub-agent started is told to the root when it
finishes, in the notice Deep Agents sends. A sub-agent without a description
is offered as "The <name> agent.", as on Deep Agents.

A native ADK YAML app (``adk_config_path``) is built by ADK and keeps
transferring (``tests/workflow/test_adk_transfer_tool.py``).

Turns run the real manager on ``FakeAPI`` (Claude); what the ``task`` tool
relies on of ADK's ``BaseTool`` has contract tests at the end.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

pytest.importorskip("google.adk")

from anthropic import AsyncAnthropic  # noqa: E402
from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.events import Event  # noqa: E402
from google.adk.models.anthropic_llm import AnthropicLlm  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import BaseTool, LongRunningFunctionTool  # noqa: E402
from google.adk.tools.agent_tool import AgentTool  # noqa: E402
from google.adk.tools.skill_toolset import SkillToolset  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools._core.delegation import TASK_TOOL  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.tools.jobs.resume import (  # noqa: E402
    JOB_NOTICE_ASK,
    JOB_NOTICE_HEADER,
    job_result_payload,
)
from agentic_cli.tools.registry import (  # noqa: E402
    ToolCategory,
    get_registry,
    identify_tool,
    register_tool,
)
from agentic_cli.tools.skills import SkillStore  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.permission_plugin import _UNVERIFIED, PermissionPlugin  # noqa: E402
from agentic_cli.workflow.adk.sessions import child_app  # noqa: E402
from agentic_cli.workflow.adk.task_tool import FollowUpTool, TaskTool  # noqa: E402
from agentic_cli.workflow.agents import SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.agents import brief as brief_module  # noqa: E402
from agentic_cli.workflow.agents.brief import (  # noqa: E402
    INHERITED_CONTEXT_NOTE,
    Brief,
    follow_up_description,
    follow_up_parameters,
    task_message,
)
from agentic_cli.workflow.agents.envelope import read_envelope  # noqa: E402
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.context import context_budget  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.prompt import DENY_CHOICE  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, http_library  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402
from tests.workflow.test_skills import _make_skill  # noqa: E402

BRIEF = "Find X. Report the number."
TURNS = ["My name is Ada and the secret code is 7.", "Please have the researcher find X.",
         "Thanks. What next?"]


def says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def calls(name: str, call_id: str, **args) -> dict:
    message = anthropic_message({"type": "tool_use", "id": call_id, "name": name, "input": args})
    message["stop_reason"] = "tool_use"
    return message


def delegation(call_id: str, objective: str = BRIEF, *, agent: str = "researcher",
               title: str = "Find X", **fields) -> dict:
    """A ``task`` call delegating ``objective`` to ``agent``."""
    return calls(TASK_TOOL, call_id, agent_type=agent, title=title, objective=objective, **fields)


def message_of(objective: str = BRIEF, *, title: str = "Find X", inherit: bool = False) -> str:
    """The task message a delegation of ``objective`` starts its run with."""
    return task_message(Brief(title=title, objective=objective), inherit=inherit)


def result_of(line: str) -> dict:
    """The envelope a ``lines`` result line holds: ``user: result "<JSON>"``."""
    return json.loads(json.loads(line.removeprefix("user: result ")))


#: The call id each start of the probe job was given.
STARTED: list[str] = []


@register_tool(category=ToolCategory.EXECUTION, capabilities=EXEMPT, long_running=True,
               description="Start a background job (an exempt test tool).")
def delegation_probe_start_job(command: str = "", tool_context=None) -> dict:
    """Start a job and return its id at once, as a job starter does."""
    STARTED.append(getattr(tool_context, "function_call_id", None))
    return {"success": True, "job_id": "j1", "state": "running"}


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


#: Each agent's system prompt begins with its prompt.
PROMPTS = {"coordinator": "You coordinate.", "researcher": "You research.",
           "middle": "You plan.", "leaf": "You look up."}


def system_text(body: dict) -> str:
    system = body.get("system") or ""
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system)
    return system


def tool_names(body: dict) -> list[str]:
    return [tool["name"] for tool in body.get("tools", [])]


def lines(body: dict) -> list[str]:
    """Each message of a Claude request as ``"<role>: <block>"``, block by block."""
    out: list[str] = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(f"{message['role']}: {content}")
            continue
        for block in content:
            if block["type"] == "text":
                out.append(f"{message['role']}: {block['text']}")
            elif block["type"] == "tool_use":
                out.append(f"{message['role']}: called {block['name']}({json.dumps(block['input'])})")
            elif block["type"] == "tool_result":
                out.append(f"{message['role']}: result {json.dumps(block['content'])}")
    return out


class Team:
    """Claude for a team: each agent answers from its own script, told apart by
    its system prompt; every request is recorded under its agent. A script's
    entry is an answer, or a function of the request's body that gives one."""

    def __init__(self, **scripts: list[dict]) -> None:
        self.scripts = {name: list(script) for name, script in scripts.items()}
        self.sent: list[tuple[str, dict]] = []

    def __call__(self, request) -> dict:
        body = json.loads(request.content)
        system = system_text(body)
        agent = next(name for name, prompt in PROMPTS.items() if system.startswith(prompt))
        self.sent.append((agent, body))
        script = self.scripts.get(agent) or []
        if not script:
            return says(f"[{agent} has no scripted answer]")
        answer = script.pop(0)
        return answer(body) if callable(answer) else answer

    @property
    def agents(self) -> list[str]:
        return [agent for agent, _ in self.sent]

    def bodies(self, agent: str) -> list[dict]:
        return [body for name, body in self.sent if name == agent]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic project (the cwd) and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path / "project"


def team(inherit_context: bool = False, *, researcher_tools=(), **coordinator) -> list[AgentConfig]:
    return [
        AgentConfig(
            name="coordinator", prompt=PROMPTS["coordinator"], sub_agents=["researcher"],
            description="Leads.", include_state_tools=False, **coordinator,
        ),
        AgentConfig(
            name="researcher", prompt=PROMPTS["researcher"], description="Finds facts.",
            tools=list(researcher_tools), include_state_tools=False, inherit_context=inherit_context,
        ),
    ]


def manager_for(configs: list[AgentConfig]) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=configs,
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


def delegating() -> Team:
    """The coordinator delegates on the second turn, and answers the third."""
    return Team(
        coordinator=[
            says("Noted, Ada."),
            delegation("toolu_c1"),
            says("The researcher says X is 42."),
            says("The coordinator answers turn 3."),
        ],
        researcher=[says("X is 42."), says("The researcher answers turn 3.")],
    )


async def three_turns(monkeypatch, configs: list[AgentConfig], answers: Team) -> list[list]:
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
    manager = manager_for(configs)
    try:
        return [await turn(manager, text) for text in TURNS]
    finally:
        await manager.cleanup()


class TestDelegation:
    async def test_the_root_is_offered_task_and_no_transfer(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(), answers)

        for body in answers.bodies("coordinator"):
            assert tool_names(body) == [TASK_TOOL]
        (declaration,) = answers.bodies("coordinator")[0]["tools"]
        schema = declaration["input_schema"]
        assert schema["required"] == ["agent_type", "title", "objective"]
        assert schema["properties"]["agent_type"]["enum"] == ["researcher"]

    async def test_a_sub_agent_is_offered_no_transfer(self, project, monkeypatch):
        answers = delegating()
        await three_turns(
            monkeypatch, team(researcher_tools=["read_file"]), answers
        )

        (body,) = answers.bodies("researcher")
        assert tool_names(body) == ["read_file"]

    async def test_the_sub_agent_answers_its_parent_in_the_same_turn(self, project, monkeypatch):
        answers = delegating()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            await turn(manager, TURNS[0])
            events = await turn(manager, TURNS[1])
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "coordinator", "researcher", "coordinator"]
        called, result = lines(answers.bodies("coordinator")[2])[-2:]
        assert called == "assistant: called task({})".format(json.dumps(
            {"agent_type": "researcher", "title": "Find X", "objective": BRIEF}
        ))
        assert result_of(result) == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#1",
            "status": "completed", "partial": False, "result": "X is 42.", "error": None,
        }
        (response,) = [
            part.function_response
            for event in session.events
            for part in (event.content.parts if event.content else [])
            if part.function_response
        ]
        assert response.name == TASK_TOOL
        assert json.loads(response.response["result"])["result"] == "X is 42."
        assert texts(events)[-1] == "The researcher says X is 42."

    async def test_the_next_turn_starts_at_the_root(self, project, monkeypatch):
        answers = delegating()
        events = await three_turns(monkeypatch, team(), answers)

        assert answers.agents[-1] == "coordinator"
        assert texts(events[2]) == ["The coordinator answers turn 3."]
        assert len(answers.bodies("researcher")) == 1

    async def test_a_fresh_sub_agent_sees_only_its_task_message(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(inherit_context=False), answers)

        (body,) = answers.bodies("researcher")
        assert lines(body) == [f"user: {message_of()}"]

    async def test_the_default_is_fresh(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(), answers)

        (body,) = answers.bodies("researcher")
        assert lines(body) == [f"user: {message_of()}"]

    async def test_an_inheriting_sub_agent_sees_the_conversation(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(inherit_context=True), answers)

        (body,) = answers.bodies("researcher")
        sent = lines(body)
        assert sent[0] == f"user: {TURNS[0]}"
        assert f"user: {TURNS[1]}" in sent
        assert any("Noted, Ada." in line for line in sent)  # the coordinator's reply, quoted
        assert sent[-1] == f"user: {message_of(inherit=True)}"
        assert "assistant" not in {line.partition(":")[0] for line in sent}

    async def test_an_inheriting_sub_agents_task_message_says_whose_conversation_it_reads(
        self, project, monkeypatch
    ):
        # The user's turns reach it as plain user messages, so without the
        # note it may take them as its own instructions and answer as the
        # coordinator. The note ends the task message; the prompt is its own.
        answers = delegating()
        await three_turns(monkeypatch, team(inherit_context=True), answers)

        (body,) = answers.bodies("researcher")
        assert lines(body)[-1].endswith(INHERITED_CONTEXT_NOTE)
        assert system_text(body).startswith(PROMPTS["researcher"])
        assert INHERITED_CONTEXT_NOTE not in system_text(body)
        for body in answers.bodies("coordinator"):
            assert INHERITED_CONTEXT_NOTE not in system_text(body)

    async def test_adk_warns_of_the_inherited_events_authors_as_the_cli_filter_expects(
        self, project, monkeypatch, caplog
    ):
        # The copied events' authors are not in the run's tree, so ADK warns
        # of each as it picks the agent to run: the CLI logs those warnings
        # at debug (adk/log_filters.py), by this logger and this text.
        from agentic_cli.workflow.adk.log_filters import _ADK_UNKNOWN_AUTHOR

        logger_name, prefix = _ADK_UNKNOWN_AUTHOR
        # ADK's own warning, as it comes: without the filter a test that ran
        # the CLI (BaseCLIApp.run) may have left on that logger.
        monkeypatch.setattr(logging.getLogger(logger_name), "filters", [])
        with caplog.at_level(logging.WARNING, logger=logger_name):
            await three_turns(monkeypatch, team(inherit_context=True), delegating())

        warned = [r.getMessage() for r in caplog.records if r.name == logger_name]
        assert warned and all(message.startswith(prefix) for message in warned)

    async def test_a_fresh_sub_agent_is_given_no_note(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(inherit_context=False), answers)

        (body,) = answers.bodies("researcher")
        assert system_text(body).startswith(PROMPTS["researcher"])
        assert INHERITED_CONTEXT_NOTE not in json.dumps(body)

    async def test_the_sub_agents_own_tool_calls_are_gated(self, project, monkeypatch):
        answers = Team(
            coordinator=[delegation("toolu_c1", "Write a note."), says("It could not.")],
            researcher=[calls("write_file", "toolu_r1", path="note.txt", content="hi"),
                        says("Writing was denied.")],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=["write_file"]))
        asked: list[str] = []

        async def approver(request):
            asked.append(request.tool_name)
            return DENY_CHOICE

        manager.set_input_callback(approver)
        try:
            await turn(manager, "Have a note written.")
        finally:
            await manager.cleanup()

        assert asked == ["write_file"]
        assert not (project / "note.txt").exists()
        result = lines(answers.bodies("researcher")[1])[-1]
        assert "Permission denied" in result

    async def test_a_middle_coordinator_delegates_in_turn(self, project, monkeypatch):
        answers = Team(
            coordinator=[delegation("toolu_c1", "Plan X.", agent="middle"), says("Planned: 42.")],
            middle=[delegation("toolu_m1", "Look X up.", agent="leaf"),
                    says("X is 42, says the leaf.")],
            leaf=[says("42.")],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for([
            AgentConfig(name="coordinator", prompt=PROMPTS["coordinator"],
                        sub_agents=["middle"], include_state_tools=False),
            AgentConfig(name="middle", prompt=PROMPTS["middle"], sub_agents=["leaf"],
                        description="Plans.", include_state_tools=False),
            AgentConfig(name="leaf", prompt=PROMPTS["leaf"], description="Looks up.",
                        include_state_tools=False),
        ])
        try:
            events = await turn(manager, "What is X?")
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "middle", "leaf", "middle", "coordinator"]
        assert tool_names(answers.bodies("middle")[0]) == [TASK_TOOL]
        # Two levels down, each sub-agent is sent its own task message alone.
        assert lines(answers.bodies("middle")[0]) == [f"user: {message_of('Plan X.')}"]
        assert lines(answers.bodies("leaf")[0]) == [f"user: {message_of('Look X up.')}"]
        assert result_of(lines(answers.bodies("middle")[1])[-1])["result"] == "42."
        assert result_of(lines(answers.bodies("coordinator")[1])[-1])["result"] == (
            "X is 42, says the leaf."
        )
        assert texts(events)[-1] == "Planned: 42."

    async def test_two_delegations_in_one_response(self, project, monkeypatch):
        both = anthropic_message(
            {"type": "tool_use", "id": "toolu_a", "name": TASK_TOOL,
             "input": {"agent_type": "researcher", "title": "X", "objective": "X?"}},
            {"type": "tool_use", "id": "toolu_b", "name": TASK_TOOL,
             "input": {"agent_type": "researcher", "title": "Z", "objective": "Z?"}},
        )
        both["stop_reason"] = "tool_use"

        def researcher(body: dict) -> dict:
            # Which of the two parallel runs asks first is a race: each run
            # answers its own task.
            return says("X is 42." if "X?" in lines(body)[-1] else "Z is 7.")

        answers = Team(coordinator=[both, says("Both.")], researcher=[researcher, researcher])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            events = await turn(manager, "Both, please.")
        finally:
            await manager.cleanup()

        # Each run sees only its own task message; both results come back.
        assert sorted(lines(b)[-1] for b in answers.bodies("researcher")) == [
            f"user: {message_of('X?', title='X')}", f"user: {message_of('Z?', title='Z')}",
        ]
        read = {
            block["tool_use_id"]: read_envelope(
                block["content"] if isinstance(block["content"], str)
                else "".join(part.get("text", "") for part in block["content"])
            )
            for message in answers.bodies("coordinator")[1]["messages"]
            if isinstance(message["content"], list)
            for block in message["content"] if block["type"] == "tool_result"
        }
        # Each call reads its own run's answer; the two runs are two instances.
        assert {call: r["result"] for call, r in read.items()} == {
            "toolu_a": "X is 42.", "toolu_b": "Z is 7.",
        }
        assert sorted(r["agent"] for r in read.values()) == ["researcher-1", "researcher-2"]
        assert texts(events)[-1] == "Both."

    async def test_a_sub_agents_long_answer_is_capped(self, project, monkeypatch):
        answers = Team(
            coordinator=[delegation("toolu_c1"), says("Done.")],
            researcher=[says("word " * 30_000)],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            await turn(manager, "Find X.")
        finally:
            await manager.cleanup()

        result = lines(answers.bodies("coordinator")[1])[-1]
        assert len(result) < 60_000
        # The envelope is cut as a dict: its JSON still reads, the note in it.
        (held,) = [
            block["content"]
            for message in answers.bodies("coordinator")[1]["messages"]
            if isinstance(message["content"], list)
            for block in message["content"] if block["type"] == "tool_result"
        ]
        if isinstance(held, list):
            held = "".join(part.get("text", "") for part in held)
        read = read_envelope(held)
        assert read is not None, held[:300]
        assert (read["status"], read["agent"], read["success"]) == ("completed", "researcher-1", True)
        assert read["truncated"].startswith("Cut: ") and "tool-output/task-" in read["truncated"]
        (saved,) = (project / ".agentic_cli" / "tool-output").glob("task-*.txt")
        assert saved.read_text().count("word") == 30_000  # all of it, in lines to page

    async def test_an_inheriting_sub_agent_is_sent_none_of_the_coordinators_thinking(
        self, project, monkeypatch
    ):
        thought = "The user is Ada; remember the code."
        answers = Team(
            coordinator=[
                anthropic_message(
                    {"type": "thinking", "thinking": thought, "signature": "sig-c1"},
                    {"type": "text", "text": "Noted, Ada."},
                ),
                delegation("toolu_c1"),
                says("The researcher says X is 42."),
            ],
            researcher=[says("X is 42.")],
        )
        await three_turns(monkeypatch, team(inherit_context=True), answers)

        # The coordinator is sent its own signed thinking back ...
        assert thought in json.dumps(answers.bodies("coordinator")[1])
        # ... while the inheriting researcher sees its reply, not its thoughts.
        (body,) = answers.bodies("researcher")
        assert any("Noted, Ada." in line for line in lines(body))
        assert thought not in json.dumps(body)
        assert "sig-c1" not in json.dumps(body)

    async def test_each_run_is_kept_in_a_child_session(self, project, monkeypatch):
        """The run's transcript: its own app's session, under the conversation's id."""
        answers = delegating()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            await turn(manager, TURNS[0])
            await turn(manager, TURNS[1])
            service = manager._session_service
            child = await service.get_session(
                app_name=child_app(manager.app_name), user_id="u", session_id="s/researcher-1"
            )
            listed = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert [e.author for e in child.events] == ["user", "researcher"]
        assert child.events[0].content.parts[0].text == message_of()
        assert [row.session_id for row in listed] == ["s"]

    async def test_a_native_adk_app_keeps_transferring(self, project, tmp_path):
        (tmp_path / "helper.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: Help.
        """))
        (tmp_path / "root_agent.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: root
            model: claude-sonnet-4-6
            instruction: Route.
            sub_agents:
              - config_path: helper.yaml
        """))
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
            adk_config_path=str(tmp_path / "root_agent.yaml"),
        )
        await manager.initialize_services()
        try:
            root = manager._root_agent
            assert root.sub_agents[0].mode == "chat"
            assert not any(isinstance(tool, (AgentTool, TaskTool)) for tool in root.tools)
            assert manager._agent_runners == {}
        finally:
            await manager.cleanup()


class TestThePromptsAreSentAsWritten:
    """ADK fills ``{name}`` in a string instruction from the session state and
    raises ``KeyError`` for a name the state lacks, so a prompt quoting code
    (``f'{value}'``) failed every turn before any request, and one naming
    ``{plan}`` had the session's plan spliced in once the state held one. A
    config-built agent's prompt, the root's and a sub-agent's, is sent as
    written, as Deep Agents sends it."""

    BRACES = " An f-string looks like f'{value}'; the plan is {plan}; JSON is {\"a\": 1}."

    async def test_braces_reach_every_agent_as_written(self, project, monkeypatch):
        configs = [
            AgentConfig(
                name="coordinator", prompt=PROMPTS["coordinator"] + self.BRACES,
                sub_agents=["researcher"], description="Leads.", include_state_tools=False,
            ),
            AgentConfig(
                name="researcher", prompt=PROMPTS["researcher"] + self.BRACES,
                description="Finds facts.", include_state_tools=False,
            ),
        ]
        answers = delegating()

        await three_turns(monkeypatch, configs, answers)

        assert answers.agents.count("researcher") == 1  # delegated once, on turn 2
        for agent in ("coordinator", "researcher"):
            for body in answers.bodies(agent):
                assert system_text(body).startswith(PROMPTS[agent] + self.BRACES)

    async def test_an_empty_prompt_sends_no_system_instruction(self, project):
        """As before: ADK adds an instruction only when there is one (Gemini
        would be sent an empty system text)."""
        agent = await built([AgentConfig(name="solo", prompt="")])

        assert agent.instruction == ""


class TestTheStream:
    """Only the root replies: a sub-agent's text is activity, and its calls'
    prompts are its brief, not the conversation's context."""

    async def test_a_sub_agents_text_and_usage_are_marked(self, project, monkeypatch):
        answers = delegating()
        events = await three_turns(monkeypatch, team(), answers)

        delegated = events[1]
        assert [(e.content, e.metadata.get("agent"), e.metadata.get("subagent"))
                for e in delegated if e.type is EventType.TEXT] == [
            ("X is 42.", "researcher", True),
            ("The researcher says X is 42.", None, None),
        ]
        assert [e.metadata.get("purpose") for e in delegated
                if e.type is EventType.LLM_USAGE] == [None, "subagent", None]

    async def test_fact_extraction_reads_only_the_user_and_the_main_agent(
        self, project, monkeypatch
    ):
        answers = delegating()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [
            {"role": "user", "content": TURNS[0]},
            {"role": "assistant", "content": "Noted, Ada."},
            {"role": "user", "content": TURNS[1]},
            {"role": "assistant", "content": "The researcher says X is 42."},
        ]

    async def test_the_processor_knows_the_root(self, project):
        manager = manager_for(team())
        await manager.initialize_services()
        try:
            assert manager._event_processor.root == "coordinator"
        finally:
            await manager.cleanup()

    async def test_a_native_apps_are_its_root_and_the_agents_it_transfers_to(
        self, project, tmp_path
    ):
        """A native ADK YAML app's chat-mode sub-agent takes the conversation
        and replies; a single-turn one is a tool, whose text is its work."""
        (tmp_path / "helper.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: Help.
        """))
        (tmp_path / "root_agent.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: root
            model: claude-sonnet-4-6
            instruction: Route.
            sub_agents:
              - config_path: helper.yaml
              - code: tests.workflow.adk.native_agents.solver
        """))
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
            adk_config_path=str(tmp_path / "root_agent.yaml"),
        )
        await manager.initialize_services()
        try:
            assert manager._event_processor.reply_agents == {"root", "helper"}
            # Fact extraction reads what the agents that reply wrote.
            service = manager._session_service
            session = await service.create_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
            for author, role, text in (
                ("user", "user", "Hi."), ("solver", "model", "4"),
                ("helper", "model", "Helper here."),
            ):
                await service.append_event(session, Event(
                    author=author, invocation_id="i1",
                    content=types.Content(role=role, parts=[types.Part.from_text(text=text)]),
                ))
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [{"role": "user", "content": "Hi."},
                          {"role": "assistant", "content": "Helper here."}]


class TestUsagePerAgent:
    """Each call's usage names the agent that made it, that agent's own model
    and its context trigger: the CLI shows only the main agent's prompt as
    the conversation's context, an inheriting sub-agent's included (its
    prompt holds the conversation, but it is not the main agent's context)."""

    @pytest.mark.parametrize("inherit_context", [False, True])
    async def test_a_delegating_turns_usage_is_each_agents_own(
        self, project, monkeypatch, inherit_context
    ):
        coordinator, researcher = team(inherit_context)
        configs = [coordinator, dataclasses.replace(researcher, model="claude-haiku-4-5")]
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(delegating()))
        manager = manager_for(configs)
        try:
            await turn(manager, TURNS[0])
            delegated = await turn(manager, TURNS[1])
            triggers = {
                name: context_budget(model, manager.settings).trigger
                for name, model in manager.agent_models.items()
            }
        finally:
            await manager.cleanup()

        assert triggers["coordinator"] != triggers["researcher"]
        usage = [
            tuple(e.metadata.get(key) for key in (
                "agent", "model", "purpose", "context_trigger", "window_known"
            ))
            for e in delegated if e.type is EventType.LLM_USAGE
        ]
        assert usage == [
            ("coordinator", "anthropic:claude-sonnet-4-6", None, triggers["coordinator"], True),
            ("researcher", "anthropic:claude-haiku-4-5", "subagent", triggers["researcher"],
             True),
            ("coordinator", "anthropic:claude-sonnet-4-6", None, triggers["coordinator"], True),
        ]

    @pytest.mark.parametrize("inherit_context", [False, True])
    async def test_the_status_bar_shows_only_the_main_agents_context(
        self, project, monkeypatch, inherit_context
    ):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[sized(delegation("toolu_c1"), 4_000),
                         sized(says("The researcher says X is 42."), 4_200)],
            # A prompt far larger than the coordinator's, as an inheriting
            # sub-agent's may be: it is still not the main agent's context.
            researcher=[sized(says("X is 42."), 150_000)],
        )))
        manager = manager_for(team(inherit_context))
        try:
            events = await turn(manager, "Have the researcher find X.")
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        tracker, shown = UsageTracker(), []
        for event in events:
            if event.type is EventType.LLM_USAGE:
                tracker.record(event.metadata)
                shown.append(tracker.format_context())
        of = f"/{format_tokens(trigger)}"
        assert shown == [
            f"ctx 4k{of} ({round(400_000 / trigger)}%)",
            f"ctx 4k{of} ({round(400_000 / trigger)}%)",  # the researcher's call
            f"ctx 4.2k{of} ({round(420_000 / trigger)}%)",
        ]

    @pytest.mark.parametrize("inherit_context", [False, True])
    async def test_with_the_raw_llm_log_a_sub_agent_never_moves_ctx(
        self, project, monkeypatch, inherit_context
    ):
        """The raw LLM log keeps its usage records in its file: each call's
        usage is streamed once, the event processor's, credited to its agent.
        A second, unattributed copy made a sub-agent's prompt the context and
        counted every call twice."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[sized(delegation("toolu_c1"), 4_000),
                         sized(says("The researcher says X is 42."), 4_200)],
            researcher=[sized(says("X is 42."), 150_000)],
        )))
        manager = GoogleADKWorkflowManager(
            agent_configs=team(inherit_context),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            events = await turn(manager, "Have the researcher find X.")
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        usage = [e.metadata for e in events if e.type is EventType.LLM_USAGE]
        assert [(u.get("agent"), u.get("purpose"), u["prompt_tokens"]) for u in usage] == [
            ("coordinator", None, 4_000),
            ("researcher", "subagent", 150_000),
            ("coordinator", None, 4_200),
        ]
        tracker, shown = UsageTracker(), []
        for metadata in usage:
            tracker.record(metadata)
            shown.append(tracker.format_context())
        of = f"/{format_tokens(trigger)}"
        assert shown == [
            f"ctx 4k{of} ({round(400_000 / trigger)}%)",
            f"ctx 4k{of} ({round(400_000 / trigger)}%)",
            f"ctx 4.2k{of} ({round(420_000 / trigger)}%)",
        ]
        assert tracker.prompt_tokens == 158_200  # each call once

    async def test_the_raw_log_names_each_calls_own_model(self, project, monkeypatch):
        coordinator, researcher = team()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[delegation("toolu_c1"), says("X is 42.")],
            researcher=[says("X is 42.")],
        )))
        manager = GoogleADKWorkflowManager(
            agent_configs=[coordinator, dataclasses.replace(researcher, model="claude-haiku-4-5")],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            await turn(manager, TURNS[1])
            log_file = manager._llm_logging_plugin._log_file
        finally:
            await manager.cleanup()

        records = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert [(r["type"], r["metadata"]["model"]) for r in records
                if r["type"] != "llm_request"] == [
            ("llm_response", "anthropic:claude-sonnet-4-6"),
            ("llm_usage", "anthropic:claude-sonnet-4-6"),
            ("llm_response", "anthropic:claude-haiku-4-5"),
            ("llm_usage", "anthropic:claude-haiku-4-5"),
            ("llm_response", "anthropic:claude-sonnet-4-6"),
            ("llm_usage", "anthropic:claude-sonnet-4-6"),
        ]
        assert [r["metadata"]["model"] for r in records if r["type"] == "llm_request"] == [
            "anthropic:claude-sonnet-4-6", "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
        ]

    async def test_a_resumed_turns_usage_is_the_main_agents(self, project, monkeypatch):
        """A finished job resumes the main agent: the resumed call's usage is
        credited to it, with its trigger, as any turn's is."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(coordinator=[
            sized(calls("delegation_probe_start_job", "toolu_j1", command="make"), 3_000),
            sized(says("Started job j1."), 3_100),
            sized(says("The build finished."), 3_300),
        ])))
        manager = manager_for(team(tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            resumed = [e async for e in manager.resume_with_job_result(
                _record(STARTED[-1]), result="built ok")]
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        assert [
            tuple(e.metadata.get(key) for key in (
                "agent", "purpose", "prompt_tokens", "context_trigger", "window_known"
            ))
            for e in resumed if e.type is EventType.LLM_USAGE
        ] == [("coordinator", None, 3_300, trigger, True)]


def sized(message: dict, prompt_tokens: int) -> dict:
    """``message`` reporting a prompt of ``prompt_tokens``."""
    return {**message, "usage": {"input_tokens": prompt_tokens, "output_tokens": 5}}


# --- what the manager builds -----------------------------------------------------------


async def built(configs: list[AgentConfig]) -> LlmAgent:
    """The root agent a manager over ``configs`` builds."""
    return (await built_all(configs))[0]


async def built_all(configs: list[AgentConfig]) -> tuple[LlmAgent, GoogleADKWorkflowManager]:
    """The root agent a manager over ``configs`` builds, and the manager (cleaned up)."""
    manager = manager_for(configs)
    await manager.initialize_services()
    try:
        return manager._root_agent, manager
    finally:
        await manager.cleanup()


class TestTheAgents:
    async def test_every_agent_is_built_on_its_own(self, project):
        manager = manager_for(team())
        await manager.initialize_services()
        try:
            root, agents = manager._root_agent, dict(manager._agents)
            runners = dict(manager._agent_runners)
        finally:
            await manager.cleanup()

        researcher = agents["researcher"]
        assert root is agents["coordinator"]
        assert (root.sub_agents, researcher.sub_agents) == ([], [])
        assert researcher.mode in (None, "chat")
        (task,) = [tool for tool in root.tools if isinstance(tool, TaskTool)]
        assert identify_tool(task) is get_registry().get(TASK_TOOL)
        assert not any(isinstance(tool, TaskTool) for tool in researcher.tools)
        # The sub-agent type's own runner: its own app, never summarized.
        (runner,) = runners.values()
        assert list(runners) == ["researcher"]
        assert runner.app.name == child_app(manager.app_name)
        assert runner.app.root_agent is researcher
        assert runner.app.events_compaction_config is None

    async def test_a_middle_coordinator_delegates_too(self, project):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["middle"]),
            AgentConfig(name="middle", prompt="p", sub_agents=["leaf"], description="Mid."),
            AgentConfig(name="leaf", prompt="p", description="Leaf."),
        ]
        manager = manager_for(configs)
        await manager.initialize_services()
        try:
            agents = dict(manager._agents)
            runners = sorted(manager._agent_runners)
        finally:
            await manager.cleanup()

        assert runners == ["leaf", "middle"]
        for parent in ("coordinator", "middle"):
            (task,) = [t for t in agents[parent].tools if isinstance(t, TaskTool)]
            assert identify_tool(task) is get_registry().get(TASK_TOOL)
        assert not any(isinstance(t, TaskTool) for t in agents["leaf"].tools)


class TestWhatTheAgentsAreOffered:
    async def test_only_the_roots_long_running_tools_are_wrapped(self, project):
        configs = team(
            researcher_tools=[delegation_probe_start_job], tools=[delegation_probe_start_job]
        )
        manager = manager_for(configs)
        await manager.initialize_services()
        try:
            root, researcher = manager._root_agent, manager._agents["researcher"]
        finally:
            await manager.cleanup()

        assert [type(t) for t in root.tools if not isinstance(t, TaskTool)] == [
            LongRunningFunctionTool
        ]
        assert researcher.tools == [delegation_probe_start_job]

    async def test_a_sub_agent_without_a_description_is_still_described(
        self, project, monkeypatch
    ):
        recorder = _Recorder()
        monkeypatch.setattr(brief_module, "logger", recorder)
        answers = Team(coordinator=[says("ok")])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        configs = team()
        configs[1].description = "  "
        manager = manager_for(configs)
        try:
            await turn(manager, "hi")
        finally:
            await manager.cleanup()

        (tool,) = answers.bodies("coordinator")[0]["tools"]
        assert "- researcher: The researcher agent." in tool["description"]  # as on Deep Agents
        assert ("warning", "subagent_without_description", {"agent": "researcher"}) in (
            recorder.calls
        )

    async def test_a_described_one_keeps_its_description(self, project, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(brief_module, "logger", recorder)
        root = await built(team())

        (task,) = [tool for tool in root.tools if isinstance(tool, TaskTool)]
        assert "- researcher: Finds facts." in task.description
        assert recorder.calls == []


def task(objective: str) -> dict:
    """An application's function named ``task``: not the framework's."""
    return {"success": True}


class TestTheNameClash:
    """A delegating agent is offered the framework's ``task``: a tool of its
    own by that name would be offered beside it, under the same name."""

    async def test_a_delegating_agents_own_task_tool_fails_the_build(self, project):
        configs = team(tools=[task])

        with pytest.raises(AgentGraphError) as exc:
            await built(configs)
        message = str(exc.value)
        assert "'coordinator'" in message and "'task'" in message

    async def test_an_agent_that_does_not_delegate_may_have_one(self, project):
        configs = [AgentConfig(name="solo", prompt="p", tools=[task])]

        root = await built(configs)
        assert not any(isinstance(tool, TaskTool) for tool in root.tools)


class TestTheParentsTools:
    """With the policy's ``follow_ups`` a delegating agent also gets
    ``message_agent``, ``stop_agent`` and ``check_agents``, each the
    framework's, bound to its declaration; by default ``task`` alone."""

    @staticmethod
    async def root_of(
        configs: list[AgentConfig], policy: SubAgentPolicy
    ) -> tuple[LlmAgent, dict[str, LlmAgent]]:
        manager = GoogleADKWorkflowManager(
            agent_configs=configs, subagents=policy,
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
        )
        await manager.initialize_services()
        try:
            return manager._root_agent, dict(manager._agents)
        finally:
            await manager.cleanup()

    async def test_by_default_task_alone(self, project):
        root, _ = await self.root_of(team(), SubAgentPolicy())

        assert [tool.name for tool in root.tools] == ["task"]

    async def test_with_follow_ups_every_delegating_agent_gets_them(self, project):
        configs = [
            team()[0],
            AgentConfig(name="researcher", prompt=PROMPTS["researcher"], description="Finds.",
                        sub_agents=["checker"], include_state_tools=False),
            AgentConfig(name="checker", prompt="You check.", description="Checks.",
                        include_state_tools=False),
        ]
        root, agents = await self.root_of(configs, SubAgentPolicy(follow_ups=True))

        names = ["task", "message_agent", "stop_agent", "check_agents"]
        assert [tool.name for tool in root.tools] == names
        assert [tool.name for tool in agents["researcher"].tools] == names
        assert agents["checker"].tools == []
        for tool in root.tools[1:]:
            assert type(tool) is FollowUpTool
            assert identify_tool(tool) is get_registry().get(tool.name)
            assert await _gate(tool) is None
            declared = tool._get_declaration()
            assert (declared.description, declared.parameters_json_schema) == (
                follow_up_description(tool.name), follow_up_parameters(tool.name)
            )

    async def test_a_subclass_is_denied(self, project):
        class Message(FollowUpTool):
            pass

        tool = Message("message_agent", lambda: None)

        assert identify_tool(tool) is None
        assert _denied(await _gate(tool))

    async def test_an_own_tool_of_one_of_their_names_fails_the_build(self, project):
        def check_agents() -> dict:
            """An application's own."""
            return {"success": True}

        with pytest.raises(AgentGraphError) as exc:
            await self.root_of(team(tools=[check_agents]), SubAgentPolicy(follow_ups=True))

        assert str(exc.value) == (
            "Agent 'coordinator' continues its sub-agents through tools named message_agent, "
            "stop_agent and check_agents, so its own tools cannot include 'check_agents'; "
            "rename that tool."
        )

    async def test_the_error_names_the_first_clash_in_the_order_of_their_names(self, project):
        """On both backends, whatever the order of the agent's own tools."""
        def check_agents() -> dict:
            """An application's own."""
            return {"success": True}

        def message_agent() -> dict:
            """An application's own."""
            return {"success": True}

        with pytest.raises(AgentGraphError, match="cannot include 'message_agent'"):
            await self.root_of(
                team(tools=[check_agents, message_agent]), SubAgentPolicy(follow_ups=True)
            )

    async def test_without_follow_ups_an_own_tool_of_one_of_their_names_is_its_own(self, project):
        def check_agents() -> dict:
            """An application's own."""
            return {"success": True}

        root, _ = await self.root_of(team(tools=[check_agents]), SubAgentPolicy())

        assert [getattr(tool, "name", None) or tool.__name__ for tool in root.tools] == [
            "check_agents", "task",
        ]


# --- permission identity ---------------------------------------------------------------


async def _gate(tool) -> dict | None:
    return await PermissionPlugin().before_tool_callback(
        tool=tool, tool_args={"agent_type": "researcher"}, tool_context=None
    )


def _denied(result: dict | None) -> bool:
    return bool(result) and result["success"] is False and "not registered" in result["error"]


class TestPermissionIdentity:
    """Only the ``task`` tool the framework built passes."""

    async def test_the_task_tool_the_framework_built_passes(self, project):
        root = await built(team())
        (tool,) = [t for t in root.tools if isinstance(t, TaskTool)]

        assert identify_tool(tool) is get_registry().get(TASK_TOOL)
        assert await _gate(tool) is None

    async def test_a_look_alike_with_the_same_name_is_denied(self, project):
        class Task(BaseTool):
            async def run_async(self, *, args, tool_context):  # pragma: no cover
                return "anything"

        await built(team())
        tool = Task(name=TASK_TOOL, description="Delegates.")
        assert PermissionPlugin._resolve_definition(tool) is _UNVERIFIED
        assert _denied(await _gate(tool))

    async def test_a_subclass_is_denied(self, project):
        class Delegate(TaskTool):
            pass

        manager = manager_for(team())
        await manager.initialize_services()
        try:
            tool = Delegate("coordinator", {"researcher": team()[1]}, lambda: manager._delegations)
        finally:
            await manager.cleanup()
        # A subclass may run anything: only an exact TaskTool is bound.
        assert identify_tool(tool) is None
        assert _denied(await _gate(tool))

    async def test_an_apps_agent_tool_is_denied(self, project):
        stranger = LlmAgent(name="researcher", description="Finds facts.")
        assert _denied(await _gate(AgentTool(stranger)))


# --- jobs -----------------------------------------------------------------------------


def _record(call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j1", tool="delegation_probe_start_job", call_id=call_id,
        call_name="delegation_probe_start_job", session_id="s", user_id="u",
        state=JobState.SUCCEEDED, exit_code=0, error=None,
    )


def _last_user_text(body: dict) -> str:
    content = body["messages"][-1]["content"]
    return content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if block["type"] == "text"
    )


class TestJobs:
    """A job a sub-agent starts does not pause its parent, and the main agent
    is told when it finishes; the main agent's own job still answers its call."""

    @staticmethod
    def _sub_agents_job() -> Team:
        return Team(
            coordinator=[delegation("toolu_c1", "Start the build."),
                         says("The researcher started job j1."),
                         says("The build finished: built ok.")],
            researcher=[calls("delegation_probe_start_job", "toolu_r1", command="make"),
                        says("Started job j1."),
                        says("[the researcher was asked again]")],
        )

    async def test_a_sub_agents_job_starter_returns_in_the_turn(self, project, monkeypatch):
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            events = await turn(manager, "Build it.")
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "researcher", "researcher", "coordinator"]
        # The sub-agent's starter returned at once, with the job's id.
        started = lines(answers.bodies("researcher")[1])[-1].removeprefix("user: result ")
        assert json.loads(json.loads(started))["job_id"] == "j1"
        assert result_of(lines(answers.bodies("coordinator")[1])[-1])["result"] == (
            "Started job j1."
        )
        assert texts(events)[-1] == "The researcher started job j1."

    async def test_its_finished_job_is_told_to_the_main_agent(self, project, monkeypatch):
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            before = len(answers.sent)
            record = _record(STARTED[-1])
            resumed = [e async for e in manager.resume_with_job_result(record, result="built ok")]
        finally:
            await manager.cleanup()

        # As on Deep Agents: the root runs on the notice, the sub-agent does not.
        assert answers.agents[before:] == ["coordinator"]
        notice = JOB_NOTICE_HEADER + "\n" + JOB_NOTICE_ASK + "\n" + json.dumps(
            job_result_payload(record, "built ok"), ensure_ascii=False
        )
        assert _last_user_text(answers.bodies("coordinator")[-1]) == notice
        assert texts(resumed) == ["The build finished: built ok."]

    async def test_fact_extraction_leaves_the_notice_out(self, project, monkeypatch):
        # The notice is a user-role message nobody typed, as on Deep Agents,
        # where its mark keeps it out.
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            record = _record(STARTED[-1])
            async for _ in manager.resume_with_job_result(record, result="built ok"):
                pass
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [
            {"role": "user", "content": "Build it."},
            {"role": "assistant", "content": "The researcher started job j1."},
            {"role": "assistant", "content": "The build finished: built ok."},
        ]

    async def test_the_main_agents_own_job_is_told_after_the_turns_since(
        self, project, monkeypatch
    ):
        """As a sub-agent's: a notice. A result for the call itself made ADK
        send the call, its result and nothing between (its rearranger), so
        the main agent answered without what the user said meanwhile."""
        answers = Team(coordinator=[
            calls("delegation_probe_start_job", "toolu_j1", command="make"),
            says("Started job j1."),
            says("Four."),
            says("The build finished."),
        ])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(tools=[delegation_probe_start_job]))
        try:
            events = await turn(manager, "Build it.")
            await turn(manager, "Meanwhile, what is 2+2?")
            asked = len(answers.sent)
            record = _record(STARTED[-1])
            resumed = [e async for e in manager.resume_with_job_result(record, result="built ok")]
        finally:
            await manager.cleanup()

        assert STARTED[-1] == "toolu_j1"
        assert texts(events) == ["Started job j1."]
        assert answers.agents[asked:] == ["coordinator"]
        sent = answers.bodies("coordinator")[-1]
        assert "user: Meanwhile, what is 2+2?" in lines(sent)
        assert "assistant: Four." in lines(sent)
        notice = JOB_NOTICE_HEADER + "\n" + JOB_NOTICE_ASK + "\n" + json.dumps(
            job_result_payload(record, "built ok"), ensure_ascii=False
        )
        assert _last_user_text(sent) == notice
        assert texts(resumed) == ["The build finished."]


# --- the ADK this rests on (contract tests) ---------------------------------------------


def claude(api: FakeAPI) -> AnthropicLlm:
    """ADK's own Claude model, its requests sent to ``api``."""
    return AnthropicLlm(
        model="claude-sonnet-4-6",
        client=AsyncAnthropic(
            api_key="a-test-key", http_client=api.http_client(http_library(AsyncAnthropic))
        ),
    )


def _user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


async def _run(root: LlmAgent, texts_in: list[str]) -> None:
    runner = Runner(
        app=App(name="contract", root_agent=root), session_service=InMemorySessionService()
    )
    session = await runner.session_service.create_session(app_name="contract", user_id="u")
    for text in texts_in:
        async for _ in runner.run_async(
            user_id="u", session_id=session.id, new_message=_user(text)
        ):
            pass


class _Probe(BaseTool):
    """A ``BaseTool`` as ``TaskTool`` is one: its declaration from a JSON schema,
    its run from ``run_async(args=, tool_context=)``."""

    def __init__(self) -> None:
        super().__init__(name="probe", description="Probe a thing.")
        self.ran: list[tuple[dict, str | None, dict]] = []

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name, description=self.description,
            parameters_json_schema={
                "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
            },
        )

    async def run_async(self, *, args, tool_context):
        tool_context.state["plan"] = "probed"
        self.ran.append((dict(args), tool_context.function_call_id,
                         {"events": len(tool_context.session.events)}))
        return json.dumps({"success": True})


class TestTheADKItRestsOn:
    async def test_a_skill_toolsets_tools_are_its_static_list(self, tmp_path):
        # The name-clash check reads SkillToolset._tools: it is what ADK offers.
        skills = SkillStore().resolve([str(_make_skill(tmp_path))])
        toolset = SkillToolset(skills=skills)

        assert all(isinstance(tool, BaseTool) for tool in toolset._tools)
        assert sorted(t.name for t in toolset._tools) == sorted(
            t.name for t in await toolset.get_tools()
        )
        assert {"list_skills", "load_skill", "load_skill_resource"} <= {
            t.name for t in toolset._tools
        }

    def test_a_base_tool_is_made_from_its_name_and_description(self):
        parameters = inspect.signature(BaseTool.__init__).parameters
        assert {"name", "description"} <= set(parameters)
        assert set(inspect.signature(BaseTool.run_async).parameters) == {
            "self", "args", "tool_context",
        }

    async def test_a_tool_is_declared_and_run_as_task_is(self):
        """ADK offers a tool by its ``_get_declaration`` (a JSON schema, sent
        as given), calls ``run_async`` with the model's arguments and a tool
        context (its call id, its state, its session), stores a text result
        as ``{"result": <text>}``, and writes its state changes with the
        result."""
        api = FakeAPI(Team(
            coordinator=[calls("probe", "toolu_p1", q="X"), says("Done.")],
        ))
        probe = _Probe()
        root = LlmAgent(name="coordinator", model=claude(api),
                        instruction=PROMPTS["coordinator"], tools=[probe])
        service = InMemorySessionService()
        runner = Runner(app=App(name="contract", root_agent=root), session_service=service)
        session = await service.create_session(app_name="contract", user_id="u")
        async for _ in runner.run_async(
            user_id="u", session_id=session.id, new_message=_user("Probe X.")
        ):
            pass
        stored = await service.get_session(app_name="contract", user_id="u", session_id=session.id)

        (declared,) = api.bodies[0]["tools"]
        assert (declared["name"], declared["input_schema"]["required"]) == ("probe", ["q"])
        assert probe.ran == [({"q": "X"}, "toolu_p1", {"events": 2})]
        (response,) = [
            part.function_response for event in stored.events
            for part in (event.content.parts if event.content else []) if part.function_response
        ]
        assert response.response == {"result": json.dumps({"success": True})}
        assert stored.state["plan"] == "probed"

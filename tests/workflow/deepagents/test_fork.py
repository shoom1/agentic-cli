"""A sub-agent's context on Deep Agents: a fresh brief, or a fork of the conversation.

A ``fresh`` sub-agent (the default) is Deep Agents' isolated
``CompiledSubAgent``: its first request holds the brief alone. An ``inherit``
one is a fork (``mode="fork"``): it gets the coordinator's effective
conversation (after any summary), without the message that called ``task``,
then Deep Agents' preamble and the brief. It keeps its own prompt, tools and
model. Deep Agents marks forks beta with a ``LangChainBetaWarning`` that
LangChain would print over the UI; that one warning is filtered and logged at
debug instead. A fork never sends the coordinator's reasoning on (signed
reasoning from one model may be refused by another), while its own goes back
to it in its tool loop; the coordinator's state keeps all of it.

Every model is scripted (``fakes.ScriptedModels``); nothing reaches a provider.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("deepagents")

from deepagents import CompiledSubAgent, SubAgentMiddleware  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_core._api import LangChainBetaWarning  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import context_budget  # noqa: E402
from agentic_cli.workflow.deepagents import agent_builder  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import ForeignReasoningMiddleware  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.deepagents.fakes import USAGE, ScriptedChat, ScriptedModels, ai, task  # noqa: E402

BRIEF = "Find X. Report the number."
TURNS = ["My name is Ada and the secret code is 7.", "Have the researcher find X."]
#: How Deep Agents opens a fork's task message.
PREAMBLE = "[The messages above are a prior conversation you are continuing as the subagent"


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def hermetic(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


def team(context: str | None, *, researcher_tools: bool = False) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt="You coordinate.", sub_agents=["researcher"],
                    description="Leads.", include_state_tools=False),
        AgentConfig(name="researcher", prompt="You research.", description="Finds facts.",
                    include_state_tools=researcher_tools, context=context),
    ]


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def manager_for(configs: list[AgentConfig]) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs,
        settings=BaseSettings(session_store="memory", default_model="anthropic:claude-sonnet-4-5"),
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def shape(messages) -> list[tuple[str, str]]:
    return [(m.type, m.text) for m in messages]


def delegating() -> dict[str, list]:
    """The coordinator delegates on the second turn."""
    return {
        "coordinator": [
            ai("Noted, Ada."),
            ai("Delegating.", calls=[task("researcher", BRIEF, "c1")]),
            ai("The researcher says X is 42."),
        ],
        "researcher": [ai("X is 42.")],
    }


async def two_turns(monkeypatch, configs, scripts) -> tuple[ScriptedModels, list]:
    models = scripted(monkeypatch, **scripts)
    manager = manager_for(configs)
    try:
        events = [await turn(manager, text) for text in TURNS]
    finally:
        await manager.cleanup()
    return models, events


class TestWhatTheSubAgentSees:
    async def test_an_inheriting_one_continues_the_conversation(self, hermetic, monkeypatch):
        models, events = await two_turns(monkeypatch, team("inherit"), delegating())
        researcher = models.chats["researcher"]

        sent = shape(researcher.conversation(0))
        # The coordinator's conversation, without the message that delegated ...
        assert sent[:3] == [("human", TURNS[0]), ("ai", "Noted, Ada."), ("human", TURNS[1])]
        assert not any("Delegating." in text for _, text in sent)
        # ... then Deep Agents' preamble and the brief.
        (kind, last), = sent[3:]
        assert kind == "human" and last.startswith(PREAMBLE) and last.endswith(BRIEF)
        # Its own prompt, not the coordinator's.
        assert researcher.system_prompt(0).startswith("You research.")
        assert [e.content for e in events[1] if e.type is EventType.TEXT][-1] == (
            "The researcher says X is 42."
        )

    @pytest.mark.parametrize("context", [None, "fresh"])
    async def test_a_fresh_one_sees_only_the_brief(self, hermetic, monkeypatch, context):
        models, _ = await two_turns(monkeypatch, team(context), delegating())

        assert shape(models.chats["researcher"].conversation(0)) == [("human", BRIEF)]


class TestUsagePerAgent:
    """Each call's usage names its agent, that agent's own model and its
    context trigger. A sub-agent's is ``purpose="subagent"`` whether it is
    fresh or inherits: a fork's prompt holds the conversation, but it is not
    the main agent's context."""

    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    async def test_a_delegating_turns_usage_is_each_agents_own(
        self, hermetic, monkeypatch, context
    ):
        coordinator, researcher = team(context)
        configs = [coordinator, dataclasses.replace(researcher, model="anthropic:claude-opus-4-7")]
        scripted(monkeypatch, **delegating())
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
            ("coordinator", "anthropic:claude-sonnet-4-5", None, triggers["coordinator"], True),
            ("researcher", "anthropic:claude-opus-4-7", "subagent", triggers["researcher"], True),
            ("coordinator", "anthropic:claude-sonnet-4-5", None, triggers["coordinator"], True),
        ]


class TestTheBetaWarning:
    def test_deep_agents_warns_of_a_fork(self):
        # What the builder quiets: if Deep Agents stops warning, or words it
        # otherwise, this fails and the filter is to be revisited.
        runnable = create_agent(model=ScriptedChat(), tools=[], name="researcher")
        fork = CompiledSubAgent(
            name="researcher", description="Finds facts.", runnable=runnable, mode="fork"
        )
        with pytest.warns(LangChainBetaWarning, match="The feature `forked subagents` is in beta"):
            SubAgentMiddleware(backend=None, subagents=[fork])

    async def test_building_a_fork_logs_it_instead(self, hermetic, monkeypatch, recwarn):
        recorder = _Recorder()
        monkeypatch.setattr(agent_builder, "logger", recorder)
        scripted(monkeypatch, **delegating())
        manager = manager_for(team("inherit"))
        try:
            await manager.initialize_services()
        finally:
            await manager.cleanup()

        assert [w for w in recwarn if issubclass(w.category, LangChainBetaWarning)] == []
        assert ("debug", "deepagents_fork_beta", {"agents": ["researcher"]}) in recorder.calls

    async def test_no_fork_logs_nothing(self, hermetic, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(agent_builder, "logger", recorder)
        scripted(monkeypatch, **delegating())
        manager = manager_for(team("fresh"))
        try:
            await manager.initialize_services()
        finally:
            await manager.cleanup()

        assert [event for _, event, _ in recorder.calls if event == "deepagents_fork_beta"] == []


# --- reasoning -------------------------------------------------------------------------


def claude(text: str, thought: str, signature: str, calls=()) -> AIMessage:
    """A Claude answer, as langchain-anthropic returns one: a signed thinking block first."""
    message = ai(text, calls)
    return AIMessage(
        content=[{"type": "thinking", "thinking": thought, "signature": signature},
                 {"type": "text", "text": text}],
        tool_calls=message.tool_calls,
        usage_metadata=USAGE,
        response_metadata={"model_provider": "anthropic"},
    )


def thoughts(messages) -> list[str]:
    return [
        block["thinking"]
        for message in messages
        if isinstance(message, AIMessage) and isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]


class TestReasoning:
    async def test_a_fork_is_sent_its_own_reasoning_and_not_the_coordinators(
        self, hermetic, monkeypatch
    ):
        models = scripted(
            monkeypatch,
            coordinator=[
                claude("Noted, Ada.", "The user is Ada.", "sig-c1"),
                claude("Delegating.", "The researcher should look.", "sig-c2",
                       calls=[task("researcher", BRIEF, "c1")]),
                claude("The researcher says X is 42.", "Relay it.", "sig-c3"),
            ],
            researcher=[
                claude("Checking the plan.", "Look at the plan first.", "sig-r1",
                       calls=[("get_plan", {}, "r1")]),
                claude("X is 42.", "Answer now.", "sig-r2"),
            ],
        )
        manager = manager_for(team("inherit", researcher_tools=True))
        try:
            for text in TURNS:
                await turn(manager, text)
            state = (await manager._root.aget_state(
                run_config(manager.session_ref("s", "u"))
            )).values
        finally:
            await manager.cleanup()
        researcher = models.chats["researcher"]

        # The inherited conversation is sent without the coordinator's thoughts ...
        first = researcher.conversation(0)
        assert ("ai", "Noted, Ada.") in shape(first)
        assert thoughts(first) == []
        assert "sig-c1" not in json.dumps([m.content for m in first])
        # ... while its own thinking goes back to it, signed, in its tool loop.
        second = researcher.conversation(1)
        assert thoughts(second) == ["Look at the plan first."]
        (own,) = [m for m in second if isinstance(m, AIMessage) and m.name == "researcher"]
        assert own.content[0]["signature"] == "sig-r1"
        # The coordinator's state keeps its reasoning.
        assert thoughts(state["messages"]) == [
            "The user is Ada.", "The researcher should look.", "Relay it.",
        ]

    async def test_the_coordinator_is_still_sent_its_own(self, hermetic, monkeypatch):
        models = scripted(
            monkeypatch,
            coordinator=[
                claude("Noted, Ada.", "The user is Ada.", "sig-c1"),
                claude("Done.", "Nothing to delegate.", "sig-c2"),
            ],
            researcher=[],
        )
        manager = manager_for(team("inherit"))
        try:
            for text in TURNS:
                await turn(manager, text)
        finally:
            await manager.cleanup()

        assert thoughts(models.chats["coordinator"].conversation(1)) == ["The user is Ada."]


class TestTheMiddleware:
    def test_another_agents_reasoning_is_left_out(self):
        own = AIMessage(
            content=[{"type": "thinking", "thinking": "Mine.", "signature": "s1"},
                     {"type": "text", "text": "A."}],
            name="researcher", response_metadata={"model_provider": "anthropic"},
        )
        parents = AIMessage(
            content=[{"type": "thinking", "thinking": "The parent's.", "signature": "s2"},
                     {"type": "text", "text": "B."}],
            name="coordinator", response_metadata={"model_provider": "anthropic"},
        )
        unnamed = AIMessage(
            content=[{"type": "thinking", "thinking": "Nobody's.", "signature": "s3"},
                     {"type": "text", "text": "C."}],
            response_metadata={"model_provider": "anthropic"},
        )
        history = [HumanMessage("hi"), parents, unnamed, own]
        request = ModelRequest(model=ScriptedChat(), messages=history, state={"messages": history})

        middleware = ForeignReasoningMiddleware(ProviderKind.ANTHROPIC, agent="researcher")
        messages = middleware.wrap_model_call(request, lambda r: r.messages)

        assert thoughts(messages) == ["Nobody's.", "Mine."]
        assert messages[1].content == [{"type": "text", "text": "B."}]
        assert thoughts(history) == ["The parent's.", "Nobody's.", "Mine."]  # untouched

    def test_without_an_agent_only_other_providers_reasoning_goes(self):
        parents = AIMessage(
            content=[{"type": "thinking", "thinking": "The parent's.", "signature": "s2"}],
            name="coordinator", response_metadata={"model_provider": "anthropic"},
        )
        request = ModelRequest(model=ScriptedChat(), messages=[parents], state={"messages": [parents]})

        messages = ForeignReasoningMiddleware(ProviderKind.ANTHROPIC).wrap_model_call(
            request, lambda r: r.messages
        )

        assert thoughts(messages) == ["The parent's."]

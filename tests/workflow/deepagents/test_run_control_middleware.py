"""Deep Agents: a sub-agent run's limits at its model calls (``RunControlMiddleware``).

What the middleware does to a request and an answer, alone: a run's limits, and the
``task`` its request offers (what the run may start). Whole turns, on
both backends, are in the contract suite (``TestTheStepLimit``,
``TestTheTokenLimit``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from agentic_cli.workflow.agents import AgentRuns, SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE, RunControl  # noqa: E402
from agentic_cli.workflow.agents.delegate import Delegations  # noqa: E402
from agentic_cli.workflow.agents.envelope import TOKEN_LIMIT_ERROR  # noqa: E402
from agentic_cli.workflow.agents.policy import AgentLimits  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    RunControlMiddleware,
    with_note,
    without_calls,
)
from agentic_cli.workflow.deepagents.task_tool import build_task_tool  # noqa: E402
from agentic_cli.workflow.sessions import RunRef, reset_active_run, set_active_run  # noqa: E402

CALL = {"name": "lookup", "args": {"q": "X"}, "id": "r1", "type": "tool_call"}


class TestTheNote:
    def test_it_ends_a_last_human_message(self):
        task = HumanMessage("# Task: Find X.")

        noted = with_note([task], STEP_LIMIT_NOTE)

        assert noted[0].content == [
            {"type": "text", "text": "# Task: Find X."},
            {"type": "text", "text": STEP_LIMIT_NOTE},
        ]
        assert task.content == "# Task: Find X."  # not changed in place

    def test_after_a_tool_result_it_is_a_human_message_of_its_own(self):
        messages = [HumanMessage("Go."), AIMessage("", tool_calls=[CALL]),
                    ToolMessage("42", tool_call_id="r1")]

        noted = with_note(messages, STEP_LIMIT_NOTE)

        assert noted[:3] == messages
        assert (type(noted[3]), noted[3].content) == (HumanMessage, STEP_LIMIT_NOTE)


class TestDroppedCalls:
    def test_the_calls_go_and_the_text_and_reasoning_stay(self):
        answer = AIMessage(
            content=[{"type": "reasoning", "reasoning": "why"},
                     {"type": "text", "text": "X is 42."},
                     {"type": "tool_use", "id": "r1", "name": "lookup", "input": {"q": "X"}}],
            tool_calls=[CALL],
            additional_kwargs={"tool_calls": [{"id": "r1"}], "other": 1},
        )

        kept = without_calls(answer)

        assert kept.content == [{"type": "reasoning", "reasoning": "why"},
                                {"type": "text", "text": "X is 42."}]
        assert (kept.tool_calls, kept.additional_kwargs) == ([], {"other": 1})
        assert answer.tool_calls == [CALL]  # not changed in place

    def test_an_answer_without_calls_is_itself(self):
        answer = AIMessage("X is 42.")

        assert without_calls(answer) is answer


async def test_the_roots_calls_are_left_alone():
    """No run is active in the root's calls: its control is none."""
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: None))
    answer = AIMessage("", tool_calls=[CALL])

    async def handler(request):
        return answer

    assert await middleware.awrap_model_call("the request", handler) is answer


REF = RunRef(agent="researcher", instance="researcher-1", run="researcher-1#1",
             parent_run=None, depth=1)
USAGE = {"input_tokens": 90, "output_tokens": 20, "total_tokens": 110}


@pytest.mark.parametrize("calls", [True, False])
async def test_an_answer_past_the_token_limit_is_the_last_only_when_it_would_go_on(calls):
    control = RunControl(REF, AgentLimits(max_tokens=100))
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: control))
    answer = AIMessage("Found it.", tool_calls=[CALL] if calls else [], usage_metadata=USAGE)

    async def handler(request):
        return answer

    kept = await middleware.awrap_model_call("the request", handler)

    if calls:
        assert control.cancelled == ("limit", TOKEN_LIMIT_ERROR)
        assert (kept.content, kept.tool_calls) == ("Found it.", [])
    else:
        assert (control.cancelled, kept) == (None, answer)


async def test_an_answer_with_no_usage_counts_none_of_its_own_tokens():
    control = RunControl(REF, AgentLimits(max_tokens=1))
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: control))
    answer = AIMessage("Found it.", tool_calls=[CALL])

    async def handler(request):
        return answer

    assert await middleware.awrap_model_call("the request", handler) is answer
    assert (control.cancelled, control.final_call) == (None, False)


async def test_an_answer_with_no_usage_still_counts_the_runs_side_calls():
    control = RunControl(REF, AgentLimits(max_tokens=100), side_tokens=lambda: 120)
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: control))
    answer = AIMessage("Found it.", tool_calls=[CALL])

    async def handler(request):
        return answer

    kept = await middleware.awrap_model_call("the request", handler)

    assert control.cancelled == ("limit", TOKEN_LIMIT_ERROR)
    assert (kept.content, kept.tool_calls) == ("Found it.", [])


async def test_the_messages_steered_since_the_last_call_join_the_runs_state():
    ref = RunRef(agent="researcher", instance="researcher-1", run="researcher-1#1",
                 parent_run=None, depth=1)
    control = RunControl(ref, AgentLimits())
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: control))
    control.steer("Also find Z.")
    control.steer("Be brief.")

    update = await middleware.abefore_model({}, None)

    assert [(type(m), m.content) for m in update["messages"]] == [
        (HumanMessage, "Also find Z."), (HumanMessage, "Be brief."),
    ]
    assert await middleware.abefore_model({}, None) is None  # given once


async def test_steered_messages_are_stored_once_a_model_call_holds_them():
    """Taken into the state before a model call (``abefore_model``), a message
    is the run's unstored one until that call's request holds it: the step
    that added it is checkpointed by then."""
    control = RunControl(REF, AgentLimits())
    middleware = RunControlMiddleware(lambda: SimpleNamespace(control=lambda run: control))
    control.steer("Also Z.")

    update = await middleware.abefore_model({"messages": []}, None)

    assert [m.content for m in update["messages"]] == ["Also Z."]
    assert control.unstored() == ["Also Z."]

    async def handler(request):
        return AIMessage("Found it.")

    await middleware.awrap_model_call(SimpleNamespace(messages=update["messages"]), handler)

    assert control.unstored() == []


# -- task, offered for each model call what the run may start -------------------------


def _hook():
    async def hook(run):  # pragma: no cover - no run starts here
        raise AssertionError("no run starts")

    return hook


def _offering(policy: SubAgentPolicy):
    """A researcher's task, an own tool, the middleware and the delegations, under ``policy``."""
    configs = [
        AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher"]),
        AgentConfig(name="researcher", prompt="p", description="Finds facts."),
        AgentConfig(name="helper", prompt="p", description="Helps.",
                    spawnable_by=frozenset({"agent"})),
    ]
    made = Delegations(runs=AgentRuns(), policy=policy,
                       types={c.name: c for c in configs}, hook=_hook())
    task = build_task_tool("researcher", lambda: made, root=False)
    lookup = StructuredTool.from_function(lambda q="": "42", name="lookup", description="Look up.")
    return made, task, lookup, RunControlMiddleware(lambda: made)


def _request(tools: list) -> ModelRequest:
    return ModelRequest(
        model=None, messages=[HumanMessage("# Task: Find X.")], system_message=None,
        tool_choice=None, tools=tools, response_format=None, state={}, runtime=None,
    )


async def _offered(middleware, request: ModelRequest, run: RunRef | None) -> list:
    """The tools ``request`` offers once the middleware has it, in ``run``."""
    seen: list = []

    async def handler(sent):
        seen.append(list(sent.tools))
        return AIMessage("Done.")

    token = set_active_run(run)
    try:
        await middleware.awrap_model_call(request, handler)
    finally:
        reset_active_run(token)
    (tools,) = seen
    return tools


DEEPER = SubAgentPolicy(spawn_sources={"agent"}, max_spawn_depth=2)


async def test_a_runs_task_offers_what_the_run_may_start():
    made, task, lookup, middleware = _offering(DEEPER)
    request = _request([lookup, task])

    tools = await _offered(middleware, request, REF)

    assert [tool.name for tool in tools] == ["lookup", "task"]
    offer = made.task_offer("researcher", 1)
    assert (tools[1].description, tools[1].args_schema) == (offer.description, offer.parameters)
    assert tools[1].args_schema["properties"]["agent_type"]["enum"] == ["helper"]
    assert request.tools == [lookup, task]  # the request it was given is as it was


async def test_a_run_that_may_start_nothing_is_offered_no_task():
    _, task, lookup, middleware = _offering(DEEPER)
    deep = RunRef(agent="researcher", instance="researcher-2", run="researcher-2#1",
                  parent_run="helper-1#1", depth=2)

    assert await _offered(middleware, _request([lookup, task]), deep) == [lookup]


async def test_an_apps_own_tool_named_task_is_left_alone():
    _, _, lookup, middleware = _offering(SubAgentPolicy())
    own = StructuredTool.from_function(lambda: "mine", name="task", description="An app's own.")

    assert await _offered(middleware, _request([lookup, own]), REF) == [lookup, own]


async def test_the_roots_calls_keep_their_task():
    _, task, lookup, middleware = _offering(DEEPER)

    assert await _offered(middleware, _request([lookup, task]), None) == [lookup, task]

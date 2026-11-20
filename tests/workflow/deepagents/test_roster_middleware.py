"""Deep Agents: the main agent's roster opens the first human message of its
requests, after the memory index; the summarization counts it.

``RosterMiddleware`` puts the roster (``workflow/agents/roster.py``) first in
the first human message of each of the main agent's requests; it sits outside
``MemoryIndexMiddleware``, which then puts the index ahead of it, and inside
the summarization, which counts it (``roster=``) as it counts the index and
drops its snapshot when it writes a summary (``on_written``).
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from agentic_cli.workflow.agents import SubAgentPolicy
from agentic_cli.workflow.agents.roster import ROSTER_CLOSE, ROSTER_HEADER, ROSTER_OPEN
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.deepagents.middleware import MemoryIndexMiddleware, RosterMiddleware
from agentic_cli.workflow.deepagents.summarization import build_summarization
from tests.workflow.deepagents.fakes import ScriptedChat, ai
from tests.workflow.deepagents.test_context import USER_TURN, context, resolved
from tests.workflow.deepagents.test_manager import _manager, ctx, hermetic, scripted  # noqa: F401
from tests.workflow.deepagents.test_memory_index_middleware import INDEX

ROSTER = "\n".join([ROSTER_OPEN, ROSTER_HEADER,
                    "- researcher-1 (researcher, idle): Find X. — X is 42.", ROSTER_CLOSE])


def _request(messages):
    return ModelRequest(model=ScriptedChat(), messages=messages, state={"messages": messages})


async def sent(middleware, messages) -> list:
    async def handler(request):
        return request.messages

    return await middleware.awrap_model_call(_request(messages), handler)


async def test_the_roster_is_the_first_block_of_the_first_human_message():
    history = [HumanMessage("Hello."), AIMessage("Hi."), HumanMessage("Again.")]

    messages = await sent(RosterMiddleware(lambda: ROSTER), history)

    assert messages[0].content == [{"type": "text", "text": ROSTER},
                                   {"type": "text", "text": "Hello."}]
    assert messages[1:] == history[1:]
    assert history[0].content == "Hello."  # the state keeps its message
    assert await sent(RosterMiddleware(lambda: ""), history) == history


async def test_outside_the_memory_index_middleware_the_index_goes_ahead_of_it():
    inner = MemoryIndexMiddleware(lambda: INDEX)
    outer = RosterMiddleware(lambda: ROSTER)

    async def _messages(request):
        return request.messages

    async def handler(request):
        return await inner.awrap_model_call(request, _messages)

    messages = await outer.awrap_model_call(_request([HumanMessage("Hello.")]), handler)

    assert [block["text"] for block in messages[0].content] == [INDEX, ROSTER, "Hello."]


async def test_the_summarization_counts_the_roster():
    chat = ScriptedChat(script=[ai("A summary.")])
    model = resolved("local:small", agent="solo")  # summarizes at 16,150 tokens
    big = "\n".join([ROSTER_OPEN, ROSTER_HEADER, "- x" * 4_000, ROSTER_CLOSE])
    plain = build_summarization(chat, model, context())
    counted = build_summarization(chat, model, context(), roster=lambda: big)
    both = build_summarization(chat, model, context(), index=lambda: INDEX, roster=lambda: big)
    history = [HumanMessage(f"Turn {n}: {USER_TURN}") for n in range(1, 4)]

    base = plain._count_tokens(history, None, None)
    roster = plain.token_counter([HumanMessage(big)])
    assert counted._count_tokens(history, None, None) == base + roster
    assert both._count_tokens(history, None, None) == (
        base + roster + plain.token_counter([HumanMessage(INDEX)])
    )


TEAM = [
    AgentConfig(name="lead", prompt="p", sub_agents=["helper"]),
    AgentConfig(name="helper", prompt="p", description="Helps."),
]


async def test_a_root_that_continues_its_sub_agents_counts_it_and_forgets_it(ctx, monkeypatch):
    """With the policy's ``follow_ups`` (``message_agent``) the root's
    summarization counts the roster, and its summary drops the roster's
    snapshot, without a memory tool too; with the policy's defaults neither."""
    scripted(monkeypatch, lead=[], helper=[])
    following = _manager(ctx.settings, TEAM, subagents=SubAgentPolicy(follow_ups=True))
    plain = _manager(ctx.settings, TEAM)
    try:
        await following.initialize_services()
        await plain.initialize_services()
        summarizer = following._root_summarization
        plain_summarizer = plain._root_summarization
    finally:
        await following.cleanup()
        await plain.cleanup()

    assert summarizer._roster == following._roster_now
    assert summarizer._on_written == following._forget_memory_index
    assert plain_summarizer._roster is None and plain_summarizer._on_written is None

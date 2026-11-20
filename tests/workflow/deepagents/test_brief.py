"""Deep Agents: an instance's brief is kept through every summary of its thread.

An instance's first run marks its task message as its brief
(``summarization.brief_message``); a follow-up's message is not marked. Its
summarization (on its own model, against its own budget) keeps the brief
ahead of the summary: the summary message Deep Agents stores opens with the
brief's text, so every request after it, and every reader of the stored
summary, holds the brief. A brief the summary leaves in the kept messages
(an inheriting instance's history summarized alone) is not restated. The
main agent's thread has no brief, and its summaries are Deep Agents' own.
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from agentic_cli.workflow.agents import SubAgentPolicy
from agentic_cli.workflow.context import context_budget
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.deepagents.sessions import instance_thread_id
from agentic_cli.workflow.deepagents.summarization import (
    SUMMARIZATION_EVENT_KEY,
    brief_message,
    brief_of,
    build_summarization,
    is_brief,
)
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX
from tests.workflow.deepagents.fakes import ScriptedChat, ai, task
from tests.workflow.deepagents.test_context import USER_TURN, context, resolved
from tests.workflow.deepagents.test_manager import _manager, ctx, hermetic, scripted, turn  # noqa: F401

BRIEF = "# Task: Find X.\n\n## Objective\nFind X."


def long_turns(first: int, last: int) -> list:
    """Turns ``first`` to ``last``, each about 3,000 tokens and an answer."""
    turns: list = []
    for n in range(first, last + 1):
        turns += [HumanMessage(f"Turn {n}: {USER_TURN}", id=f"h{n}"), AIMessage(f"A{n}.", id=f"a{n}")]
    return turns


async def summarized(chat: ScriptedChat, messages: list, event=None) -> tuple[list, dict]:
    """One model call of a ``local:small`` agent (summarized at 16,150 tokens)
    on ``messages``: what the model was sent, and the summary event stored."""
    middleware = build_summarization(chat, resolved("local:small", agent="worker"), context())
    state = {"messages": messages}
    if event is not None:
        state[SUMMARIZATION_EVENT_KEY] = event
    seen: list = []

    async def handler(request):
        seen.append(list(request.messages))
        return AIMessage("Done.")

    response = await middleware.awrap_model_call(
        ModelRequest(model=chat, messages=messages, state=state), handler
    )
    update = getattr(getattr(response, "command", None), "update", None) or {}
    return seen[-1], update.get(SUMMARIZATION_EVENT_KEY)


def blocks(message) -> list[str]:
    content = message.content
    return [content] if isinstance(content, str) else [block["text"] for block in content]


class TestTheMark:
    def test_a_brief_is_marked_and_found_the_last_one_wins(self):
        parents = brief_message("# Task: The parent's.", id="t1")
        own = brief_message(BRIEF, id="t2")
        thread = [parents, AIMessage("Done."), own, AIMessage("X."), HumanMessage("Now Y.")]

        assert is_brief(own) and not is_brief(HumanMessage(BRIEF))
        assert own.response_metadata == {"agentic_cli": {"kind": "brief"}}
        assert brief_of(thread) is own
        assert brief_of([HumanMessage("Hi.")]) is None

    async def test_an_instances_first_message_is_its_brief_and_a_follow_up_is_not(
        self, ctx, monkeypatch
    ):
        scripted(
            monkeypatch,
            lead=[ai(calls=[task("helper", "Find X.", "c1")]), ai("Asked."),
                  ai(calls=[("message_agent", {"agent": "helper-1", "message": "Now Y."}, "c2")]),
                  ai("Done.")],
            helper=[ai("X is 42."), ai("Y is 7.")],
        )
        manager = _manager(ctx.settings, [
            AgentConfig(name="lead", prompt="p", sub_agents=["helper"]),
            AgentConfig(name="helper", prompt="p", description="Helps."),
        ], subagents=SubAgentPolicy(follow_ups=True))
        try:
            await turn(manager, "Go.")
            await turn(manager, "Again.")
            thread = instance_thread_id(manager.session_ref("s", "u"), "helper-1")
            saved = await manager._checkpointer.aget_tuple({"configurable": {"thread_id": thread}})
        finally:
            await manager.cleanup()

        messages = saved.checkpoint["channel_values"]["messages"]
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        assert [is_brief(m) for m in humans] == [True, False]
        assert humans[0].text.startswith("# Task: Find X.") and humans[1].text == "Now Y."
        assert brief_of(messages) is humans[0]


class TestTheSummaries:
    async def test_a_summary_opens_with_the_brief(self):
        chat = ScriptedChat(label="worker", script=[ai("A summary.")])
        brief = brief_message(BRIEF, id="t1")
        messages = [brief, AIMessage("Reading.", id="a0"), *long_turns(1, 6),
                    HumanMessage("Now find Y.", id="h7")]

        sent, event = await summarized(chat, messages)

        written = f"{SUMMARY_PREFIX}\n\nA summary."
        assert blocks(sent[0]) == [BRIEF, written]
        assert sent[0].additional_kwargs == {"lc_source": "summarization"}
        assert blocks(event["summary_message"]) == [BRIEF, written]
        assert [m.text for m in sent].count(BRIEF) == 0  # not as a message of its own too
        assert sent[-1].text == "Now find Y."

    async def test_every_summary_of_the_thread_keeps_it(self):
        chat = ScriptedChat(label="worker", script=[ai("A summary."), ai("Another.")])
        messages = [brief_message(BRIEF, id="t1"), AIMessage("Reading.", id="a0"),
                    *long_turns(1, 6), HumanMessage("Now find Y.", id="h7")]
        _, first = await summarized(chat, messages)

        grown = [*messages, AIMessage("Y.", id="a7"), *long_turns(8, 13),
                 HumanMessage("Now find Z.", id="h14")]
        sent, second = await summarized(chat, grown, first)

        assert blocks(second["summary_message"]) == [BRIEF, f"{SUMMARY_PREFIX}\n\nAnother."]
        assert blocks(sent[0])[0] == BRIEF
        assert second["cutoff_index"] > first["cutoff_index"]

    async def test_a_brief_the_summary_keeps_is_not_restated(self):
        """An inheriting instance's history before its brief, summarized alone."""
        chat = ScriptedChat(label="worker", script=[ai("The history.")])
        brief = brief_message(BRIEF, id="t1")
        messages = [*long_turns(1, 6), brief, AIMessage("Short.", id="a7")]

        sent, event = await summarized(chat, messages)

        assert event is not None and brief in sent
        assert blocks(sent[0]) == [f"{SUMMARY_PREFIX}\n\nThe history."]

    async def test_a_thread_without_one_gets_deep_agents_own_summary(self):
        chat = ScriptedChat(label="worker", script=[ai("A summary.")])
        messages = [HumanMessage("Find X.", id="h0"), *long_turns(1, 6),
                    HumanMessage("Now find Y.", id="h7")]

        sent, event = await summarized(chat, messages)

        assert sent[0].content == f"{SUMMARY_PREFIX}\n\nA summary."
        assert event["summary_message"].content == f"{SUMMARY_PREFIX}\n\nA summary."


class TestALongBrief:
    BIG = "Brief. " + "word " * 12_600  # about 14,000 tokens; local:small summarizes at 16,150

    async def test_one_summary_not_one_per_call(self):
        """A brief near the trigger stays in the request after its summary: a
        later call whose summary would save little (under ``MIN_SAVING`` of
        the trigger) does not write another."""
        chat = ScriptedChat(label="worker", script=[ai(f"S{n}.") for n in range(1, 10)])
        messages = [brief_message(self.BIG, id="t1"), AIMessage("Reading.", id="a0"),
                    HumanMessage("Turn 1: " + "word " * 2_400, id="h1"), AIMessage("A1.", id="a1")]
        _, event = await summarized(chat, messages)
        written = [event is not None]
        for n in range(2, 10):
            messages = [*messages, HumanMessage(f"Short {n}. " + "word " * 200, id=f"h{n}"),
                        AIMessage(f"A{n}.", id=f"a{n}")]
            _, new = await summarized(chat, messages, event)
            written.append(new is not None)
            event = new or event

        assert written == [True] + [False] * 8
        assert blocks(event["summary_message"])[0] == self.BIG

    async def test_never_a_request_past_the_budget(self):
        """The skip is for requests under the input budget (here 19,000 tokens,
        the window not being known to the model's profile): once the brief and
        the steps pass it, a summary is written whatever it saves."""
        chat = ScriptedChat(label="worker", script=[ai(f"S{n}.") for n in range(1, 10)])
        middleware = build_summarization(chat, resolved("local:small", agent="worker"), context())
        budget = context_budget(resolved("local:small", agent="worker"), context()).budget
        messages = [brief_message(self.BIG, id="t1"), AIMessage("Reading.", id="a0"),
                    HumanMessage("Turn 1: " + "word " * 2_400, id="h1"), AIMessage("A1.", id="a1")]
        sent, event = await summarized(chat, messages)
        sizes = [middleware.token_counter(sent)]
        for n in range(2, 10):
            messages = [*messages, HumanMessage(f"Short {n}. " + "word " * 400, id=f"h{n}"),
                        AIMessage(f"A{n}.", id=f"a{n}")]
            sent, new = await summarized(chat, messages, event)
            sizes.append(middleware.token_counter(sent))
            event = new or event

        assert max(sizes) <= budget, sizes

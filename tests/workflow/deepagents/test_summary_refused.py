"""A summary the model wrote no text for is never applied, on Deep Agents.

Claude can end a call with ``stop_reason: "refusal"`` and no text, a summary
call included (seen live). Deep Agents would then store the empty summary in
place of the conversation it stood for, and the model would never see those
messages again. Instead nothing is summarized: the request goes on in full,
the user is told once a turn, and the next turn tries again; within the
turn, only once the range to summarize has grown. After a refused request,
one that cannot be summarized ends in the clear overflow message.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, overflow_message  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import build_summarization  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents, reporting  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai  # noqa: E402
from tests.workflow.deepagents.test_context import (  # noqa: E402,F401
    USER_TURN,
    _agent,
    _gemini_overflow,
    _manager,
    cache_probe,
    context,
    hermetic,
    resolved,
    small,
    turn,
)


async def _turns(settings, monkeypatch, script, count: int) -> tuple[list, ScriptedChat]:
    models = ScriptedModels(solo=script)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    manager = _manager(settings, [_agent("solo", "Answer.")])
    try:
        turns = [await turn(manager, f"Turn {n}: {USER_TURN}") for n in range(1, count + 1)]
    finally:
        await manager.cleanup()
    return turns, models.chats["solo"]


#: What the user reads, once a turn, when a summary could not be written.
SUMMARY_FAILED = "Could not summarize the earlier conversation; it is sent in full instead."


def _types(events) -> list:
    return [event.type for event in events]


class TestAnEmptySummary:
    @pytest.mark.parametrize("empty", ["", " \n "], ids=["empty", "blank"])
    async def test_it_is_not_applied_and_the_request_goes_on_in_full(
        self, small, monkeypatch, empty
    ):
        # Six questions of about 3,000 tokens pass the 16,150-token trigger
        # at the sixth; its summary call comes back empty.
        script = [ai(f"Answer {n}.") for n in range(1, 6)]
        script += [ai(empty), ai("Answer 6."), ai("A summary."), ai("Answer 7.")]
        turns, chat = await _turns(small, monkeypatch, script, 7)

        sixth, seventh = turns[5], turns[6]
        assert EventType.CONTEXT_TRIMMED not in _types(sixth)
        # Calls 0-4 answered turns 1-5, call 5 was the summary, call 6 the
        # answer: the whole conversation, from the first question on.
        answer = chat.conversation(6)
        assert len(answer) == 11
        assert answer[0].text.startswith("Turn 1:")
        assert [e.content for e in sixth if e.type is EventType.TEXT] == ["Answer 6."]
        warnings = [e for e in sixth if e.type is EventType.ERROR]
        assert [(e.content, e.metadata["error_code"], e.metadata["recoverable"])
                for e in warnings] == [(SUMMARY_FAILED, "summary_failed", True)]

        # The next turn summarizes again, from the first question: nothing lost.
        assert "Turn 1:" in chat.conversation(7)[0].text
        assert [e.metadata["source"] for e in seventh
                if e.type is EventType.CONTEXT_TRIMMED] == ["summarization"]
        assert EventType.ERROR not in _types(seventh)
        assert [e.content for e in seventh if e.type is EventType.TEXT] == ["Answer 7."]

    async def test_after_a_refused_request_it_ends_in_the_clear_message(
        self, small, monkeypatch
    ):
        # Two turns under the trigger, then the provider refuses the third
        # request as too long; Deep Agents summarizes to retry, and the
        # summary comes back empty.
        turns, chat = await _turns(small, monkeypatch, [
            ai("Answer 1."), ai("Answer 2."), _gemini_overflow(), ai(""),
        ], 3)

        third = turns[2]
        errors = [e for e in third if e.type is EventType.ERROR]
        assert [(e.content, e.metadata["error_code"]) for e in errors] == [
            (overflow_message("local:small"), OVERFLOW_ERROR_CODE)
        ]
        assert EventType.CONTEXT_TRIMMED not in _types(third)
        assert len(chat.requests) == 4  # the refused request and the summary; no resend


def is_summary_request(messages) -> bool:
    return len(messages) == 1 and "<messages>" in messages[0].text


class WritesNoSummaries(ScriptedChat):
    """Ends every summary call with no text, and answers the agent's calls
    from the script.

    Attributes:
        summaries: The messages of each summary call.
    """

    summaries: list[Any] = Field(default_factory=list)

    def _answer(self, messages):
        if is_summary_request(messages):
            self.summaries.append(list(messages))
            return ChatResult(generations=[ChatGeneration(message=ai(""))])
        return super()._answer(messages)


def _called(call_id: str, message_id: str) -> AIMessage:
    return AIMessage("", id=message_id, tool_calls=[
        {"name": "cache_probe", "args": {"x": call_id}, "id": call_id, "type": "tool_call"}
    ])


class TestWithinATurn:
    """Every model call of a turn past the trigger checks for a summary: one
    that wrote nothing would be paid for again at each tool step, over the
    same range."""

    async def test_it_is_tried_once_a_turn(self, small, monkeypatch):
        chat = WritesNoSummaries(label="solo", script=[
            *(ai(f"Answer {n}.") for n in range(1, 7)),
            *(ai(calls=[("cache_probe", {"x": str(n)})]) for n in (1, 2, 3)),
            ai("Answer 7."), ai("Answer 8."),
        ])
        models = ScriptedModels()
        models.chats["solo"] = chat
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = _manager(small, [_agent("solo", "Answer.", tools=[cache_probe])])
        texts = [f"Turn {n}: {USER_TURN}" for n in range(1, 7)]
        texts += ["Turn 7: look three times.", "Turn 8?"]
        turns, tried = [], []
        try:
            for text in texts:
                turns.append(await turn(manager, text))
                tried.append(len(chat.summaries))
        finally:
            await manager.cleanup()

        # The sixth turn passes the trigger; the seventh tries once for its
        # four model calls; the eighth tries once more.
        assert tried == [0, 0, 0, 0, 0, 1, 2, 3]
        for events in turns[5:]:
            assert [(e.content, e.metadata["error_code"]) for e in events
                    if e.type is EventType.ERROR] == [(SUMMARY_FAILED, "summary_failed")]
        # Its calls go on with the conversation in full.
        assert [e.content for e in turns[6] if e.type is EventType.TEXT] == ["Answer 7."]
        assert len(chat.requests) == 11
        assert chat.conversation(9)[0].text.startswith("Turn 1:")

    async def test_it_is_tried_again_once_its_range_has_grown(self):
        """A short tool result leaves the range as it was; a long one moves the
        kept tail past it, so the question and the result join the range: a
        fifth of the trigger and more."""
        chat = WritesNoSummaries(label="solo")
        middleware = build_summarization(chat, resolved("local:small", agent="solo"), context())
        history = []
        for n in range(1, 7):
            history += [HumanMessage(f"Turn {n}: {USER_TURN}", id=f"h{n}"),
                        AIMessage(f"Answer {n}.", id=f"a{n}")]
        asked = [*history, HumanMessage("Turn 7.", id="h7")]
        short = [*asked, _called("c1", "a7"),
                 ToolMessage("A short note.", tool_call_id="c1", id="t1")]
        grown = [*short, _called("c2", "a8"),
                 ToolMessage("word " * 3_000, tool_call_id="c2", id="t2"),
                 _called("c3", "a9"), ToolMessage("A short note.", tool_call_id="c3", id="t3")]

        async def handler(request):
            return "the answer"

        with reporting(LiveEvents()):
            for messages in (asked, short, grown):
                request = ModelRequest(model=chat, messages=messages, state={"messages": messages})
                assert await middleware.awrap_model_call(request, handler) == "the answer"

        assert len(chat.summaries) == 2
        first, second = (summary[0].text for summary in chat.summaries)
        assert "Turn 7." not in first
        assert "Turn 7." in second

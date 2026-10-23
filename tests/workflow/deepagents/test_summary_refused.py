"""A summary the model wrote no text for is never applied, on Deep Agents.

Claude can end a call with ``stop_reason: "refusal"`` and no text, a summary
call included (seen live). Deep Agents would then store the empty summary in
place of the conversation it stood for, and the model would never see those
messages again. Instead nothing is summarized: the request goes on in full,
the user is told once a turn, and the next call tries again. After a refused
request, one that cannot be summarized ends in the clear overflow message.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ModelRequest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, overflow_message  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import build_summarization  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai  # noqa: E402
from tests.workflow.deepagents.test_context import (  # noqa: E402,F401
    USER_TURN,
    _agent,
    _gemini_overflow,
    _manager,
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

    def test_the_sync_path_sends_the_request_in_full_too(self):
        chat = ScriptedChat(script=[ai("")])
        middleware = build_summarization(chat, resolved("local:small", agent="solo"), context())
        history = []
        for n in range(1, 7):
            history += [HumanMessage(f"Turn {n}: {USER_TURN}"), AIMessage(f"Answer {n}.")]
        history.append(HumanMessage("Turn 7."))
        request = ModelRequest(model=chat, messages=history, state={"messages": history})
        sent = []

        def handler(request):
            sent.append(request.messages)
            return "the answer"

        assert middleware.wrap_model_call(request, handler) == "the answer"
        assert len(chat.requests) == 1  # the summary call, which wrote nothing
        assert sent == [history]

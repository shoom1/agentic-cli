"""``compact()`` on Deep Agents: the main agent's conversation summarized now.

The manager keeps the root agent's summarization middleware
(``build_agent_tree(..., summarizers=...)``). ``compact`` reads the thread's
state, applies the summary it holds to get the conversation the model is
sent, and cuts it as ADK's ``/compact`` does: about the middleware's ``keep``
tokens kept word for word, the cut where a user's message starts, never
between a call and its results, never inside the newest turn. The summary
is written by a copy of the middleware's LangChain helper with the focus in
its prompt, under a run callback (so its start and usage are reported), and
stored as Deep Agents stores its own: a new ``_summarization_event``, written
as the model node would (``aupdate_state(..., as_node="model")``). The next
request applies it. Private Deep Agents methods; contract tests pin them.

Turns run the real manager on ``ScriptedModels``, on the 20,000-token
endpoint model of ``test_context.py``: summarize at 16,150 tokens, keep the
newest 1,938; each long question is about 3,000 tokens.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import httpx
import pytest

pytest.importorskip("deepagents")

from deepagents.middleware.summarization import (  # noqa: E402
    SUMMARIZATION_EVENT_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
)
from langchain.agents.middleware.summarization import (  # noqa: E402
    SummarizationMiddleware as LangChainSummarization,
)
from langchain_core.exceptions import ContextOverflowError, ModelInvalidRequestError  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.context import CompactResult  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.event_processor import RunCallback  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.deepagents import summarization  # noqa: E402
from agentic_cli.workflow.deepagents.summarization import (  # noqa: E402
    ReportingSummarizationMiddleware,
)
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.context import is_context_overflow  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.compacting import compact_now  # noqa: E402
from pydantic import Field  # noqa: E402

from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai  # noqa: E402
from tests.workflow.deepagents.test_context import (  # noqa: E402,F401
    SMALL,
    TEAM,
    USER_TURN,
    GatedChat,
    _agent,
    _build,
    _manager,
    context,
    created,
    hermetic,
    pumps,
    resolved,
    small,
    turn,
)

#: Three long questions, each under the trigger: nothing is summarized on its own.
THREE = [f"Turn {n}: {USER_TURN}" for n in (1, 2, 3)]
SUMMARY = f"{SUMMARY_PREFIX}\n\nA summary."


def solo(monkeypatch, settings, script: list) -> tuple[DeepAgentsWorkflowManager, ScriptedChat]:
    """A one-agent manager on ``script`` (summary calls answer from it too)."""
    models = ScriptedModels(solo=script)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return _manager(settings, [_agent("solo", "Answer.")]), models.chats["solo"]


def gated(monkeypatch, settings, script: list, gate_at: int):
    chat = GatedChat(label="solo", script=script, gate_at=gate_at)
    models = ScriptedModels()
    models.chats["solo"] = chat
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return _manager(settings, [_agent("solo", "Answer.")]), chat


async def state_of(manager, session: str = "s") -> dict:
    return (await manager._root.aget_state(run_config(manager.session_ref(session, "u")))).values


def answers(script_end: int) -> list:
    return [ai(f"Answer {n}.") for n in range(1, script_end + 1)]


def is_summary_request(messages) -> bool:
    return len(messages) == 1 and "<messages>" in messages[0].text


#: Set while ``compact_probe_wait`` waits.
WAITING: list[asyncio.Event] = []


@register_tool(capabilities=EXEMPT)
async def compact_probe_wait() -> dict:
    """Waits until cancelled, as a long tool does (an exempt test tool)."""
    WAITING[-1].set()
    await asyncio.Event().wait()
    return {"success": True}


class TestCompactNow:
    async def test_the_earlier_turns_are_summarized_now_and_sent_from_then_on(
        self, small, monkeypatch
    ):
        manager, chat = solo(monkeypatch, small, [*answers(3), ai("A summary."), ai("Answer 4.")])
        try:
            for text in THREE:
                await turn(manager, text)
            result, _ = await compact_now(manager)
            state = await state_of(manager)
            await turn(manager, "Turn 4?")
            recent = await manager.recent_messages("s", 50, user_id="u")
        finally:
            await manager.cleanup()

        # Turns one and two are summarized; the third, kept word for word, is the tail.
        summary_request = chat.requests[3]
        assert is_summary_request(summary_request)
        assert "Turn 1:" in summary_request[0].text and "Turn 3:" not in summary_request[0].text
        assert isinstance(result, CompactResult)
        assert result.messages_summarized == 4
        assert 0 < result.tokens_after < result.tokens_before
        event = state[SUMMARIZATION_EVENT_KEY]
        assert (event["cutoff_index"], event["file_path"]) == (4, None)
        assert event["summary_message"].text == SUMMARY
        # The next request: the summary, once, then the kept turn and the new one.
        fourth = chat.conversation(4)
        assert [m.text[:9] for m in fourth] == [SUMMARY[:9], "Turn 3: w", "Answer 3.", "Turn 4?"]
        assert sum(SUMMARY_PREFIX in m.text for m in fourth) == 1
        # The checkpoint keeps every message.
        assert len(recent) == 8

    async def test_the_event_records_how_many_messages_it_was_written_on(
        self, small, monkeypatch
    ):
        # Deep Agents 0.7.30's created_at_index: the messages a summary kept
        # from before it lose their reasoning, which Claude binds to the
        # conversation before them. An older Deep Agents ignores the key.
        manager, _ = solo(monkeypatch, small, [*answers(3), ai("A summary.")])
        try:
            for text in THREE:
                await turn(manager, text)
            await compact_now(manager)
            state = await state_of(manager)
        finally:
            await manager.cleanup()

        assert state[SUMMARIZATION_EVENT_KEY]["created_at_index"] == len(state["messages"]) == 6

    async def test_the_focus_reaches_the_summary_prompt(self, small, monkeypatch):
        """Braces in the focus reach the model as written (the helper fills
        its prompt with ``str.format``)."""
        manager, chat = solo(monkeypatch, small, [*answers(3), ai("A summary.")])
        try:
            for text in THREE:
                await turn(manager, text)
            await compact_now(manager, "keep the {API} decisions")
        finally:
            await manager.cleanup()

        prompt = chat.requests[3][0].text
        assert "Focus the summary on: keep the {API} decisions" in prompt
        assert prompt.index("Focus the summary on:") < prompt.index("<messages>")

    async def test_it_reports_as_a_summary_in_a_turn_does(self, small, monkeypatch):
        manager, _ = solo(monkeypatch, small, [*answers(3), ai("A summary.")])
        try:
            for text in THREE:
                await turn(manager, text)
            result, events = await compact_now(manager)
        finally:
            await manager.cleanup()

        assert [event.type for event in events] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED
        ]
        summarizing, usage, trimmed = events
        assert summarizing.metadata["agent"] == "solo"
        assert (usage.metadata["purpose"], usage.metadata["agent"], usage.metadata["model"],
                usage.metadata["context_trigger"], usage.metadata["window_known"]) == (
            "summarization", "solo", "local:small", 16_150, True
        )
        assert (trimmed.metadata["source"], trimmed.metadata["agent"]) == ("compact", "solo")
        assert (trimmed.metadata["messages_before"], trimmed.metadata["messages_after"]) == (6, 3)
        assert (trimmed.metadata["messages_summarized"], trimmed.metadata["tokens_before"],
                trimmed.metadata["tokens_after"]) == (
            result.messages_summarized, result.tokens_before, result.tokens_after
        )

    async def test_a_short_conversation_has_nothing_to_summarize(self, small, monkeypatch):
        manager, chat = solo(monkeypatch, small, answers(2))
        try:
            for text in ["Hello.", "And again?"]:
                await turn(manager, text)
            result, events = await compact_now(manager)
            state = await state_of(manager)
        finally:
            await manager.cleanup()

        assert (result, events) == (None, [])
        assert len(chat.requests) == 2  # no summary call
        assert SUMMARIZATION_EVENT_KEY not in state

    async def test_before_the_first_turn_there_is_nothing_to_summarize(
        self, small, monkeypatch
    ):
        manager, chat = solo(monkeypatch, small, [])
        try:
            result, events = await compact_now(manager)
            exists = await manager.session_exists("s", user_id="u")
            listed = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert (result, events) == (None, [])
        assert chat.requests == []
        assert (exists, listed) == (False, [])  # no conversation is started

    async def test_ctrl_c_during_the_summary_writes_nothing(self, small, monkeypatch):
        manager, chat = gated(monkeypatch, small, [*answers(3), ai("A summary.")], gate_at=4)
        seen: list = []
        try:
            for text in THREE:
                await turn(manager, text)
            compacting = asyncio.create_task(compact_now(manager, events=seen))
            await asyncio.wait_for(chat._started.wait(), 5)
            await asyncio.sleep(0.05)
            while_summarizing = [event.type for event in seen]
            compacting.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(compacting, 5)
            left = pumps()
            after_cancel = await state_of(manager)
            chat._release.set()
            # The turn is free: a /compact again summarizes.
            again, _ = await asyncio.wait_for(compact_now(manager), 5)
        finally:
            chat._release.set()
            await manager.cleanup()

        assert while_summarizing == [EventType.CONTEXT_SUMMARIZING]
        assert left == []
        assert SUMMARIZATION_EVENT_KEY not in after_cancel
        assert again.messages_summarized == 4

    async def test_it_waits_for_a_running_turn(self, small, monkeypatch):
        """Turn admission: a /compact never interleaves with a turn (or a job's
        resume), so it summarizes the turn that ran before it, all of it."""
        manager, chat = gated(monkeypatch, small, [*answers(3), ai("A summary.")], gate_at=3)
        try:
            for text in THREE[:2]:
                await turn(manager, text)
            third = asyncio.create_task(turn(manager, THREE[2]))
            await asyncio.wait_for(chat._started.wait(), 5)
            compacting = asyncio.create_task(compact_now(manager))
            await asyncio.sleep(0.1)
            waited = not compacting.done() and len(chat.requests) == 2
            chat._release.set()
            answered = [e.content for e in await asyncio.wait_for(third, 5)
                        if e.type is EventType.TEXT]
            result, _ = await asyncio.wait_for(compacting, 5)
        finally:
            chat._release.set()
            await manager.cleanup()

        assert waited
        assert answered == ["Answer 3."]
        assert is_summary_request(chat.requests[3])
        assert result.messages_summarized == 4

    async def test_a_summary_that_fails_is_raised_and_nothing_is_written(
        self, small, monkeypatch
    ):
        """Asked for, a summary that fails is the caller's to report: it is
        raised, with no warning of its own."""
        refused = ModelInvalidRequestError("max_tokens: 8000 > 4096")
        manager, chat = solo(monkeypatch, small, [*answers(3), refused])
        seen: list = []
        try:
            for text in THREE:
                await turn(manager, text)
            with pytest.raises(ModelInvalidRequestError, match="max_tokens: 8000 > 4096"):
                await compact_now(manager, events=seen)
            state = await state_of(manager)
        finally:
            await manager.cleanup()

        assert len(chat.requests) == 4  # one summary call, not retried
        assert [event.type for event in seen] == [EventType.CONTEXT_SUMMARIZING]
        assert SUMMARIZATION_EVENT_KEY not in state

    async def test_an_empty_summary_is_a_failure(self, small, monkeypatch):
        manager, _ = solo(monkeypatch, small, [*answers(3), ai("  ")])
        try:
            for text in THREE:
                await turn(manager, text)
            with pytest.raises(ValueError, match="empty summary"):
                await compact_now(manager)
            state = await state_of(manager)
        finally:
            await manager.cleanup()

        assert SUMMARIZATION_EVENT_KEY not in state

    async def test_a_second_compact_after_a_turn_replaces_the_first(self, small, monkeypatch):
        manager, chat = solo(monkeypatch, small, [
            *answers(3), ai("First summary."), ai("Answer 4."), ai("Answer 5."),
            ai("Second summary."), ai("Answer 6."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            await compact_now(manager)
            for text in [f"Turn 4: {USER_TURN}", "Turn 5?"]:
                await turn(manager, text)
            second, _ = await compact_now(manager)
            state = await state_of(manager)
            await turn(manager, "Turn 6?")
        finally:
            await manager.cleanup()

        # The second summary reads the first, and the turn kept after it.
        prompt = chat.requests[6][0].text
        assert "First summary." in prompt and "Turn 3:" in prompt and "Turn 4:" not in prompt
        assert second.messages_summarized == 3  # the first summary, Turn 3, Answer 3
        assert state[SUMMARIZATION_EVENT_KEY]["cutoff_index"] == 6
        # The next request holds exactly one summary, the second.
        sixth = chat.conversation(7)
        assert [m.text[:9] for m in sixth] == [
            SUMMARY_PREFIX[:9], "Turn 4: w", "Answer 4.", "Turn 5?", "Answer 5.", "Turn 6?"
        ]
        assert sixth[0].text == f"{SUMMARY_PREFIX}\n\nSecond summary."
        assert sum(SUMMARY_PREFIX in m.text for m in sixth) == 1
        assert "First summary." not in json.dumps([m.text for m in sixth])

    async def test_on_a_coordinator_the_thread_settles_and_sessions_see_it(
        self, small, monkeypatch
    ):
        """Written as the model node on a coordinator (its stack adds the
        delegation middleware), the summary leaves no pending step, and its
        checkpoint names the session and user, so /sessions moves its time."""
        models = ScriptedModels(
            lead=[*answers(3), ai("A summary."), ai("Answer 4.")], helper=[],
        )
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = _manager(small, [
            _agent("lead", "Lead.", sub_agents=["helper"]),
            _agent("helper", "Help.", description="Helps."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            (before,) = await manager.list_sessions(user_id="u")
            await asyncio.sleep(0.01)
            result, _ = await compact_now(manager)
            snapshot = await manager._root.aget_state(run_config(manager.session_ref("s", "u")))
            (after,) = await manager.list_sessions(user_id="u")
            fourth = await turn(manager, "Turn 4?")
        finally:
            await manager.cleanup()

        assert result is not None
        assert snapshot.next == ()
        assert (snapshot.metadata["app_name"], snapshot.metadata["user_id"],
                snapshot.metadata["session_id"]) == (manager.app_name, "u", "s")
        assert after.updated > before.updated
        assert after.turns == before.turns == 3
        assert [e.content for e in fourth if e.type is EventType.TEXT] == ["Answer 4."]

    async def test_after_ctrl_c_mid_tool_the_next_turn_still_runs_once(
        self, small, monkeypatch
    ):
        """A call Ctrl+C left without a result keeps the thread's next step at
        the tools node; the summary written as the model node leaves it so,
        and the next turn answers the call (PatchToolCallsMiddleware) rather
        than run it again."""
        WAITING.append(asyncio.Event())
        models = ScriptedModels(solo=[
            ai("Answer 1."), ai("Answer 2."), ai("", [("compact_probe_wait", {}, "call_w")]),
            ai("A summary."), ai("Answer 4."),
        ])
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = _manager(small, [_agent("solo", "Answer.", tools=[compact_probe_wait])])
        try:
            for text in THREE[:2]:
                await turn(manager, text)
            waiting = asyncio.create_task(turn(manager, "Wait, please."))
            await asyncio.wait_for(WAITING[-1].wait(), 5)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, 5)
            result, _ = await compact_now(manager)
            fourth = await turn(manager, "Turn 4?")
        finally:
            await manager.cleanup()

        chat = models.chats["solo"]
        assert result.messages_summarized == 2
        assert [e.content for e in fourth if e.type is EventType.TEXT] == ["Answer 4."]
        assert [e for e in fourth if e.type is EventType.TOOL_CALL] == []  # not run again
        last = chat.conversation(4)
        assert [type(m) for m in last] == [
            HumanMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, ToolMessage,
            HumanMessage,
        ]
        assert sum(SUMMARY_PREFIX in m.text for m in last) == 1



# --- the count reported before a /compact (review I1) -------------------------------------


def reported(text: str, total: int) -> AIMessage:
    """An answer as a real provider's: its usage, and its provider named
    (LangChain trusts a reported count only from the model's own provider)."""
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": total - 100, "output_tokens": 100, "total_tokens": total},
        response_metadata={"model_provider": "scripted"},
    )


#: Three answers; the third call's total (prompt and answer) is past the
#: 16,150-token trigger, as it is when ctx reads near the top or the answer
#: was long.
NEAR_THE_TRIGGER = [
    reported("Answer 1.", 4_000), reported("Answer 2.", 7_500), reported("Answer 3.", 16_600),
]


class TestTheCountBeforeACompact:
    async def test_without_a_compact_the_next_turn_summarizes(self, small, monkeypatch):
        """The control: past the trigger by the reported count, the next turn
        summarizes (LangChain's reported-tokens check is live here)."""
        manager, chat = solo(monkeypatch, small, [
            *NEAR_THE_TRIGGER, ai("A summary."), reported("Answer 4.", 5_000),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            fourth = await turn(manager, "Turn 4?")
        finally:
            await manager.cleanup()

        assert is_summary_request(chat.requests[3])
        assert [e.type for e in fourth].count(EventType.CONTEXT_TRIMMED) == 1

    async def test_after_a_compact_the_next_turn_is_not_summarized_again(
        self, small, monkeypatch
    ):
        """The newest answer's count is the prompt the summary replaced: no
        evidence, as on ADK, whose engine stops at a summary. The next turn
        sends the focused summary instead of summarizing it again."""
        manager, chat = solo(monkeypatch, small, [
            *NEAR_THE_TRIGGER, ai("A focused summary."),
            reported("Answer 4.", 5_000), reported("Answer 5.", 17_000), ai("A later summary."),
            ai("Answer 6."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            result, _ = await compact_now(manager, "keep the decisions")
            fourth = await turn(manager, "Turn 4?")
            state = await state_of(manager)
            fifth = await turn(manager, "Turn 5?")
            sixth = await turn(manager, "Turn 6?")
        finally:
            await manager.cleanup()

        assert result is not None and is_summary_request(chat.requests[3])
        assert not is_summary_request(chat.requests[4]), "summarized again right after /compact"
        assert chat.conversation(4)[0].text == f"{SUMMARY_PREFIX}\n\nA focused summary."
        assert EventType.CONTEXT_TRIMMED not in [e.type for e in fourth]
        # The mark: the newest answer the summary was written after.
        mark = state[SUMMARIZATION_EVENT_KEY]["summary_message"].response_metadata["agentic_cli"]
        assert mark == {"kind": "compact", "after": chat.conversation(4)[2].id}
        assert EventType.CONTEXT_TRIMMED not in [e.type for e in fifth]
        # Once a newer answer reports a count, it is evidence again.
        assert is_summary_request(chat.requests[6])
        assert [e.type for e in sixth].count(EventType.CONTEXT_TRIMMED) == 1


# --- a summary the model refuses as too long (review I2) -----------------------------------


def openrouter_overflow() -> Exception:
    """ChatOpenRouter's error: the openrouter SDK's, which LangChain maps to nothing."""
    from openrouter.errors import OpenRouterError

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return OpenRouterError(
        "This endpoint's maximum context length is 32768 tokens. However, you requested "
        "about 41000 tokens (33000 of text input, 8000 in the output).",
        raw_response=httpx.Response(400, request=request),
    )


def server_overflow() -> Exception:
    """An OpenAI-compatible server's (vLLM) 400, as langchain-openai maps it."""
    from langchain_openai.chat_models.base import OpenAIInvalidRequestError

    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    message = ("This model's maximum context length is 32768 tokens. However, you requested "
               "41000 tokens (33000 in the messages, 8000 in the completion).")
    return OpenAIInvalidRequestError(
        message=message, response=httpx.Response(400, request=request),
        body={"message": message, "type": "BadRequestError", "code": 400},
    )


def langchain_overflow() -> Exception:
    return ContextOverflowError("prompt is too long: 41000 tokens > 32768 maximum")


class TooLongForSummaries(ScriptedChat):
    """Refuses a summary request as too long unless its middle was cut out
    (LangChain's overflow retry); answers everything else from the script."""

    error: Any = None
    refused: list = Field(default_factory=list)

    def _get_ls_params(self, stop=None, **kwargs):
        return {**super()._get_ls_params(stop=stop, **kwargs), "ls_provider": "scripted"}

    def _answer(self, messages):
        if is_summary_request(messages) and "middle of conversation omitted" not in messages[0].text:
            self.refused.append(messages)
            raise self.error()
        return super()._answer(messages)


def refusing(monkeypatch, settings, error, script: list):
    chat = TooLongForSummaries(label="solo", error=error, script=script)
    models = ScriptedModels()
    models.chats["solo"] = chat
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return _manager(settings, [_agent("solo", "Answer.")]), chat


OVERFLOWS = [openrouter_overflow, server_overflow]


class TestASummaryTooLongForTheModel:
    @pytest.mark.parametrize("error", OVERFLOWS, ids=["openrouter", "server"])
    async def test_a_compact_is_retried_smaller(self, small, monkeypatch, error):
        """After a switch to a smaller model, or on a server whose window is
        not declared: our table knows the refusal, which LangChain's partner
        package does not map, and the summary is retried once, smaller
        (OpenRouter's is no longer sent three times first)."""
        assert is_context_overflow(error())
        manager, chat = refusing(monkeypatch, small, error, [
            ai("Answer 1."), ai("Answer 2."), ai("Answer 3."), ai("A shorter summary."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            result, _ = await compact_now(manager)
        finally:
            await manager.cleanup()

        assert result is not None
        assert len(chat.refused) == 1  # refused once, then retried with the middle cut out
        assert "middle of conversation omitted" in chat.requests[3][0].text

    @pytest.mark.parametrize("error", OVERFLOWS, ids=["openrouter", "server"])
    async def test_a_summary_in_a_turn_is_retried_smaller(self, small, monkeypatch, error):
        """The same refusal in a turn: the summary is retried smaller and the
        turn answers, where it ended in "This conversation no longer fits …"."""
        manager, chat = refusing(monkeypatch, small, error, [
            *NEAR_THE_TRIGGER, ai("A shorter summary."), ai("Answer 4."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            fourth = await turn(manager, "Turn 4?")
        finally:
            await manager.cleanup()

        assert [e.content for e in fourth if e.type is EventType.ERROR] == []
        assert [e.content for e in fourth if e.type is EventType.TEXT] == ["Answer 4."]
        assert len(chat.refused) == 1

    async def test_langchains_own_overflow_is_retried_smaller_as_before(
        self, small, monkeypatch
    ):
        """The control: an overflow the partner package maps (Anthropic's,
        OpenAI's code, Gemini's text) was always retried smaller."""
        manager, chat = refusing(monkeypatch, small, langchain_overflow, [
            ai("Answer 1."), ai("Answer 2."), ai("Answer 3."), ai("A shorter summary."),
        ])
        try:
            for text in THREE:
                await turn(manager, text)
            result, _ = await compact_now(manager)
        finally:
            await manager.cleanup()

        assert result is not None and len(chat.refused) == 1


class Failing(ScriptedChat):
    """Raises ``error`` on every call."""

    error: Any = None

    def _answer(self, messages):
        self.requests.append(list(messages))
        raise self.error


class TestOverflowMapped:
    @pytest.mark.parametrize("error", OVERFLOWS, ids=["openrouter", "server"])
    async def test_a_known_overflow_is_langchains(self, error):
        original = error()
        mapped = summarization.OverflowMapped(Failing(error=original))

        with pytest.raises(ContextOverflowError) as raised:
            await mapped.ainvoke("Summarize.")
        with pytest.raises(ContextOverflowError) as raised_sync:
            mapped.invoke("Summarize.")

        assert raised.value.__cause__ is original and raised_sync.value.__cause__ is original
        assert str(original) in str(raised.value)

    async def test_anything_else_is_unchanged(self):
        other, overflow = ValueError("no"), langchain_overflow()

        for error in (other, overflow):
            mapped = summarization.OverflowMapped(Failing(error=error))
            with pytest.raises(type(error)) as raised:
                await mapped.ainvoke("Summarize.")
            assert raised.value is error
            with pytest.raises(type(error)) as raised_sync:
                mapped.invoke("Summarize.")
            assert raised_sync.value is error

    async def test_an_answer_passes_through(self):
        mapped = summarization.OverflowMapped(ScriptedChat(script=[ai("A summary.")]))

        assert (await mapped.ainvoke("Summarize.")).text == "A summary."

    def test_every_agents_summaries_go_through_it_inside_the_retry(self):
        helper = _middleware()._lc_helper

        assert isinstance(helper._summary_model.bound, summarization.OverflowMapped)
        assert helper._summary_model.bound.model is helper._summary_chat_model


# --- the cut -----------------------------------------------------------------------------


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _summary(text: str = "S.") -> HumanMessage:
    return HumanMessage(
        content=f"{SUMMARY_PREFIX}\n\n{text}", additional_kwargs={"lc_source": "summarization"}
    )


def _calls(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "probe", "args": {}, "id": call_id, "type": "tool_call"}
    ])


LONG = "word " * 800  # 4,000 characters, 1,000 tokens at 4 characters a token


class TestTheCut:
    def test_it_keeps_the_tail_and_cuts_where_a_turn_starts(self):
        messages = [_human(f"One {LONG}"), AIMessage("A1"), _human(f"Two {LONG}"),
                    AIMessage("A2"), _human("Three?"), AIMessage("A3")]

        # About 1,000 tokens kept: back from A3 into Two, whose turn starts at 2.
        assert summarization.compact_cutoff(messages, keep_tokens=500, kind=ProviderKind.OPENAI) == 2

    def test_the_newest_turn_is_never_summarized(self):
        messages = [_human(f"One {LONG}"), AIMessage("A1"), _human(f"Two {LONG}"),
                    AIMessage(LONG)]

        assert summarization.compact_cutoff(messages, keep_tokens=10, kind=ProviderKind.OPENAI) == 2

    def test_all_of_it_in_the_tail_is_nothing_to_summarize(self):
        messages = [_human("One?"), AIMessage("A1"), _human("Two?"), AIMessage("A2")]

        assert summarization.compact_cutoff(messages, keep_tokens=1_000, kind=ProviderKind.OPENAI) is None

    def test_the_summary_alone_before_the_cut_is_nothing_to_summarize(self):
        messages = [_summary(), _human(f"One {LONG}"), AIMessage("A1"), _human("Two?")]

        # Only "Two?" kept: the summary so far and the turn after it.
        assert summarization.compact_cutoff(messages, keep_tokens=1, kind=ProviderKind.OPENAI) == 3
        # The long turn kept too: the summary so far alone is no saving.
        assert summarization.compact_cutoff(messages, keep_tokens=10, kind=ProviderKind.OPENAI) is None

    def test_a_call_is_never_parted_from_its_result(self):
        """A message between a call and its result (no such state arises; the
        cut checks anyway) moves the cut back to a turn that parts nothing."""
        messages = [
            _human(f"One {LONG}"), AIMessage("A1"), _human(f"Two {LONG}"), _calls("c1"),
            _human(f"Interjected {LONG}"), ToolMessage("done", tool_call_id="c1"),
            AIMessage("A2"), _human("Three?"),
        ]

        # The walk stops in the interjected turn (4), where c1 is still open.
        assert summarization.compact_cutoff(messages, keep_tokens=1_000, kind=ProviderKind.OPENAI) == 2

    def test_a_call_without_a_result_is_never_summarized(self):
        """A call a cancelled turn left open is answered before the next run
        (PatchToolCallsMiddleware), right after it: summarized, its answer
        would open the next request."""
        messages = [_human(f"One {LONG}"), _calls("c1"), _human(f"Two {LONG}"),
                    AIMessage("A2"), _human("Three?")]

        assert summarization.compact_cutoff(messages, keep_tokens=1_000, kind=ProviderKind.OPENAI) is None

    def test_claude_counts_more_tokens_a_character(self):
        messages = [_human(f"One {LONG}"), AIMessage("A1"), _human(f"Two {LONG}"),
                    AIMessage("A2"), _human(f"Three {LONG}"), AIMessage("A3")]

        # 1,100 tokens kept: one long question at 4 characters a token, not
        # at Claude's 3.3.
        assert summarization.compact_cutoff(messages, keep_tokens=1_100, kind=ProviderKind.OPENAI) == 2
        assert summarization.compact_cutoff(messages, keep_tokens=1_100, kind=ProviderKind.ANTHROPIC) == 4


# --- the builder keeps every agent's middleware -------------------------------------------


class TestTheBuilder:
    def test_each_agents_summarization_is_handed_back(self, created, tmp_path, monkeypatch):
        from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree

        monkeypatch.chdir(tmp_path)
        summarizers: dict = {}
        models = {"lead": resolved("anthropic:claude-sonnet-4-5", "lead"),
                  "helper": resolved("google:gemini-2.5-flash", "helper")}
        build_agent_tree(
            TEAM,
            chat_models={c.name: ScriptedChat(label=c.name) for c in TEAM},
            resolved=models, tools={}, prompts={c.name: "p" for c in TEAM},
            settings=context(), checkpointer=None, summarizers=summarizers,
        )

        assert set(summarizers) == {"lead", "helper"}
        for name in ("lead", "helper"):
            in_stack = [m for m in created[name]["middleware"]
                        if isinstance(m, ReportingSummarizationMiddleware)]
            assert in_stack == [summarizers[name]]

    async def test_the_manager_keeps_the_roots(self, small, monkeypatch):
        manager, _ = solo(monkeypatch, small, [])
        try:
            await manager.initialize_services()
            kept = manager._root_summarization
            supports = manager.supports_compact
        finally:
            await manager.cleanup()

        assert isinstance(kept, ReportingSummarizationMiddleware)
        assert kept.agent == "solo"
        assert supports is True
        assert manager._root_summarization is None  # released with the agents


# --- the Deep Agents it rests on ------------------------------------------------------------


class TestTheDeepAgentsItRestsOn:
    """Each private piece of Deep Agents' summarization that ``compact`` uses."""

    def test_the_event_lives_in_state_under_its_key(self):
        assert SUMMARIZATION_EVENT_KEY == "_summarization_event"
        # Deep Agents 0.7.30 added created_at_index, which an event may lack.
        assert set(SummarizationEvent.__annotations__) in (
            {"cutoff_index", "summary_message", "file_path"},
            {"cutoff_index", "summary_message", "file_path", "created_at_index"},
        )

    def test_an_event_is_applied_as_the_summary_and_the_messages_after_its_cut(self):
        apply = SummarizationMiddleware._apply_event_to_messages
        messages = [_human("One?"), AIMessage("A1"), _human("Two?"), AIMessage("A2")]
        summary = _summary()

        assert list(inspect.signature(apply).parameters) == ["messages", "event"]
        assert apply(messages, None) == messages
        assert apply(messages, {"cutoff_index": 2, "summary_message": summary,
                                "file_path": None}) == [summary, *messages[2:]]

    def test_a_cut_of_the_applied_messages_is_moved_to_the_states(self):
        cutoff = SummarizationMiddleware._compute_state_cutoff

        assert list(inspect.signature(cutoff).parameters) == ["event", "effective_cutoff"]
        assert cutoff(None, 4) == 4
        # The summary stands at 0 in the applied messages.
        assert cutoff({"cutoff_index": 4, "summary_message": _summary(),
                       "file_path": None}, 3) == 6

    def test_the_summary_message_without_an_archive(self):
        middleware = _middleware()
        (message,) = middleware._build_new_messages_with_path("A summary.", None)

        assert isinstance(message, HumanMessage)
        assert message.content == f"{SUMMARY_PREFIX}\n\nA summary."
        assert message.additional_kwargs == {"lc_source": "summarization"}

    def test_the_langchain_helper_writes_the_summary_from_its_prompt(self):
        helper = _middleware()._lc_helper

        assert isinstance(helper, LangChainSummarization)
        assert isinstance(helper.summary_prompt, str) and "{messages}" in helper.summary_prompt
        assert inspect.iscoroutinefunction(helper._acreate_summary)
        assert list(inspect.signature(helper._acreate_summary).parameters) == [
            "messages_to_summarize"
        ]

    async def test_a_summary_is_decided_by_should_summarize(self, small, monkeypatch):
        """Deep Agents decides with ``_should_summarize(messages, total_tokens)``,
        on the messages it sends (the summary so far first) and its own
        estimate of the request; the middleware's override reads both."""
        calls: list = []
        decide = ReportingSummarizationMiddleware._should_summarize

        def spy(self, messages, total_tokens):
            calls.append((list(messages), total_tokens))
            return decide(self, messages, total_tokens)

        monkeypatch.setattr(ReportingSummarizationMiddleware, "_should_summarize", spy)
        manager, _ = solo(monkeypatch, small, answers(1))
        try:
            await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        assert list(inspect.signature(SummarizationMiddleware._should_summarize).parameters) == [
            "self", "messages", "total_tokens"
        ]
        ((messages, total),) = calls
        assert [m.text for m in messages] == ["Hello."]
        assert isinstance(total, int) and total > 0

    def test_the_helpers_summary_model_wraps_its_chat_model(self):
        helper = _middleware()._lc_helper

        assert hasattr(helper, "_summary_model") and hasattr(helper, "_summary_chat_model")
        assert helper._summary_chat_model is helper.model

    async def test_a_too_long_summary_is_retried_with_the_middle_cut_out(self):
        chat = ScriptedChat(label="solo", script=[langchain_overflow(), ai("A shorter one.")])
        helper = _middleware(chat)._lc_helper

        summary = await helper._acreate_summary([_human("One?"), AIMessage("A1")])

        assert hasattr(helper, "_overflow_retry_prompt")
        assert summary == "A shorter one."
        assert "middle of conversation omitted" in chat.requests[1][0].text

    async def test_the_helpers_call_is_a_summary_the_run_callback_reports(self):
        """The call carries ``lc_source="summarization"``, so a run callback
        in the context reports its start and usage."""
        from langchain_core.runnables.config import set_config_context

        from agentic_cli.workflow import live_events

        chat = ScriptedChat(label="solo", script=[ai("A summary.")])
        helper = _middleware(chat)._lc_helper
        live = live_events.LiveEvents()
        callback = RunCallback(
            root="solo", agent_models={"solo": resolved("local:small", "solo")}, live=live
        )
        with set_config_context({"callbacks": [callback]}) as context:
            summary = await asyncio.create_task(
                helper._acreate_summary([_human("One?"), AIMessage("A1")]), context=context
            )
        reported = []
        while not live._queue.empty():
            reported.append(live._queue.get_nowait()[1].type)

        assert summary == "A summary."
        assert "One?" in chat.requests[0][0].text
        assert reported == [EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE]


def _middleware(chat: ScriptedChat | None = None) -> ReportingSummarizationMiddleware:
    from agentic_cli.workflow.deepagents.summarization import build_summarization

    model = resolved("local:small", "solo")
    return build_summarization(chat or ScriptedChat(label="solo"), model, context())

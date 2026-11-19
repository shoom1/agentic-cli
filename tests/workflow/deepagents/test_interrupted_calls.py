"""Deep Agents: a provider's reused call id, and the calls a history leaves open.

A provider that numbers its call ids per response (``call_0`` in every one)
reuses an id the agent's conversation holds. ``InterruptedCallsMiddleware``
gives such a call an id of its own as the answer arrives, wherever the
answer names it, so its result, the record and every later request pair it
by that id. A conversation stored before may still hold an id twice: a call
is answered by a result of its id that follows its model message before the
next one. What the middleware does alone; whole turns, on both backends, are
in the contract suite (``TestAReusedCallId``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ModelRequest, ModelResponse  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)

from agentic_cli.workflow.deepagents.middleware import InterruptedCallsMiddleware  # noqa: E402
from agentic_cli.workflow.interrupts import INTERRUPTED, OpenCalls, interrupted  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat  # noqa: E402


def _call(name: str, call_id: str) -> dict[str, Any]:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def _answer(*made: tuple[str, str]) -> AIMessage:
    """An answer that makes ``made`` calls, ``(name, id)`` each, as the adapters
    write one: Claude's ``tool_use`` blocks, Chat Completions' raw calls, and
    LangChain's tool calls."""
    return AIMessage(
        content=[{"type": "tool_use", "id": call_id, "name": name, "input": {}}
                 for name, call_id in made],
        additional_kwargs={"tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for name, call_id in made
        ]},
        tool_calls=[_call(name, call_id) for name, call_id in made],
    )


def _ids(message: AIMessage) -> dict[str, list[str]]:
    """Each place ``message`` names its calls' ids."""
    return {
        "tool_calls": [call["id"] for call in message.tool_calls],
        "content": [block["id"] for block in message.content],
        "additional_kwargs": [call["id"] for call in message.additional_kwargs["tool_calls"]],
    }


async def _arrived(
    history: list[AnyMessage], answer: AIMessage, *, sent: list[AnyMessage] | None = None
) -> AIMessage:
    """``answer`` as it leaves the middleware, to an agent whose state holds
    ``history`` and whose request sends ``sent`` (all of it by default)."""
    request = ModelRequest(
        model=ScriptedChat(), messages=history if sent is None else sent,
        state={"messages": history},
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[answer])

    response = await InterruptedCallsMiddleware(OpenCalls(), root=True).awrap_model_call(
        request, handler
    )
    (message,) = response.result
    return message


#: A conversation whose first turn's ``call_0`` has its result.
FIRST_TURN: list[AnyMessage] = [
    HumanMessage("One."),
    _answer(("contract_lookup", "call_0")),
    ToolMessage('{"success": true}', tool_call_id="call_0"),
    AIMessage("First."),
    HumanMessage("Two."),
]


class TestAReusedCallId:
    async def test_an_id_the_history_holds_is_renamed_wherever_the_answer_names_it(self):
        arrived = await _arrived(FIRST_TURN, _answer(("read_file", "call_0"), ("lookup", "call_1")))

        ids = _ids(arrived)
        renamed, kept = ids["tool_calls"]
        assert renamed.startswith("call_0-") and kept == "call_1"
        assert ids["content"] == ids["additional_kwargs"] == [renamed, "call_1"]

    async def test_a_new_id_is_kept_as_the_provider_gave_it(self):
        answer = _answer(("contract_lookup", "call_0"))

        assert await _arrived([HumanMessage("One.")], answer) is answer

    async def test_two_calls_of_one_id_in_one_answer_get_their_own(self):
        arrived = await _arrived(
            [HumanMessage("One.")], _answer(("lookup", "call_0"), ("lookup", "call_0")),
        )

        ids = _ids(arrived)
        first, second = ids["tool_calls"]
        assert first == "call_0" and second.startswith("call_0-")
        assert ids["content"] == ids["additional_kwargs"] == [first, second]

    async def test_an_invalid_call_and_a_valid_one_of_one_id_each_keep_their_block(self):
        """The model made the invalid ``lookup`` first and the valid
        ``read_file`` second, both ``call_0``: LangChain lists them apart
        (valid first), the content blocks in the model's order. Each call
        and its block end with one id."""
        answer = AIMessage(
            content=[
                {"type": "tool_use", "id": "call_0", "name": "lookup", "input": {}},
                {"type": "tool_use", "id": "call_0", "name": "read_file",
                 "input": {"path": "n.txt"}},
            ],
            tool_calls=[{"name": "read_file", "args": {"path": "n.txt"}, "id": "call_0",
                         "type": "tool_call"}],
            invalid_tool_calls=[{"name": "lookup", "args": '{"q": ', "id": "call_0",
                                 "error": None, "type": "invalid_tool_call"}],
        )

        arrived = await _arrived([HumanMessage("One.")], answer)

        blocks = {block["name"]: block["id"] for block in arrived.content}
        assert arrived.invalid_tool_calls[0]["id"] == blocks["lookup"] == "call_0"
        assert arrived.tool_calls[0]["id"] == blocks["read_file"] != "call_0"

    async def test_raw_calls_keep_their_calls_ids_too(self):
        """Chat Completions' raw calls, in the model's order, name each call's new id."""
        answer = AIMessage(
            content="",
            additional_kwargs={"tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "lookup", "arguments": '{"q": '}},
                {"id": "call_0", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "n.txt"}'}},
            ]},
            tool_calls=[{"name": "read_file", "args": {"path": "n.txt"}, "id": "call_0",
                         "type": "tool_call"}],
            invalid_tool_calls=[{"name": "lookup", "args": '{"q": ', "id": "call_0",
                                 "error": None, "type": "invalid_tool_call"}],
        )

        arrived = await _arrived([HumanMessage("One.")], answer)

        raw = {
            call["function"]["name"]: call["id"]
            for call in arrived.additional_kwargs["tool_calls"]
        }
        assert arrived.invalid_tool_calls[0]["id"] == raw["lookup"] == "call_0"
        assert arrived.tool_calls[0]["id"] == raw["read_file"] != "call_0"

    async def test_an_answer_without_calls_is_left_as_it_is(self):
        answer = AIMessage("Done.")

        assert await _arrived(FIRST_TURN, answer) is answer

    async def test_the_whole_history_counts_not_only_what_is_sent(self):
        """After a summary the request sends less than the state holds; the
        state's calls are still the conversation's."""
        arrived = await _arrived(
            FIRST_TURN, _answer(("read_file", "call_0")),
            sent=[HumanMessage("Here is a summary of the conversation to date.")],
        )

        (renamed,) = _ids(arrived)["tool_calls"]
        assert renamed.startswith("call_0-")


def _patched(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The history ``before_agent`` leaves: ``messages`` with the open calls answered."""
    update = InterruptedCallsMiddleware(OpenCalls(), root=True).before_agent(
        {"messages": messages}, None
    )
    return messages if update is None else update["messages"][1:]


class TestPositionalPairing:
    def test_a_reused_id_left_open_is_answered_and_the_answered_one_is_not(self):
        """A conversation stored before reused ids were renamed: the second
        turn's ``call_0`` failed its turn with no result. A result answers the
        call of its id in the model message before it, so the first call keeps
        its one result and the second gets one."""
        reused = _answer(("read_file", "call_0"))

        patched = _patched([*FIRST_TURN, reused, HumanMessage("Three.")])

        assert patched[:6] == [*FIRST_TURN, reused]
        answer = patched[6]
        assert isinstance(answer, ToolMessage) and answer.tool_call_id == "call_0"
        assert json.loads(answer.content) == interrupted(INTERRUPTED)
        assert patched[7:] == [HumanMessage("Three.")]

    def test_two_calls_of_one_id_with_one_result_get_one_more(self):
        made = _answer(("lookup", "call_0"), ("lookup", "call_0"))
        result = ToolMessage('{"success": true}', tool_call_id="call_0")

        patched = _patched([HumanMessage("One."), made, result, HumanMessage("Two.")])

        results = [m for m in patched if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in results] == ["call_0", "call_0"]

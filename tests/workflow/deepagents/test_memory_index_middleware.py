"""Deep Agents: the memory index opens the first human message of each of the
main agent's requests; the summarization counts it (spec §11.3)."""

from __future__ import annotations

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from agentic_cli.config import BaseSettings
from agentic_cli.memory import INDEX_OPEN, MemoryStore
from agentic_cli.tools import memory_tools
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.deepagents import manager as manager_module
from agentic_cli.workflow.deepagents.middleware import MemoryIndexMiddleware
from agentic_cli.workflow.deepagents.summarization import NoSummaryWritten, build_summarization
from tests.conftest import MockContext
from tests.workflow.compacting import compact_now
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai, task
from tests.workflow.deepagents.test_context import (
    SMALL,
    SMALL_MODELS,
    USER_TURN,
    _agent,
    context,
    resolved,
)
from tests.workflow.deepagents.test_context import _manager as _small_manager
from tests.workflow.deepagents.test_manager import _manager, hermetic, scripted, turn  # noqa: F401

INDEX = f"{INDEX_OPEN}\n- [a1] 2026-11-08 feedback: Prefers tabs\n</memory-index>"


def _request(messages):
    return ModelRequest(model=ScriptedChat(), messages=messages, state={"messages": messages})


async def test_the_index_is_the_first_block_of_the_first_human_message():
    history = [HumanMessage("Hello."), AIMessage("Hi."), HumanMessage("Again.")]
    seen = []

    async def handler(request):
        seen.append(request.messages)
        return "response"

    assert await MemoryIndexMiddleware(lambda: INDEX).awrap_model_call(_request(history), handler) == "response"
    first = seen[0][0]
    assert first.content == [{"type": "text", "text": INDEX}, {"type": "text", "text": "Hello."}]
    assert seen[0][1:] == history[1:]
    assert history[0].content == "Hello."  # the state keeps its message


def test_the_sync_path_and_no_index():
    history = [HumanMessage("Hello.")]
    assert MemoryIndexMiddleware(lambda: "").wrap_model_call(_request(history), lambda r: r.messages) == history
    sent = MemoryIndexMiddleware(lambda: INDEX).wrap_model_call(_request(history), lambda r: r.messages)
    assert sent[0].content[0]["text"] == INDEX


def test_a_summary_message_gets_it_first():
    summary = HumanMessage([{"type": "text", "text": "Here is a summary of the conversation to date: ..."}])
    sent = MemoryIndexMiddleware(lambda: INDEX).wrap_model_call(
        _request([summary, AIMessage("ok")]), lambda r: r.messages
    )
    assert [block["text"][:12] for block in sent[0].content] == [INDEX[:12], "Here is a su"]


SOLO = [AgentConfig(name="solo", prompt="Answer.", tools=[memory_tools.read_memory])]


async def test_each_request_of_the_main_agent_opens_with_the_same_index(hermetic, monkeypatch):
    models = scripted(monkeypatch, solo=[ai("One."), ai("Two.")])
    with MockContext(google_api_key="test-key") as context:
        MemoryStore(context.settings.workspace_dir / "memory").save(
            "Prefers tabs", "Tabs.", "feedback", scope="user")
        manager = _manager(context.settings, SOLO)
        try:
            await turn(manager, "Hello.")
            await turn(manager, "Again.")
        finally:
            await manager.cleanup()
    chat = models.chats["solo"]
    first, second = chat.conversation(0)[0], chat.conversation(1)[0]
    assert first.content[0]["text"].startswith(INDEX_OPEN)
    assert second.content[0]["text"] == first.content[0]["text"]
    assert first.content[1]["text"] == "Hello."


@pytest.mark.parametrize("case", ["disabled", "no memory tool", "no memories"])
async def test_no_index(hermetic, monkeypatch, case):
    models = scripted(monkeypatch, solo=[ai("One.")])
    values = {"memory_index_enabled": False} if case == "disabled" else {}
    tools = [] if case == "no memory tool" else [memory_tools.read_memory]
    with MockContext(google_api_key="test-key", **values) as context:
        if case != "no memories":
            MemoryStore(context.settings.workspace_dir / "memory").save(
                "Prefers tabs", "Tabs.", "feedback", scope="user")
        manager = _manager(context.settings, [AgentConfig(name="solo", prompt="Answer.", tools=tools)])
        try:
            await turn(manager, "Hello.")
        finally:
            await manager.cleanup()
    assert models.chats["solo"].conversation(0)[0].content == "Hello."


async def test_a_sub_agent_gets_no_index(hermetic, monkeypatch):
    configs = [
        AgentConfig(name="lead", prompt="Lead.", tools=[memory_tools.read_memory], sub_agents=["helper"]),
        AgentConfig(name="helper", prompt="Help.", description="Helps."),
    ]
    models = scripted(monkeypatch, lead=[ai("", calls=[task("helper", "Do it.")]), ai("Done.")],
                      helper=[ai("Helped.")])
    with MockContext(google_api_key="test-key") as context:
        MemoryStore(context.settings.workspace_dir / "memory").save(
            "Prefers tabs", "Tabs.", "feedback", scope="user")
        manager = _manager(context.settings, configs)
        try:
            await turn(manager, "Go.")
        finally:
            await manager.cleanup()
    helper_first = models.chats["helper"].conversation(0)[0]
    assert INDEX_OPEN not in str(helper_first.content)


# -- the summarization counts it; a summary takes a new one ----------------------------------


def test_the_summarization_counts_the_index():
    """The index is added after the summarization, which must count it: a
    conversation just under the trigger goes over it with a 3,000-token index."""
    chat = ScriptedChat(script=[ai("A summary.")])
    model = resolved("local:small", agent="solo")  # summarizes at 16,150 tokens
    big = f"{INDEX_OPEN}\n{'x' * 12_000}\n</memory-index>"
    plain = build_summarization(chat, model, context())
    counted = build_summarization(chat, model, context(), index=lambda: big)
    empty = build_summarization(chat, model, context(), index=lambda: "")
    history = []
    for n in range(1, 6):
        history += [HumanMessage(f"Turn {n}: {USER_TURN}"), AIMessage(f"Answer {n}.")]
    history.append(HumanMessage("Turn 6."))

    base = plain._count_tokens(history, None, None)
    assert empty._count_tokens(history, None, None) == base < 16_150
    assert counted._count_tokens(history, None, None) == base + plain.token_counter([HumanMessage(big)])
    request = ModelRequest(model=chat, messages=history, state={"messages": history})
    assert plain.wrap_model_call(request, lambda r: "the answer") == "the answer"
    assert chat.requests == []  # under the trigger: no summary call
    counted.wrap_model_call(request, lambda r: "the answer")
    assert len(chat.requests) == 1  # over it with the index: summarized


async def test_a_summary_written_calls_back_and_one_refused_does_not():
    """``on_written`` runs once a summary has text, in the sync and the async
    path, and never for one the model wrote no text for (nothing is stored then)."""
    written: list[int] = []
    chat = ScriptedChat(script=[ai("A summary."), ai("Another."), ai(""), ai("")])
    middleware = build_summarization(
        chat, resolved("local:small", agent="solo"), context(), on_written=lambda: written.append(1)
    )
    history = [HumanMessage("Hi."), AIMessage("Hello.")]
    assert middleware._create_summary(history) == "A summary."
    assert await middleware._acreate_summary(history) == "Another."
    assert written == [1, 1]
    with pytest.raises(NoSummaryWritten):
        middleware._create_summary(history)
    with pytest.raises(NoSummaryWritten):
        await middleware._acreate_summary(history)
    assert written == [1, 1]


SMALL_ROOT = [_agent("solo", "Answer.", tools=[memory_tools.read_memory])]


def _small(root, monkeypatch, script):
    """A one-agent manager on the 20,000-token endpoint model (summarize at
    16,150 tokens), its memory in ``root``, its model on ``script``."""
    settings = BaseSettings(session_store="memory", workspace_dir=root / "w", model_providers=SMALL,
                            models=SMALL_MODELS, default_model="local:small")
    models = ScriptedModels(solo=script)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    store = MemoryStore(settings.workspace_dir / "memory")
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    return _small_manager(settings, SMALL_ROOT), models.chats["solo"], store


def _blocks(message) -> list[str]:
    return [block["text"] for block in message.content]


async def test_a_summary_in_a_turn_takes_a_new_index_with_the_request_that_carries_it(
    hermetic, monkeypatch
):
    """The index changes only where the prompt prefix already does (spec §11.4):
    the request the summary is applied to carries the new index, so the new
    prefix is written to the provider's cache once, not twice."""
    manager, chat, store = _small(hermetic, monkeypatch, [
        *(ai(f"Answer {n}.") for n in range(1, 6)), ai("A summary."), ai("Answer 6."), ai("Answer 7."),
    ])
    try:
        for n in range(1, 6):
            await turn(manager, f"Turn {n}: {USER_TURN}")
        store.save("Prefers dark mode", "Dark.", "user", scope="user")
        await turn(manager, f"Turn 6: {USER_TURN}")  # summarized
        await turn(manager, "Turn 7?")
    finally:
        await manager.cleanup()
    old = _blocks(chat.conversation(0)[0])[0]
    assert old.startswith(INDEX_OPEN) and "Prefers dark mode" not in old
    assert _blocks(chat.conversation(4)[0])[0] == old  # fixed until the summary
    summary_request = chat.requests[5]
    assert len(summary_request) == 1 and "<messages>" in summary_request[0].text
    assert INDEX_OPEN not in summary_request[0].text  # never summarized into itself
    # The request the summary is applied to: the summary, behind the index as the files are now.
    sixth = _blocks(chat.conversation(6)[0])
    assert sixth[0].startswith(INDEX_OPEN) and "Prefers dark mode" in sixth[0]
    assert sixth[1].startswith("Here is a summary")
    # And it stays that one: the next request's prefix is the sixth's.
    seventh = _blocks(chat.conversation(7)[0])
    assert seventh[:2] == sixth[:2]


async def test_a_small_window_root_gets_an_index_that_leaves_room(hermetic, monkeypatch):
    """The 20,000-token model's input budget is 19,000 tokens: the index may take
    a tenth of it (1,900 tokens, about 6,900 bytes), not the 25,000 bytes of the setting."""
    manager, chat, store = _small(hermetic, monkeypatch, [ai("One.")])
    for n in range(150):
        store.save(f"fact {n:03d} " + ("lorem ipsum " * 12)[:139], "c", "user", scope="user")
    try:
        await turn(manager, "Hello.")
    finally:
        await manager.cleanup()
    index = _blocks(chat.conversation(0)[0])[0]
    assert index.startswith(INDEX_OPEN) and index.endswith("</memory-index>")
    assert 5_000 < len(index.encode()) <= 6_909
    assert "more not shown" in index


async def test_compact_takes_a_new_index_for_the_next_request(hermetic, monkeypatch):
    manager, chat, store = _small(hermetic, monkeypatch, [
        *(ai(f"Answer {n}.") for n in range(1, 4)), ai("A summary."), ai("Answer 4."),
    ])
    try:
        for n in range(1, 4):
            await turn(manager, f"Turn {n}: {USER_TURN}")
        store.save("Prefers dark mode", "Dark.", "user", scope="user")
        await compact_now(manager)
        await turn(manager, "Turn 4?")
    finally:
        await manager.cleanup()
    old = _blocks(chat.conversation(0)[0])[0]
    fourth = _blocks(chat.conversation(4)[0])
    assert "Prefers dark mode" not in old
    assert fourth[0].startswith(INDEX_OPEN) and "Prefers dark mode" in fourth[0]
    assert fourth[1].startswith("Here is a summary")

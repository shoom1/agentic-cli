"""ADK: the memory index opens each of the main agent's requests, is never
dropped, and is folded into the first user text (spec §11.2)."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.memory import INDEX_OPEN, MemoryStore  # noqa: E402
from agentic_cli.tools import memory_tools  # noqa: E402
from agentic_cli.workflow.adk.fit_guard_plugin import (  # noqa: E402
    FIT_NOTE,
    FitGuardPlugin,
    drop_turns,
    fold_summary,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.memory_index_plugin import MemoryIndexPlugin, index_content  # noqa: E402
from agentic_cli.workflow.adk.overflow_plugin import _with_summary  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import ContextBudget  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.adk.test_fit_guard import (  # noqa: E402
    CLAUDE,
    LONG,
    NOTE_ROOM,
    Recorded,
    call_context,
    guarded,
    model,
    tokens,
    user,
)
from tests.workflow.test_adk_compaction import (  # noqa: E402,F401
    SUMMARY,
    Claude,
    project,
    says,
    serve,
    turn,
)
from tests.workflow.test_adk_overflow import SHAPES  # noqa: E402

INDEX = f"{INDEX_OPEN}\nNotes saved in earlier conversations.\n- [a1] 2026-11-08 feedback: Prefers tabs\n</memory-index>"


def texts(content) -> list[str]:
    return [part.text for part in content.parts]


class TestThePlugin:
    async def test_the_index_goes_first_for_the_main_agent_only(self):
        plugin = MemoryIndexPlugin(root="solo", index_for=lambda: INDEX)
        mine, theirs = LlmRequest(contents=[user("Hi.")]), LlmRequest(contents=[user("Hi.")])
        await plugin.before_model_callback(callback_context=call_context("solo"), llm_request=mine)
        await plugin.before_model_callback(callback_context=call_context("researcher"), llm_request=theirs)
        assert [texts(c) for c in mine.contents] == [[INDEX], ["Hi."]]
        assert [texts(c) for c in theirs.contents] == [["Hi."]]

    async def test_no_index_no_change(self):
        plugin = MemoryIndexPlugin(root="solo", index_for=lambda: "")
        request = LlmRequest(contents=[user("Hi.")])
        await plugin.before_model_callback(callback_context=call_context("solo"), llm_request=request)
        assert [texts(c) for c in request.contents] == [["Hi."]]


class TestFolding:
    def test_index_then_user_text(self):
        assert [texts(c) for c in fold_summary([index_content(INDEX), user("Hi.")])] == [[INDEX, "Hi."]]

    def test_index_then_summary_then_user_text(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        folded = fold_summary([index_content(INDEX), summary, user("Two?"), model("A2.")])
        assert [texts(c) for c in folded] == [[INDEX, summary.parts[0].text, "Two?"], ["A2."]]

    def test_index_then_summary_then_no_user_text(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        folded = fold_summary([index_content(INDEX), summary, model("A2.")])
        assert [texts(c) for c in folded] == [[INDEX, summary.parts[0].text], ["A2."]]

    def test_folding_twice_changes_nothing(self):
        once = fold_summary([index_content(INDEX), user("Hi."), model("A.")])
        assert fold_summary(once) == once

    def test_dropping_turns_keeps_the_index_and_a_summary(self):
        summary = user(f"{SUMMARY_PREFIX}\n\nThe user is Ada.")
        contents = [index_content(INDEX), summary, user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
        # What stays (the index and the summary, the last two turns) with room for the note.
        target = tokens([*contents[:2], *contents[4:]]) + NOTE_ROOM
        kept, dropped = drop_turns(contents, target_tokens=target, kind=CLAUDE)
        assert dropped == 1
        assert texts(kept[0]) == [INDEX] and kept[1] == summary
        assert texts(kept[2])[0] == FIT_NOTE

    def test_the_overflow_retry_keeps_the_index_first(self):
        old = user(f"{SUMMARY_PREFIX}\n\nOld summary.")
        retry = _with_summary([index_content(INDEX), old, user("Two?")], f"{SUMMARY_PREFIX}\n\nNew summary.")
        assert texts(retry[0]) == [INDEX, f"{SUMMARY_PREFIX}\n\nNew summary.", "Two?"]


async def test_the_index_survives_dropped_turns_and_stays_identical():
    """Review Focus 5: over the budget, the oldest turn goes but the index stays
    first, and two requests carry byte-identical index text."""
    history = [user(LONG), model("A1."), user(LONG), model("A2."), user("Three?")]
    with_index = [user(INDEX), *history]
    over = tokens(with_index) - 1
    budget = ContextBudget(window=over * 2, window_known=True, budget=over,
                           trigger=tokens([user(INDEX), *history[2:]]) + NOTE_ROOM, keep=0)
    index_plugin = MemoryIndexPlugin(root="solo", index_for=lambda: INDEX)
    guard = FitGuardPlugin({"solo": budget}, {"solo": CLAUDE})
    sent = []
    for _ in range(2):
        request = LlmRequest(contents=list(history))
        await index_plugin.before_model_callback(callback_context=call_context(), llm_request=request)
        await guarded(guard, request, Recorded())
        sent.append(request.contents)
    for contents in sent:
        assert texts(contents[0])[:2] == [INDEX, FIT_NOTE]
    assert sent[0][0].parts[0].text == sent[1][0].parts[0].text


# -- on the real manager -------------------------------------------------------


def _solo(project, tools=(memory_tools.read_memory, memory_tools.search_memory), **values):
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.", tools=list(tools),
                                   include_state_tools=False)],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                              workspace_dir=project / "w", **values),
    )


def first_blocks(body: dict) -> list[str]:
    content = body["messages"][0]["content"]
    return [content] if isinstance(content, str) else [block.get("text", "") for block in content]


async def test_each_request_of_the_main_agent_opens_with_the_same_index(project, monkeypatch):
    MemoryStore(project / "w" / "memory").save("Prefers tabs", "Tabs.", "feedback", scope="user")
    answers = serve(monkeypatch, Claude(solo=[says("One."), says("Two.")]))
    manager = _solo(project)
    try:
        await turn(manager, "Hello.")
        MemoryStore(project / "w" / "memory").save("Prefers dark mode", "Dark.", "user", scope="user")
        await turn(manager, "Again.")
    finally:
        await manager.cleanup()
    first, second = answers.bodies("solo")
    assert first_blocks(first)[0].startswith(INDEX_OPEN)
    assert "Prefers tabs" in first_blocks(first)[0]
    assert first_blocks(second)[0] == first_blocks(first)[0]  # fixed for the conversation
    assert first_blocks(first)[1] == "Hello."


async def test_the_cache_marks_are_where_they_were(project, monkeypatch):
    """The index is no breakpoint of its own: the tools, the system prompt and
    the conversation's last block are marked, as without it (spec §11.4)."""
    MemoryStore(project / "w" / "memory").save("Prefers tabs", "Tabs.", "feedback", scope="user")
    answers = serve(monkeypatch, Claude(solo=[says("One.")]))
    manager = _solo(project)
    try:
        await turn(manager, "Hello.")
    finally:
        await manager.cleanup()
    [body] = answers.bodies("solo")
    assert "cache_control" in body["system"][-1]
    assert "cache_control" in body["tools"][-1]
    assert "cache_control" in body["messages"][-1]["content"][-1]
    assert body["messages"][0]["content"][0]["text"].startswith(INDEX_OPEN)
    assert "cache_control" not in body["messages"][0]["content"][0]


@pytest.mark.parametrize("case", ["disabled", "no memory tool", "no memories"])
async def test_no_index(project, monkeypatch, case):
    if case != "no memories":
        MemoryStore(project / "w" / "memory").save("Prefers tabs", "Tabs.", "feedback", scope="user")
    answers = serve(monkeypatch, Claude(solo=[says("One.")]))
    values = {"memory_index_enabled": False} if case == "disabled" else {}
    tools = () if case == "no memory tool" else (memory_tools.read_memory,)
    manager = _solo(project, tools=tools, **values)
    try:
        await turn(manager, "Hello.")
    finally:
        await manager.cleanup()
    [body] = answers.bodies("solo")
    assert first_blocks(body) == ["Hello."]


async def test_the_plugin_comes_before_the_overflow_handler_and_the_fit_guard(project, monkeypatch):
    serve(monkeypatch, Claude())
    manager = _solo(project)
    try:
        await manager.initialize_services()
        names = [plugin.name for plugin in manager._runner.app.plugins]
    finally:
        await manager.cleanup()
    assert names.index("memory_index") < names.index("context_overflow") < names.index("fit_guard")
    assert names[-2:] == ["context_overflow", "fit_guard"]  # test_adk_overflow's order still holds


class TestAfterAnOverflow:
    """The retry is built from the request as the fit guard saw it: the index
    stays first in it, folded in with the summary and the note."""

    async def test_the_trimmed_retry_opens_with_the_index_and_the_note_in_one_message(
        self, project, monkeypatch
    ):
        MemoryStore(project / "w" / "memory").save("Prefers tabs", "Tabs.", "feedback", scope="user")
        _, status, body = SHAPES["claude-400"]
        answers = serve(monkeypatch, Claude(solo=[
            says("A1."), says("A2."), httpx.Response(status, json=body), says("A3."),
        ]))
        manager = _solo(project)
        try:
            for text in ("One.", "Two.", "Three?"):
                await turn(manager, text)
        finally:
            await manager.cleanup()
        _, _, refused, retry = answers.bodies("solo")
        assert first_blocks(refused)[0].startswith(INDEX_OPEN)
        assert first_blocks(retry) == [first_blocks(refused)[0], FIT_NOTE, "Two."]

    async def test_a_summary_written_takes_a_new_index_for_the_conversations_next_request(
        self, project, monkeypatch
    ):
        store = MemoryStore(project / "w" / "memory")
        store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
        _, status, body = SHAPES["claude-400"]
        answers = serve(monkeypatch, Claude(solo=[
            says("A1."), says("A2."), httpx.Response(status, json=body), says("A3."), says("A4."),
        ]))
        manager = _solo(project, context_max_tokens=16_000)
        try:
            for text in (f"One: {LONG}", f"Two: {LONG}"):
                await turn(manager, text)
            store.save("Prefers dark mode", "Dark.", "user", scope="user")
            await turn(manager, "Three?")  # refused, summarized, retried
            await turn(manager, "Four?")
        finally:
            await manager.cleanup()
        first, _, refused, retry, fourth = answers.bodies("solo")
        old = first_blocks(first)[0]
        summarized = f"{SUMMARY_PREFIX}\n\n{SUMMARY}"
        assert "Prefers dark mode" not in old
        # The retry is the refused request's own index, the summary and the
        # turn it kept; the next request takes the index as the files are now.
        assert first_blocks(refused)[0] == old
        assert first_blocks(retry) == [old, summarized, FIT_NOTE, f"Two: {LONG}"]
        new = first_blocks(fourth)[0]
        assert new.startswith(INDEX_OPEN) and "Prefers dark mode" in new
        assert first_blocks(fourth)[1:] == [summarized, f"Two: {LONG}"]

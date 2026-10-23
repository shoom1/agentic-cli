"""``/compact`` on Deep Agents, through a real ``BaseCLIApp``.

The app's controller holds the real Deep Agents manager on ``ScriptedModels``
(the 20,000-token endpoint model of ``tests/workflow/deepagents/``), and its
session is a fake: while the summary is held, the events box's status line
says so; then one "Context summarized: …", and the next request holds one
summary. Ctrl+C says "Operation cancelled..." once and writes nothing.
"""

from __future__ import annotations

import asyncio
import re

import pytest

pytest.importorskip("deepagents")

from agentic_cli.cli.message_processor import CANCELLED_LINE  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    NOTHING,
    SUMMARIZING,
    app_on,
    quiet_app,
    said,
)
from tests.workflow.deepagents.fakes import ai  # noqa: E402
from tests.workflow.deepagents.test_compact import THREE, gated  # noqa: E402
from tests.workflow.deepagents.test_context import hermetic, small  # noqa: E402,F401


async def test_the_line_shows_while_it_summarizes_then_what_it_did(small, monkeypatch):
    manager, chat = gated(monkeypatch, small, [
        ai("Answer 1."), ai("Answer 2."), ai("Answer 3."), ai("A summary."), ai("Answer 4."),
    ], gate_at=4)
    await manager.initialize_services()
    app = app_on(manager, small)
    try:
        for text in THREE:
            await app.process_input(text)
        compacting = asyncio.create_task(app.process_input("/compact"))
        await asyncio.wait_for(chat._started.wait(), 5)
        await asyncio.sleep(0.05)
        status = app.session.start_thinking.call_args_list[-1].args[0]
        while_summarizing = status()
        chat._release.set()
        await asyncio.wait_for(compacting, 5)
        await app.process_input("Turn 4?")
    finally:
        chat._release.set()
        await manager.cleanup()

    assert while_summarizing == SUMMARIZING
    (warning,) = said(app.session, "add_warning")
    assert re.fullmatch(r"Context summarized: 4 messages, \S+ → \S+ tokens", warning)
    assert said(app.session, "add_error") == []
    assert sum(SUMMARY_PREFIX in m.text for m in chat.conversation(4)) == 1


async def test_ctrl_c_says_so_once_and_writes_nothing(small, monkeypatch):
    manager, chat = gated(monkeypatch, small, [
        ai("Answer 1."), ai("Answer 2."), ai("Answer 3."), ai("A summary."),
    ], gate_at=4)
    await manager.initialize_services()
    app = app_on(manager, small)
    try:
        for text in THREE:
            await app.process_input(text)
        compacting = asyncio.create_task(app.process_input("/compact"))
        await asyncio.wait_for(chat._started.wait(), 5)
        compacting.cancel()
        with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
            await asyncio.wait_for(compacting, 5)
        ref = manager.session_ref(app.session_id, app.settings.default_user)
        state = (await manager._root.aget_state(run_config(ref))).values
        chat._release.set()
        await asyncio.wait_for(app.process_input("/compact"), 5)
        state_after = (await manager._root.aget_state(run_config(ref))).values
    finally:
        chat._release.set()
        await manager.cleanup()

    messages = said(app.session, "add_message")
    assert messages.count(CANCELLED_LINE) == 1
    assert NOTHING not in messages
    assert len(state["messages"]) == 6 and "_summarization_event" not in state
    assert state_after["_summarization_event"]["cutoff_index"] == 4

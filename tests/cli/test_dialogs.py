"""One dialog at a time (``cli/dialogs.py``): the session's dialog methods take
one lock, which the same task may take again."""

from __future__ import annotations

import asyncio

from thinking_prompt import ThinkingPromptSession

from agentic_cli.cli.dialogs import DIALOG_METHODS, dialog_lock, serialize_dialogs
from tests.cli.dialog_fakes import autospec_session


def test_a_real_sessions_dialog_methods_are_wrapped_once():
    session = ThinkingPromptSession(message=">>> ")
    originals = {name: getattr(session, name) for name in DIALOG_METHODS}

    assert serialize_dialogs(session) is session
    wrapped = {name: getattr(session, name) for name in DIALOG_METHODS}
    serialize_dialogs(session)

    assert all(wrapped[name] is not originals[name] for name in DIALOG_METHODS)
    assert all(getattr(session, name) is wrapped[name] for name in DIALOG_METHODS)  # not again


def test_a_stand_in_whose_methods_are_mocks_is_left_alone():
    session = autospec_session()
    shown = session.show_dialog

    serialize_dialogs(session)

    assert session.show_dialog is shown


async def test_the_same_task_may_take_the_lock_again_and_another_waits():
    lock = dialog_lock(object.__new__(type("Ui", (), {})))
    order: list[str] = []

    async def other() -> None:
        async with lock:
            order.append("other")

    async with lock:
        async with lock:  # the same task: let through
            order.append("inner")
        task = asyncio.create_task(other())
        await asyncio.sleep(0)
        order.append("still held")
    await task

    assert order == ["inner", "still held", "other"]

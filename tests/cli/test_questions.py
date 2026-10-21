"""The questions a tool asks the user, as thinking-prompt 0.4 dialogs.

A free-text question (``ask_clarification`` without options) called
``ui.input_dialog()``, which no thinking-prompt release has, so on a real
session it always failed; the tests' hand-written session defined the method.
It is now a ``Dialog`` with one ``TextField`` and OK and Cancel; the field says
to press Enter before typing, since 0.4 opens it for viewing and drops keys
typed at it until then. A choice and a yes/no question keep 0.4's
``dropdown_dialog`` and ``yes_no_dialog``. Each is drawn where
``dialog_placement`` says when it is asked: the session's own placement is
the one it started with.

The session is specced from ``ThinkingPromptSession`` and the dialog is the
real ``Dialog`` (``edited_dialog``), so neither can grow what 0.4 lacks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from thinking_prompt import ButtonConfig, TextField

from agentic_cli.cli import message_processor
from agentic_cli.cli.message_processor import MessageProcessor
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent
from tests.cli.dialog_fakes import autospec_session, edited_dialog, opening_boxes


@pytest.fixture
def ui() -> MagicMock:
    return autospec_session()


def _text(default: str | None = None, tool_name: str = "ask_clarification") -> UserInputRequest:
    return UserInputRequest(
        request_id="r",
        tool_name=tool_name,
        prompt="Which dataset should I use?",
        input_type=InputType.TEXT,
        default=default,
    )


def _choice() -> UserInputRequest:
    return UserInputRequest(
        request_id="r", tool_name="pick", prompt="Which?",
        input_type=InputType.CHOICE, choices=["a", "b"], default="a",
    )


def _confirmation() -> UserInputRequest:
    return UserInputRequest(
        request_id="r", tool_name="approve", prompt="Go?", input_type=InputType.CONFIRM
    )


async def _ask(
    ui,
    request: UserInputRequest,
    monkeypatch,
    edits: dict | None = None,
    *,
    ok=True,
    placement="box",
):
    """Ask ``request`` in ``placement``; the user makes ``edits`` and presses
    OK (or Cancel). Returns the answer and the dialogs shown."""
    dialog_cls = edited_dialog(edits or {})
    monkeypatch.setattr("agentic_cli.cli.message_processor.Dialog", dialog_cls)
    ui.show_dialog.return_value = True if ok else None

    answer = await MessageProcessor()._prompt_user_input(request, ui, placement)

    return answer, dialog_cls.built


# --- a free-text question ----------------------------------------------------------------


async def test_a_free_text_question_is_one_dialog_with_a_text_field(ui, monkeypatch):
    _, shown = await _ask(ui, _text(default="iris"), monkeypatch)

    (dialog,) = shown
    ui.show_dialog.assert_awaited_once_with(dialog)
    assert dialog.title == "ask_clarification - Input Required"
    assert dialog.body == "Which dataset should I use?"
    assert list(dialog.fields) == [
        TextField(
            "answer",
            "Answer",
            description="Press Enter to type your answer, Enter again to keep it",
            default="iris",
            edit_width=60,
        )
    ]
    assert list(dialog.buttons) == [ButtonConfig("OK", result=True), ButtonConfig("Cancel")]


def test_the_answer_field_says_how_to_type_in_it():
    """Keys typed before Enter are lost: the hint under the field says so."""
    assert message_processor.ANSWER_HINT == (
        "Press Enter to type your answer, Enter again to keep it"
    )


async def test_a_tool_without_a_name_is_named_tool(ui, monkeypatch):
    _, (dialog,) = await _ask(ui, _text(tool_name=""), monkeypatch)

    assert dialog.title == "Tool - Input Required"


async def test_ok_returns_the_answer(ui, monkeypatch):
    answer, _ = await _ask(ui, _text(), monkeypatch, {"answer": "the penguins"})

    assert answer == "the penguins"


async def test_ok_on_the_default_returns_it(ui, monkeypatch):
    answer, _ = await _ask(ui, _text(default="iris"), monkeypatch)

    assert answer == "iris"


async def test_ok_with_the_field_empty_returns_nothing(ui, monkeypatch):
    answer, _ = await _ask(ui, _text(), monkeypatch)

    assert answer == ""


async def test_cancel_or_escape_returns_nothing(ui, monkeypatch):
    answer, _ = await _ask(ui, _text(default="iris"), monkeypatch, {"answer": "typed"}, ok=False)

    assert answer == ""


# --- choice and yes/no ---------------------------------------------------------------------


async def test_a_choice_is_a_dropdown_dialog(ui, monkeypatch):
    ui.dropdown_dialog.return_value = "b"

    answer, shown = await _ask(ui, _choice(), monkeypatch)

    assert answer == "b"
    assert shown == []
    ui.dropdown_dialog.assert_awaited_once_with(
        title="pick - Input Required",
        text="Which?",
        options=["a", "b"],
        default="a",
        placement="box",
    )
    ui.show_dialog.assert_not_awaited()


async def test_a_cancelled_choice_is_its_default(ui, monkeypatch):
    ui.dropdown_dialog.return_value = None

    answer, _ = await _ask(ui, _choice(), monkeypatch)

    assert answer == "a"


@pytest.mark.parametrize(("pressed", "answer"), [(True, "yes"), (False, "no")])
async def test_a_confirmation_is_a_yes_no_dialog(ui, monkeypatch, pressed, answer):
    ui.yes_no_dialog.return_value = pressed

    assert (await _ask(ui, _confirmation(), monkeypatch))[0] == answer
    ui.yes_no_dialog.assert_awaited_once_with(
        title="approve - Confirmation", text="Go?", placement="box"
    )


# --- where they are drawn ----------------------------------------------------------------


@pytest.mark.parametrize("placement", ["box", "inline"])
async def test_every_question_is_drawn_where_it_is_asked_to_be(ui, monkeypatch, placement):
    ui.dropdown_dialog.return_value = "b"
    ui.yes_no_dialog.return_value = True

    _, (dialog,) = await _ask(ui, _text(), monkeypatch, placement=placement)
    await _ask(ui, _choice(), monkeypatch, placement=placement)
    await _ask(ui, _confirmation(), monkeypatch, placement=placement)

    assert dialog.placement == placement
    assert ui.dropdown_dialog.await_args.kwargs["placement"] == placement
    assert ui.yes_no_dialog.await_args.kwargs["placement"] == placement


class _AskingWorkflow:
    """Each turn asks one free-text question through the input callback."""

    def __init__(self) -> None:
        self.callback = None

    def set_input_callback(self, callback) -> None:
        self.callback = callback

    def clear_input_callback(self) -> None:
        self.callback = None

    def source(self, _workflow):
        async def _turn():
            answer = await self.callback(_text())
            yield WorkflowEvent.text(answer)

        return _turn()


async def test_a_turn_asks_where_the_settings_say_now(monkeypatch):
    """The placement is read at each question, so a switch between turns
    applies to the next one; the session's own stays the startup value."""
    ui = opening_boxes(autospec_session())
    ui.show_dialog.return_value = True
    dialog_cls = edited_dialog({})
    monkeypatch.setattr("agentic_cli.cli.message_processor.Dialog", dialog_cls)
    workflow = _AskingWorkflow()
    controller = SimpleNamespace(workflow=workflow, update_status_bar=lambda ui: None)
    settings = SimpleNamespace(
        verbose_thinking=False, default_user="u", dialog_placement="box"
    )
    processor = MessageProcessor()

    await processor._run_turn(workflow.source, controller, ui, settings)
    settings.dialog_placement = "inline"  # a /settings switch between turns
    await processor._run_turn(workflow.source, controller, ui, settings)

    assert [dialog.placement for dialog in dialog_cls.built] == ["box", "inline"]

"""The task-list box on thinking-prompt 0.4.

0.4 changed three things the box relies on:

- An overflowing box keeps its newest lines (``overflow="tail"``); a task
  list keeps its first tasks in view, as before, with ``overflow="head"``.
- A box can end without us: the session ends the boxes a handler leaves open,
  and Ctrl+C at an idle prompt clears every box. A write to an ended box is
  dropped (with a logged warning), so an ended box counts as no box.
- The session ends a box a handler leaves open with ``add_to_history=True``:
  the task list would enter the transcript unseen, and a repaint would print
  it. So the box ends with each turn, out of the transcript, and the next turn
  rebuilds it, showing the tasks as the last one did.

The session and the boxes are specced from thinking-prompt's classes, so an
argument 0.4 does not take fails the test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, create_autospec

import pytest
from thinking_prompt import ThinkingBox

from agentic_cli.cli.message_processor import (
    MessageProcessor,
    _EventProcessingState,
    _richify_task_display,
)
from agentic_cli.workflow.events import WorkflowEvent
from tests.cli.dialog_fakes import autospec_session
from tests.event_replay import ReplayController, ReplayWorkflow

SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False)
OUT_OF_THE_TRANSCRIPT = call(add_to_history=False, echo_to_console=False)


def _box(finished: bool = False) -> MagicMock:
    box = create_autospec(ThinkingBox, instance=True)
    box.is_finished = finished
    return box


@pytest.fixture
def ui() -> MagicMock:
    """A session whose every ``start_thinking`` opens a new box (``ui.boxes``)."""
    ui = autospec_session()
    ui.boxes = []

    def start_thinking(*args, **kwargs):
        box = _box()
        ui.boxes.append(box)
        return box

    ui.start_thinking.side_effect = start_thinking
    return ui


def _progress(content: str, completed: int, total: int) -> WorkflowEvent:
    return WorkflowEvent.task_progress(
        display=content,
        progress={"completed": completed, "total": total, "pending": total - completed},
    )


async def _on_progress(processor: MessageProcessor, ui, event: WorkflowEvent) -> None:
    await processor._handle_task_progress(
        event, _EventProcessingState(), ui, SETTINGS, MagicMock()
    )


async def _turn(processor: MessageProcessor, ui, events=(), fails: bool = False) -> None:
    """One user turn whose workflow streams ``events`` (then raises, if ``fails``)."""
    workflow = ReplayWorkflow(list(events))
    if fails:
        streamed = workflow.process

        async def process(*args, **kwargs):
            async for event in streamed(*args, **kwargs):
                yield event
            raise RuntimeError("the model is gone")

        workflow.process = process
    await processor.process("hi", ReplayController(workflow), ui, SETTINGS)


def _task_box_calls(ui) -> list:
    """The ``start_thinking`` calls that opened a task box (order 100)."""
    return [c for c in ui.start_thinking.call_args_list if c.kwargs.get("order") == 100]


def _task_boxes(ui) -> list[MagicMock]:
    """The task boxes opened, in order."""
    calls = ui.start_thinking.call_args_list
    return [box for box, c in zip(ui.boxes, calls, strict=True) if c.kwargs.get("order") == 100]


# --- overflow="head" -------------------------------------------------------------


async def test_the_task_box_keeps_its_first_tasks_in_view(ui):
    processor = MessageProcessor()

    await _on_progress(processor, ui, _progress("[▸] A\n[ ] B", 0, 2))

    ui.start_thinking.assert_called_once_with(
        title="Tasks: 0/2", order=100, content_format="ansi", overflow="head"
    )


async def test_the_task_box_rebuilt_at_a_turn_s_start_does_too(ui):
    processor = MessageProcessor()
    processor._last_task_progress = "[▸] A\n[ ] B"

    await _turn(processor, ui)

    assert _task_box_calls(ui) == [
        call(title="Tasks", order=100, content_format="ansi", overflow="head")
    ]


# --- an ended box is no box ------------------------------------------------------------


async def test_progress_after_the_box_ended_opens_a_new_one(ui):
    processor = MessageProcessor()
    ended = _box(finished=True)
    processor._task_box = ended
    processor._last_task_content = "[▸] A\n[ ] B"

    # The same tasks as the ended box showed: the new box shows them anyway.
    await _on_progress(processor, ui, _progress("[▸] A\n[ ] B", 0, 2))

    (new,) = ui.boxes
    assert processor._task_box is new
    new.append.assert_called_once()
    assert ended.method_calls == []


async def test_finishing_the_tasks_leaves_an_ended_box_alone(ui):
    processor = MessageProcessor()
    ended = _box(finished=True)
    processor._task_box = ended
    processor._last_task_content = "[▸] A"

    await _on_progress(processor, ui, _progress("[✓] A", 1, 1))

    assert ended.method_calls == []
    assert (processor._task_box, processor._last_task_content) == (None, None)


async def test_a_turn_rebuilds_a_box_that_ended(ui):
    processor = MessageProcessor()
    processor._task_box = _box(finished=True)
    processor._last_task_progress = "[▸] A"

    await _turn(processor, ui)

    assert len(_task_box_calls(ui)) == 1


# --- the box ends with the turn --------------------------------------------------------


@pytest.mark.parametrize("fails", [False, True], ids=["completed", "failed"])
async def test_the_box_ends_with_the_turn_out_of_the_transcript(ui, fails):
    processor = MessageProcessor()

    await _turn(processor, ui, [_progress("[▸] A\n[ ] B", 0, 2)], fails=fails)

    (task_box,) = _task_boxes(ui)
    assert task_box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert processor._task_box is None
    assert processor._last_task_progress == "[▸] A\n[ ] B"


async def test_the_next_turn_shows_the_tasks_again(ui):
    processor = MessageProcessor()
    await _turn(processor, ui, [_progress("[▸] A\n[ ] B", 0, 2)])

    await _turn(processor, ui)

    assert _task_box_calls(ui)[1:] == [
        call(title="Tasks", order=100, content_format="ansi", overflow="head")
    ]
    live, rebuilt = _task_boxes(ui)
    # As the last turn's box showed them: marks and colors, not the plain list.
    assert rebuilt.append.call_args_list == live.append.call_args_list == [
        call(_richify_task_display("[▸] A\n[ ] B"))
    ]
    assert rebuilt.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]


async def test_a_turn_without_tasks_leaves_nothing_to_rebuild(ui):
    processor = MessageProcessor()

    await _turn(processor, ui)

    assert _task_box_calls(ui) == []
    assert processor._last_task_progress is None


# --- the tasks are done ----------------------------------------------------------------


async def test_done_tasks_end_the_box_and_the_next_turn_has_none(ui):
    processor = MessageProcessor()

    await _turn(processor, ui, [_progress("[▸] A", 0, 1), _progress("[✓] A", 1, 1)])

    (task_box,) = _task_boxes(ui)
    assert task_box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert (processor._task_box, processor._last_task_content) == (None, None)
    assert processor._last_task_progress is None

    await _turn(processor, ui)

    assert len(_task_box_calls(ui)) == 1  # the first turn's only

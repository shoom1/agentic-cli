"""A sub-agent run stopped while its prompt is on screen, the turn going on.

A run's limit (its time limit, while the user is slow to answer), its
parent or the user cancels the run alone: the task awaiting its question is
cancelled, not the turn's. The dialog is withdrawn (thinking-prompt closes a
dialog whose awaiter is cancelled, in either placement), the turn's events
box comes back, since the turn goes on, and the turn's next question
reaches the user. A cancel of the turn itself (Ctrl+C) still leaves the box
closed: the turn is unwinding (``test_turn_boundary.py``).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from thinking_prompt import ThinkingPromptSession

from agentic_cli.cli.message_processor import MessageProcessor, TurnStatus
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.loop_guard import no_blocking_calls


def _question(request_id: str) -> UserInputRequest:
    return UserInputRequest(
        request_id=request_id, tool_name="write_file", prompt="Allow it?",
        input_type=InputType.CONFIRM,
    )


class _Workflow:
    def __init__(self) -> None:
        self.callback = None

    def set_input_callback(self, callback) -> None:
        self.callback = callback

    def clear_input_callback(self, token=None) -> None:
        self.callback = None


def _settings(placement: str = "box") -> SimpleNamespace:
    return SimpleNamespace(verbose_thinking=False, default_user="u", dialog_placement=placement)


def _controller(workflow: _Workflow) -> SimpleNamespace:
    return SimpleNamespace(workflow=workflow, update_status_bar=lambda ui: None)


async def test_the_events_box_comes_back_and_the_next_question_is_asked():
    ui = opening_boxes(autospec_session())
    on_screen = asyncio.Event()
    titles: list[str] = []
    seen: dict = {}

    async def yes_no_dialog(title, text, *args, **kwargs) -> bool:
        titles.append(title)
        if len(titles) == 1:
            on_screen.set()
            await asyncio.Event().wait()  # the user is slow to answer
        return True

    ui.yes_no_dialog.side_effect = yes_no_dialog

    def source(workflow):
        async def turn():
            # The sub-agent run's task asks; its limit cancels that task alone.
            run = asyncio.create_task(workflow.callback(_question("r1")))
            await on_screen.wait()
            run.cancel()
            with suppress(asyncio.CancelledError):
                await run
            seen["box open"] = not ui.boxes[-1].is_finished
            yield WorkflowEvent.text("The researcher ran out of time.")
            answer = await workflow.callback(_question("c2"))
            yield WorkflowEvent.text(f"Asked again: {answer}.")

        return turn()

    with no_blocking_calls():
        result = await MessageProcessor()._run_turn(
            source, _controller(_Workflow()), ui, _settings()
        )

    assert result.status is TurnStatus.COMPLETED
    assert seen == {"box open": True}
    assert len(titles) == 2
    assert [c.args[0] for c in ui.add_response.call_args_list] == [
        "The researcher ran out of time.", "Asked again: yes.",
    ]
    # Opened, finished at the question, back after it was withdrawn,
    # finished at the next question, back after its answer: each once.
    assert [box.finish.call_count for box in ui.boxes] == [1, 1, 1]


@pytest.mark.parametrize("placement", ["box", "inline"])
async def test_on_a_real_session_the_dialog_is_withdrawn_and_the_next_one_shown(placement):
    workflow = _Workflow()
    seen: dict = {}

    def source(_workflow):
        async def turn():
            run = asyncio.create_task(workflow.callback(_question("r1")))
            for _ in range(500):  # until the dialog has the focus
                if not ui.app.layout.has_focus(ui.default_buffer):
                    break
                await asyncio.sleep(0.01)
            seen["shown"] = not ui.app.layout.has_focus(ui.default_buffer)
            run.cancel()
            with suppress(asyncio.CancelledError):
                await run
            seen["withdrawn"] = ui.app.layout.has_focus(ui.default_buffer)
            seen["answer"] = await workflow.callback(_question("c2"))
            yield WorkflowEvent.text("done")

        return turn()

    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        ui = ThinkingPromptSession(message="> ")
        ui.app.ttimeoutlen = 0.01  # a lone Esc is read as Esc at once
        turn_done = asyncio.Event()

        async def handler(_line: str) -> None:
            await MessageProcessor()._run_turn(
                source, _controller(workflow), ui, _settings(placement)
            )
            turn_done.set()

        running = asyncio.create_task(ui.run_async(handler))
        try:
            for _ in range(500):  # a session that fails to start ends the wait
                if ui.app.is_running or running.done():
                    break
                await asyncio.sleep(0.01)
            assert ui.app.is_running, "the session did not start"
            keys.send_text("go\r")
            for _ in range(50):  # Esc answers the second question "no"
                with suppress(TimeoutError):
                    await asyncio.wait_for(turn_done.wait(), 0.2)
                if turn_done.is_set():
                    break
                if "withdrawn" in seen:
                    keys.send_text("\x1b")
        finally:
            ui.exit()
            await asyncio.wait_for(running, 5)

    assert seen == {"shown": True, "withdrawn": True, "answer": "no"}

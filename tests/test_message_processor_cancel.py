"""Ctrl+C during a turn: thinking-prompt 0.4 cancels the turn's task.

0.4's Ctrl+C cancels the input handler's task, and every turn runs inside the
handler (a user turn, then the job-resume turns after it), so a cancel of
``process()`` or ``process_resume()`` is the user's Ctrl+C. The turn ends its
boxes out of the transcript, prints 0.4's own "Operation cancelled..." line
once in the session's place, updates the status bar and re-raises
``CancelledError``; the next turn starts without a task box. A turn that is
still waiting for the workflow to start says so too. (0.3.1 only ended the
boxes, which a watcher polled for; it is gone.)

Each test runs the turn as a task over a scripted workflow and cancels it, as
the session does. The session is specced from ``ThinkingPromptSession``.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from agentic_cli.cli import message_processor as message_processor_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import (
    CANCELLED_LINE,
    MessageProcessor,
    _EventProcessingState,
)
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.tools.jobs import JobState
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import InputType, UserInputRequest, WorkflowEvent
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.event_replay import ReplayController

SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False, dialog_placement="box")
OUT_OF_THE_TRANSCRIPT = call(add_to_history=False, echo_to_console=False)


class _LogRecorder:
    """Stands in for the module's logger and records every call.

    ``structlog.testing.capture_logs()`` sees nothing once an earlier test has
    run ``configure_logging()`` (the logger is cached on first use); recording
    the calls keeps the test independent of test order.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    recorder = _LogRecorder()
    monkeypatch.setattr(message_processor_module, "logger", recorder)
    return recorder.events


@pytest.fixture
def ui() -> MagicMock:
    """A session whose every ``start_thinking`` opens a new box (``ui.boxes``)."""
    return opening_boxes(autospec_session())


class Workflow:
    """Streams a turn's events, then (``hang``) waits until it is cancelled.

    ``asks`` makes the turn ask the user a yes/no question first, through the
    input callback the turn installs.
    """

    def __init__(self, events=(), *, hang: bool = True, asks: bool = False) -> None:
        self.events = list(events)
        self.hang = hang
        self.asks = asks
        self.callback = None
        self.streaming = asyncio.Event()
        self.answers: list[str] = []

    def set_input_callback(self, callback) -> None:
        self.callback = callback

    def clear_input_callback(self) -> None:
        self.callback = None

    async def _stream(self):
        if self.asks:
            request = UserInputRequest(
                request_id="r",
                tool_name="approve",
                prompt="Go ahead?",
                input_type=InputType.CONFIRM,
            )
            self.answers.append(await self.callback(request))
        for event in self.events:
            yield event
        self.streaming.set()
        if self.hang:
            await asyncio.Event().wait()

    def process(self, message: str, user_id: str, session_id=None):
        return self._stream()

    async def can_resume(self, record) -> bool:
        return True

    def resume_with_job_result(self, record):
        return self._stream()


def _progress(content: str, completed: int, total: int) -> WorkflowEvent:
    return WorkflowEvent.task_progress(
        display=content,
        progress={"completed": completed, "total": total, "pending": total - completed},
    )


def _task_boxes(ui) -> list[MagicMock]:
    calls = ui.start_thinking.call_args_list
    return [box for box, c in zip(ui.boxes, calls, strict=True) if c.kwargs.get("order") == 100]


def _cancelled_lines(ui) -> list:
    return [c for c in ui.add_message.call_args_list if c == call("system", CANCELLED_LINE)]


async def _cancel_once_streaming(workflow: Workflow, turn: asyncio.Task) -> None:
    await asyncio.wait_for(workflow.streaming.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn


# --- a user turn -----------------------------------------------------------------------


async def test_a_cancelled_turn_ends_its_boxes_says_so_once_and_re_raises(ui, logged):
    workflow = Workflow([_progress("[▸] A\n[ ] B", 0, 2), WorkflowEvent.thinking("hmm")])
    controller = ReplayController(workflow)
    processor = MessageProcessor()

    turn = asyncio.create_task(processor.process("hi", controller, ui, SETTINGS))
    await _cancel_once_streaming(workflow, turn)

    events_box, task_box = ui.boxes
    assert events_box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert task_box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]
    assert CANCELLED_LINE == "Operation cancelled..."
    ui.add_warning.assert_not_called()  # no "Cancelled." of our own
    assert controller.status_bar_updates == 1
    assert workflow.callback is None
    assert [e["event"] for e in logged].count("message_cancelled_by_user") == 1


async def test_the_next_turn_starts_without_a_task_box(ui):
    workflow = Workflow([_progress("[▸] A\n[ ] B", 0, 2)])
    controller = ReplayController(workflow)
    processor = MessageProcessor()
    turn = asyncio.create_task(processor.process("hi", controller, ui, SETTINGS))
    await _cancel_once_streaming(workflow, turn)

    workflow.events, workflow.hang = [], False
    result = await processor.process("again", controller, ui, SETTINGS)

    assert result.delivered
    assert len(_task_boxes(ui)) == 1  # the cancelled turn's own
    assert (processor._task_box, processor._last_task_progress) == (None, None)


async def test_a_turn_cancelled_before_any_task_progress(ui):
    workflow = Workflow()
    turn = asyncio.create_task(
        MessageProcessor().process("hi", ReplayController(workflow), ui, SETTINGS)
    )
    await _cancel_once_streaming(workflow, turn)

    (events_box,) = ui.boxes
    assert events_box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]


# --- background jobs go on -------------------------------------------------------------


class _Jobs:
    """A job manager's listing: the records it was given."""

    def __init__(self, *records) -> None:
        self.records = list(records)

    def list(self, *, active_only: bool = False, **_):
        active = (JobState.QUEUED, JobState.RUNNING)
        return [r for r in self.records if not active_only or r.state in active]


def _job(job_id: str, name: str, state=JobState.RUNNING, *, ago: float = 0.0):
    return SimpleNamespace(job_id=job_id, name=name, state=state,
                           submitted_at=time.time() - ago)


async def test_a_cancelled_turn_names_the_background_jobs_it_started(ui):
    """Ctrl+C stops the request, not the jobs it put in the background: each
    one still queued or running is named, with how to stop it."""
    workflow = Workflow()
    workflow.job_manager = _Jobs(_job("old", "earlier", ago=60))
    turn = asyncio.create_task(
        MessageProcessor().process("hi", ReplayController(workflow), ui, SETTINGS)
    )
    await asyncio.wait_for(workflow.streaming.wait(), timeout=2)
    workflow.job_manager.records += [
        _job("j1", "build"), _job("j2", "train", JobState.QUEUED),
        _job("j3", "quick", JobState.SUCCEEDED),
    ]
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    said = [c.args[1] for c in ui.add_message.call_args_list if c.args[0] == "system"]
    assert said == [
        CANCELLED_LINE,
        "Still running in the background: build (j1), train (j2). "
        "Stop one with /jobs cancel <id>.",
    ]


async def test_a_cancelled_turn_that_started_no_job_says_only_that(ui):
    workflow = Workflow()
    workflow.job_manager = _Jobs(_job("old", "earlier", ago=60))
    turn = asyncio.create_task(
        MessageProcessor().process("hi", ReplayController(workflow), ui, SETTINGS)
    )
    await _cancel_once_streaming(workflow, turn)

    said = [c.args[1] for c in ui.add_message.call_args_list if c.args[0] == "system"]
    assert said == [CANCELLED_LINE]


# --- with a question open (Review Focus 1) ---------------------------------------------


async def test_a_cancel_with_a_question_open_closes_it_and_the_next_one_opens(ui):
    """With a question on screen no box is open, so 0.4 would end nothing and
    print nothing: the turn says it, once, and the dialog's await ends."""
    opened = asyncio.Event()
    closed: list[str] = []

    async def first_question(*args, **kwargs) -> bool:
        opened.set()
        try:
            await asyncio.Event().wait()  # the user has not answered
        except asyncio.CancelledError:
            closed.append("cancelled")
            raise
        return True  # pragma: no cover

    ui.yes_no_dialog.side_effect = first_question
    workflow = Workflow(asks=True)
    controller = ReplayController(workflow)
    processor = MessageProcessor()
    turn = asyncio.create_task(processor.process("hi", controller, ui, SETTINGS))
    await asyncio.wait_for(opened.wait(), timeout=2)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert closed == ["cancelled"]
    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]
    assert all(box.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT] for box in ui.boxes)

    ui.yes_no_dialog.side_effect = None
    ui.yes_no_dialog.return_value = True
    workflow.hang = False
    result = await processor.process("again", controller, ui, SETTINGS)

    assert result.delivered
    assert workflow.answers == ["yes"]
    assert ui.yes_no_dialog.await_count == 2
    assert len(_cancelled_lines(ui)) == 1


# --- a job-resume turn (Review Focus 2) ------------------------------------------------


class JobManager:
    def __init__(self) -> None:
        self.completed: list[tuple[str, bool, str | None]] = []

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        self.completed.append((job_id, delivered, error))


async def test_a_cancelled_resume_turn_says_so_once_records_the_failure_and_re_raises(ui):
    workflow = Workflow([WorkflowEvent.thinking("reading the result")])
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._workflow_controller = ReplayController(workflow)
    app._turn_lock = asyncio.Lock()
    app._message_processor = MessageProcessor()
    app.session = ui
    app._settings = SETTINGS
    app._usage_tracker = None
    app._session_id = "s"
    jm = JobManager()
    record = SimpleNamespace(job_id="j1", name="build", state=SimpleNamespace(value="succeeded"))

    turn = asyncio.create_task(app._deliver_resume(jm, record))
    await _cancel_once_streaming(workflow, turn)

    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]
    assert jm.completed == [("j1", False, "resume cancelled")]
    assert not app._turn_lock.locked()


class CheckingWorkflow(Workflow):
    """Its conversation lookup (``can_resume``) is slow: the resume waits."""

    def __init__(self) -> None:
        super().__init__()
        self.checking = asyncio.Event()

    async def can_resume(self, record) -> bool:
        self.checking.set()
        await asyncio.Event().wait()  # a slow session lookup
        return True  # pragma: no cover


async def test_a_resume_cancelled_while_it_checks_its_conversation_says_so_once(ui, logged):
    """Between the workflow's start and its turn, a resume looks its
    conversation up, with no box open. Ctrl+C there stops the request too, and
    the line says so once."""
    workflow = CheckingWorkflow()
    record = SimpleNamespace(job_id="j1", name="build", state=SimpleNamespace(value="succeeded"))
    turn = asyncio.create_task(
        MessageProcessor().process_resume(record, ReplayController(workflow), ui, SETTINGS)
    )
    await asyncio.wait_for(workflow.checking.wait(), timeout=2)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert ui.boxes == []
    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]
    assert [e["event"] for e in logged].count("message_cancelled_by_user") == 1


# --- the watcher is gone ----------------------------------------------------------------


# --- while the turn waits for the workflow to start ------------------------------------


class StartingManager:
    """A manager whose initialization has not finished: the turn waits."""

    backend_type = "adk"
    model = "google:gemini-2.5-flash"

    def __init__(self) -> None:
        self.is_initialized = False
        self.starting = asyncio.Event()

    async def initialize_services(self) -> None:
        self.starting.set()
        await asyncio.Event().wait()

    async def cleanup(self) -> None:
        self.is_initialized = False


@pytest.fixture
async def starting(tmp_path, monkeypatch):
    """A real controller, its manager still initializing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = StartingManager()
    controller = WorkflowController(
        [AgentConfig(name="a", prompt="p")], BaseSettings(session_store="memory")
    )
    controller._create_fn = lambda: manager
    await controller.start_background_init()
    await asyncio.wait_for(manager.starting.wait(), timeout=2)
    yield controller
    await controller.close()


@pytest.mark.parametrize("entry", ["process", "process_resume"])
async def test_a_turn_cancelled_while_it_waits_for_the_workflow_says_so_once(
    ui, logged, starting, entry
):
    """The controller's "Waiting for initialization..." box ends out of the
    transcript, so 0.4 has no box to end and prints nothing: the line is the
    turn's."""
    processor = MessageProcessor()
    record = SimpleNamespace(job_id="j1", name="build", state=SimpleNamespace(value="succeeded"))
    turn = asyncio.create_task(
        processor.process("hi", starting, ui, SETTINGS)
        if entry == "process"
        else processor.process_resume(record, starting, ui, SETTINGS)
    )
    for _ in range(100):  # until the turn is waiting
        if ui.boxes:
            break
        await asyncio.sleep(0.01)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    (waiting,) = ui.boxes
    assert ui.start_thinking.call_args.args[0]() == "Waiting for initialization..."
    assert waiting.finish.call_args_list == [OUT_OF_THE_TRANSCRIPT]
    assert _cancelled_lines(ui) == [call("system", CANCELLED_LINE)]
    assert [e["event"] for e in logged].count("message_cancelled_by_user") == 1
    assert not starting._init_task.done(), "the cancel reached the initialization"


def test_nothing_polls_for_a_cancel_any_more():
    assert not hasattr(MessageProcessor, "_watch_for_cancel")
    assert not hasattr(_EventProcessingState(), "in_hitl")

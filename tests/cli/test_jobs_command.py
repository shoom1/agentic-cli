"""``/jobs``: the job list draws in colors every theme can read; ``/jobs resume``
runs the resumes of finished jobs (what ``/resume`` did before it opened
conversations)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from rich.table import Table

from agentic_cli.cli.builtin_commands import JobsCommand
from tests.cli.dialog_fakes import autospec_session


async def test_the_job_list_names_jobs_in_the_terminal_s_own_color():
    """A fixed white is a light grey on a light terminal (the light theme)."""
    record = SimpleNamespace(
        job_id="j1",
        name="build",
        backend="subprocess",
        state=SimpleNamespace(value="running"),
        elapsed_s=lambda: 3.0,
        exit_code=None,
    )
    manager = MagicMock()
    manager.list.return_value = [record]
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("", app)

    (table,) = app.session.add_rich.call_args.args
    assert isinstance(table, Table)
    name = next(column for column in table.columns if column.header == "Name")
    assert "white" not in str(name.style)


async def test_cancelling_a_job_that_keeps_running_says_so():
    """A running in-process job cannot be stopped: it is not reported cancelled."""
    record = SimpleNamespace(state=SimpleNamespace(value="running"), cancel_requested=True)
    manager = MagicMock()
    manager.cancel.return_value = record
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    app.session.add_success.assert_not_called()
    (text,) = app.session.add_warning.call_args.args
    assert "still running" in text and "j1" in text


async def test_a_cancelled_job_is_reported_cancelled():
    record = SimpleNamespace(state=SimpleNamespace(value="cancelled"), cancel_requested=False)
    manager = MagicMock()
    manager.cancel.return_value = record
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    app.session.add_success.assert_called_once_with("Job 'j1' → cancelled.")


async def test_resume_runs_the_finished_jobs_resumes():
    app = SimpleNamespace(
        workflow=SimpleNamespace(job_manager=MagicMock()), session=autospec_session(),
        resume_finished_jobs=AsyncMock(return_value=2),
    )

    await JobsCommand().execute("resume", app)

    app.resume_finished_jobs.assert_awaited_once_with()
    app.session.add_message.assert_not_called()


async def test_resume_with_nothing_finished_says_so():
    app = SimpleNamespace(
        workflow=SimpleNamespace(job_manager=MagicMock()), session=autospec_session(),
        resume_finished_jobs=AsyncMock(return_value=0),
    )

    await JobsCommand().execute("resume", app)

    app.session.add_message.assert_called_once_with(
        "system", "No finished background jobs to resume."
    )


def _not_cancelled(
    state: str, *, registered: bool = True
) -> tuple[MagicMock, SimpleNamespace]:
    """A manager whose ``cancel`` returns the job as it was, still ``state``."""
    record = SimpleNamespace(
        state=SimpleNamespace(value=state), cancel_requested=False, backend="remote"
    )
    manager = MagicMock()
    manager.cancel.return_value = record
    manager.has_backend.return_value = registered
    return manager, record


async def test_a_job_that_was_not_cancelled_is_a_warning():
    """Another process runs it, out of reach: it is not reported cancelled."""
    manager, _ = _not_cancelled("running")
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    app.session.add_success.assert_not_called()
    (text,) = app.session.add_warning.call_args.args
    assert "'j1' was not cancelled" in text and "still running" in text


async def test_a_queued_job_that_was_not_cancelled_is_still_queued():
    manager, _ = _not_cancelled("queued")
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    (text,) = app.session.add_warning.call_args.args
    assert "still queued" in text


async def test_a_job_on_a_backend_this_app_has_not_registered_says_so():
    manager, _ = _not_cancelled("running", registered=False)
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    manager.has_backend.assert_called_once_with("remote")
    (text,) = app.session.add_warning.call_args.args
    assert "runs on the 'remote' backend, which this app has not registered" in text


async def test_a_job_that_had_already_finished_says_nothing_was_cancelled():
    record = SimpleNamespace(state=SimpleNamespace(value="succeeded"), cancel_requested=False)
    manager = MagicMock()
    manager.cancel.return_value = record
    app = SimpleNamespace(workflow=SimpleNamespace(job_manager=manager), session=autospec_session())

    await JobsCommand().execute("cancel j1", app)

    app.session.add_success.assert_not_called()
    app.session.add_message.assert_called_once_with(
        "system", "Job 'j1' had already finished (succeeded): nothing was cancelled."
    )

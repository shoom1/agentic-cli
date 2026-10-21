"""``/jobs``: the job list draws in colors every theme can read."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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

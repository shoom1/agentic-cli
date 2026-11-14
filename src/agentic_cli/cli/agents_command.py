"""``/agents``: the open conversation's sub-agent runs.

``/agents`` lists them in a table, each run followed by the runs it
started, a level deeper: its instance (``researcher-2``), its type, its
status (``running`` while it runs; a completion cut at its output limit
``completed (partial)``), its run, its title, the tokens it used, and how
long it ran (or has been running). ``/agents <id>`` shows one instance
(``researcher-2``) or one run (``researcher-2#1``): its type, its parent,
its title, then each run's status, the first lines of its result, its error
and its tokens by model.

The runs are the manager's records (``BaseWorkflowManager.agent_runs``,
``workflow/agents/runs.py``) and the tokens its usage ledger's
(``usage_of``), so a run still going shows what it has used so far. Both
live in memory with the manager: a conversation resumed from an earlier run
of the app, or after a backend switch (a new manager), lists none. Each
instance runs once.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.cli.usage_tracker import format_tokens, format_usage
from agentic_cli.workflow.agents import RUN_COMPLETED, RUN_STARTED
from agentic_cli.workflow.agents.runs import run_number

if TYPE_CHECKING:
    from agentic_cli.workflow.agents import RunRecord
    from agentic_cli.workflow.usage import UsageLedger, UsageTotals

#: How many lines of a run's result ``/agents <id>`` shows.
RESULT_LINES = 3
#: What ``/agents`` says for a conversation with no run recorded.
NO_RUNS = (
    "No sub-agent runs in this conversation yet. Runs are kept while the app "
    "runs, by its backend: a conversation resumed from an earlier run of the "
    "app, or after a backend switch, lists none."
)


def _now() -> datetime:
    """The time a running run's time is measured to (the tests fix it)."""
    return datetime.now()


class AgentsCommand(Command):
    """List the open conversation's sub-agent runs, or show one."""

    def __init__(self) -> None:
        super().__init__(
            name="agents",
            description="List this conversation's sub-agent runs, or show one",
            usage="/agents [<id>]",
            examples=["/agents", "/agents researcher-2", "/agents researcher-2#1"],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """The runs in a table, or, with an instance or run id, that one."""
        workflow = _workflow(app)
        if workflow is None:
            return
        conversation = workflow.session_ref(getattr(app, "session_id", None))
        runs = workflow.agent_runs
        records = runs.records(conversation)
        ledger = workflow.usage_of(conversation)
        parts = self.parse_args(args).positional.split()
        if parts:
            wanted = parts[0]
            found = runs.record(conversation, wanted)
            if found is None:
                app.session.add_warning(
                    f"No sub-agent {wanted} in this conversation. Type /agents for the list."
                )
                return
            # A run's id shows that run; an instance's, each of its runs.
            mine = [found] if found.ref.run == wanted else [
                r for r in records if r.ref.instance == found.ref.instance
            ]
            app.session.add_rich(_one(mine, ledger))
            return
        if not records:
            app.session.add_message("system", NO_RUNS)
            return
        app.session.add_rich(_table(records, ledger))


def _workflow(app: Any) -> Any:
    """The app's workflow; None, and the user told why, when none runs (as ``/status`` says it)."""
    try:
        workflow = app.workflow
    except (RuntimeError, AttributeError):
        workflow = None
    if workflow is None:
        failed = getattr(app, "workflow_error", None) is not None
        state = "failed to start" if failed else "initializing..."
        app.session.add_warning(f"Sub-agent runs not available ({state}).")
    return workflow


def _table(records: list["RunRecord"], ledger: "UsageLedger") -> Table:
    """``/agents``'s table: every run, each followed by the runs it started."""
    table = Table(title="Sub-agents", show_lines=False, padding=(0, 1))
    table.add_column("Agent", style="bold cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Run", style="dim", no_wrap=True)
    table.add_column("Title")  # the terminal's own color: readable on every theme
    table.add_column("Tokens in/out", style="dim", no_wrap=True, justify="right")
    table.add_column("Time", style="dim", no_wrap=True, justify="right")
    now = _now()
    for level, record in _in_tree(records):
        # Names and titles are data: Text, never markup.
        table.add_row(
            Text("  " * level + record.ref.instance),
            Text(record.ref.agent),
            _status(record),
            _run_number(record),
            Text(record.title),
            _tokens(ledger.totals(run=record.ref.run)),
            _duration(record, now),
        )
    return table


def _one(records: list["RunRecord"], ledger: "UsageLedger") -> Panel:
    """``/agents <id>``'s panel: one instance's runs (one run's, for a run id)."""
    first = records[0]
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Type", Text(first.ref.agent))
    table.add_row("Parent", Text(first.ref.parent_run or "root"))
    table.add_row("Title", Text(first.title or "—"))
    now = _now()
    for record in records:
        table.add_row(f"Run {_run_number(record)}", f"{_status(record)} · {_duration(record, now)}")
        if record.result:
            table.add_row("  Result", Text(_first_lines(record.result)))
        if record.error:
            table.add_row("  Error", Text(record.error))
        by_model = ledger.by_model(run=record.ref.run)
        for index, (model, totals) in enumerate(by_model.items()):
            table.add_row("  Tokens" if index == 0 else "", Text(f"{model}: {format_usage(totals)}"))
    return Panel(
        table, title=f"[bold]{escape(first.ref.instance)}[/bold]", border_style="cyan", expand=False
    )


def _in_tree(records: list["RunRecord"]) -> list[tuple[int, "RunRecord"]]:
    """The runs in the order they started, each followed by the runs it
    started, one level deeper. A run whose parent is not listed (the root's,
    or one gone) is at the top."""
    listed = {record.ref.run for record in records}
    started: dict[str | None, list["RunRecord"]] = {}
    for record in records:
        parent = record.ref.parent_run if record.ref.parent_run in listed else None
        started.setdefault(parent, []).append(record)
    ordered: list[tuple[int, "RunRecord"]] = []

    def follow(parent: str | None, level: int) -> None:
        for record in started.get(parent, ()):
            ordered.append((level, record))
            follow(record.ref.run, level + 1)

    follow(None, 0)
    return ordered


def _status(record: "RunRecord") -> str:
    """A run's status as ``/agents`` says it: ``running`` while it runs, and
    a completion cut at its output limit ``completed (partial)``."""
    if record.status == RUN_STARTED:
        return "running"
    if record.status == RUN_COMPLETED and record.partial:
        return "completed (partial)"
    return record.status


def _run_number(record: "RunRecord") -> str:
    """The run's number within its instance: ``#1`` of ``researcher-2#1``."""
    return "#" + run_number(record.ref.run or "")


def _tokens(totals: "UsageTotals") -> str:
    """A run's tokens in and out; ``—`` before its first model call ends."""
    if not totals.calls:
        return "—"
    return f"{format_tokens(totals.input)} / {format_tokens(totals.output)}"


def _duration(record: "RunRecord", now: datetime) -> str:
    """How long a run ran, or has been running: ``12s``, ``3m 05s``, ``1h 02m``."""
    seconds = max(0, int(((record.ended or now) - record.started).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def _first_lines(text: str) -> str:
    """The first ``RESULT_LINES`` non-blank lines of a result, ``…`` after them when there are more."""
    lines = [line for line in text.splitlines() if line.strip()]
    shown = "\n".join(lines[:RESULT_LINES])
    return shown + "\n…" if len(lines) > RESULT_LINES else shown

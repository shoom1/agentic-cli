"""``/agents`` and ``/agent``: the open conversation's sub-agents.

``/agents`` is a picker of the conversation's instances: each one's id
(``researcher-2``), type, status (``running``, ``idle``, ``stopped`` or
``closed``) and title, each followed by the instances it started, a level
deeper. Enter shows the one picked, as ``/agents <id>`` does: one instance
(``researcher-2``) or one run (``researcher-2#1``): its type, its parent, its
title, then each run's status (while it works ``running``, ``queued`` waiting
for a free working slot, or ``needs you`` waiting on the user; a completion
cut at its output limit ``completed (partial)``) and how long it ran (or has
been running), the fields of its brief its parent filled (its objective,
deliverable, constraints, what was known, when it is done; a follow-up's
message in their place), the first lines of its result, the plan and the
task list it left (its own: a run's plan and tasks are not its parent's),
its error and its tokens by model.

``/agent <id>`` shows one as well; ``/agent stop <id>`` is the user's stop
of an instance (its run, if one works, cancelled by the user; only the user
may start its next run) and ``/agent close <id>`` the user's close of it and
its own sub-agents (no more runs; its work stays readable). Each says what it
did, or why not.

The runs are the manager's records (``BaseWorkflowManager.agent_runs``,
``workflow/agents/runs.py``), kept beside the conversation in its store and
loaded with it (``load_agents``, asked here too, so a conversation resumed
from an earlier run of the app lists its instances before its first turn),
and the tokens its usage ledger's (``usage_of``), so a run still going shows
what it has used so far; a run from before a restart shows the totals its
record keeps, by model. An instance runs again when it is followed up:
``/agents <id>`` of an instance shows every run of it. The conversation the
commands act on is the app's own (``app.session_id``).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from thinking_prompt import ChoiceField, Dialog

from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.cli.usage_tracker import format_tokens, format_usage
from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.agents import RUN_COMPLETED, RUN_INPUT_REQUIRED, RUN_STARTED
from agentic_cli.workflow.agents.brief import BRIEF_FIELDS
from agentic_cli.workflow.agents.runs import run_number

if TYPE_CHECKING:
    from thinking_prompt import Placement

    from agentic_cli.workflow.agents import InstanceRecord, RunRecord
    from agentic_cli.workflow.sessions import SessionRef
    from agentic_cli.workflow.usage import UsageLedger

#: How many lines of a run's result ``/agents <id>`` shows.
RESULT_LINES = 3
#: What ``/agents`` says for a conversation with no run recorded.
NO_RUNS = "No sub-agent runs in this conversation yet."
#: The title of ``/agents``'s picker.
PICKER_TITLE = "Sub-agents"
#: What ``/agents <id>`` and ``/agent`` say for an id the conversation has no record of.
NO_SUCH_AGENT = "No sub-agent {id} in this conversation. Type /agents for the list."
#: What ``/agent`` alone says, and ``/agent stop`` or ``/agent close`` without an id.
AGENT_USAGE = "Usage: /agent <id> | /agent stop <id> | /agent close <id>"
SUBCOMMAND_USAGE = "Usage: /agent {subcommand} <id>"
#: What ``/agent stop`` says it did.
STOPPED_TEXT = "Stopped {instance}: only you can start its next run."
#: What ``/agent close`` says it did (``whose``: ``its`` for one id, ``their`` for more).
CLOSED_TEXT = "Closed {ids}; /agents {instance} still shows {whose} work."
#: What ``/agent stop`` and ``/agent close`` say for an instance that is that already.
ALREADY_TEXT = "{instance} is {status} already."


def _now() -> datetime:
    """The time a running run's time is measured to (the tests fix it)."""
    return datetime.now()


class AgentsCommand(Command):
    """Pick one of the open conversation's sub-agents, or show one."""

    def __init__(self) -> None:
        super().__init__(
            name="agents",
            description="Pick one of this conversation's sub-agents, or show one",
            usage="/agents [<id>]",
            examples=["/agents", "/agents researcher-2", "/agents researcher-2#1"],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """The instances in a picker (Enter shows one), or, with an id, that one."""
        workflow = _workflow(app)
        if workflow is None:
            return
        conversation = workflow.session_ref(getattr(app, "session_id", None))
        await _load(app, workflow, conversation)
        parts = self.parse_args(args).positional.split()
        if parts:
            show(app, workflow, conversation, parts[0])
            return
        instances = workflow.agent_runs.instances(conversation)
        if not instances:
            app.session.add_message("system", NO_RUNS)
            return
        dialog, ids = picker(instances, placement=_placement(app))
        picked = await app.session.show_dialog(dialog)
        if picked is not None and picked in ids:
            show(app, workflow, conversation, ids[picked])


class AgentCommand(Command):
    """Show, stop or close one of the open conversation's sub-agents."""

    def __init__(self) -> None:
        super().__init__(
            name="agent",
            description="Show, stop or close one of this conversation's sub-agents",
            usage="/agent <id> | stop <id> | close <id>",
            examples=["/agent researcher-2", "/agent stop researcher-2", "/agent close researcher-2"],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """``/agent <id>`` shows one; ``stop <id>`` and ``close <id>`` stop and close it."""
        workflow = _workflow(app)
        if workflow is None:
            return
        conversation = workflow.session_ref(getattr(app, "session_id", None))
        parts = self.parse_args(args).positional.split()
        if not parts:
            app.session.add_warning(AGENT_USAGE)
            return
        word = parts[0]
        if word not in ("stop", "close"):
            await _load(app, workflow, conversation)
            show(app, workflow, conversation, word)
            return
        if len(parts) < 2:
            app.session.add_warning(SUBCOMMAND_USAGE.format(subcommand=word))
            return
        wanted = parts[1]
        if not await _load(app, workflow, conversation):
            return  # records that could not be read: nothing is tried on them
        ids = {"session_id": conversation.session_id, "user_id": conversation.user_id}
        if word == "stop":
            if await workflow.stop_instance(wanted, **ids):
                app.session.add_success(STOPPED_TEXT.format(instance=wanted))
                return
        else:
            closed = await workflow.close_instance(wanted, **ids)
            if closed:
                app.session.add_success(CLOSED_TEXT.format(
                    ids=", ".join(closed), instance=wanted,
                    whose="its" if len(closed) == 1 else "their",
                ))
                return
        _left_as_it_is(app, workflow, conversation, wanted)


def picker(
    instances: "list[InstanceRecord]", *, placement: "Placement | None" = None
) -> tuple[Dialog, dict[str, str]]:
    """The dialog that lists ``instances``, and the instance id each option stands for.

    One option per instance, ``<id> · <type> · <status> · <title>``, each
    followed by the instances it started, two spaces deeper. Its only field
    is a choice list, so Enter on an option closes it with that option, and
    Esc with None. ``placement`` is where it is drawn (None: the session's).
    """
    ids: dict[str, str] = {}
    for level, one in _instances_in_tree(instances):
        label = f"{'  ' * level}{one.instance} · {one.agent} · {one.status}"
        ids[f"{label} · {one.title}" if one.title else label] = one.instance
    dialog = Dialog(
        PICKER_TITLE, fields=[ChoiceField("agent", "", options=list(ids))], placement=placement
    )
    return dialog, ids


def show(app: Any, workflow: Any, conversation: "SessionRef", wanted: str) -> None:
    """Show the instance or run ``wanted`` (``researcher-2``, ``researcher-2#1``), or say there is none."""
    runs = workflow.agent_runs
    found = runs.record(conversation, wanted)
    if found is None:
        app.session.add_warning(NO_SUCH_AGENT.format(id=wanted))
        return
    # A run's id shows that run; an instance's, each of its runs.
    mine = [found] if found.ref.run == wanted else [
        r for r in runs.records(conversation) if r.ref.instance == found.ref.instance
    ]
    app.session.add_rich(_one(mine, workflow.usage_of(conversation)))


async def _load(app: Any, workflow: Any, conversation: "SessionRef") -> bool:
    """Load the conversation's records from its store; whether they were read.

    Not read: the user is told, and the runs the manager holds are shown all the same.
    """
    try:
        await workflow.load_agents(conversation)
    except Exception as error:  # noqa: BLE001 - the runs held are shown all the same
        app.session.add_warning(
            f"Sub-agent records not read from the store: {str(error) or type(error).__name__}"
        )
        return False
    return True


def _left_as_it_is(app: Any, workflow: Any, conversation: "SessionRef", wanted: str) -> None:
    """Say why ``/agent stop`` or ``/agent close`` did nothing to ``wanted``."""
    record = workflow.agent_runs.instance(conversation, wanted)
    if record is None:
        app.session.add_warning(NO_SUCH_AGENT.format(id=wanted))
        return
    app.session.add_message(
        "system", ALREADY_TEXT.format(instance=wanted, status=record.status)
    )


def _placement(app: Any) -> "Placement | None":
    """Where ``/agents``'s picker is drawn: the ``dialog_placement`` setting now
    (``/settings`` may have changed it since the session started); the session's own without one."""
    settings = getattr(app, "settings", None)
    return None if settings is None else settings.dialog_placement


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
        for name in BRIEF_FIELDS:
            if record.brief.get(name):
                table.add_row(f"  {_label(name)}", Text(record.brief[name]))
        if record.input:
            table.add_row("  Message", Text(_first_lines(record.input)))
        if record.result:
            table.add_row("  Result", Text(_first_lines(record.result)))
        if record.plan:
            table.add_row("  Plan", Text(_first_lines(record.plan)))
        progress = task_progress_data(record.tasks)
        if progress is not None:
            counts = progress["progress"]
            shown = _first_lines(progress["display"])
            table.add_row("  Tasks", Text(f"{counts['completed']}/{counts['total']}\n{shown}"))
        if record.error:
            table.add_row("  Error", Text(record.error))
        for index, line in enumerate(_tokens_by_model(record, ledger)):
            table.add_row("  Tokens" if index == 0 else "", Text(line))
    return Panel(
        table, title=f"[bold]{escape(first.ref.instance)}[/bold]", border_style="cyan", expand=False
    )


def _instances_in_tree(instances: "list[InstanceRecord]") -> list[tuple[int, "InstanceRecord"]]:
    """The instances in the order they started, each followed by the ones it
    started, one level deeper. One whose parent is not listed (the root's,
    or one gone) is at the top."""
    listed = {one.instance for one in instances}
    started: dict[str | None, list["InstanceRecord"]] = {}
    for one in instances:
        parent = one.parent if one.parent in listed else None
        started.setdefault(parent, []).append(one)
    ordered: list[tuple[int, "InstanceRecord"]] = []

    def follow(parent: str | None, level: int) -> None:
        for one in started.get(parent, ()):
            ordered.append((level, one))
            follow(one.instance, level + 1)

    follow(None, 0)
    return ordered


def _status(record: "RunRecord") -> str:
    """A run's status as ``/agents`` says it: as recorded (``running``,
    ``queued``, an end), but ``needs you`` while it waits on the user, and
    a completion cut at its output limit ``completed (partial)``."""
    if record.status == RUN_STARTED:
        return "running"
    if record.status == RUN_INPUT_REQUIRED:
        return "needs you"
    if record.status == RUN_COMPLETED and record.partial:
        return "completed (partial)"
    return record.status


def _label(field: str) -> str:
    """A brief field's label: ``done_when`` is ``Done when``."""
    return field.replace("_", " ").capitalize()


def _run_number(record: "RunRecord") -> str:
    """The run's number within its instance: ``#1`` of ``researcher-2#1``."""
    return "#" + run_number(record.ref.run or "")


def _tokens_by_model(record: "RunRecord", ledger: "UsageLedger") -> list[str]:
    """A run's tokens, a line per model: the ledger's, else its record's (no call count)."""
    by_model = ledger.by_model(run=record.ref.run)
    if by_model:
        return [f"{model}: {format_usage(totals)}" for model, totals in by_model.items()]
    return [
        f"{model}: {format_tokens(totals.get('input', 0))} in / "
        f"{format_tokens(totals.get('output', 0))} out"
        for model, totals in record.usage.items()
    ]


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

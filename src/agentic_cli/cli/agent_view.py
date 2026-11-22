"""A sub-agent's view: ``/agent <id>`` and ``/back`` (provisional).

The user works with one instance of the open conversation directly, as with
the main agent. Entering its view (``/agent <id>``, or Enter in ``/agents``'
picker) prints a rule, ``── researcher-2 · researcher · idle ──``, and a
recap of it: its brief's title and objective, its last run's status, the
first lines of its result and its tokens; nothing is replayed. The prompt
becomes ``researcher-2> `` and the status bar names it and its status, its
context (``ctx``: its own last prompt against its own summary trigger, the
numbers of its model choice when it runs on one) and
its tokens, the jobs segment kept.

A line typed there is the user's message to the instance: while it is idle
(or stopped: only the user may continue one) its next run (the manager's
``run_instance``), drawn as a turn is, its text the reply; while its run
works, given to that run before its next model step (``steer_instance``),
never queued as a turn. The run works in a task of its own, outside the
input handler: thinking-prompt takes no line while a handler runs, so the
prompt stays open for the lines that steer it, and Ctrl+C stops it through
the binding ``bind_ctrl_c`` adds (the user's stop: the instance is
``stopped``). The app's exit interrupts a run still working
(``AgentView.interrupt``), as a process exit interrupts any run: no stop of
the user's, so the instance stays ``idle``. A closed instance's view is
read-only: no prompt, and a line typed there is refused.

``/back`` leaves: it waits for a run still working, prints ``── back to main
──``, and, when a run the user started there ended since the main agent was
last told, tells the main agent once, in a notice turn (``tell_root`` with
``agents/envelope.py::view_notice``) drawn as a job's notice is; then the
finished jobs that waited for the conversation are told
(``job_auto_resume``). While a view is open the main agent's notices wait,
counted in ``↻N``. ``/compact`` is refused in a view; ``/new`` and
``/resume`` leave it first; ``/status`` shows the instance's context.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from rich.table import Table
from rich.text import Text

from agentic_cli.cli.agents_command import (
    NO_SUCH_AGENT,
    _first_lines,
    _load,
    _run_number,
    _status,
    _workflow,
)
from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.cli.message_processor import MessageProcessor
from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens
from agentic_cli.logging import Loggers
from agentic_cli.workflow.agents import INSTANCE_CLOSED, INSTANCE_RUNNING
from agentic_cli.workflow.agents.envelope import view_notice
from agentic_cli.workflow.usage import PURPOSE_SUBAGENT

if TYPE_CHECKING:
    from agentic_cli.workflow.agents import InstanceRecord, RunRecord
    from agentic_cli.workflow.sessions import SessionRef

logger = Loggers.cli()

#: The rule a view opens with.
VIEW_RULE = "── {instance} · {agent} · {status} ──"
#: The rule ``/back`` prints.
BACK_RULE = "── back to main ──"
#: The prompt in an instance's view (a closed one's has none).
VIEW_PROMPT = "{instance}> "
#: What a line typed in a closed instance's view is told.
READ_ONLY = "{instance} is closed: its view is read-only. /back returns to the main conversation."
#: What ``/compact`` says in a view.
COMPACT_IN_VIEW = "/compact works in the main conversation only."
#: What ``/back`` says outside a view.
NOT_IN_VIEW = "Not in a sub-agent's view: /agent <id> opens one."
#: What a line given to the run in view says.
STEERED = "Given to {instance}'s run before its next step."
#: The line before the main agent's notice turn after a view.
TELLING = "↻ Telling the main agent what you did with {instance}."
#: What ``/jobs resume`` says in a view.
JOBS_WAIT = "In a sub-agent's view the main agent's notices wait: /back tells them."


class AgentView:
    """One instance's view, open in the app (``BaseCLIApp.agent_view``).

    Args:
        app: The app it is open in.
        instance: The instance in view (``researcher-2``).
        conversation: The conversation it belongs to (the app's).
        told: The run of the user's whose envelope the main agent knows of
            (``instance_result``'s, as the view opened): ``/back`` tells it
            only of a later one.
    """

    def __init__(
        self, app: Any, instance: str, conversation: "SessionRef", told: str | None
    ) -> None:
        self.app = app
        self.instance = instance
        self.conversation = conversation
        self.told = told
        #: The view's own message processor: its runs' task list is the
        #: instance's, never the main agent's task box.
        self.processor = MessageProcessor()
        #: The prompt the app had: put back by ``/back``.
        self.prompt_before: Any = None
        self._runs: set[asyncio.Task[None]] = set()
        # The run started last: a line typed before it ends is the next run,
        # which waits for it here (and not for the turn lock, where a Ctrl+C
        # would make it say "Operation cancelled..." once more).
        self._last: asyncio.Task[None] | None = None

    # -- what it shows ---------------------------------------------------------------

    def record(self) -> "InstanceRecord | None":
        """The instance's record now; None when it is gone (or no workflow runs)."""
        try:
            return self.app.workflow.agent_runs.instance(self.conversation, self.instance)
        except (RuntimeError, AttributeError):
            return None

    def prompt(self) -> str:
        """``researcher-2> ``; none for a closed instance's view, which is read-only."""
        record = self.record()
        if record is not None and record.status == INSTANCE_CLOSED:
            return ""
        return VIEW_PROMPT.format(instance=self.instance)

    def status_segment(self) -> str:
        """What the status bar says of the instance: ``researcher-2 · idle |
        ctx 4k/170k (2%) | Tokens: 4.2k in / 310 out`` (its context and tokens
        once it has any). Never raises: the bar is drawn from timers too."""
        try:
            workflow = self.app.workflow
            record = workflow.agent_runs.instance(self.conversation, self.instance)
            if record is None:
                return self.instance
            parts = [f"{self.instance} · {record.status}"]
            context = _context(workflow, self.conversation, record)
            if context:
                parts.append(context)
            tokens_in, tokens_out = instance_tokens(workflow, self.conversation, self.instance)
            if tokens_in or tokens_out:
                parts.append(
                    f"Tokens: {format_tokens(tokens_in)} in / {format_tokens(tokens_out)} out"
                )
            return " | ".join(parts)
        except Exception:  # noqa: BLE001 - the bar names the instance at least
            logger.debug("view_status_failed", instance=self.instance, exc_info=True)
            return self.instance

    def context_rows(self) -> list[tuple[str, str]]:
        """``/status``' rows for the instance's context, in place of the main agent's.

        The instance, its type and status; its model, the model's window,
        its input budget and the prompt size its transcript is summarized
        (or trimmed) at, when its manager manages its context; its last
        prompt and its tokens.
        """
        from agentic_cli.workflow.context import UNKNOWN_WINDOW

        workflow = self.app.workflow
        record = workflow.agent_runs.instance(self.conversation, self.instance)
        if record is None:
            return [("Sub-agent", f"{self.instance} (gone)")]
        rows = [("Sub-agent", f"{self.instance} ({record.agent}, {record.status})")]
        model = _model(workflow, record)
        if model is not None:
            rows.append(("Model", str(model.ref)))
        budget = _budget(workflow, record)
        if budget is not None:
            window = (
                format_tokens(budget.window) if budget.window_known
                else f"unknown, {format_tokens(UNKNOWN_WINDOW)} floor in use"
            )
            rows += [
                ("Context Window", window),
                ("Input Budget", format_tokens(budget.budget)),
                ("Summarize At", format_tokens(budget.trigger)),
            ]
        last = _last_prompt(workflow, self.conversation, record)
        rows.append(("Last Prompt", format_tokens(last) if last else "—"))
        tokens_in, tokens_out = instance_tokens(workflow, self.conversation, self.instance)
        rows.append(("Tokens", f"{format_tokens(tokens_in)} in / {format_tokens(tokens_out)} out"))
        return rows

    def show(self, record: "InstanceRecord") -> None:
        """Print the view's rule and the instance's recap."""
        session = self.app.session
        session.add_message("system", VIEW_RULE.format(
            instance=self.instance, agent=record.agent, status=record.status,
        ))
        session.add_rich(recap(self.app.workflow, self.conversation, self.instance))

    def refresh(self) -> None:
        """Draw its prompt and status bar again, while it is the app's view:
        the instance's status may have changed (a run ended, ``/agent
        close``)."""
        app = self.app
        if app.agent_view is not self:
            return
        app.session.message = self.prompt()
        app._workflow_controller.update_status_bar(app.session)

    # -- its runs ----------------------------------------------------------------------

    @property
    def working(self) -> bool:
        """Whether a run the user started here is still going (or waits to start)."""
        return any(not run.done() for run in self._runs)

    async def take(self, line: str) -> None:
        """A line the user typed in the view: given to the run that works, or
        the instance's next run, started in a task of its own (``settled``
        waits for it). A closed instance's view refuses it."""
        app = self.app
        # A workflow that is down (a model switch whose rebuild failed) has no
        # record to read: the line is the next run all the same, whose start
        # retries it as a turn of the main conversation does.
        record = self.record()
        if record is not None and record.status == INSTANCE_CLOSED:
            app.session.add_warning(READ_ONLY.format(instance=self.instance))
            return
        app.session.add_message("user", line)
        if (
            record is not None
            and record.status == INSTANCE_RUNNING
            and app.workflow.steer_instance(
                self.instance,
                line,
                session_id=self.conversation.session_id,
                user_id=self.conversation.user_id,
            )
        ):
            app.session.add_message("system", STEERED.format(instance=self.instance))
            return
        run = asyncio.create_task(
            self._run(line, self._last), name=f"view run of {self.instance}"
        )
        self._last = run
        self._runs.add(run)
        run.add_done_callback(self._ended)

    async def _run(self, line: str, after: "asyncio.Task[None] | None") -> None:
        """The instance's next run on ``line``, once the run before it ended: a
        write to the conversation under its turn lock
        (``ConversationCore.run_instance``). A stop while it waits is silent:
        the run that works says it."""
        app = self.app
        if after is not None:
            await asyncio.wait({after})
        parts = dataclasses.replace(app._turn_parts(), processor=self.processor)
        try:
            await app._conversation.run_instance(self.instance, line, parts)
        finally:
            self.refresh()

    def _ended(self, run: "asyncio.Task[None]") -> None:
        """A run's task ended: forgotten; a failure the turn did not report is said."""
        self._runs.discard(run)
        if run.cancelled():
            return
        error = run.exception()
        if error is not None:
            logger.info(
                "view_run_failed", instance=self.instance, error=str(error),
                error_type=type(error).__name__,
            )
            self.app.session.add_error(f"Error: {str(error) or type(error).__name__}")

    def stop(self) -> bool:
        """Stop the run in view (the user's stop: the instance is ``stopped``).
        Whether one was working."""
        working = [run for run in self._runs if not run.done()]
        for run in working:
            run.cancel()
        return bool(working)

    def interrupt(self) -> bool:
        """Interrupt the run in view, as a process exit does: no stop of the user's.

        The manager ends it as a turn's cancel (``interrupt_instance``): the
        run ``interrupted``, the instance ``idle``, so its parent may continue
        it. A line waiting to run is cancelled before it starts. Whether one
        was working; ``settled`` waits for it.
        """
        working = [run for run in self._runs if not run.done()]
        if not working:
            return False
        try:
            self.app.workflow.interrupt_instance(
                self.instance,
                session_id=self.conversation.session_id,
                user_id=self.conversation.user_id,
            )
        except (RuntimeError, AttributeError):
            pass  # no workflow runs: no run of it works
        for run in working:
            run.cancel()
        return True

    async def settled(self) -> None:
        """Wait until no run the user started here works (none is stopped)."""
        while self._runs:
            await asyncio.wait(set(self._runs))


# -- entering and leaving ------------------------------------------------------------


async def open_view(app: Any, instance: str) -> bool:
    """Open ``instance``'s view in ``app`` (``BaseCLIApp.open_agent_view``).

    The conversation's records are loaded first (one that cannot be read is
    said, and no view opens), and an unknown id is said
    (``NO_SUCH_AGENT``). In its own view it prints the rule and recap again;
    in another's it leaves that one first, as ``/back`` does.

    Returns:
        Whether the view is open.
    """
    workflow = _workflow(app)
    if workflow is None:
        return False
    conversation = workflow.session_ref(getattr(app, "session_id", None))
    if not await _load(app, workflow, conversation):
        return False
    record = workflow.agent_runs.instance(conversation, instance)
    if record is None:
        app.session.add_warning(NO_SUCH_AGENT.format(id=instance))
        return False
    current = app.agent_view
    if current is not None:
        if current.instance == instance and current.conversation == conversation:
            current.show(record)
            current.refresh()
            return True
        await leave_view(app)
    latest = workflow.instance_result(
        instance, session_id=conversation.session_id, user_id=conversation.user_id
    )
    view = AgentView(app, instance, conversation, told=(latest or {}).get("run"))
    view.show(record)
    view.prompt_before = app.session.message
    app._agent_view = view
    app._workflow_controller.view_segment = view.status_segment
    view.refresh()
    logger.info("agent_view_opened", instance=instance)
    return True


async def leave_view(app: Any) -> bool:
    """``/back`` (``BaseCLIApp.leave_agent_view``): leave the view open, back
    to the main conversation; False, said, when none is.

    A run still working there ends first (Ctrl+C stops it); then the rule,
    the app's prompt and status bar; then, when a run the user started there
    ended since the main agent last heard of the instance, one notice turn
    (``ConversationCore.tell_root``) with its latest envelope; then the
    finished jobs that waited (``job_auto_resume``).
    """
    view = app.agent_view
    if view is None:
        app.session.add_message("system", NOT_IN_VIEW)
        return False
    await view.settled()
    app._agent_view = None
    controller = app._workflow_controller
    controller.view_segment = None
    app.session.message = view.prompt_before
    app.session.add_message("system", BACK_RULE)
    controller.update_status_bar(app.session)
    logger.info("agent_view_left", instance=view.instance)
    # A workflow that is down (a model switch whose rebuild failed) is started
    # again, as a turn would: the run the user did there is still to be told.
    # Left down, the notice cannot be sent: the main agent is not told of that
    # run, and the warning says only that the workflow did not start.
    if not controller.is_ready:
        await controller.ensure_initialized(app.session)
    workflow = _workflow(app)
    if workflow is None:
        return True
    latest = workflow.instance_result(
        view.instance, session_id=view.conversation.session_id,
        user_id=view.conversation.user_id,
    )
    if latest is not None and latest.get("run") != view.told:
        app.session.add_message("system", TELLING.format(instance=view.instance))
        await app._conversation.tell_root(
            view_notice(view.instance, latest), app._turn_parts()
        )
    await app._tell_its_finished_jobs()
    return True


def bind_ctrl_c(session: Any, view: Callable[[], AgentView | None]) -> bool:
    """Make Ctrl+C stop the run in view (``AgentView.stop``) while one works.

    thinking-prompt's Ctrl+C cancels the input handler's task, and a view's
    run works in a task of its own (the prompt stays open to steer it), so
    the session's key bindings get one more, active only while that run
    works; prompt_toolkit runs the last binding added whose filter holds.
    Any other time Ctrl+C is thinking-prompt's.

    Returns:
        Whether it was bound: a session whose application has no key
        bindings of prompt_toolkit's own (a fake) gets none.
    """
    bindings = getattr(getattr(session, "app", None), "key_bindings", None)
    if not isinstance(bindings, KeyBindings):
        return False

    def working() -> bool:
        current = view()
        return current is not None and current.working

    @bindings.add("c-c", filter=Condition(working))
    def stop_the_run_in_view(event: Any) -> None:
        current = view()
        if current is not None:
            current.stop()

    return True


class BackCommand(Command):
    """Leave a sub-agent's view, back to the main conversation."""

    def __init__(self) -> None:
        super().__init__(
            name="back",
            description="Leave a sub-agent's view (/agent <id>), back to the main conversation",
            usage="/back",
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Leave the view open (``leave_agent_view``), or say there is none."""
        await app.leave_agent_view()


# -- what the view reads ---------------------------------------------------------------


def recap(workflow: Any, conversation: "SessionRef", instance: str) -> Table:
    """A view's recap of ``instance``: its title, its first brief's objective,
    its last run's status, the first lines of its result, and its tokens."""
    runs = _runs_of(workflow, conversation, instance)
    record = workflow.agent_runs.instance(conversation, instance)
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Title", Text((record.title if record is not None else "") or "—"))
    if runs:
        objective = runs[0].brief.get("objective")
        if objective:
            table.add_row("Objective", Text(_first_lines(objective)))
        last = runs[-1]
        table.add_row("Last run", Text(f"{_run_number(last)} · {_status(last)}"))
        if last.result:
            table.add_row("Result", Text(_first_lines(last.result)))
    tokens_in, tokens_out = instance_tokens(workflow, conversation, instance)
    table.add_row(
        "Tokens", Text(f"{format_tokens(tokens_in)} in / {format_tokens(tokens_out)} out")
    )
    return table


def instance_tokens(workflow: Any, conversation: "SessionRef", instance: str) -> tuple[int, int]:
    """The tokens ``instance``'s runs used, in and out: each run's from the
    usage ledger, or its record's totals (a run from before a restart)."""
    ledger = workflow.usage_of(conversation)
    tokens_in = tokens_out = 0
    for record in _runs_of(workflow, conversation, instance):
        by_model = ledger.by_model(run=record.ref.run)
        if by_model:
            tokens_in += sum(totals.input for totals in by_model.values())
            tokens_out += sum(totals.output for totals in by_model.values())
        else:
            tokens_in += sum(totals.get("input", 0) for totals in record.usage.values())
            tokens_out += sum(totals.get("output", 0) for totals in record.usage.values())
    return tokens_in, tokens_out


def _runs_of(workflow: Any, conversation: "SessionRef", instance: str) -> list["RunRecord"]:
    """``instance``'s runs, oldest first."""
    return [
        record for record in workflow.agent_runs.records(conversation)
        if record.ref.instance == instance
    ]


def _model(workflow: Any, record: "InstanceRecord") -> Any:
    """The model the instance runs on: its model choice's when it has one, else its type's."""
    if record.model is not None:
        return getattr(workflow, "choice_models", {}).get((record.agent, record.model))
    return getattr(workflow, "agent_models", {}).get(record.agent)


def _budget(workflow: Any, record: "InstanceRecord") -> Any:
    """The instance's context budget (its model choice's, when it has one); None when its manager manages none."""
    try:
        if record.model is not None:
            return workflow.context_budget(record.agent, model=record.model)
        return workflow.context_budget(record.agent)
    except (KeyError, RuntimeError, AttributeError):
        return None


def _last_prompt(workflow: Any, conversation: "SessionRef", record: "InstanceRecord") -> int:
    """The instance's last prompt, in tokens: its last own step's in the ledger (0: none yet)."""
    runs = set(record.runs)
    return next(
        (
            row.input for row in reversed(workflow.usage_of(conversation).rows)
            if row.run in runs and row.purpose == PURPOSE_SUBAGENT
        ),
        0,
    )


def _context(workflow: Any, conversation: "SessionRef", record: "InstanceRecord") -> str:
    """The status bar's ``ctx`` for the instance: its last prompt against its
    own trigger (its model choice's, when it has one), as the main agent's is shown (``UsageTracker.format_context``)."""
    budget = _budget(workflow, record)
    return UsageTracker(
        last_prompt_tokens=_last_prompt(workflow, conversation, record),
        context_trigger=budget.trigger if budget is not None else None,
        window_known=budget.window_known if budget is not None else None,
    ).format_context()

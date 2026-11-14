"""Built-in slash commands for the CLI."""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.cli.conversation import NOT_KEPT
from agentic_cli.cli.message_processor import _says_so_when_stopped
from agentic_cli.cli.session_resume import NO_PROMPT

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp
    from agentic_cli.models import ModelCatalog, ModelRecord, ProviderStatus, ThinkingRule


class HelpCommand(Command):
    """Display help information about available commands."""

    def __init__(self) -> None:
        super().__init__(
            name="help",
            description="Show available commands and usage information",
            aliases=[],
            usage="/help [command]",
            examples=["/help", "/help settings"],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: Any) -> None:
        """The commands by category, or, with a name, that command's help."""
        registry = app.command_registry
        name = args.strip().lstrip("/")
        if not name:
            app.session.add_rich(_command_list(registry.all_commands()))
            return
        command = registry.get(name)
        if command is None:
            app.session.add_warning(f"No command /{name}. Type /help for the list.")
            return
        app.session.add_rich(_command_help(command))


def _command_list(commands: list[Command]) -> Panel:
    """Every command under its category, the categories in their declared order,
    then any command of another category (an app's) under "Other"; nothing is
    read as markup."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Aliases", style="dim", no_wrap=True)
    table.add_column("Description")
    by_category: dict[Any, list[Command]] = {}
    for command in commands:
        by_category.setdefault(command.category, []).append(command)
    groups = [(category.value.capitalize(), by_category.pop(category, [])) for category in CommandCategory]
    groups.append(("Other", [command for rest in by_category.values() for command in rest]))
    for heading, group in groups:
        if not group:
            continue
        if table.row_count:
            table.add_row("", "", "")
        table.add_row(Text(heading, style="bold"), "", "")
        for command in sorted(group, key=lambda c: c.name):
            aliases = ", ".join(f"/{alias}" for alias in command.aliases)
            table.add_row(Text(f"/{command.name}"), Text(aliases), Text(command.description))
    body = Group(table, Text(""), Text("/help <command> shows its usage and examples.", style="dim"))
    return Panel(body, title="[bold]Available Commands[/bold]", border_style="cyan")


def _command_help(command: Command) -> Panel:
    """One command: what it does, its aliases, usage and examples (never read as markup)."""
    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim", no_wrap=True)
    details.add_column()
    details.add_row("Usage", Text(command.usage))
    if command.aliases:
        details.add_row("Aliases", Text(", ".join(f"/{alias}" for alias in command.aliases)))
    if command.examples:
        details.add_row("Examples", Text("\n".join(command.examples)))
    title = Text(f"/{command.name}", style="bold")
    return Panel(Group(Text(command.description), Text(""), details), title=title, border_style="cyan")


class ClearCommand(Command):
    """Clear the screen."""

    def __init__(self) -> None:
        super().__init__(
            name="clear",
            description="Clear the screen",
            aliases=[],
            category=CommandCategory.GENERAL,
            silent=True,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Clear the console."""
        app.session.clear()


class ExitCommand(Command):
    """Exit the application."""

    def __init__(self) -> None:
        super().__init__(
            name="exit",
            description="Exit the application",
            aliases=["quit"],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Exit the application."""
        app.session.add_message("system", "Exiting...")
        app.stop()


def _context_rows(workflow: Any, tracker: Any) -> list[tuple[str, str]]:
    """The main agent's context, as ``/status`` rows.

    The agent and its model, the model's window (or the 128k floor in use),
    the input budget, the prompt size its conversation is summarized at, and
    its last prompt. A manager with no budget for its root agent (not
    initialized yet; LangGraph, which manages no context; a native ADK YAML
    app whose root's model did not resolve; one without ``context_budget``)
    gives only the last prompt, once there is one. Any other error is the manager's, and is not hidden here.
    """
    from agentic_cli.cli.usage_tracker import format_tokens
    from agentic_cli.workflow.context import UNKNOWN_WINDOW

    last = getattr(tracker, "last_prompt_tokens", 0) or 0
    last_prompt = ("Last Prompt", format_tokens(last) if last else "—")
    context_budget = getattr(workflow, "context_budget", None)
    if context_budget is None:
        return [last_prompt] if last else []
    try:
        budget = context_budget()
    except (KeyError, RuntimeError):
        return [last_prompt] if last else []
    root = workflow.root_agent_name
    model = workflow.agent_models[root]
    window = (
        format_tokens(budget.window) if budget.window_known
        else f"unknown, {format_tokens(UNKNOWN_WINDOW)} floor in use"
    )
    return [
        ("Main Agent", f"{root} ({model.ref})"),
        ("Context Window", window),
        ("Input Budget", format_tokens(budget.budget)),
        ("Summarize At", format_tokens(budget.trigger)),
        last_prompt,
    ]


#: How ``/status`` names the side calls of each purpose (``workflow/usage.py``).
_SIDE_CALLS = {
    "summarization": "summaries",
    "tool": "tools' summaries",
    "memory_extraction": "memory proposals",
    "side_call": "other side calls",
}


def _usage_rows(workflow: Any) -> list[tuple[str, str]]:
    """The open conversation's tokens by agent, as ``/status`` rows.

    The main agent first, then each sub-agent, with how many runs it made
    (``agent_runs``), then the side calls by purpose, from the manager's
    ledger (``usage``); none before the first call, or from a manager
    without one.
    """
    from agentic_cli.cli.usage_tracker import format_usage

    ledger = getattr(workflow, "usage", None)
    if ledger is None:
        return []
    by_agent, by_purpose = ledger.by_agent(), ledger.by_purpose()
    if not by_agent and not by_purpose:
        return []
    runs: dict[str, int] = {}
    for record in workflow.agent_runs.records(workflow.session_ref()):
        runs[record.ref.agent] = runs.get(record.ref.agent, 0) + 1
    root = workflow.root_agent_name
    rows: list[tuple[str, str]] = [("", ""), ("Tokens by Agent", "")]
    # The main agent first; the others as they first ran (sorted is stable).
    for name in sorted(by_agent, key=lambda name: name != root):
        count = runs.get(name, 0)
        label = f"  {name}" if not count else f"  {name} ({count} run{'s' if count != 1 else ''})"
        rows.append((label, format_usage(by_agent[name])))
    for purpose, totals in by_purpose.items():
        rows.append((f"  {_SIDE_CALLS.get(purpose, purpose)}", format_usage(totals)))
    return rows


class StatusCommand(Command):
    """Show current session status."""

    def __init__(self) -> None:
        super().__init__(
            name="status",
            description="Show current session and workflow status",
            aliases=[],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Display current session status."""
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value")

        # Workflow manager status
        try:
            workflow = app.workflow
            if workflow is None:
                raise RuntimeError("Workflow not initialized")
            table.add_row("Model", workflow.model)
            table.add_row("App Name", workflow.app_name)
            table.add_row("Session ID", workflow.session_id)
            services = "Initialized" if workflow.is_initialized else "Not initialized"
            table.add_row("Services", services)
            for label, value in _context_rows(workflow, getattr(app, "usage_tracker", None)):
                table.add_row(label, value)
        except (RuntimeError, AttributeError):
            # Why it did not start (or its last rebuild failed), when it did
            # not: BaseCLIApp.workflow_error. An app object without it is
            # taken as still starting.
            error = getattr(app, "workflow_error", None)
            state = "failed to start" if error is not None else "initializing..."
            table.add_row("Workflow", Text(f"Not available ({state})", style="yellow"))
            if error is not None:
                # As text, not markup: an error may hold brackets.
                table.add_row("Error", Text(str(error) or type(error).__name__, style="red"))

        # Persistent session info (none on the memory store: it is the only one)
        sid = app.session_id
        if sid and not _keeps_conversations(app):
            table.add_row("Session", f"{sid} (not kept: session_store is memory)")
        elif sid:
            table.add_row("Session", f"{sid} (persistent)")
        else:
            table.add_row("Session", "ephemeral (not saved)")

        # Token usage breakdown
        tracker = getattr(app, "usage_tracker", None)
        if tracker is not None:
            for label, value in tracker.format_detail_rows():
                table.add_row(label, value)

        # By agent: the root, each sub-agent and the side calls
        try:
            by_agent = _usage_rows(app.workflow)
        except (RuntimeError, AttributeError):
            by_agent = []  # no workflow: its rows above say why
        for label, value in by_agent:
            table.add_row(label, value)

        panel = Panel(table, title="[bold]Session Status[/bold]", border_style="cyan")
        app.session.add_rich(panel)


class SandboxCommand(Command):
    """Manage sandbox sessions."""

    def __init__(self) -> None:
        super().__init__(
            name="sandbox",
            description="List and manage sandbox sessions",
            aliases=["sb"],
            usage="/sandbox [reset [session_id] [--all]]",
            examples=["/sandbox", "/sandbox reset", "/sandbox reset my_session", "/sandbox reset --all"],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """List or reset sandbox sessions."""
        # Get sandbox manager from workflow
        try:
            workflow = app.workflow
            manager = getattr(workflow, "sandbox_manager", None)
        except (RuntimeError, AttributeError):
            manager = None

        if manager is None:
            app.session.add_warning(
                "Sandbox not available. Add sandbox tools to your agent config to enable it."
            )
            return

        parsed = self.parse_args(args)
        subcommand = parsed.positional.strip()

        if subcommand.startswith("reset"):
            # Parse session_id from after "reset"
            rest = subcommand[len("reset"):].strip()
            if parsed.has_flag("all"):
                # Reset all sessions
                sessions = manager.list_sessions()
                if not sessions:
                    app.session.add_message("system", "No active sandbox sessions.")
                    return
                for s in sessions:
                    manager.reset_session(s["session_id"])
                app.session.add_success(f"Reset {len(sessions)} sandbox session(s).")
            elif rest:
                was_active = manager.reset_session(rest)
                if was_active:
                    app.session.add_success(f"Sandbox session '{rest}' reset.")
                else:
                    app.session.add_warning(f"Sandbox session '{rest}' was not active.")
            else:
                # No id: session ids are namespaced per conversation (conv-<id>),
                # so a literal "default" rarely matches. Reset the current sandbox
                # (the single active session), or ask the user to pick if ambiguous.
                sessions = manager.list_sessions()
                if not sessions:
                    app.session.add_message("system", "No active sandbox sessions.")
                elif len(sessions) == 1:
                    sid = sessions[0]["session_id"]
                    manager.reset_session(sid)
                    app.session.add_success(f"Sandbox session '{sid}' reset.")
                else:
                    ids = ", ".join(s["session_id"] for s in sessions)
                    app.session.add_warning(
                        f"Multiple sandbox sessions active ({ids}). "
                        "Reset one with /sandbox reset <id>, or all with /sandbox reset --all."
                    )
        else:
            # List sessions
            sessions = manager.list_sessions()
            if not sessions:
                app.session.add_message("system", "No active sandbox sessions.")
                return

            table = Table(title="Sandbox Sessions", show_lines=False, padding=(0, 1))
            table.add_column("Session ID", style="bold cyan", no_wrap=True)
            table.add_column("Working Dir", style="dim")
            table.add_column("Executions", style="dim", no_wrap=True, justify="right")

            for s in sessions:
                table.add_row(
                    s["session_id"],
                    s["working_dir"],
                    str(s["execution_count"]),
                )

            app.session.add_rich(table)


class JobsCommand(Command):
    """List and manage long-running jobs."""

    def __init__(self) -> None:
        super().__init__(
            name="jobs",
            description="List and manage long-running jobs",
            usage="/jobs [all | <id> | cancel <id> | clean | resume]",
            examples=["/jobs", "/jobs all", "/jobs <id>", "/jobs cancel <id>", "/jobs clean",
                      "/jobs resume"],
            category=CommandCategory.WORKFLOW,
        )

    async def execute(self, args: str, app: Any) -> None:
        """List running jobs, show one job, cancel, clean completed jobs, or resume finished ones.

        Each call to the job manager runs on a worker thread (``_ask``): it
        takes the jobs directory's lock, which another CLI of the app may
        hold, and a cancel waits for the job's processes to exit. Ctrl+C
        stops waiting for it, not the call (a cancel or a clean under way
        finishes), and says the request was stopped.
        """
        try:
            manager = getattr(app.workflow, "job_manager", None)
        except (RuntimeError, AttributeError):
            manager = None

        if manager is None:
            app.session.add_warning(
                "Jobs not available. Add a long-running job tool (and the job_* tools) "
                "to your agent config to enable them."
            )
            return

        parts = self.parse_args(args).positional.split()

        if parts and parts[0] == "cancel":
            if len(parts) < 2:
                app.session.add_warning("Usage: /jobs cancel <id>")
                return
            from agentic_cli.tools.jobs.backends import TERMINAL_STATES

            rec = await self._ask(app, manager.cancel, parts[1])
            if rec is None:
                app.session.add_warning(f"No such job: {parts[1]}")
                return
            state = rec.state.value
            if state == "cancelled":
                app.session.add_success(f"Job '{parts[1]}' → cancelled.")
            elif state in {s.value for s in TERMINAL_STATES}:
                app.session.add_message(
                    "system",
                    f"Job '{parts[1]}' had already finished ({state}): "
                    "nothing was cancelled.",
                )
            elif rec.cancel_requested:
                app.session.add_warning(
                    f"Job '{parts[1]}' is still {state}: it cannot be interrupted now, "
                    "and is recorded as cancelled when it ends."
                )
            elif not manager.has_backend(rec.backend):
                app.session.add_warning(
                    f"Job '{parts[1]}' was not cancelled: it is still {state}, and "
                    f"runs on the '{rec.backend}' backend, which this app has not "
                    "registered."
                )
            else:
                app.session.add_warning(
                    f"Job '{parts[1]}' was not cancelled: it is still {state}, and "
                    "could not be stopped from here (another process of this app may "
                    "be running it, or its record could not be updated just now)."
                )
            return

        if parts and parts[0] == "clean":
            n = await self._ask(app, manager.clean)
            app.session.add_success(f"Removed {n} completed job(s).")
            return

        if parts and parts[0] == "resume":
            # Each finished, resume-flagged job resumes its agent (a turn).
            resume = getattr(app, "resume_finished_jobs", None)
            if resume is None:
                app.session.add_warning("Resume is not supported by this app.")
            elif await resume() == 0:
                app.session.add_message("system", "No finished background jobs to resume.")
            return

        # A bare id → details + recent logs for that one job.
        if parts and parts[0] != "all":
            rec = await self._ask(app, manager.get, parts[0])
            if rec is None:
                app.session.add_warning(f"No such job: {parts[0]}")
                return
            tail = await self._ask(app, manager.tail, rec.job_id, 15, "stdout")
            self._render_detail(app, rec, tail)
            return

        # Otherwise list: running+queued by default, everything with "all".
        jobs = await self._ask(
            app, manager.list, active_only=not (parts and parts[0] == "all")
        )
        if not jobs:
            scope = "" if (parts and parts[0] == "all") else "running "
            app.session.add_message("system", f"No {scope}jobs.")
            return

        table = Table(title="Jobs", show_lines=False, padding=(0, 1))
        table.add_column("ID", style="bold cyan", no_wrap=True)
        table.add_column("Name")  # the terminal's own color: readable on every theme
        table.add_column("Backend", style="dim", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("Elapsed", style="dim", no_wrap=True, justify="right")
        table.add_column("Exit", style="dim", no_wrap=True, justify="right")
        for r in jobs:
            table.add_row(
                r.job_id,
                r.name,
                r.backend,
                r.state.value,
                f"{r.elapsed_s():.0f}s",
                "" if r.exit_code is None else str(r.exit_code),
            )
        app.session.add_rich(table)

    @staticmethod
    async def _ask(app: Any, call: Any, *args: Any, **kwargs: Any) -> Any:
        """``call(*args, **kwargs)`` on a worker thread.

        A Ctrl+C that stops the wait says "Operation cancelled...", once:
        the wait opens no box for thinking-prompt to end and say it.
        """
        async with _says_so_when_stopped(app.session):
            return await asyncio.to_thread(call, *args, **kwargs)

    def _render_detail(self, app: Any, rec: Any, tail: list[str]) -> None:
        lines = [
            f"id:       {rec.job_id}",
            f"tool:     {rec.tool}",
            f"name:     {rec.name}",
            f"backend:  {rec.backend}",
            f"state:    {rec.state.value}",
            f"elapsed:  {rec.elapsed_s():.0f}s",
            f"exit:     {rec.exit_code if rec.exit_code is not None else '-'}",
        ]
        if rec.error:
            lines.append(f"error:    {rec.error}")
        body = "\n".join(lines)
        if tail:
            body += "\n\n-- last stdout --\n" + "\n".join(tail)
        app.session.add_rich(Panel(body, title=f"Job {rec.job_id}", expand=False))


class PapersCommand(Command):
    """List documents in the knowledge base."""

    def __init__(self) -> None:
        super().__init__(
            name="papers",
            description="List documents in the knowledge base",
            aliases=["docs"],
            usage="/papers [query] [--source=arxiv|web|local|user] [--global]",
            examples=["/papers", "/papers transformer", "/papers --source=arxiv", "/papers --global"],
            category=CommandCategory.SEARCH,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Display knowledge base documents in a table."""
        from agentic_cli.memory.kb import SourceType

        parsed = self.parse_args(args)
        query = parsed.positional or ""
        source_filter = parsed.get_option("source", "", str) or ""
        use_global = parsed.has_flag("global")
        scope = "user" if use_global else "project"
        scope_label = "User" if use_global else "Project"

        # The workflow's knowledge base, or one opened for this command
        kb = None
        try:
            workflow = app.workflow
            if workflow:
                kb = getattr(workflow, "knowledge_base", None)
        except (RuntimeError, AttributeError):
            pass
        if kb is not None:
            try:
                kb.manager(scope)
            except KeyError:
                kb = None
        if kb is None:
            from agentic_cli.workflow.memory_services import open_knowledge_base

            kb = open_knowledge_base(app.settings, summarizer=None, scopes=(scope,))

        # Parse source type filter (an unknown one lists everything, as before)
        st_filter = None
        if source_filter:
            try:
                st_filter = SourceType(source_filter)
            except ValueError:
                pass

        # Filtered before the cut at 50, as kb_list does
        docs = [
            listed.document
            for listed in kb.list_documents(
                title_or_author=query, source_type=st_filter, scope=scope, limit=50
            )
        ]

        if not docs:
            app.session.add_message("system", f"No documents found in {scope_label.lower()} knowledge base.")
            return

        table = Table(title=f"{scope_label} Knowledge Base Documents", show_lines=False, padding=(0, 1))
        table.add_column("ID", style="dim", no_wrap=True, max_width=8)
        table.add_column("Title", style="bold", max_width=50)
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Chunks", style="dim", no_wrap=True, justify="right")
        table.add_column("Created", style="dim", no_wrap=True)

        for d in docs:
            table.add_row(
                d.id[:8],
                d.title[:50],
                d.source_type.value,
                str(d.chunk_count),
                d.created_at.strftime("%Y-%m-%d"),
            )

        app.session.add_rich(table)


def _keeps_conversations(app: Any) -> bool:
    """Whether the app keeps the conversations it leaves: not on the memory
    store (``app.session_store``), where it is in one at a time."""
    return getattr(app, "session_store", None) != "memory"


def _session_workflow(app: Any) -> Any:
    """The app's workflow when it keeps sessions; otherwise None, and the user is told why."""
    try:
        workflow = app.workflow
    except (RuntimeError, AttributeError):
        workflow = None
    if workflow is None:
        app.session.add_warning("Sessions not available yet — workflow is initializing.")
        return None
    # An empty list would read as "you have no saved sessions"; say plainly
    # that this backend keeps none.
    if not getattr(workflow, "supports_sessions", False):
        app.session.add_warning(
            f"The {getattr(workflow, 'backend_type', 'current')} backend does not "
            "persist sessions, so there are none to list, resume or delete."
        )
        return None
    return workflow


class SessionsCommand(Command):
    """List and manage saved sessions.

    One row per stored conversation of every installed backend
    (``app.list_conversations()``: a ``SessionInfo`` each, newest first): its
    id (the conversation the app is in, its id on its backend, marked
    ``* ``) and its title (the first prompt), both shown as given, never as
    markup, its backend, its turns and when it last changed.
    ``--delete=<id>`` deletes one on the backend that keeps it, never the
    conversation the user is in nor one open in another process; an id two
    backends keep needs ``--backend=<name>``. On the memory store, which
    keeps no conversation once left, it lists none (the one the app is in,
    ``/status`` names) and deletes none: it says why (``NOT_KEPT``).
    """

    def __init__(self) -> None:
        super().__init__(
            name="sessions",
            description="List saved sessions",
            aliases=["sess"],
            usage="/sessions [--delete=<id> [--backend=<backend>]]",
            examples=["/sessions", "/sessions --delete=my-research"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Display persisted sessions (from every backend's store) or delete one."""
        from rich.text import Text

        parsed = self.parse_args(args)
        delete_id = parsed.get_option("delete", "", str) or ""
        backend = parsed.get_option("backend", "", str) or None

        if not _keeps_conversations(app):  # said before the workflow is needed
            if delete_id:
                app.session.add_error(NOT_KEPT)
            else:
                app.session.add_message("system", NOT_KEPT)
            return

        workflow = _session_workflow(app)
        if workflow is None:
            return

        if delete_id:
            await self._delete(app, delete_id, backend)
            return

        sessions = await app.list_conversations()
        if not sessions:
            app.session.add_message("system", "No saved sessions.")
            return

        table = Table(title="Saved Sessions", show_lines=False, padding=(0, 1))
        table.add_column("Session ID", style="bold cyan", no_wrap=True)
        table.add_column("Backend", style="dim", no_wrap=True)
        table.add_column("Title")  # the terminal's own color: readable on every theme
        table.add_column("Turns", style="dim", justify="right")
        table.add_column("Last Update", style="dim", no_wrap=True)

        current = (app.session_id, app.session_backend)

        for info in sessions:
            sid = info.session_id
            # Text: an id and a title are shown as typed, neither markup nor
            # emoji codes (":x:"), so an id can be copied back.
            label = Text(f"* {sid}" if (sid, info.backend) == current else sid)
            if info.title:
                title = Text(info.title)
            elif info.turns is not None:  # stored, and no prompt in it yet
                title = Text(NO_PROMPT, style="dim")
            else:  # a backend that cannot tell (LangGraph)
                title = Text("")
            turns = "" if info.turns is None else str(info.turns)
            updated = (
                info.updated.astimezone().strftime("%Y-%m-%d %H:%M") if info.updated else ""
            )
            table.add_row(label, Text(info.backend or ""), title, turns, updated)

        app.session.add_rich(table)

    @staticmethod
    async def _delete(app: Any, session_id: str, backend: str | None) -> None:
        """Delete ``session_id`` on the backend that keeps it (``backend``, when named)."""
        from agentic_cli.workflow.settings import OrchestratorType

        names = sorted(o.value for o in OrchestratorType)
        if backend is not None and backend not in names:
            app.session.add_error(
                f"Unknown backend '{backend}': {', '.join(names[:-1])} or {names[-1]}."
            )
            return
        if session_id == app.session_id and backend in (None, app.session_backend):
            twins = sorted(
                info.backend for info in await app.list_conversations()
                if info.session_id == session_id and info.backend != app.session_backend
            )
            line = "You are in this conversation; /new or /resume another first."
            if twins:
                line += (
                    f" {twins[0]} keeps one with this id too: /sessions "
                    f"--delete={session_id} --backend={twins[0]} deletes that one."
                )
            app.session.add_error(line)
            return
        rows = [
            info for info in await app.list_conversations()
            if info.session_id == session_id and backend in (None, info.backend)
        ]
        if not rows:
            app.session.add_error(f"Session '{session_id}' not found.")
            return
        if len(rows) > 1:
            keeping = sorted(info.backend or "?" for info in rows)
            names, choices = " and ".join(keeping), " or ".join(keeping)
            app.session.add_error(
                f"'{session_id}' is kept by {names}: "
                f"/sessions --delete={session_id} --backend=<{choices}> deletes one."
            )
            return
        (info,) = rows
        # Held while it is deleted: one open in another process would be
        # recreated there, empty, at its next turn, and that process not told.
        with app.conversation_held(session_id, backend=info.backend) as holder:
            if holder is not None:
                app.session.add_error(
                    f"Could not delete '{session_id}': it is open in another "
                    f"process ({holder})."
                )
                return
            deleted = await app.delete_conversation(session_id, backend=info.backend)
        if deleted:
            app.session.add_success(f"Session '{session_id}' deleted.")
        else:
            app.session.add_error(f"Session '{session_id}' not found.")


class ResumeSessionCommand(Command):
    """Reopen a stored conversation: from a picker, or the one named. On the
    memory store, which keeps none, it is refused (``NOT_KEPT``)."""

    def __init__(self) -> None:
        super().__init__(
            name="resume",
            description="Resume a saved conversation (pick one, or name its id)",
            usage="/resume [<id>]",
            examples=["/resume", "/resume 3f2a9c1d0b7e"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Resume the conversation named, or the one picked (``pick_session``)."""
        if not _keeps_conversations(app):  # said before the workflow is needed
            app.session.add_error(NOT_KEPT)
            return
        if _session_workflow(app) is None:
            return
        session_id = self.parse_args(args).positional.strip()
        if not session_id:
            session_id = await app.pick_session()
            if session_id is None:
                return
        await app.resume_session(session_id)


class CompactCommand(Command):
    """Summarize the conversation now, to free the context window."""

    def __init__(self) -> None:
        super().__init__(
            name="compact",
            description="Summarize the conversation now, keeping the newest turns word for word",
            usage="/compact [focus]",
            examples=["/compact", "/compact keep the API decisions and open questions"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Summarize the main agent's conversation, as a turn runs (``app.compact``).

        One request at a time (the app's turn lock), with the turn's events
        box and status line; Ctrl+C stops it and nothing is written. The rest
        of the line, if any, is what the summary should keep above all.
        """
        await app.compact(args.strip() or None)


class NewCommand(Command):
    """Start a new conversation, on another backend if one is named; the one
    left stays in /sessions (on the memory store it is dropped)."""

    def __init__(self) -> None:
        super().__init__(
            name="new",
            description=(
                "Start a new conversation (the current one stays in /sessions, "
                "unless session_store is memory); name a backend to run it there"
            ),
            usage="/new [adk|deepagents|langgraph]",
            examples=["/new", "/new deepagents"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Start a new conversation in this process (``start_new_session``),
        on the backend named, if any: an unknown one, or one whose extra is
        not installed, is refused, and nothing changes."""
        from agentic_cli.workflow.factory import check_backend
        from agentic_cli.workflow.settings import OrchestratorType

        backend = self.parse_args(args).positional.strip()
        if not backend:
            try:
                await app.start_new_session()  # on the settings' orchestrator
            except ImportError as error:  # its extra is not installed
                app.session.add_error(str(error).split("\n")[0])
            return
        if not _takes_a_backend(app.start_new_session):
            app.session.add_error("This app's /new takes no backend.")
            return
        try:
            check_backend(backend)
        except ValueError:
            names = sorted(o.value for o in OrchestratorType)
            app.session.add_error(
                f"Unknown backend '{backend}': {', '.join(names[:-1])} or {names[-1]}."
            )
            return
        except ImportError as error:
            app.session.add_error(str(error).split("\n")[0])
            return
        await app.start_new_session(backend=backend)


def _takes_a_backend(start_new_session: Any) -> bool:
    """Whether an app's ``start_new_session`` takes ``backend``: one an app
    overrode without it (0.7.0 added the keyword) is called as before."""
    import inspect

    parameters = inspect.signature(start_new_session).parameters.values()
    return any(p.name == "backend" or p.kind is p.VAR_KEYWORD for p in parameters)


class ModelsCommand(Command):
    """Show the list of models, find a model in the providers' listings, or list again."""

    def __init__(self) -> None:
        super().__init__(
            name="models",
            description="Show the models the pickers offer; find a model, or list again",
            usage="/models [<text> | refresh]",
            examples=["/models", "/models qwen", "/models refresh"],
            category=CommandCategory.SETTINGS,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Show the list and the providers in use, search the providers' full
        listings for ``args``, or (``refresh``) list them again."""
        from agentic_cli.workflow import model_services

        text = args.strip()
        try:
            workflow = app.workflow
        except (RuntimeError, AttributeError):
            workflow = None
        providers = model_services.providers_from_settings(app.settings)
        catalog = getattr(workflow, "catalog", None)
        owned = catalog is None
        if owned:
            # The catalog the workflow would load: the model cache, and the
            # listings a start would wait on (ModelCatalog.load); any other
            # provider in use shows as not listed.
            if workflow is None:
                app.session.add_message(
                    "system", "The workflow is not initialized yet: showing the model cache."
                )
            catalog = model_services.build_catalog(
                app.settings, providers, agent_configs=app.agent_configs
            )
            await catalog.load()
        try:
            if text.lower() == "refresh":
                await _refresh_models(app, catalog, providers)
                return
            default = getattr(workflow, "model", None)
            roles = model_services.role_models(app.settings, catalog, providers, default)
            if text:
                await _search_models(app, catalog, roles, text)
            else:
                _show_models(app, catalog, providers, roles)
        finally:
            if owned:
                await catalog.aclose()


# --- /models: the tables ---------------------------------------------------------

_LISTING_STATES = {
    "unattempted": "not listed",
    "fresh": "fresh",
    "stale": "stale",
    "degraded": "failed",
}
_THINKING_NAMES = {
    "gemini_budget": "budget",
    "anthropic_budget": "budget",
    "gemini_level": "level",
    "anthropic_adaptive": "adaptive",
    "effort": "effort",
    "chat_template": "on/off",
}
# Where a record's values came from, highest precedence first.
_SOURCES = ("config", "listing", "table", "default")
# A search's Status for a match the list offers.
_IN_THE_LIST = "in the list"


def _show_models(
    app: Any, catalog: "ModelCatalog", providers: Any, roles: dict[str, list[str]]
) -> None:
    """The list, then each model the settings name that it leaves out, with why;
    below, the providers in use.

    The list is what the pickers offer: ``offered()`` with the roles'
    models (``model_services.role_models``). What it leaves out is
    ``not_offered()``: a ``models`` entry, ``default_model``, a role's model,
    an agent's or the manager's own, each once.
    """
    offered = catalog.offered(extra=list(roles))
    left_out = catalog.not_offered()
    if offered or left_out:
        title = f"Models ({len(offered)})"
        if left_out:
            title = f"Models ({len(offered)} offered, {len(left_out)} not)"
        table = _models_table(title)
        for ref in offered:
            _add_model_row(table, ref, _record(catalog, ref), roles, "")
        for ref, reason in left_out.items():
            _add_model_row(table, ref, _record(catalog, ref), roles, reason)
        app.session.add_rich(table)
    else:
        app.session.add_message("system", "No models: no provider has a key.")
    statuses = catalog.status()
    if statuses:
        app.session.add_rich(_status_table(statuses, providers))


async def _search_models(
    app: Any, catalog: "ModelCatalog", roles: dict[str, list[str]], text: str
) -> None:
    """The models of the providers' full listings whose reference contains
    ``text``, each marked when the list offers it: how a user finds an exact
    id to copy into ``models``. A provider whose listing failed is named, so
    "no match" never stands for a listing that could not be read."""
    found = await catalog.search(text)
    for status in catalog.status():
        if status.has_credentials and status.state.value == "degraded":
            app.session.add_warning(
                f"Could not list {status.provider} models ({status.error}): "
                "the search could not look in its listing."
            )
    if not found:
        app.session.add_message("system", f"No model matches {text!r}.")
        return
    offered = set(catalog.offered(extra=list(roles)))
    table = _models_table(f"Models matching {text!r} ({len(found)})")
    for record in found:
        ref = str(record.ref)
        _add_model_row(table, ref, record, roles, _IN_THE_LIST if ref in offered else "")
    app.session.add_rich(table)


async def _refresh_models(app: Any, catalog: "ModelCatalog", providers: Any) -> None:
    """List the providers in use, and those searched, again, and report what changed."""
    before = {s.provider: {r.model for r in catalog.records(s.provider)} for s in catalog.status()}
    statuses = await catalog.refresh()
    listed = changed = 0
    for status in statuses:
        provider = providers.get(status.provider)
        if not status.has_credentials or (provider is not None and not provider.list_models):
            continue
        listed += 1
        if status.state.value == "degraded":
            app.session.add_warning(f"{status.provider}: listing failed ({status.error}).")
            continue
        old = before[status.provider]
        new = {record.model for record in catalog.records(status.provider)}
        if new != old:
            changed += 1
            app.session.add_success(_change(status.provider, old, new))
    if not changed and all(s.state.value != "degraded" for s in statuses if s.has_credentials):
        app.session.add_message(
            "system", f"Listed {listed} provider(s) again: their models are unchanged."
        )
    app.session.add_rich(_status_table(catalog.status(), providers))


def _models_table(title: str) -> Table:
    table = Table(title=title, show_lines=False, padding=(0, 1))
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Max output", justify="right", no_wrap=True)
    table.add_column("Thinking")
    table.add_column("Tools", no_wrap=True)
    table.add_column("Cache", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("Roles", style="green")
    table.add_column("Status", style="yellow")
    return table


def _add_model_row(
    table: Table,
    ref: str,
    record: "ModelRecord | None",
    roles: dict[str, list[str]],
    status: str,
) -> None:
    """One model's row; without a record (an entry left out), its reference only."""
    used_by = ", ".join(roles.get(ref, ()))
    if record is None:
        table.add_row(ref, "—", "—", "—", "—", "—", "", used_by, status)
        return
    table.add_row(
        ref,
        _tokens(record.context_window),
        _tokens(record.max_output),
        _thinking(record.thinking, record.reasoning_text),
        "yes" if record.tools else "no",
        "—" if record.cache.value == "none" else record.cache.value,
        _sources(record),
        used_by,
        status,
    )


def _record(catalog: "ModelCatalog", ref: str) -> "ModelRecord | None":
    """The catalog's record of ``ref``; None for a provider it does not know,
    or a model a fresh listing lacks."""
    from agentic_cli.models import ModelResolutionError, parse_ref

    try:
        return catalog.record(parse_ref(ref))
    except ModelResolutionError:
        return None


def _status_table(statuses: "list[ProviderStatus]", providers: Any) -> Table:
    """One row per provider. Never a key: only whether there is one, or none is needed."""
    now = time.time()
    table = Table(title="Providers", show_lines=False, padding=(0, 1))
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("Credentials", no_wrap=True)
    table.add_column("Listing", no_wrap=True)
    table.add_column("Fetched", style="dim", no_wrap=True)
    table.add_column("Models", justify="right", no_wrap=True)
    table.add_column("Error", style="red")
    for status in statuses:
        provider = providers.get(status.provider)
        if provider is not None and not provider.builtin and provider.key_env is None:
            credentials = "none needed"
        else:
            credentials = "key" if status.has_credentials else "no key"
        fetched = "—" if status.fetched_at is None else _ago(now - status.fetched_at)
        table.add_row(
            status.provider,
            credentials,
            _LISTING_STATES[status.state.value],
            fetched,
            str(status.model_count) if status.has_credentials else "—",
            status.error or "",
        )
    return table


def _thinking(rule: "ThinkingRule", reasoning_text: bool) -> str:
    """``budget 4096/12288/24576``, ``adaptive low/medium/high/max``, ``—``,
    then ``· no reasoning text`` for a model that thinks but does not send its
    reasoning as text. Most thinking models do, so only the exception is said."""
    name = _THINKING_NAMES.get(rule.mechanism.value)
    if name is None:
        return "—"
    if rule.budgets:
        detail = "/".join(str(budget) for budget in rule.budgets)
    else:
        detail = "/".join(rule.levels)
    text = f"{name} {detail}".rstrip()
    if not rule.can_disable:
        text += " (always on)"
    return text if reasoning_text else f"{text} · no reasoning text"


def _sources(record: "ModelRecord") -> str:
    used = set(record.sources.values())
    return ", ".join(source for source in _SOURCES if source in used) or "table"


def _tokens(count: int | None) -> str:
    return "—" if count is None else f"{count:,}"


def _ago(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 2 * 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def _change(provider: str, old: set[str], new: set[str]) -> str:
    """``google: 2 → 6 models (added a, b, c)``."""
    parts = []
    for verb, names in (("added", new - old), ("removed", old - new)):
        if names:
            shown = sorted(names)
            more = f" and {len(shown) - 5} more" if len(shown) > 5 else ""
            parts.append(f"{verb} {', '.join(shown[:5])}{more}")
    return f"{provider}: {len(old)} → {len(new)} models ({'; '.join(parts)})"

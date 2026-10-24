"""Built-in slash commands for the CLI."""

import time
from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table

from agentic_cli.cli.commands import Command, CommandCategory

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
        """Display help information."""
        commands = app.command_registry.all_commands()

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Aliases", style="dim", no_wrap=True)
        table.add_column("Description")

        for cmd in sorted(commands, key=lambda c: c.name):
            aliases = ", ".join(f"/{a}" for a in cmd.aliases) if cmd.aliases else ""
            table.add_row(f"/{cmd.name}", aliases, cmd.description)

        panel = Panel(table, title="[bold]Available Commands[/bold]", border_style="cyan")
        app.session.add_rich(panel)


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
    initialized yet; LangGraph or a native ADK YAML app, which manage no
    context; one without ``context_budget``) gives only the last prompt, once
    there is one. Any other error is the manager's, and is not hidden here.
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
            table.add_row("Workflow", "[yellow]Not available (initializing...)[/yellow]")
            init_error = getattr(app, "_init_error", None) or getattr(app, "workflow_error", None)
            if init_error:
                table.add_row("Error", f"[red]{init_error}[/red]")

        # Persistent session info
        sid = app.session_id
        if sid:
            table.add_row("Session", f"{sid} (persistent)")
        else:
            table.add_row("Session", "ephemeral (not saved)")

        # Token usage breakdown
        tracker = getattr(app, "usage_tracker", None)
        if tracker is not None:
            for label, value in tracker.format_detail_rows():
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
        """List running jobs, show one job, cancel, clean completed jobs, or resume finished ones."""
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
            rec = manager.cancel(parts[1])
            if rec is None:
                app.session.add_warning(f"No such job: {parts[1]}")
            elif rec.cancel_requested and rec.state.value == "running":
                app.session.add_warning(
                    f"Job '{parts[1]}' is still running: it cannot be interrupted now, "
                    "and is recorded as cancelled when it ends."
                )
            else:
                app.session.add_success(f"Job '{parts[1]}' → {rec.state.value}.")
            return

        if parts and parts[0] == "clean":
            n = manager.clean()
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
            rec = manager.get(parts[0])
            if rec is None:
                app.session.add_warning(f"No such job: {parts[0]}")
                return
            self._render_detail(app, manager, rec)
            return

        # Otherwise list: running+queued by default, everything with "all".
        jobs = manager.list() if parts and parts[0] == "all" else manager.list(active_only=True)
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

    def _render_detail(self, app: Any, manager: Any, rec: Any) -> None:
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
        tail = manager.tail(rec.job_id, 15, "stdout")
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
                str(len(d.chunks)),
                d.created_at.strftime("%Y-%m-%d"),
            )

        app.session.add_rich(table)


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

    One row per stored conversation (``SessionInfo``): its id (the current
    one marked ``* ``) and its title (the first prompt), both shown as given,
    never as markup, its turns and when it last changed. ``--delete=<id>``
    deletes one, never the conversation the user is in.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sessions",
            description="List saved sessions",
            aliases=["sess"],
            usage="/sessions [--delete=<id>]",
            examples=["/sessions", "/sessions --delete=my-research"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Display persisted sessions (from the native store) or delete one."""
        from rich.markup import escape

        parsed = self.parse_args(args)
        delete_id = parsed.get_option("delete", "", str) or ""

        workflow = _session_workflow(app)
        if workflow is None:
            return

        if delete_id:
            if delete_id == app.session_id:
                app.session.add_error("You are in this conversation; /new or /resume another first.")
                return
            if await workflow.delete_session(delete_id):
                app.session.add_success(f"Session '{delete_id}' deleted.")
            else:
                app.session.add_error(f"Session '{delete_id}' not found.")
            return

        sessions = await workflow.list_sessions()
        if not sessions:
            app.session.add_message("system", "No saved sessions.")
            return

        table = Table(title="Saved Sessions", show_lines=False, padding=(0, 1))
        table.add_column("Session ID", style="bold cyan", no_wrap=True)
        table.add_column("Title")  # the terminal's own color: readable on every theme
        table.add_column("Turns", style="dim", justify="right")
        table.add_column("Last Update", style="dim", no_wrap=True)

        current_sid = app.session_id

        for info in sessions:
            sid = info.session_id
            label = escape(f"* {sid}" if sid == current_sid else sid)  # an id is no markup
            title = escape(info.title) if info.title else "[dim](no prompt yet)[/dim]"
            turns = "" if info.turns is None else str(info.turns)
            updated = (
                info.updated.astimezone().strftime("%Y-%m-%d %H:%M") if info.updated else ""
            )
            table.add_row(label, title, turns, updated)

        app.session.add_rich(table)


class ResumeSessionCommand(Command):
    """Reopen a stored conversation: from a picker, or the one named."""

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
    """Start a new conversation; the one left stays in /sessions."""

    def __init__(self) -> None:
        super().__init__(
            name="new",
            description="Start a new conversation (the current one stays in /sessions)",
            usage="/new",
            examples=["/new"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Start a new conversation in this process (``start_new_session``)."""
        await app.start_new_session()


class ModelsCommand(Command):
    """Show what the model catalog knows: each provider's listing, and its models."""

    def __init__(self) -> None:
        super().__init__(
            name="models",
            description="Show each provider's models and listing status",
            usage="/models [<text> | refresh]",
            examples=["/models", "/models flash", "/models refresh"],
            category=CommandCategory.SETTINGS,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Show the catalog, filter its models by reference, or list every provider again."""
        from agentic_cli.workflow import model_services

        text = args.strip()
        try:
            workflow = app.workflow
        except (RuntimeError, AttributeError):
            workflow = None
        endpoints = model_services.endpoints_from_settings(app.settings)
        catalog = getattr(workflow, "catalog", None)
        owned = catalog is None
        if owned:
            # The catalog the workflow would load: the model cache, and a
            # listing of each provider with a key and no cache entry.
            if workflow is None:
                app.session.add_message(
                    "system", "The workflow is not initialized yet: showing the model cache."
                )
            catalog = model_services.build_catalog(app.settings, endpoints)
            await catalog.load()
        try:
            if text.lower() == "refresh":
                await _refresh_models(app, catalog, endpoints)
            else:
                default = getattr(workflow, "model", None)
                _show_models(app, catalog, endpoints, default, text)
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
_SOURCES = ("declared", "listing", "table", "default")


def _show_models(
    app: Any, catalog: "ModelCatalog", endpoints: Any, default: str | None, text: str
) -> None:
    """The status rows, then the models of each provider with credentials."""
    statuses = catalog.status()
    app.session.add_rich(_status_table(statuses, endpoints))
    records = [
        record
        for status in statuses
        if status.has_credentials
        for record in sorted(catalog.records(status.provider), key=lambda r: r.model)
    ]
    if text:
        records = [r for r in records if text.lower() in str(r.ref).lower()]
    if not records:
        app.session.add_message(
            "system",
            f"No model matches {text!r}." if text else "No models: no provider has a key.",
        )
        return
    roles = _roles_by_model(app.settings, catalog, endpoints, default)
    title = f"Models matching {text!r}" if text else "Models"
    table = Table(title=f"{title} ({len(records)})", show_lines=False, padding=(0, 1))
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Max output", justify="right", no_wrap=True)
    table.add_column("Thinking")
    table.add_column("Tools", no_wrap=True)
    table.add_column("Cache", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("Roles", style="green")
    for record in records:
        table.add_row(
            str(record.ref),
            _tokens(record.context_window),
            _tokens(record.max_output),
            _thinking(record.thinking),
            "yes" if record.tools else "no",
            "—" if record.cache.value == "none" else record.cache.value,
            _sources(record),
            ", ".join(roles.get(str(record.ref), ())),
        )
    app.session.add_rich(table)


async def _refresh_models(app: Any, catalog: "ModelCatalog", endpoints: Any) -> None:
    """List every provider with credentials again, and report what changed."""
    before = {s.provider: {r.model for r in catalog.records(s.provider)} for s in catalog.status()}
    statuses = await catalog.refresh()
    listed = changed = 0
    for status in statuses:
        if not status.has_credentials:
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
    app.session.add_rich(_status_table(catalog.status(), endpoints))


def _status_table(statuses: "list[ProviderStatus]", endpoints: Any) -> Table:
    """One row per provider and endpoint. Never a key: only whether there is one."""
    now = time.time()
    table = Table(title="Providers", show_lines=False, padding=(0, 1))
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("Credentials", no_wrap=True)
    table.add_column("Listing", no_wrap=True)
    table.add_column("Fetched", style="dim", no_wrap=True)
    table.add_column("Models", justify="right", no_wrap=True)
    table.add_column("Error", style="red")
    for status in statuses:
        endpoint = endpoints.get(status.provider)
        if endpoint is not None and endpoint.kind.value == "openai_compatible":
            credentials = "endpoint"
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


def _roles_by_model(
    settings: Any, catalog: "ModelCatalog", endpoints: Any, default: str | None
) -> dict[str, list[str]]:
    """Each model's roles: the built-in ones (mapped or computed) and the mapped app roles."""
    from agentic_cli.models import BUILTIN_ROLES, ModelResolutionError
    from agentic_cli.workflow import model_services

    try:
        role_map = model_services.role_map_from_settings(settings)
    except ValueError:
        return {}
    roles: dict[str, list[str]] = {}
    for role in dict.fromkeys([*BUILTIN_ROLES, *settings.model_roles]):
        if role == "default" and default:
            ref = default
        else:
            try:
                ref = str(role_map.resolve(role, catalog=catalog, endpoints=endpoints))
            except (ModelResolutionError, ValueError):
                continue
        roles.setdefault(ref, []).append(role)
    return roles


def _thinking(rule: "ThinkingRule") -> str:
    """``budget 4096/12288/24576``, ``adaptive low/medium/high/max``, ``—``."""
    name = _THINKING_NAMES.get(rule.mechanism.value)
    if name is None:
        return "—"
    if rule.budgets:
        detail = "/".join(str(budget) for budget in rule.budgets)
    else:
        detail = "/".join(rule.levels)
    text = f"{name} {detail}".rstrip()
    return text if rule.can_disable else f"{text} (always on)"


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

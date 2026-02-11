"""Built-in slash commands for the CLI."""

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp


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
            services = "Initialized" if workflow.runner else "Not initialized"
            table.add_row("Services", services)
        except (RuntimeError, AttributeError):
            table.add_row("Workflow", "[yellow]Not available (initializing...)[/yellow]")
            init_error = getattr(app, "_init_error", None) or getattr(app, "workflow_error", None)
            if init_error:
                table.add_row("Error", f"[red]{init_error}[/red]")

        # Message history stats
        table.add_row("Messages", str(len(app.message_history)))

        # Token usage breakdown
        tracker = getattr(app, "usage_tracker", None)
        if tracker is not None and tracker.invocation_count > 0:
            from agentic_cli.cli.usage_tracker import format_tokens

            table.add_row("", "")  # Spacer
            table.add_row("LLM Invocations", str(tracker.invocation_count))
            table.add_row("Input Tokens", format_tokens(tracker.prompt_tokens))
            table.add_row("Output Tokens", format_tokens(tracker.completion_tokens))
            table.add_row("Total Tokens", format_tokens(tracker.total_tokens))
            if tracker.cached_tokens > 0:
                table.add_row("Cached Tokens", format_tokens(tracker.cached_tokens))
            if tracker.cache_creation_tokens > 0:
                table.add_row("Cache Creation", format_tokens(tracker.cache_creation_tokens))
            if tracker.thinking_tokens > 0:
                table.add_row("Thinking Tokens", format_tokens(tracker.thinking_tokens))
            if tracker.total_latency_ms > 0:
                avg_ms = tracker.total_latency_ms / tracker.invocation_count
                table.add_row("Avg Latency", f"{avg_ms:.0f}ms")

        panel = Panel(table, title="[bold]Session Status[/bold]", border_style="cyan")
        app.session.add_rich(panel)


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
        from pathlib import Path
        from agentic_cli.knowledge_base import KnowledgeBaseManager
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.constants import format_size

        parsed = self.parse_args(args)
        query = parsed.positional or ""
        source_filter = parsed.get_option("source", "", str) or ""
        use_global = parsed.has_flag("global")

        # Determine which KB dir to use
        if use_global:
            kb_dir = app.settings.knowledge_base_dir
            scope_label = "User"
        else:
            kb_dir = Path.cwd() / f".{app.settings.app_name}" / "knowledge_base"
            scope_label = "Project"

        # Get KB manager from workflow or create temporary one
        kb = None
        try:
            workflow = app.workflow
            if workflow:
                if use_global and hasattr(workflow, "user_kb_manager") and workflow.user_kb_manager:
                    kb = workflow.user_kb_manager
                elif not use_global and hasattr(workflow, "kb_manager") and workflow.kb_manager:
                    kb = workflow.kb_manager
        except (RuntimeError, AttributeError):
            pass

        if kb is None:
            kb = KnowledgeBaseManager(
                settings=app.settings,
                use_mock=getattr(app.settings, "knowledge_base_use_mock", True),
                base_dir=kb_dir,
            )

        # Parse source type filter
        st_filter = None
        if source_filter:
            try:
                st_filter = SourceType(source_filter)
            except ValueError:
                pass

        docs = kb.list_documents(source_type=st_filter, limit=50)

        # Apply query filter
        if query:
            query_lower = query.lower()
            docs = [
                d for d in docs
                if query_lower in d.title.lower()
                or any(query_lower in a.lower() for a in d.metadata.get("authors", []))
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

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
    """List saved research papers."""

    def __init__(self) -> None:
        super().__init__(
            name="papers",
            description="List saved research papers",
            aliases=[],
            usage="/papers [query] [--source=arxiv|web|local]",
            examples=["/papers", "/papers transformer", "/papers --source=arxiv"],
            category=CommandCategory.SEARCH,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Display saved papers in a table."""
        from agentic_cli.tools.paper_tools import PaperStore

        parsed = self.parse_args(args)
        query = parsed.positional or None
        source_filter = parsed.get_option("source", "", str) or None

        # Get paper store from workflow manager or create temporary one
        store = None
        try:
            workflow = app.workflow
            if workflow and hasattr(workflow, "paper_store") and workflow.paper_store:
                store = workflow.paper_store
        except (RuntimeError, AttributeError):
            pass

        if store is None:
            store = PaperStore(app.settings)

        papers = store.list_papers(query=query, source_type=source_filter)

        if not papers:
            app.session.add_message("system", "No saved papers found.")
            return

        table = Table(title="Saved Papers", show_lines=False, padding=(0, 1))
        table.add_column("ID", style="dim", no_wrap=True, max_width=8)
        table.add_column("Title", style="bold", max_width=50)
        table.add_column("Authors", max_width=30)
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Added", style="dim", no_wrap=True)
        table.add_column("Size", style="dim", no_wrap=True, justify="right")

        for p in papers:
            # Format authors (truncate if too many)
            authors = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors += f" +{len(p.authors) - 3}"

            # Format date
            added = p.added_at[:10] if p.added_at else ""

            # Format size
            if p.file_size_bytes >= 1048576:
                size = f"{p.file_size_bytes / 1048576:.1f} MB"
            elif p.file_size_bytes >= 1024:
                size = f"{p.file_size_bytes / 1024:.0f} KB"
            else:
                size = f"{p.file_size_bytes} B"

            table.add_row(
                p.id[:8],
                p.title[:50],
                authors,
                p.source_type,
                added,
                size,
            )

        app.session.add_rich(table)

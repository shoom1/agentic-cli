"""Built-in slash commands for the CLI."""

from datetime import datetime
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
            aliases=["h", "?"],
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
            aliases=["cls"],
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
            aliases=["quit", "q"],
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
            aliases=["st"],
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

        panel = Panel(table, title="[bold]Session Status[/bold]", border_style="cyan")
        app.session.add_rich(panel)


class SaveCommand(Command):
    """Save current session."""

    def __init__(self) -> None:
        super().__init__(
            name="save",
            description="Save current session",
            aliases=[],
            usage="/save [name]",
            examples=["/save", "/save my_research_session"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Save current session state."""
        from agentic_cli.persistence import SessionPersistence

        # Generate session name if not provided
        session_name = (
            args.strip()
            if args.strip()
            else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        persistence = SessionPersistence(app.settings)

        # Gather metadata
        metadata = {}
        try:
            workflow = app.workflow
            metadata["model"] = workflow.model
            metadata["app_name"] = workflow.app_name
        except RuntimeError:
            pass

        try:
            saved_path = persistence.save_session(
                session_id=session_name,
                message_history=app.message_history,
                metadata=metadata,
            )
            msg_count = len(app.message_history)
            app.session.add_success(f"Session saved: {session_name}")
            app.session.add_message("system", f"Messages saved: {msg_count}")
            app.session.add_message("system", f"Path: {saved_path}")
        except Exception as e:
            app.session.add_error(f"Failed to save session: {e}")


class LoadCommand(Command):
    """Load a saved session."""

    def __init__(self) -> None:
        super().__init__(
            name="load",
            description="Load a saved session",
            aliases=[],
            usage="/load <name>",
            examples=["/load my_session", "/load session_20240115_120000"],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Load a saved session."""
        from agentic_cli.persistence import SessionPersistence

        session_name = args.strip()
        if not session_name:
            app.session.add_error("Please provide a session name.")
            app.session.add_message("system", "Usage: /load <session_name>")
            app.session.add_message(
                "system", "Use /sessions to list available sessions."
            )
            return

        persistence = SessionPersistence(app.settings)

        try:
            snapshot = persistence.load_session(session_name)
            if snapshot is None:
                app.session.add_error(f"Session not found: {session_name}")
                app.session.add_message(
                    "system", "Use /sessions to list available sessions."
                )
                return

            # Restore messages to history
            msg_count = persistence.restore_to_history(snapshot, app.message_history)

            app.session.add_success(f"Session loaded: {session_name}")
            app.session.add_message("system", f"Messages restored: {msg_count}")

            # Show session info
            if snapshot.metadata:
                if "model" in snapshot.metadata:
                    app.session.add_message(
                        "system", f"Original model: {snapshot.metadata['model']}"
                    )

        except Exception as e:
            app.session.add_error(f"Failed to load session: {e}")


class SessionsCommand(Command):
    """List saved sessions."""

    def __init__(self) -> None:
        super().__init__(
            name="sessions",
            description="List all saved sessions",
            aliases=[],
            category=CommandCategory.SESSION,
        )

    async def execute(self, args: str, app: Any) -> None:
        """List all saved sessions."""
        from agentic_cli.persistence import SessionPersistence

        persistence = SessionPersistence(app.settings)
        sessions = persistence.list_sessions()

        if not sessions:
            app.session.add_message("system", "No saved sessions found.")
            app.session.add_message(
                "system", "Use /save [name] to save the current session."
            )
            return

        table = Table(show_header=True, box=None, padding=(0, 2, 0, 0))
        table.add_column("Session", style="bold cyan")
        table.add_column("Messages", justify="right")
        table.add_column("Saved", style="dim")

        for sess in sessions:
            saved_at = datetime.fromisoformat(sess["saved_at"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            table.add_row(sess["session_id"], str(sess["message_count"]), saved_at)

        panel = Panel(table, title="[bold]Saved Sessions[/bold]", border_style="cyan")
        app.session.add_rich(panel)
        app.session.add_message("system", "Use /load <name> to restore a session.")



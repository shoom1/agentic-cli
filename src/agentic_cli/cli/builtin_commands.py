"""Built-in slash commands for the CLI."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style

from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.config import THINKING_EFFORT_LEVELS

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp


# Dialog style for settings
DIALOG_STYLE = Style.from_dict({
    "dialog": "bg:#1a1a2e",
    "dialog frame.label": "bg:#16213e fg:#e94560",
    "dialog.body": "bg:#1a1a2e fg:#eaeaea",
    "dialog shadow": "bg:#0f0f1a",
    "button": "bg:#16213e fg:#eaeaea",
    "button.focused": "bg:#e94560 fg:#ffffff",
    "radio-list": "bg:#1a1a2e fg:#eaeaea",
    "radio": "fg:#e94560",
    "radio-selected": "fg:#e94560 bold",
})


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

        help_text = "**Available Commands**\n\n"

        for cmd in sorted(commands, key=lambda c: c.name):
            help_text += f"**/{cmd.name}**"
            if cmd.aliases:
                help_text += f" (aliases: {', '.join(f'/{a}' for a in cmd.aliases)})"
            help_text += f"\n  {cmd.description}\n\n"

        app.session.add_response(help_text, markdown=True)


class ClearCommand(Command):
    """Clear the screen."""

    def __init__(self) -> None:
        super().__init__(
            name="clear",
            description="Clear the screen",
            aliases=["cls"],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Clear the console."""
        app.session.clear_history()
        app.console.clear()


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
        status_text = "**Session Status**\n\n"

        # Workflow manager status
        try:
            workflow = app.workflow
            if workflow is None:
                raise RuntimeError("Workflow not initialized")
            status_text += f"**Model:** {workflow.model}\n"
            status_text += f"**App Name:** {workflow.app_name}\n"
            status_text += f"**Session ID:** {workflow.session_id}\n"

            # Check if services are initialized
            if workflow.runner:
                status_text += "**Services:** Initialized\n"
            else:
                status_text += "**Services:** Not initialized\n"
        except (RuntimeError, AttributeError):
            status_text += "**Workflow:** Not available (initializing...)\n"
            init_error = getattr(app, "_init_error", None) or getattr(app, "workflow_error", None)
            if init_error:
                status_text += f"**Error:** {init_error}\n"

        # Message history stats
        msg_count = len(app.message_history)
        status_text += f"\n**Messages in history:** {msg_count}\n"

        app.session.add_response(status_text, markdown=True)


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

        output = "**Saved Sessions**\n\n"
        for sess in sessions:
            saved_at = datetime.fromisoformat(sess["saved_at"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            output += f"- **{sess['session_id']}**\n"
            output += f"  Messages: {sess['message_count']} | Saved: {saved_at}\n\n"

        app.session.add_response(output, markdown=True)
        app.session.add_message("system", "Use /load <name> to restore a session.")


class SettingsCommand(Command):
    """Open settings dialog to configure model and thinking effort."""

    def __init__(self) -> None:
        super().__init__(
            name="settings",
            description="Configure model and thinking effort settings",
            aliases=["set", "config"],
            usage="/settings [model|thinking|show]",
            examples=["/settings", "/settings model", "/settings show"],
            category=CommandCategory.SETTINGS,
        )

    async def execute(self, args: str, app: Any) -> None:
        """Open interactive settings dialog."""
        args = args.strip().lower()

        if args == "model":
            await self._select_model(app)
            return

        if args == "thinking":
            await self._select_thinking_effort(app)
            return

        if args == "show":
            self._show_current_settings(app)
            return

        if args == "":
            await self._show_settings_menu(app)
            return

        app.session.add_error(f"Unknown settings option: {args}")
        app.session.add_message("system", "Usage: /settings [model|thinking|show]")

    async def _show_settings_menu(self, app: Any) -> None:
        """Show main settings menu."""
        options = [
            ("model", "Change Model"),
            ("thinking", "Change Thinking Effort"),
            ("show", "Show Current Settings"),
            ("cancel", "Cancel"),
        ]

        result = await radiolist_dialog(
            title="Settings",
            text="Select a setting to configure:",
            values=options,
            style=DIALOG_STYLE,
        ).run_async()

        if result == "model":
            await self._select_model(app)
        elif result == "thinking":
            await self._select_thinking_effort(app)
        elif result == "show":
            self._show_current_settings(app)

    async def _select_model(self, app: Any) -> None:
        """Show model selection dialog."""
        settings = app.settings
        available_models = settings.get_available_models()

        if not available_models:
            app.session.add_error("No models available. Please configure API keys.")
            return

        current_model = settings.get_model()

        # Build options with current model marked
        options = []
        for model in available_models:
            label = f"{model} (current)" if model == current_model else model
            options.append((model, label))

        result = await radiolist_dialog(
            title="Select Model",
            text=f"Current model: {current_model}\nSelect a new model:",
            values=options,
            default=current_model,
            style=DIALOG_STYLE,
        ).run_async()

        if result and result != current_model:
            try:
                settings.set_model(result)
                # Reinitialize workflow manager with new model
                try:
                    workflow = app.workflow
                    await workflow.reinitialize(model=result, preserve_sessions=True)
                    app.session.add_success(f"Model changed to: {result}")
                    if settings.supports_thinking_effort(result):
                        app.session.add_message(
                            "system",
                            f"Thinking effort: {settings.thinking_effort}"
                        )
                except RuntimeError:
                    # Workflow not yet available - just update settings
                    app.session.add_success(f"Model set to: {result}")
            except ValueError as e:
                app.session.add_error(str(e))
        elif result == current_model:
            app.session.add_message("system", "Model unchanged.")

    async def _select_thinking_effort(self, app: Any) -> None:
        """Show thinking effort selection dialog."""
        settings = app.settings
        current_effort = settings.thinking_effort
        current_model = settings.get_model()

        if not settings.supports_thinking_effort():
            app.session.add_warning(
                f"Model '{current_model}' may not support thinking effort."
            )

        # Build options
        effort_descriptions = {
            "none": "No extended thinking",
            "low": "Brief reasoning",
            "medium": "Balanced thinking (default)",
            "high": "Deep, thorough reasoning",
        }

        options = []
        for effort in THINKING_EFFORT_LEVELS:
            desc = effort_descriptions.get(effort, effort)
            label = f"{effort.capitalize()}: {desc}"
            if effort == current_effort:
                label += " (current)"
            options.append((effort, label))

        result = await radiolist_dialog(
            title="Thinking Effort",
            text=f"Current: {current_effort}\nSelect thinking effort level:",
            values=options,
            default=current_effort,
            style=DIALOG_STYLE,
        ).run_async()

        if result and result != current_effort:
            try:
                settings.set_thinking_effort(result)
                # Reinitialize workflow to apply new planner config
                try:
                    workflow = app.workflow
                    await workflow.reinitialize(preserve_sessions=True)
                    app.session.add_success(f"Thinking effort changed to: {result}")
                except RuntimeError:
                    # Workflow not yet available - just update settings
                    app.session.add_success(f"Thinking effort set to: {result}")
            except ValueError as e:
                app.session.add_error(str(e))
        elif result == current_effort:
            app.session.add_message("system", "Thinking effort unchanged.")

    def _show_current_settings(self, app: Any) -> None:
        """Display current settings."""
        settings = app.settings
        current_model = settings.get_model()

        output = "**Current Settings**\n\n"
        output += f"**Model:** {current_model}\n"
        output += f"**Thinking Effort:** {settings.thinking_effort}\n"

        if settings.supports_thinking_effort():
            output += "  _(Model supports thinking effort)_\n"
        else:
            output += "  _(Model may not support thinking effort)_\n"

        output += "\n**API Keys:**\n"
        output += f"  Google: {'configured' if settings.has_google_key else 'not set'}\n"
        output += f"  Anthropic: {'configured' if settings.has_anthropic_key else 'not set'}\n"

        output += "\n**Available Models:**\n"
        for model in settings.get_available_models():
            marker = "> " if model == current_model else "  "
            output += f"{marker}{model}\n"

        app.session.add_response(output, markdown=True)

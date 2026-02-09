"""Base CLI Application for agentic applications.

This module provides the base CLI application that:
1. Uses ThinkingPromptSession for all UI (thinking boxes, messages, etc.)
2. Connects to domain-specific workflow managers
3. Tracks message history for persistence
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from thinking_prompt import ThinkingPromptSession, AppInfo
from thinking_prompt.styles import ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.cli.message_processor import MessageProcessor, MessageHistory, MessageType
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.logging import Loggers, configure_logging

if TYPE_CHECKING:
    from pathlib import Path
    from agentic_cli.workflow import GoogleADKWorkflowManager, EventType, WorkflowEvent
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.cli()


# === Slash Command Completer ===


class SlashCommandCompleter(Completer):
    """Completer that only triggers for slash commands."""

    def __init__(self, commands: list[str]) -> None:
        """Initialize with a list of command names (without leading slash).

        Args:
            commands: List of command names, e.g., ["help", "settings", "quit"]
        """
        self.commands = sorted(commands)

    def get_completions(self, document: Document, complete_event):
        """Yield completions only when text starts with /."""
        text = document.text_before_cursor

        # Only complete if input starts with /
        if not text.startswith("/"):
            return

        # Get the partial command (without the leading /)
        partial = text[1:].lower()

        # Find matching commands
        for cmd in self.commands:
            if cmd.lower().startswith(partial):
                # Calculate how much to complete (replace everything after /)
                yield Completion(
                    text=f"/{cmd}",
                    start_position=-len(text),
                    display=f"/{cmd}",
                )


# === Default Styles ===


def get_default_styles() -> ThinkingPromptStyles:
    """Get default styles for ThinkingPromptSession."""
    return ThinkingPromptStyles(
        thinking_box="fg:#6c757d",
        thinking_box_border="fg:#495057",
        thinking_box_hint="fg:#6c757d italic",
        user_message="fg:#17a2b8",
        assistant_message="fg:#e9ecef",
        system_message="fg:#6c757d italic",
        error_message="fg:#dc3545 bold",
        warning_message="fg:#ffc107",
        success_message="fg:#28a745",
    )


# === Base CLI Application ===


class BaseCLIApp:
    """
    Base CLI Application for agentic applications.

    Domain-specific applications extend this class and provide via constructor:
    - app_info: Application name, version, welcome message
    - agent_configs: List of agent configurations for the workflow
    - settings: Application settings instance

    Optional overrides:
    - register_commands(): Register domain-specific commands
    - get_styles(): Customize UI colors
    - get_ui_setting_keys(): Customize settings dialog fields
    - apply_settings(): Handle additional custom settings
    """

    def __init__(
        self,
        app_info: AppInfo,
        agent_configs: list["AgentConfig"],
        settings: BaseSettings,
    ) -> None:
        """Initialize the CLI application.

        Args:
            app_info: Application info (name, version, welcome message)
            agent_configs: List of agent configurations for the workflow
            settings: Application settings instance
        """
        # === Configuration ===
        self._app_info = app_info
        self._settings = settings
        configure_logging(self._settings)

        logger.info("app_starting", app_name=self._settings.app_name)

        # === Command Registry ===
        self.command_registry = CommandRegistry()
        self._register_builtin_commands()
        self.register_commands()  # Domain-specific commands

        # === Components ===
        self._workflow_controller = WorkflowController(
            agent_configs=agent_configs,
            settings=settings,
        )
        self._message_processor = MessageProcessor()

        # === UI: ThinkingPromptSession ===
        completer = SlashCommandCompleter(self.command_registry.get_completions())

        self.session = ThinkingPromptSession(
            message=">>> ",
            app_info=self._app_info,
            styles=self.get_styles(),
            history=InMemoryHistory(),
            completer=completer,
            complete_while_typing=True,
            completions_menu_height=8,
            enable_status_bar=True,
            status_text="Ctrl+C: cancel | Ctrl+D: exit | /help: commands",
            echo_input=False,  # We handle echoing manually based on command type
        )

        # === State ===
        self.should_exit = False

        logger.debug("app_initialized_fast")

    def get_styles(self) -> ThinkingPromptStyles:
        """Get styles for ThinkingPromptSession.

        Override to customize styles.
        """
        return get_default_styles()

    def register_commands(self) -> None:
        """Register domain-specific commands.

        Override to register additional commands.
        """
        pass

    def get_ui_setting_keys(self) -> list[str]:
        """Get field names to display in the settings dialog.

        Override to customize which settings appear in the UI.
        Default: model, thinking_effort, log_activity

        Returns:
            List of field names that should appear in the settings UI
        """
        return ["model", "thinking_effort", "log_activity"]

    def _build_ui_items(self) -> list[Any]:
        """Build UI items from settings fields using introspection.

        Uses get_ui_setting_keys() to determine which fields to show,
        then converts each field to an appropriate UI control using
        the field's type annotation and metadata.

        Returns:
            List of thinking_prompt UI items sorted by ui_order
        """
        from agentic_cli.cli.settings_introspection import field_to_ui_item, get_ui_order

        items: list[tuple[int, Any]] = []

        for key in self.get_ui_setting_keys():
            # Handle special 'model' field with dynamic options
            if key == "model":
                available_models = list(self._settings.get_available_models())
                if not available_models:
                    continue  # Skip if no models available

                # Get current model
                try:
                    current_model = self._settings.get_model()
                except RuntimeError:
                    current_model = None

                # Create a synthetic field for model selection
                from pydantic.fields import FieldInfo
                model_field = FieldInfo(
                    default=None,
                    title="Model",
                    description="Select the AI model to use",
                    json_schema_extra={"ui_order": 10},
                )
                item = field_to_ui_item(
                    key="model",
                    field=model_field,
                    current_value=current_model,
                    dynamic_options=available_models,
                )
                items.append((10, item))
                continue

            # Handle regular fields from settings
            # Access model_fields from class to avoid Pydantic deprecation warning
            settings_cls = type(self._settings)
            if key in settings_cls.model_fields:
                field = settings_cls.model_fields[key]
                current = getattr(self._settings, key, None)
                item = field_to_ui_item(key, field, current)
                order = get_ui_order(field)
                items.append((order, item))

        # Sort by order and return items only
        return [item for _, item in sorted(items, key=lambda x: x[0])]

    async def save_settings(self) -> "Path":
        """Save current settings to project config file (./.{app_name}/settings.json).

        Uses SettingsPersistence to save non-default settings to the
        project-level config file. Secrets (API keys) are never saved.

        Returns:
            Path to the saved config file
        """
        from pathlib import Path
        from agentic_cli.settings_persistence import SettingsPersistence

        persistence = SettingsPersistence(self._settings.app_name)
        return persistence.save(self._settings)

    @property
    def workflow(self) -> "BaseWorkflowManager":
        """Get the workflow manager.

        Raises:
            RuntimeError: If workflow is not yet initialized
        """
        return self._workflow_controller.workflow

    @property
    def message_history(self) -> MessageHistory:
        """Get the message history (for persistence)."""
        return self._message_processor.history

    @property
    def settings(self) -> BaseSettings:
        """Get the application settings."""
        return self._settings

    def stop(self) -> None:
        """Stop the application."""
        self.should_exit = True
        self.session.exit()

    async def apply_settings(self, changes: dict[str, Any]) -> None:
        """Apply changed settings and reinitialize workflow if needed.

        Settings are applied in two ways:
        1. Special settings (model, thinking_effort) use dedicated setters
           and may trigger workflow reinitialization
        2. All other settings are applied directly to the settings object

        Args:
            changes: Dictionary of changed settings (key -> new_value)
        """
        if not changes:
            self.session.add_message("system", "No changes made.")
            return

        needs_reinit = False
        new_model = changes.get("model")

        # Settings that require special handling
        special_settings = {"model", "thinking_effort"}

        # Apply model change (requires dedicated setter + reinit)
        if new_model:
            try:
                self._settings.set_model(new_model)
                needs_reinit = True
            except ValueError as e:
                self.session.add_error(f"Failed to set model: {e}")
                return

        # Apply thinking effort change (requires dedicated setter + reinit)
        if "thinking_effort" in changes:
            try:
                self._settings.set_thinking_effort(changes["thinking_effort"])
                needs_reinit = True
            except ValueError as e:
                self.session.add_error(f"Failed to set thinking effort: {e}")
                return

        # Apply all other settings directly
        for key, value in changes.items():
            if key not in special_settings:
                object.__setattr__(self._settings, key, value)

        # Reinitialize workflow if needed
        if needs_reinit and self._workflow_controller.is_ready:
            try:
                await self._workflow_controller.reinitialize(model=new_model)
            except Exception as e:
                self.session.add_error(f"Failed to reinitialize workflow: {e}")
                return

        # Report changes
        for key, value in changes.items():
            label = key.replace("_", " ").title()
            self.session.add_success(f"{label}: {value}")

    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        from agentic_cli.cli.builtin_commands import (
            HelpCommand,
            ClearCommand,
            ExitCommand,
            StatusCommand,
            PapersCommand,
        )
        from agentic_cli.cli.settings_command import SettingsCommand

        self.command_registry.register(HelpCommand())
        self.command_registry.register(ClearCommand())
        self.command_registry.register(ExitCommand())
        self.command_registry.register(StatusCommand())
        self.command_registry.register(SettingsCommand())
        self.command_registry.register(PapersCommand())

    async def process_input(self, user_input: str) -> None:
        """Process user input.

        Args:
            user_input: The raw user input string
        """
        user_input = user_input.strip()

        if not user_input:
            return

        # Route to appropriate handler
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            await self._handle_message(user_input)

    async def _handle_command(self, user_input: str) -> None:
        """Handle slash command execution.

        Args:
            user_input: The command string starting with /
        """
        parts = user_input[1:].split(maxsplit=1)
        command_name = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        command = self.command_registry.get(command_name)
        if command:
            # Echo command if not silent
            if not command.silent:
                self.session.add_message("user", user_input)
            logger.debug("executing_command", command=command.name, args=args)
            try:
                await command.execute(args, self)
                logger.debug("command_completed", command=command.name)
            except Exception as e:
                self.session.add_error(f"Error executing command: {e}")
        else:
            # Echo unknown command before error
            self.session.add_message("user", user_input)
            self.session.add_error(f"Unknown command: /{command_name}")
            self.session.add_message("system", "Type /help to see available commands")

    async def _handle_message(self, message: str) -> None:
        """Route message through agentic workflow.

        Delegates to MessageProcessor for event stream handling.

        Args:
            message: User message to process
        """
        # Echo user input for regular messages
        self.session.add_message("user", message)

        # Delegate to message processor
        await self._message_processor.process(
            message=message,
            workflow_controller=self._workflow_controller,
            ui=self.session,
            settings=self._settings,
        )

    async def _save_activity_log(self) -> None:
        """Save activity log to file on exit."""
        from datetime import datetime
        from agentic_cli.persistence import SessionPersistence

        session_name = f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        persistence = SessionPersistence(self._settings)

        try:
            metadata = {"app_name": self._settings.app_name}
            if self._workflow_controller.is_ready:
                metadata["model"] = self._workflow_controller.model

            saved_path = persistence.save_session(
                session_id=session_name,
                message_history=self.message_history,
                metadata=metadata,
            )
            logger.info("activity_log_saved", path=str(saved_path))
        except Exception as e:
            logger.error("activity_log_save_failed", error=str(e))

    async def run(self) -> None:
        """Run the main application loop."""
        logger.info("repl_starting")

        async with self._workflow_controller.background_init(self.session):
            # Register input handler
            @self.session.on_input
            async def handle_input(text: str) -> None:
                if self.should_exit:
                    return
                await self.process_input(text)

            # Run the session - user sees prompt immediately!
            await self.session.run_async()

        # Auto-save activity log if enabled
        if self._settings.log_activity and len(self.message_history) > 0:
            await self._save_activity_log()

        logger.info("app_ending")
        self.session.add_message("system", "Goodbye!")

"""Base CLI Application for agentic applications.

This module provides the base CLI application that:
1. Uses ThinkingPromptSession for all UI (thinking boxes, messages, etc.)
2. Connects to domain-specific workflow managers
3. Tracks message history for persistence
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from thinking_prompt import ThinkingPromptSession, AppInfo
from thinking_prompt.styles import ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.config import BaseSettings
from agentic_cli.logging import Loggers, configure_logging, bind_context

if TYPE_CHECKING:
    from pathlib import Path
    from agentic_cli.workflow import GoogleADKWorkflowManager, EventType, WorkflowEvent
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.cli()

# Thread pool for background initialization (single worker)
_init_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-init")


def create_workflow_manager_from_settings(
    agent_configs: list["AgentConfig"],
    settings: BaseSettings,
    app_name: str | None = None,
    model: str | None = None,
    **kwargs,
) -> "BaseWorkflowManager":
    """Factory function to create the appropriate workflow manager based on settings.

    Creates either a GoogleADKWorkflowManager or LangGraphWorkflowManager
    based on the settings.orchestrator configuration.

    Args:
        agent_configs: List of agent configurations.
        settings: Application settings (determines orchestrator type).
        app_name: Application name for services.
        model: Model override.
        **kwargs: Additional arguments passed to the specific manager.

    Returns:
        BaseWorkflowManager instance (ADK or LangGraph based on settings).

    Raises:
        ImportError: If LangGraph is selected but not installed.

    Example:
        settings = MySettings(orchestrator="langgraph")
        configs = [AgentConfig(name="agent", prompt="...")]
        manager = create_workflow_manager_from_settings(configs, settings)
    """
    orchestrator = getattr(settings, "orchestrator", "adk")

    if orchestrator == "langgraph":
        try:
            from agentic_cli.workflow.langgraph_manager import (
                LangGraphWorkflowManager,
            )

            checkpointer = getattr(settings, "langgraph_checkpointer", "memory")
            return LangGraphWorkflowManager(
                agent_configs=agent_configs,
                settings=settings,
                app_name=app_name,
                model=model,
                checkpointer=checkpointer,
                **kwargs,
            )
        except ImportError as e:
            raise ImportError(
                f"LangGraph orchestrator selected but dependencies not installed. "
                f"Install with: pip install agentic-cli[langgraph]\n"
                f"Original error: {e}"
            ) from e

    else:  # Default to ADK
        from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

        return GoogleADKWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            **kwargs,
        )


# === Message History for Persistence ===


class MessageType(Enum):
    """Types of messages in history."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    THINKING = "thinking"


@dataclass
class Message:
    """A message stored in history for persistence."""

    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class MessageHistory:
    """Simple message history for persistence."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(
        self,
        content: str,
        message_type: MessageType | str,
        timestamp: datetime | None = None,
        **metadata: object,
    ) -> None:
        """Add a message to history."""
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        self._messages.append(
            Message(
                content=content,
                message_type=message_type,
                timestamp=timestamp or datetime.now(),
                metadata=dict(metadata),
            )
        )

    def get_all(self) -> list[Message]:
        """Get all messages."""
        return list(self._messages)

    def get_by_type(self, message_type: MessageType) -> list[Message]:
        """Get messages of a specific type."""
        return [m for m in self._messages if m.message_type == message_type]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


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


class BaseCLIApp(ABC):
    """
    Base CLI Application for agentic applications.

    Domain-specific applications extend this class and implement:
    - get_app_info(): Provide app name, version, welcome message
    - get_settings(): Provide domain-specific settings
    - create_workflow_manager(): Create domain-specific workflow manager
    - register_commands(): Register domain-specific commands (optional)
    """

    def __init__(self, settings: BaseSettings | None = None) -> None:
        """Initialize the CLI application.

        Args:
            settings: Optional settings override
        """
        # === Configuration ===
        self._settings = settings or self.get_settings()
        configure_logging(self._settings)

        logger.info("app_starting", app_name=self._settings.app_name)

        # === Message History (for persistence) ===
        self.message_history = MessageHistory()

        # === Command Registry ===
        self.command_registry = CommandRegistry()
        self._register_builtin_commands()
        self.register_commands()  # Domain-specific commands

        # === Workflow Manager (initialized in background) ===
        self._workflow: BaseWorkflowManager | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None

        # === UI: ThinkingPromptSession ===
        completer = SlashCommandCompleter(self.command_registry.get_completions())

        self.session = ThinkingPromptSession(
            message=">>> ",
            app_info=self.get_app_info(),
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

    @abstractmethod
    def get_app_info(self) -> AppInfo:
        """Get the application info for ThinkingPromptSession.

        Domain projects implement this to provide their app name,
        version, and welcome message.
        """
        ...

    @abstractmethod
    def get_settings(self) -> BaseSettings:
        """Get the application settings.

        Domain projects implement this to provide their settings class.
        """
        ...

    @abstractmethod
    def create_workflow_manager(self) -> "BaseWorkflowManager":
        """Create the workflow manager for this domain.

        Domain projects implement this to create their workflow manager
        with domain-specific agents.

        Use the factory function `create_workflow_manager_from_settings()` to
        automatically select the orchestrator based on settings.orchestrator.

        Example:
            def create_workflow_manager(self) -> BaseWorkflowManager:
                from agentic_cli.cli.app import create_workflow_manager_from_settings
                return create_workflow_manager_from_settings(
                    agent_configs=self._get_agent_configs(),
                    settings=self._settings,
                    app_name=self._settings.app_name,
                )
        """
        ...

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
        """Save current settings to project config file (./settings.json).

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
        if self._workflow is None:
            raise RuntimeError("Workflow not initialized yet")
        return self._workflow

    @property
    def default_user(self) -> str:
        """Get the default user name."""
        return self._settings.default_user

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

        Args:
            changes: Dictionary of changed settings (key -> new_value)
        """
        if not changes:
            self.session.add_message("system", "No changes made.")
            return

        needs_reinit = False
        new_model = changes.get("model")

        # Apply model change
        if new_model:
            try:
                self._settings.set_model(new_model)
                needs_reinit = True
            except ValueError as e:
                self.session.add_error(f"Failed to set model: {e}")
                return

        # Apply thinking effort change
        if "thinking_effort" in changes:
            try:
                self._settings.set_thinking_effort(changes["thinking_effort"])
                needs_reinit = True
            except ValueError as e:
                self.session.add_error(f"Failed to set thinking effort: {e}")
                return

        # Apply log_activity change
        if "log_activity" in changes:
            object.__setattr__(self._settings, "log_activity", changes["log_activity"])

        # Reinitialize workflow if needed
        if needs_reinit and self._workflow is not None:
            try:
                await self._workflow.reinitialize(model=new_model, preserve_sessions=True)
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
        )
        from agentic_cli.cli.settings_command import SettingsCommand

        self.command_registry.register(HelpCommand())
        self.command_registry.register(ClearCommand())
        self.command_registry.register(ExitCommand())
        self.command_registry.register(StatusCommand())
        self.command_registry.register(SettingsCommand())

    async def _background_init(self) -> None:
        """Initialize workflow manager in background thread."""
        loop = asyncio.get_running_loop()

        def _create_workflow() -> "BaseWorkflowManager":
            return self.create_workflow_manager()

        try:
            logger.debug("background_init_starting")
            self._workflow = await loop.run_in_executor(
                _init_executor, _create_workflow
            )

            # Update status bar to show ready
            model = self._workflow.model
            self.session.status_text = f"{model} | Ctrl+C: cancel | /help: commands"
            logger.info("background_init_complete", model=model)

        except Exception as e:
            self._init_error = e
            # Don't log to console - error will be shown via UI when user interacts
            self.session.status_text = "Init failed - check API keys"

    async def _ensure_initialized(self) -> bool:
        """Wait for background initialization to complete.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._workflow is not None:
            return True

        if self._init_task is None:
            return False

        if not self._init_task.done():
            # Show user we're waiting for initialization
            self.session.start_thinking(lambda: "Waiting for initialization...")
            try:
                await self._init_task
            finally:
                self.session.finish_thinking(add_to_history=False)

        if self._init_error:
            self.session.add_error(f"Initialization failed: {self._init_error}")
            return False

        return self._workflow is not None

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

    async def _prompt_user_input(self, event: "WorkflowEvent") -> str:
        """Prompt user for input requested by a tool.

        Args:
            event: USER_INPUT_REQUIRED event with prompt details

        Returns:
            User's response string
        """
        from agentic_cli.workflow import WorkflowEvent  # noqa: F811

        input_type = event.metadata.get("input_type", "text")
        tool_name = event.metadata.get("tool_name", "Tool")
        choices = event.metadata.get("choices")
        default = event.metadata.get("default")

        if input_type == "choice" and choices:
            # Use choice dialog for multiple options
            result = await self.session.choice_dialog(
                title=f"{tool_name} - Input Required",
                text=event.content,
                choices=choices,
            )
            return result or (default or "")

        elif input_type == "confirm":
            # Use yes/no dialog for confirmation
            result = await self.session.yes_no_dialog(
                title=f"{tool_name} - Confirmation",
                text=event.content,
            )
            return "yes" if result else "no"

        else:
            # Use text input dialog for free-form input
            result = await self.session.input_dialog(
                title=f"{tool_name} - Input Required",
                text=event.content,
                default=default or "",
            )
            return result or ""

    async def _handle_message(self, message: str) -> None:
        """Route message through agentic workflow.

        Args:
            message: User message to process
        """
        # Echo user input for regular messages
        self.session.add_message("user", message)

        # Wait for initialization if needed
        if not await self._ensure_initialized():
            self.session.add_error(
                "Cannot process message - workflow not initialized. "
                "Please check your API keys (GOOGLE_API_KEY or ANTHROPIC_API_KEY)."
            )
            return

        # Import EventType here (workflow module is now loaded)
        from agentic_cli.workflow import EventType

        bind_context(user_id=self._settings.default_user)
        logger.info("handling_message", message_length=len(message))

        # Track message in history (if logging enabled)
        if self._settings.log_activity:
            self.message_history.add(message, MessageType.USER)

        # Status line for thinking box (single line updates)
        status_line = "Processing..."
        thinking_started = False

        # Accumulate content for history
        thinking_content: list[str] = []
        response_content: list[str] = []

        def get_status() -> str:
            return status_line

        try:
            self.session.start_thinking(get_status)
            thinking_started = True

            async for event in self.workflow.process(
                message=message,
                user_id=self._settings.default_user,
            ):
                if event.type == EventType.TEXT:
                    # Stream response directly to console
                    self.session.add_response(event.content, markdown=True)
                    response_content.append(event.content)

                elif event.type == EventType.THINKING:
                    # Stream thinking directly to console
                    status_line = "Thinking..."
                    self.session.add_message("system", event.content)
                    thinking_content.append(event.content)

                elif event.type == EventType.TOOL_CALL:
                    # Update status line in thinking box
                    tool_name = event.metadata.get("tool_name", "unknown")
                    status_line = f"Calling: {tool_name}"

                elif event.type == EventType.TOOL_RESULT:
                    # Display tool result summary
                    tool_name = event.metadata.get("tool_name", "unknown")
                    success = event.metadata.get("success", True)
                    duration = event.metadata.get("duration_ms")
                    icon = "✓" if success else "✗"
                    duration_str = f" ({duration}ms)" if duration else ""
                    status_line = f"{icon} {tool_name}: {event.content}{duration_str}"
                    # Also show in message area for visibility
                    style = "green" if success else "red"
                    self.session.add_message(
                        "system",
                        f"[{style}]{icon}[/{style}] {tool_name}: {event.content}{duration_str}"
                    )

                elif event.type == EventType.USER_INPUT_REQUIRED:
                    # Pause thinking, prompt user, resume
                    if thinking_started:
                        self.session.finish_thinking(add_to_history=False)
                        thinking_started = False

                    response = await self._prompt_user_input(event)
                    self.workflow.provide_user_input(
                        event.metadata["request_id"],
                        response,
                    )

                    # Resume thinking box
                    self.session.start_thinking(get_status)
                    thinking_started = True

                elif event.type == EventType.CODE_EXECUTION:
                    # Update status with execution result
                    result_preview = (
                        event.content[:40] + "..."
                        if len(event.content) > 40
                        else event.content
                    )
                    status_line = f"Result: {result_preview}"

                elif event.type == EventType.EXECUTABLE_CODE:
                    # Update status when executing code
                    lang = event.metadata.get("language", "python")
                    status_line = f"Running {lang} code..."

                elif event.type == EventType.FILE_DATA:
                    # Update status with file info
                    status_line = f"File: {event.content}"

            # Finish thinking box (don't add status to history)
            if thinking_started:
                self.session.finish_thinking(add_to_history=False)

            # Add accumulated content to message history (if logging enabled)
            if self._settings.log_activity:
                if thinking_content:
                    self.message_history.add(
                        "".join(thinking_content), MessageType.THINKING
                    )
                if response_content:
                    self.message_history.add(
                        "".join(response_content), MessageType.ASSISTANT
                    )

            logger.debug("message_handled_successfully")

        except Exception as e:
            if thinking_started:
                self.session.finish_thinking(add_to_history=False)
            self.session.add_error(f"Workflow error: {e}")
            if self._settings.log_activity:
                self.message_history.add(str(e), MessageType.ERROR)

    async def _save_activity_log(self) -> None:
        """Save activity log to file on exit."""
        from datetime import datetime
        from agentic_cli.persistence import SessionPersistence

        session_name = f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        persistence = SessionPersistence(self._settings)

        try:
            metadata = {"app_name": self._settings.app_name}
            if self._workflow:
                metadata["model"] = self._workflow.model

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

        # Start background initialization (non-blocking)
        self._init_task = asyncio.create_task(self._background_init())

        # Register input handler
        @self.session.on_input
        async def handle_input(text: str) -> None:
            if self.should_exit:
                return
            await self.process_input(text)

        # Run the session - user sees prompt immediately!
        await self.session.run_async()

        # Cleanup: cancel init task if still running
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass

        # Auto-save activity log if enabled
        if self._settings.log_activity and len(self.message_history) > 0:
            await self._save_activity_log()

        logger.info("app_ending")
        self.session.add_message("system", "Goodbye!")

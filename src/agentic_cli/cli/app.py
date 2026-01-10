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

from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from thinking_prompt import ThinkingPromptSession, AppInfo
from thinking_prompt.styles import ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.config import BaseSettings
from agentic_cli.logging import Loggers, configure_logging, bind_context

if TYPE_CHECKING:
    from agentic_cli.workflow import WorkflowManager, EventType

logger = Loggers.cli()

# Thread pool for background initialization (single worker)
_init_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-init")


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
        self._workflow: WorkflowManager | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None

        # === UI: ThinkingPromptSession ===
        command_completions = [
            "/" + name for name in self.command_registry.get_completions()
        ]
        completer = WordCompleter(command_completions, ignore_case=True)

        self.session = ThinkingPromptSession(
            message=">>> ",
            app_info=self.get_app_info(),
            styles=self.get_styles(),
            history=InMemoryHistory(),
            completer=completer,
            enable_status_bar=True,
            status_text="Ctrl+C: cancel | Ctrl+D: exit | /help: commands",
        )

        # === State ===
        self.should_exit = False

        # === Rich Console (for commands that need direct console access) ===
        self.console = Console()

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
    def create_workflow_manager(self) -> "WorkflowManager":
        """Create the workflow manager for this domain.

        Domain projects implement this to create their workflow manager
        with domain-specific agents.
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

    @property
    def workflow(self) -> "WorkflowManager":
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

    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        from agentic_cli.cli.builtin_commands import (
            HelpCommand,
            ClearCommand,
            ExitCommand,
            StatusCommand,
            SaveCommand,
            LoadCommand,
            SessionsCommand,
            SettingsCommand,
        )

        self.command_registry.register(HelpCommand())
        self.command_registry.register(ClearCommand())
        self.command_registry.register(ExitCommand())
        self.command_registry.register(StatusCommand())
        self.command_registry.register(SaveCommand())
        self.command_registry.register(LoadCommand())
        self.command_registry.register(SessionsCommand())
        self.command_registry.register(SettingsCommand())

    async def _background_init(self) -> None:
        """Initialize WorkflowManager in background thread."""
        loop = asyncio.get_running_loop()

        def _create_workflow() -> "WorkflowManager":
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
            logger.debug("executing_command", command=command.name, args=args)
            try:
                self.session.add_response(f"Executing command: /{command.name} {args}")
                await command.execute(args, self)
                logger.debug("command_completed", command=command.name)
            except Exception as e:
                self.session.add_error(f"Error executing command: {e}")
        else:
            self.session.add_error(f"Unknown command: /{command_name}")
            self.session.add_message("system", "Type /help to see available commands")

    async def _handle_message(self, message: str) -> None:
        """Route message through agentic workflow.

        Args:
            message: User message to process
        """
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

        # Track message in history
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

            # Add accumulated content to message history
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
            self.message_history.add(str(e), MessageType.ERROR)

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

        logger.info("app_ending")
        self.session.add_message("system", "Goodbye!")

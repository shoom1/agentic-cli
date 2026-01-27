"""Main application for the Research Demo.

Showcases all P0/P1 features through a research assistant agent
with memory, planning, file operations, shell commands, and HITL.
"""

from typing import Any

from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp
from agentic_cli.cli import AppInfo
from agentic_cli.cli.app import MessageType
from agentic_cli.hitl import ApprovalManager, ApprovalRule, CheckpointManager, HITLConfig
from agentic_cli.logging import bind_context, Loggers
from agentic_cli.memory import MemoryManager
from agentic_cli.planning import TaskGraph

from examples.research_demo.agents import AGENT_CONFIGS
from examples.research_demo.commands import DEMO_COMMANDS
from examples.research_demo.settings import ResearchDemoSettings, get_settings
from examples.research_demo.tools import set_demo_state

logger = Loggers.cli()


def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("Research Demo\n\n", style="bold cyan")
    text.append("A research assistant with memory and planning capabilities.\n\n", style="dim")
    text.append("Features:\n", style="bold")
    text.append("  - Working memory (session context)\n", style="dim")
    text.append("  - Long-term memory (persistent learnings)\n", style="dim")
    text.append("  - Task planning (research plans)\n", style="dim")
    text.append("  - File operations (save/compare findings)\n", style="dim")
    text.append("  - Shell commands (safe execution)\n", style="dim")
    text.append("\n")
    text.append("Commands:\n", style="bold")
    text.append("  /memory  - Show memory state\n", style="cyan")
    text.append("  /plan    - Show task plan\n", style="cyan")
    text.append("  /files   - List saved findings\n", style="cyan")
    text.append("  /help    - All commands\n", style="cyan")

    return AppInfo(
        name="Research Demo",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="cyan"),
        echo_thinking=True,
    )


class ResearchDemoApp(BaseCLIApp):
    """Research Demo CLI Application.

    Demonstrates all P0/P1 features:
    - Memory: Working memory (session) + Long-term memory (persistent)
    - Planning: Task graphs with dependencies
    - File Operations: Save/read/compare findings
    - Shell Commands: Safe command execution
    - HITL: Approval gates and checkpoints
    """

    def __init__(self, settings: ResearchDemoSettings | None = None) -> None:
        # Initialize feature managers before super().__init__
        # These will be configured after settings are available

        self._memory_manager: MemoryManager | None = None
        self._task_graph: TaskGraph | None = None
        self._approval_manager: ApprovalManager | None = None
        self._checkpoint_manager: CheckpointManager | None = None

        # Call parent __init__ with app_info, agent_configs, and settings
        super().__init__(
            app_info=_create_app_info(),
            agent_configs=AGENT_CONFIGS,
            settings=settings or get_settings(),
        )

        # Now initialize our managers with the resolved settings
        self._init_feature_managers()

    def _init_feature_managers(self) -> None:
        """Initialize memory, planning, and HITL managers."""
        # Memory Manager
        self._memory_manager = MemoryManager(self._settings)

        # Task Graph
        self._task_graph = TaskGraph()

        # HITL Config with approval rules
        hitl_config = HITLConfig(
            approval_rules=[
                # Require approval for shell commands (except safe ones)
                ApprovalRule(
                    tool="shell_executor",
                    operations=None,  # All operations
                    auto_approve_patterns=[
                        "ls*", "pwd", "cat *", "head *", "tail *",
                        "grep *", "find *", "echo *", "date", "whoami",
                    ],
                ),
                # Require approval for file writes
                ApprovalRule(
                    tool="file_manager",
                    operations=["write", "delete", "move"],
                    auto_approve_patterns=[],
                ),
            ],
            checkpoint_enabled=True,
            feedback_enabled=True,
        )

        self._approval_manager = ApprovalManager(hitl_config)
        self._checkpoint_manager = CheckpointManager()

        # Share state with tools module
        set_demo_state(
            memory_manager=self._memory_manager,
            task_graph=self._task_graph,
            approval_manager=self._approval_manager,
            checkpoint_manager=self._checkpoint_manager,
            settings=self._settings,
        )

    @property
    def memory_manager(self) -> MemoryManager | None:
        """Access the memory manager."""
        return self._memory_manager

    @property
    def task_graph(self) -> TaskGraph | None:
        """Access the task graph."""
        return self._task_graph

    @property
    def approval_manager(self) -> ApprovalManager | None:
        """Access the approval manager."""
        return self._approval_manager

    @property
    def checkpoint_manager(self) -> CheckpointManager | None:
        """Access the checkpoint manager."""
        return self._checkpoint_manager

    def register_commands(self) -> None:
        """Register demo-specific commands."""
        for command_class in DEMO_COMMANDS:
            self.command_registry.register(command_class())

    def get_ui_setting_keys(self) -> list[str]:
        """Get field names to display in the settings dialog.

        Includes verbose_output in addition to standard settings.
        """
        return ["model", "thinking_effort", "log_activity", "verbose_output"]

    async def apply_settings(self, changes: dict[str, Any]) -> None:
        """Apply settings including verbose_output.

        Args:
            changes: Dictionary of changed settings (key -> new_value)
        """
        # Handle verbose_output change before calling parent
        if "verbose_output" in changes:
            object.__setattr__(self._settings, "verbose_output", changes["verbose_output"])

        # Call parent for standard settings
        await super().apply_settings(changes)

    async def _handle_message(self, message: str) -> None:
        """Route message through agentic workflow, respecting verbose setting.

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
                    # Only show thinking content if verbose mode is enabled
                    status_line = "Thinking..."
                    if self._settings.verbose_output:
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

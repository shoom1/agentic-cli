"""Main application for the Research Demo.

Showcases framework features through a research assistant agent
with memory, planning, knowledge base, web fetching, code execution,
file operations, shell commands, and HITL.

Feature managers (MemoryManager, TaskGraph, CheckpointManager,
ApprovalManager) are auto-created by the workflow manager based on
tool requirements.
"""

from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp
from agentic_cli.cli import AppInfo
from agentic_cli.logging import Loggers

from examples.research_demo.agents import AGENT_CONFIGS
from examples.research_demo.commands import DEMO_COMMANDS
from examples.research_demo.settings import ResearchDemoSettings, get_settings

logger = Loggers.cli()


def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("Research Demo\n\n", style="bold cyan")
    text.append("A research assistant with memory, planning, knowledge base, and HITL capabilities.\n\n", style="dim")
    text.append("Features:\n", style="bold")
    text.append("  - Working & long-term memory\n", style="dim")
    text.append("  - Task planning with dependencies\n", style="dim")
    text.append("  - Knowledge base (search & ingest)\n", style="dim")
    text.append("  - Web search & content fetching\n", style="dim")
    text.append("  - Academic research (arXiv)\n", style="dim")
    text.append("  - Python code execution\n", style="dim")
    text.append("  - File operations & shell commands\n", style="dim")
    text.append("  - Human-in-the-loop (checkpoints & approvals)\n", style="dim")
    text.append("\n")
    text.append("Commands:\n", style="bold")
    text.append("  /memory     - Show memory state\n", style="cyan")
    text.append("  /plan       - Show task plan\n", style="cyan")
    text.append("  /files      - List saved findings\n", style="cyan")
    text.append("  /approvals  - Show pending approvals\n", style="cyan")
    text.append("  /help       - All commands\n", style="cyan")

    return AppInfo(
        name="Research Demo",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="cyan"),
        echo_thinking=False,
    )


class ResearchDemoApp(BaseCLIApp):
    """Research Demo CLI Application.

    Demonstrates framework features:
    - Memory: Working memory (session) + Long-term memory (persistent)
    - Planning: Task graphs with dependencies and mid-execution revision
    - Knowledge Base: Search and ingest documents
    - Web: Search and fetch/summarize URL content
    - Code Execution: Sandboxed Python execution
    - File Operations: Save/read/compare findings
    - Shell Commands: Safe command execution
    - User Interaction: Clarification questions
    - HITL: Checkpoints, approvals, and review workflows

    Feature managers are auto-created by the workflow manager based on
    the @requires decorators on framework tools.
    """

    def __init__(self, settings: ResearchDemoSettings | None = None) -> None:
        super().__init__(
            app_info=_create_app_info(),
            agent_configs=AGENT_CONFIGS,
            settings=settings or get_settings(),
        )

    def register_commands(self) -> None:
        """Register demo-specific commands."""
        for command_class in DEMO_COMMANDS:
            self.command_registry.register(command_class())

    def get_ui_setting_keys(self) -> list[str]:
        """Get field names to display in the settings dialog.

        Includes verbose_thinking in addition to standard settings.
        """
        return ["model", "thinking_effort", "log_activity", "verbose_thinking"]

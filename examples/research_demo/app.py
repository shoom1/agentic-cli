"""Main application for the Research Demo.

Multi-agent research assistant showcasing framework features:
- research_coordinator: Root agent with memory, planning, tasks, HITL, web, code, files
- arxiv_specialist: Sub-agent for focused arXiv paper search, analysis, and cataloging

Feature managers (MemoryStore, PlanStore, TaskStore, ApprovalManager)
are auto-created by the workflow manager based on tool requirements.
"""

from rich.panel import Panel
from rich.text import Text

from agentic_cli import BaseCLIApp
from agentic_cli.cli import AppInfo
from agentic_cli.logging import Loggers

from examples.research_demo.agents import AGENT_CONFIGS
from examples.research_demo.commands import DEMO_COMMANDS
from examples.research_demo.settings import ResearchDemoSettings

logger = Loggers.cli()


def _create_app_info() -> AppInfo:
    """Create the application info for the welcome message."""
    text = Text()
    text.append("Research Demo\n\n", style="bold cyan")
    text.append("A multi-agent research assistant powered by the agentic-cli framework.\n\n", style="dim")
    text.append("Agents:\n", style="bold")
    text.append("  - research_coordinator — plans, coordinates, manages workflow\n", style="dim")
    text.append("  - arxiv_specialist — arXiv paper search, analysis, cataloging\n", style="dim")
    text.append("\nFeatures:\n", style="bold")
    text.append("  - Persistent memory across sessions\n", style="dim")
    text.append("  - Planning (markdown checkboxes)\n", style="dim")
    text.append("  - Task tracking (status & priority)\n", style="dim")
    text.append("  - Knowledge base (search & ingest)\n", style="dim")
    text.append("  - Web search & content fetching (incl. PDF)\n", style="dim")
    text.append("  - Academic research (arXiv)\n", style="dim")
    text.append("  - Python code execution\n", style="dim")
    text.append("  - File operations (read, write, search)\n", style="dim")
    text.append("  - Human-in-the-loop (approvals)\n", style="dim")
    text.append("\n")
    text.append("Commands:\n", style="bold")
    text.append("  /memory     - Show saved memories\n", style="cyan")
    text.append("  /plan       - Show current plan\n", style="cyan")
    text.append("  /tasks      - Show execution tasks\n", style="cyan")
    text.append("  /files      - List workspace files\n", style="cyan")
    text.append("  /approvals  - Show approval history\n", style="cyan")
    text.append("  /help       - All commands\n", style="cyan")

    return AppInfo(
        name="Research Demo",
        version="0.1.0",
        welcome_message=lambda: Panel(text, border_style="cyan"),
        echo_thinking=False,
    )


class ResearchDemoApp(BaseCLIApp):
    """Research Demo CLI Application.

    Multi-agent architecture:
    - research_coordinator: Root agent with memory, planning, tasks, HITL,
      web search, code execution, file operations
    - arxiv_specialist: Sub-agent for arXiv paper search, analysis, and
      knowledge base ingestion (incl. PDF full-text extraction)

    Feature managers are auto-created by the workflow manager based on
    the @requires decorators on framework tools.
    """

    def __init__(self, settings: ResearchDemoSettings | None = None) -> None:
        super().__init__(
            app_info=_create_app_info(),
            agent_configs=AGENT_CONFIGS,
            settings=settings or ResearchDemoSettings(),
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

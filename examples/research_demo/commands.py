"""Status commands for the Research Demo application.

Provides commands for inspecting memory, plan, files, approvals, and checkpoints.
Managers are accessed via app.workflow which auto-creates them based on tool requirements.
"""

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from examples.research_demo.app import ResearchDemoApp


class MemoryCommand(Command):
    """Show persistent memory contents."""

    def __init__(self) -> None:
        super().__init__(
            name="memory",
            description="Show persistent memory contents",
            aliases=[],
            usage="/memory",
            examples=["/memory"],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        memory_store = app.workflow.memory_manager if app.workflow else None

        table = Table(title="Persistent Memory", show_header=True)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Content", style="white")
        table.add_column("Tags", style="dim")

        if memory_store:
            for item in memory_store.search("", limit=20):
                content_str = item.content
                if len(content_str) > 60:
                    content_str = content_str[:60] + "..."
                tags_str = ", ".join(item.tags) if item.tags else ""
                table.add_row(item.id[:8], content_str, tags_str)

        if table.row_count == 0:
            table.add_row("(empty)", "", "")

        app.session.add_rich(table)


class PlanCommand(Command):
    """Show current task plan."""

    def __init__(self) -> None:
        super().__init__(
            name="plan",
            description="Show current research task graph",
            aliases=[],
            usage="/plan",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        plan_store = app.workflow.task_graph if app.workflow else None

        if plan_store is None:
            app.session.add_message("system", "Plan store not initialized")
            return

        if plan_store.is_empty():
            app.session.add_message("system", "No plan created yet. Ask the agent to create a research plan.")
            return

        panel = Panel(plan_store.get(), title="Research Plan", border_style="blue")
        app.session.add_rich(panel)


class ApprovalsCommand(Command):
    """Show approval history."""

    def __init__(self) -> None:
        super().__init__(
            name="approvals",
            description="Show approval history",
            aliases=["approve"],
            usage="/approvals",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        approval_manager = app.workflow.approval_manager if app.workflow else None

        if approval_manager is None:
            app.session.add_message("system", "Approval manager not initialized")
            return

        history = approval_manager.history

        if not history:
            app.session.add_message("system", "No approval history")
            return

        table = Table(title="Approval History", show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Approved", style="cyan")
        table.add_column("Reason", style="white")

        for result in history:
            table.add_row(
                result.request_id,
                "Yes" if result.approved else "No",
                result.reason or "",
            )

        app.session.add_rich(table)


class CheckpointsCommand(Command):
    """Show checkpoint history."""

    def __init__(self) -> None:
        super().__init__(
            name="checkpoints",
            description="Show checkpoint review history",
            aliases=[],
            usage="/checkpoints",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        checkpoint_manager = app.workflow.checkpoint_manager if app.workflow else None

        if checkpoint_manager is None:
            app.session.add_message("system", "Checkpoint manager not initialized")
            return

        history = checkpoint_manager.history

        if not history:
            app.session.add_message("system", "No checkpoint history")
            return

        table = Table(title="Checkpoint History", show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Action", style="cyan")
        table.add_column("Feedback", style="white")

        for result in history:
            table.add_row(
                result.checkpoint_id,
                result.action,
                result.feedback or "",
            )

        app.session.add_rich(table)


class FilesCommand(Command):
    """List files in workspace."""

    def __init__(self) -> None:
        super().__init__(
            name="files",
            description="List files in workspace (findings, artifacts)",
            aliases=[],
            usage="/files [--dir=DIR]",
            examples=[
                "/files",
                "/files --dir=findings",
            ],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        parsed = self.parse_args(args)
        subdir = parsed.get_option("dir", "findings")

        from agentic_cli.tools.glob_tool import list_dir
        from agentic_cli.tools.registry import ToolError

        workspace = app.settings.workspace_dir
        target_dir = workspace / subdir

        if not target_dir.exists():
            app.session.add_message("system", f"Directory does not exist: {target_dir}")
            return

        try:
            result = list_dir(str(target_dir))
        except ToolError as e:
            app.session.add_error(e.message)
            return

        table = Table(title=f"Files in {subdir}/", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Size", style="dim", justify="right")

        # Add directories
        for item in result["directories"]:
            table.add_row(item["name"], "directory", "")

        # Add files
        for item in result["files"]:
            size_str = ""
            if item["size"] is not None:
                size_str = f"{item['size']:,} bytes"
            table.add_row(item["name"], "file", size_str)

        if table.row_count == 0:
            table.add_row("(empty)", "", "")

        app.session.add_rich(table)
        app.session.add_message("system", f"Total: {result['total']} items")


class ClearMemoryCommand(Command):
    """No-op — persistent memory cannot be bulk-cleared from the CLI."""

    def __init__(self) -> None:
        super().__init__(
            name="clear-memory",
            description="(Deprecated) Memory is now persistent-only",
            aliases=["clearmem"],
            usage="/clear-memory",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        app.session.add_warning(
            "Working memory has been removed. Persistent memories "
            "are managed via save_memory/search_memory tools."
        )


class ClearPlanCommand(Command):
    """Clear the task plan."""

    def __init__(self) -> None:
        super().__init__(
            name="clear-plan",
            description="Clear the current task plan",
            aliases=["clearplan"],
            usage="/clear-plan",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        plan_store = app.workflow.task_graph if app.workflow else None

        if plan_store:
            plan_store.clear()
            app.session.add_success("Plan cleared")
        else:
            app.session.add_error("Plan store not initialized")


# Export all commands for registration
# Note: Settings command is now handled by the base SettingsCommand
# with get_ui_setting_keys() override in ResearchDemoApp
DEMO_COMMANDS = [
    MemoryCommand,
    PlanCommand,
    ApprovalsCommand,
    CheckpointsCommand,
    FilesCommand,
    ClearMemoryCommand,
    ClearPlanCommand,
]

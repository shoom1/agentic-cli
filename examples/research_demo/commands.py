"""Status commands for the Research Demo application.

Provides commands for inspecting memory, plan, files, approvals, and checkpoints.
Managers are accessed via app.workflow which auto-creates them based on tool requirements.
"""

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        # Access task graph from workflow
        task_graph = app.workflow.task_graph if app.workflow else None

        if task_graph is None:
            app.session.add_message("system", "No task graph initialized")
            return

        # Get progress statistics
        progress = task_graph.get_progress()

        if progress["total"] == 0:
            app.session.add_message("system", "No tasks in the plan. Ask the agent to create a research plan.")
            return

        # Progress summary
        completed = progress["completed"]
        total = progress["total"]
        in_progress = progress["in_progress"]
        pending = progress["pending"]
        failed = progress["failed"]

        progress_text = Text()
        progress_text.append(f"Progress: {completed}/{total} completed")
        if in_progress > 0:
            progress_text.append(f", {in_progress} in progress", style="yellow")
        if pending > 0:
            progress_text.append(f", {pending} pending", style="dim")
        if failed > 0:
            progress_text.append(f", {failed} failed", style="red")

        app.session.add_rich(progress_text)

        # Task display
        display = task_graph.to_display()
        panel = Panel(display, title="Research Plan", border_style="blue")
        app.session.add_rich(panel)


class ApprovalsCommand(Command):
    """Show pending approvals."""

    def __init__(self) -> None:
        super().__init__(
            name="approvals",
            description="Show pending approval requests",
            aliases=["approve"],
            usage="/approvals",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        # Access approval manager from workflow
        approval_manager = app.workflow.approval_manager if app.workflow else None

        if approval_manager is None:
            app.session.add_message("system", "Approval manager not initialized")
            return

        pending = approval_manager.get_pending_requests()

        if not pending:
            app.session.add_message("system", "No pending approval requests")
            return

        table = Table(title="Pending Approvals", show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Operation", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Risk", style="red")

        for request in pending:
            table.add_row(
                request.id,
                request.tool,
                request.operation,
                request.description,
                request.risk_level,
            )

        app.session.add_rich(table)


class CheckpointsCommand(Command):
    """Show checkpoints awaiting review."""

    def __init__(self) -> None:
        super().__init__(
            name="checkpoints",
            description="Show checkpoints awaiting review",
            aliases=[],
            usage="/checkpoints",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        # Access checkpoint manager from workflow
        checkpoint_manager = app.workflow.checkpoint_manager if app.workflow else None

        if checkpoint_manager is None:
            app.session.add_message("system", "Checkpoint manager not initialized")
            return

        unresolved = checkpoint_manager.get_unresolved()

        if not unresolved:
            app.session.add_message("system", "No checkpoints awaiting review")
            return

        for checkpoint in unresolved:
            content_preview = str(checkpoint.content)
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."

            panel = Panel(
                content_preview,
                title=f"Checkpoint: {checkpoint.name} ({cp_id})",
                subtitle=f"Type: {checkpoint.content_type}",
                border_style="yellow",
            )
            app.session.add_rich(panel)


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
        # Access task graph from workflow
        task_graph = app.workflow.task_graph if app.workflow else None

        if task_graph:
            task_graph.clear()
            app.session.add_success("Task plan cleared")
        else:
            app.session.add_error("Task graph not initialized")


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

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
        plan_store = app.workflow.plan_store if app.workflow else None

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
            description="List files in workspace directory",
            aliases=[],
            usage="/files [--dir=DIR]",
            examples=[
                "/files",
                "/files --dir=tasks",
            ],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        parsed = self.parse_args(args)
        subdir = parsed.get_option("dir")

        from agentic_cli.tools.glob_tool import list_dir

        workspace = app.settings.workspace_dir
        target_dir = workspace / subdir if subdir else workspace

        if not target_dir.exists():
            app.session.add_message("system", f"Directory does not exist: {target_dir}")
            return

        result = list_dir(str(target_dir))
        if not result.get("success"):
            app.session.add_error(result.get("error", "Failed to list directory"))
            return

        title = f"Files in {subdir}/" if subdir else "Workspace files"
        table = Table(title=title, show_header=True)
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


class TasksCommand(Command):
    """Show execution tasks."""

    def __init__(self) -> None:
        super().__init__(
            name="tasks",
            description="Show execution tasks and progress",
            aliases=[],
            usage="/tasks [--status=STATUS]",
            examples=[
                "/tasks",
                "/tasks --status=pending",
                "/tasks --status=in_progress",
            ],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        task_store = app.workflow.task_store if app.workflow else None

        if task_store is None:
            app.session.add_message("system", "Task store not initialized")
            return

        parsed = self.parse_args(args)
        status_filter = parsed.get_option("status")

        tasks = task_store.list_tasks(status=status_filter or None)

        if not tasks:
            app.session.add_message("system", "No tasks found")
            return

        table = Table(title="Execution Tasks", show_header=True)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Description", style="white")
        table.add_column("Status", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Tags", style="dim")

        for task in tasks:
            status_style = {
                "pending": "dim",
                "in_progress": "bold cyan",
                "completed": "green",
                "cancelled": "red",
            }.get(task.status, "white")
            tags_str = ", ".join(task.tags) if task.tags else ""
            table.add_row(
                task.id[:8],
                task.description,
                f"[{status_style}]{task.status}[/]",
                task.priority,
                tags_str,
            )

        app.session.add_rich(table)


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
        plan_store = app.workflow.plan_store if app.workflow else None

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
    TasksCommand,
    ApprovalsCommand,
    CheckpointsCommand,
    FilesCommand,
    ClearPlanCommand,
]

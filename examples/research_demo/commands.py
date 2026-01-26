"""Status commands for the Research Demo application.

Provides commands for inspecting memory, plan, files, approvals, and checkpoints.
"""

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from examples.research_demo.app import ResearchDemoApp


class MemoryCommand(Command):
    """Show current memory state."""

    def __init__(self) -> None:
        super().__init__(
            name="memory",
            description="Show working and long-term memory contents",
            aliases=[],
            usage="/memory [--type=TYPE]",
            examples=[
                "/memory",
                "/memory --type=learning",
            ],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        parsed = self.parse_args(args)
        memory_type = parsed.get_option("type")

        # Working Memory Section
        working_table = Table(title="Working Memory (Session)", show_header=True)
        working_table.add_column("Key", style="cyan")
        working_table.add_column("Value", style="white")
        working_table.add_column("Tags", style="dim")

        if app.memory_manager:
            working_mem = app.memory_manager.working
            for key in working_mem.list():
                entry = working_mem._entries.get(key)
                if entry:
                    value_str = str(entry.value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    tags_str = ", ".join(entry.tags) if entry.tags else ""
                    working_table.add_row(key, value_str, tags_str)

        if working_table.row_count == 0:
            working_table.add_row("(empty)", "", "")

        app.session.add_rich(working_table)

        # Long-term Memory Section
        longterm_table = Table(title="Long-term Memory (Persistent)", show_header=True)
        longterm_table.add_column("ID", style="dim", width=8)
        longterm_table.add_column("Type", style="yellow")
        longterm_table.add_column("Content", style="white")
        longterm_table.add_column("Tags", style="dim")

        if app.memory_manager:
            from agentic_cli.memory.longterm import MemoryType

            longterm_mem = app.memory_manager.longterm

            # Get all entries, optionally filtered by type
            if memory_type:
                try:
                    mem_type = MemoryType(memory_type)
                    entries = longterm_mem.recall("", type=mem_type)
                except ValueError:
                    app.session.add_error(f"Invalid memory type: {memory_type}")
                    return
            else:
                entries = longterm_mem.recall("")

            for entry in entries[:20]:  # Limit display
                content_str = entry.content
                if len(content_str) > 60:
                    content_str = content_str[:60] + "..."
                tags_str = ", ".join(entry.tags) if entry.tags else ""
                longterm_table.add_row(
                    entry.id[:8],
                    entry.type.value,
                    content_str,
                    tags_str,
                )

        if longterm_table.row_count == 0:
            longterm_table.add_row("(empty)", "", "", "")

        app.session.add_rich(longterm_table)


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
        if app.task_graph is None:
            app.session.add_message("system", "No task graph initialized")
            return

        # Get progress statistics
        progress = app.task_graph.get_progress()

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
        display = app.task_graph.to_display()
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
        if app.approval_manager is None:
            app.session.add_message("system", "Approval manager not initialized")
            return

        pending = app.approval_manager._pending

        if not pending:
            app.session.add_message("system", "No pending approval requests")
            return

        table = Table(title="Pending Approvals", show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Operation", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("Risk", style="red")

        for request_id, request in pending.items():
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
        if app.checkpoint_manager is None:
            app.session.add_message("system", "Checkpoint manager not initialized")
            return

        checkpoints = app.checkpoint_manager._checkpoints
        resolved = app.checkpoint_manager._results

        # Filter to unresolved checkpoints
        unresolved = {
            cp_id: cp
            for cp_id, cp in checkpoints.items()
            if cp_id not in resolved
        }

        if not unresolved:
            app.session.add_message("system", "No checkpoints awaiting review")
            return

        for cp_id, checkpoint in unresolved.items():
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

        from agentic_cli.tools.file_ops import file_manager

        workspace = app.settings.workspace_dir
        target_dir = workspace / subdir

        if not target_dir.exists():
            app.session.add_message("system", f"Directory does not exist: {target_dir}")
            return

        result = file_manager("list", str(target_dir))

        if not result["success"]:
            app.session.add_error(result.get("error", "Failed to list directory"))
            return

        table = Table(title=f"Files in {subdir}/", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Size", style="dim", justify="right")

        for name, info in result["entries"].items():
            size_str = ""
            if info["size"] is not None:
                size_str = f"{info['size']:,} bytes"
            table.add_row(name, info["type"], size_str)

        if table.row_count == 0:
            table.add_row("(empty)", "", "")

        app.session.add_rich(table)
        app.session.add_message("system", f"Total: {result['count']} items")


class ClearMemoryCommand(Command):
    """Clear working memory."""

    def __init__(self) -> None:
        super().__init__(
            name="clear-memory",
            description="Clear working memory (session context)",
            aliases=["clearmem"],
            usage="/clear-memory",
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        if app.memory_manager:
            app.memory_manager.clear_working()
            app.session.add_success("Working memory cleared")
        else:
            app.session.add_error("Memory manager not initialized")


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
        if app.task_graph:
            app.task_graph._tasks.clear()
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

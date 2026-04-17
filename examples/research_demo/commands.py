"""Status commands for the Research Demo application.

Provides commands for inspecting memory and workspace files.
"""

from typing import TYPE_CHECKING

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


class KbBackfillCommand(Command):
    """Pre-bake markdown sidecars for any KB docs that don't have one yet."""

    def __init__(self) -> None:
        super().__init__(
            name="kb-backfill",
            description="Generate missing sidecars for every KB doc (batch LLM call)",
            aliases=[],
            usage="/kb-backfill",
            examples=["/kb-backfill"],
            category=CommandCategory.GENERAL,
        )

    async def execute(self, args: str, app: "ResearchDemoApp") -> None:
        from agentic_cli.workflow.service_registry import set_service_registry

        workflow = app.workflow
        if workflow is None:
            app.session.add_error("Workflow not initialized")
            return

        project_kb = workflow.kb_manager
        user_kb = workflow.user_kb_manager

        if project_kb is None:
            app.session.add_warning(
                "No knowledge base configured. Ingest a document first, "
                "or enable KB tools in the agent config."
            )
            return

        kbs: list[tuple[str, object]] = [("project", project_kb)]
        if user_kb is not None and user_kb is not project_kb:
            kbs.append(("user", user_kb))

        # backfill_sidecars() → generate_sidecar_payload() reads LLM_SUMMARIZER
        # from the service-registry ContextVar, which is normally only set
        # during agent message processing. Push it here so the LLM is actually
        # called (otherwise the fallback fires and we write truncated content).
        registry_token = set_service_registry(workflow._services)
        try:
            total_written = 0
            for label, kb in kbs:
                app.session.add_message(
                    "system", f"Backfilling {label} KB at {kb.kb_dir}…"
                )
                try:
                    written = await kb.backfill_sidecars()
                except Exception as e:
                    app.session.add_error(f"{label} KB backfill failed: {e}")
                    continue
                total_written += written
                app.session.add_success(
                    f"{label} KB: {written} sidecar(s) generated"
                )
        finally:
            registry_token.var.reset(registry_token)

        if total_written == 0:
            app.session.add_message("system", "All documents already have sidecars.")


DEMO_COMMANDS = [
    MemoryCommand,
    FilesCommand,
    KbBackfillCommand,
]

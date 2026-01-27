"""App-specific tools for the Research Demo application.

This module only contains tools specific to the research demo:
- File operations for saving/reading findings
- Shell command execution

Memory, planning, and HITL tools are provided by the framework:
- agentic_cli.tools.memory_tools
- agentic_cli.tools.planning_tools
- agentic_cli.tools.hitl_tools
"""

from typing import Any

from agentic_cli.config import get_context_settings


# =============================================================================
# File Tools
# =============================================================================


def save_finding(filename: str, content: str) -> dict[str, Any]:
    """Save a research finding to a file in the workspace.

    Args:
        filename: Name of the file (will be saved in workspace/findings/).
        content: Content to write.

    Returns:
        A dict with the saved file path.
    """
    settings = get_context_settings()
    if settings is None:
        return {"success": False, "error": "Settings not available"}

    from agentic_cli.tools.file_ops import file_manager

    # Determine the full path
    findings_dir = settings.workspace_dir / "findings"
    file_path = findings_dir / filename

    # Create findings directory if needed
    findings_dir.mkdir(parents=True, exist_ok=True)

    # Write the file
    result = file_manager("write", str(file_path), content=content)

    if result["success"]:
        return {
            "success": True,
            "path": str(file_path),
            "size": result["size"],
            "message": f"Saved finding to {filename}",
        }

    return result


def read_finding(filename: str) -> dict[str, Any]:
    """Read a previously saved finding.

    Args:
        filename: Name of the file to read.

    Returns:
        A dict with the file content.
    """
    settings = get_context_settings()
    if settings is None:
        return {"success": False, "error": "Settings not available"}

    from agentic_cli.tools.file_ops import file_manager

    findings_dir = settings.workspace_dir / "findings"
    file_path = findings_dir / filename

    return file_manager("read", str(file_path))


def list_findings() -> dict[str, Any]:
    """List all saved findings.

    Returns:
        A dict with the list of findings files.
    """
    settings = get_context_settings()
    if settings is None:
        return {"success": False, "error": "Settings not available"}

    from agentic_cli.tools.file_ops import file_manager

    findings_dir = settings.workspace_dir / "findings"

    if not findings_dir.exists():
        return {
            "success": True,
            "path": str(findings_dir),
            "entries": {},
            "count": 0,
        }

    return file_manager("list", str(findings_dir))


def compare_versions(file_a: str, file_b: str) -> dict[str, Any]:
    """Compare two versions of a document.

    Args:
        file_a: First file path or content.
        file_b: Second file path or content.

    Returns:
        A dict with the diff and similarity score.
    """
    from agentic_cli.tools.file_ops import diff_compare

    return diff_compare(file_a, file_b, mode="unified")


# =============================================================================
# Shell Tool
# =============================================================================


def run_safe_command(command: str) -> dict[str, Any]:
    """Run a safe shell command.

    Only allows safe read-only commands. Dangerous commands are blocked.

    Args:
        command: The shell command to run.

    Returns:
        A dict with command output or error message.
    """
    from agentic_cli.tools.shell import shell_executor

    return shell_executor(command, timeout=30)

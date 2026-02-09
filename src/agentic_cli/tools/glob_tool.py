"""File pattern matching tool (glob).

Provides file discovery using glob patterns:
- glob: Find files matching a pattern (also serves as directory listing)

This is a read-only tool with PermissionLevel.SAFE.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    register_tool,
)


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="Find files by name pattern (e.g. '**/*.py'). Use this to discover files in a directory tree. For searching inside file contents, use grep instead.",
)
def glob(
    pattern: str = "*",
    path: str = ".",
    include_hidden: bool = False,
    include_dirs: bool = True,
    max_results: int = 1000,
    sort_by: Literal["name", "size", "modified"] = "name",
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Find files matching a glob pattern.

    Use this tool to discover files by name/extension. Supports recursive
    search with '**'. Use pattern='*' for a directory listing.
    For searching inside file contents, use grep instead.
    For detailed directory listing with type separation, use list_dir.

    Args:
        pattern: Glob pattern to match (default "*" lists all files).
            E.g. "*" (all files), "*.py" (Python files),
            "**/*.py" (recursive), "src/**/*.{js,ts}" (multiple extensions).
        path: Directory to search in (default "." = current directory).
        include_hidden: Include hidden files (starting with .) (default False).
        include_dirs: Include directories in results (default True).
        max_results: Maximum number of results to return (default 1000).
        sort_by: Sort results by "name", "size", or "modified" (default "name").
        include_metadata: Include file size and modified time (default False).

    Returns:
        dict with:
        - success: True if search completed
        - files: List of matching files/directories
        - count: Number of matches found
        - truncated: True if results were truncated
        - path: Resolved search path
    """
    search_path = Path(path).resolve()

    if not search_path.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}",
            "path": str(search_path),
        }

    if not search_path.is_dir():
        return {
            "success": False,
            "error": f"Not a directory: {path}",
            "path": str(search_path),
        }

    # Determine if recursive
    is_recursive = "**" in pattern

    # Find matching files
    if is_recursive:
        matches = list(search_path.glob(pattern))
    else:
        matches = list(search_path.glob(pattern))

    # Filter results
    filtered = []
    for match in matches:
        # Skip hidden files if not requested
        if not include_hidden and match.name.startswith("."):
            continue

        # Skip directories if not requested
        if not include_dirs and match.is_dir():
            continue

        filtered.append(match)

    # Sort results
    if sort_by == "name":
        filtered.sort(key=lambda p: p.name.lower())
    elif sort_by == "size":
        filtered.sort(key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)
    elif sort_by == "modified":
        filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Truncate if needed
    truncated = len(filtered) > max_results
    filtered = filtered[:max_results]

    # Format output
    if include_metadata:
        files = []
        for p in filtered:
            try:
                stat = p.stat()
                entry = {
                    "path": str(p.relative_to(search_path)),
                    "type": "directory" if p.is_dir() else "file",
                    "size": stat.st_size if p.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                files.append(entry)
            except (OSError, PermissionError):
                # Skip files we can't stat
                files.append({
                    "path": str(p.relative_to(search_path)),
                    "type": "directory" if p.is_dir() else "file",
                    "size": None,
                    "modified": None,
                })
    else:
        files = [str(p.relative_to(search_path)) for p in filtered]

    return {
        "success": True,
        "files": files,
        "count": len(files),
        "truncated": truncated,
        "path": str(search_path),
    }


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="List directory contents organized by type (directories first, then files). Use this when you need a structured overview of a directory; for pattern-based file search use glob.",
)
def list_dir(
    path: str = ".",
    include_hidden: bool = False,
    sort_by: Literal["name", "size", "modified", "type"] = "name",
) -> dict[str, Any]:
    """List directory contents with detailed metadata.

    Returns directories and files separately with size and modification time.
    Use this for understanding project structure. For finding files by
    pattern across subdirectories, use glob with '**' instead.

    Args:
        path: Directory to list (default "." = current directory).
        include_hidden: Include hidden files (default False).
        sort_by: Sort by "name", "size", "modified", or "type" (default "name").

    Returns:
        dict with:
        - success: True
        - path: Resolved directory path
        - directories: List of directory entries
        - files: List of file entries
        - total: Total count of entries
    """
    # Delegate to glob for the actual file listing
    result = glob(
        pattern="*",
        path=path,
        include_hidden=include_hidden,
        include_dirs=True,
        include_metadata=True,
        sort_by=sort_by if sort_by in ("name", "size", "modified") else "name",
    )

    if not result.get("success"):
        return result

    # Separate directories and files, reformat entries
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    for entry in result.get("files", []):
        item = {
            "name": entry["path"],
            "size": entry.get("size"),
            "modified": entry.get("modified"),
        }
        if entry.get("type") == "directory":
            directories.append(item)
        else:
            files.append(item)

    return {
        "success": True,
        "path": result["path"],
        "directories": directories,
        "files": files,
        "total": len(directories) + len(files),
    }

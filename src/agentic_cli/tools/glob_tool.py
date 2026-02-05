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
    ToolError,
    ErrorCode,
    register_tool,
)


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="Find files matching a glob pattern",
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

    Supports standard glob patterns and can also serve as a directory
    listing when pattern="*".

    Args:
        pattern: Glob pattern to match (default "*" lists all files).
            Common patterns:
            - "*": All files in directory
            - "*.py": All Python files
            - "**/*.py": All Python files recursively
            - "src/**/*.{js,ts}": JS/TS files under src/
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

    File entry format (when include_metadata=False):
        str (file path relative to search path)

    File entry format (when include_metadata=True):
        {
            "path": str,
            "type": "file" | "directory",
            "size": int | None,  # bytes, None for directories
            "modified": str,     # ISO format timestamp
        }

    Examples:
        glob()                          # List current directory
        glob("*.py")                    # Find Python files
        glob("**/*.test.ts", "src")     # Find test files under src/
        glob("*", include_metadata=True) # List with file details
    """
    search_path = Path(path).resolve()

    if not search_path.exists():
        raise ToolError(
            message=f"Path not found: {path}",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=False,
            details={"path": str(search_path)},
        )

    if not search_path.is_dir():
        raise ToolError(
            message=f"Not a directory: {path}",
            error_code=ErrorCode.INVALID_INPUT,
            recoverable=False,
            details={"path": str(search_path)},
        )

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
    description="List directory contents with metadata",
)
def list_dir(
    path: str = ".",
    include_hidden: bool = False,
    sort_by: Literal["name", "size", "modified", "type"] = "name",
) -> dict[str, Any]:
    """List directory contents with detailed metadata.

    A convenience wrapper around glob() that always returns metadata
    and organizes output by file type.

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

    Each entry contains:
        {
            "name": str,
            "size": int | None,
            "modified": str,
        }
    """
    search_path = Path(path).resolve()

    if not search_path.exists():
        raise ToolError(
            message=f"Path not found: {path}",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=False,
            details={"path": str(search_path)},
        )

    if not search_path.is_dir():
        raise ToolError(
            message=f"Not a directory: {path}",
            error_code=ErrorCode.INVALID_INPUT,
            recoverable=False,
            details={"path": str(search_path)},
        )

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    try:
        entries = list(search_path.iterdir())
    except PermissionError:
        raise ToolError(
            message=f"Permission denied: {path}",
            error_code=ErrorCode.PERMISSION_DENIED,
            recoverable=False,
            details={"path": str(search_path)},
        )

    for entry in entries:
        # Skip hidden files if not requested
        if not include_hidden and entry.name.startswith("."):
            continue

        try:
            stat = entry.stat()
            item = {
                "name": entry.name,
                "size": stat.st_size if entry.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except (OSError, PermissionError):
            item = {
                "name": entry.name,
                "size": None,
                "modified": None,
            }

        if entry.is_dir():
            directories.append(item)
        else:
            files.append(item)

    # Sort based on sort_by
    def sort_key(item: dict[str, Any]) -> Any:
        if sort_by == "name":
            return item["name"].lower()
        elif sort_by == "size":
            return item.get("size") or 0
        elif sort_by == "modified":
            return item.get("modified") or ""
        return item["name"].lower()

    reverse = sort_by in ("size", "modified")

    if sort_by != "type":
        directories.sort(key=sort_key, reverse=reverse)
        files.sort(key=sort_key, reverse=reverse)

    return {
        "success": True,
        "path": str(search_path),
        "directories": directories,
        "files": files,
        "total": len(directories) + len(files),
    }

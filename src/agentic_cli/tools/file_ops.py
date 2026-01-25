"""File operations tools for agentic CLI applications.

Provides file_manager for common file operations and diff_compare for text comparison.
"""

import difflib
import os
import shutil
from pathlib import Path
from typing import Any


def file_manager(
    operation: str,
    path: str,
    content: str | None = None,
    destination: str | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    """Perform file operations.

    Args:
        operation: Operation to perform. One of:
            - "read": Read file contents
            - "write": Write content to file
            - "list": List directory contents
            - "delete": Delete file or directory
            - "move": Move file or directory
            - "copy": Copy file or directory
        path: Path to the file or directory to operate on.
        content: Content to write (required for "write" operation).
        destination: Destination path (required for "move" and "copy" operations).
        recursive: Whether to operate recursively on directories
            (for "delete" and "copy" of directories).

    Returns:
        dict with operation results:
        - read: {"success": True, "content": str, "size": int, "path": str}
        - write: {"success": True, "path": str, "size": int}
        - list: {"success": True, "path": str, "entries": {...}, "count": int}
        - delete: {"success": True, "deleted": str}
        - move: {"success": True, "source": str, "destination": str}
        - copy: {"success": True, "source": str, "destination": str}
        - error: {"success": False, "error": str}
    """
    try:
        if operation == "read":
            return _read_file(path)
        elif operation == "write":
            return _write_file(path, content)
        elif operation == "list":
            return _list_directory(path)
        elif operation == "delete":
            return _delete_path(path, recursive)
        elif operation == "move":
            return _move_path(path, destination)
        elif operation == "copy":
            return _copy_path(path, destination, recursive)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _read_file(path: str) -> dict[str, Any]:
    """Read file contents."""
    file_path = Path(path)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {path}"}
    if not file_path.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    content = file_path.read_text()
    size = file_path.stat().st_size
    return {
        "success": True,
        "content": content,
        "size": size,
        "path": str(file_path),
    }


def _write_file(path: str, content: str | None) -> dict[str, Any]:
    """Write content to file."""
    if content is None:
        return {"success": False, "error": "Content is required for write operation"}

    file_path = Path(path)
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    size = file_path.stat().st_size
    return {
        "success": True,
        "path": str(file_path),
        "size": size,
    }


def _list_directory(path: str) -> dict[str, Any]:
    """List directory contents."""
    dir_path = Path(path)
    if not dir_path.exists():
        return {"success": False, "error": f"Directory not found: {path}"}
    if not dir_path.is_dir():
        return {"success": False, "error": f"Not a directory: {path}"}

    entries: dict[str, dict[str, Any]] = {}
    for entry in dir_path.iterdir():
        if entry.is_file():
            entries[entry.name] = {
                "type": "file",
                "size": entry.stat().st_size,
            }
        elif entry.is_dir():
            entries[entry.name] = {
                "type": "directory",
                "size": None,
            }
        else:
            entries[entry.name] = {
                "type": "other",
                "size": None,
            }

    return {
        "success": True,
        "path": str(dir_path),
        "entries": entries,
        "count": len(entries),
    }


def _delete_path(path: str, recursive: bool) -> dict[str, Any]:
    """Delete file or directory."""
    target_path = Path(path)
    if not target_path.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        if recursive:
            shutil.rmtree(target_path)
        else:
            # Try to remove empty directory
            try:
                target_path.rmdir()
            except OSError as e:
                return {
                    "success": False,
                    "error": f"Directory not empty. Use recursive=True to delete: {e}",
                }
    else:
        target_path.unlink()

    return {
        "success": True,
        "deleted": str(target_path),
    }


def _move_path(path: str, destination: str | None) -> dict[str, Any]:
    """Move file or directory."""
    if destination is None:
        return {"success": False, "error": "Destination is required for move operation"}

    source_path = Path(path)
    dest_path = Path(destination)

    if not source_path.exists():
        return {"success": False, "error": f"Source not found: {path}"}

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(dest_path))

    return {
        "success": True,
        "source": str(source_path),
        "destination": str(dest_path),
    }


def _copy_path(path: str, destination: str | None, recursive: bool) -> dict[str, Any]:
    """Copy file or directory."""
    if destination is None:
        return {"success": False, "error": "Destination is required for copy operation"}

    source_path = Path(path)
    dest_path = Path(destination)

    if not source_path.exists():
        return {"success": False, "error": f"Source not found: {path}"}

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        shutil.copy2(str(source_path), str(dest_path))
    elif source_path.is_dir():
        if recursive:
            shutil.copytree(str(source_path), str(dest_path))
        else:
            return {
                "success": False,
                "error": "Use recursive=True to copy directories",
            }
    else:
        shutil.copy2(str(source_path), str(dest_path))

    return {
        "success": True,
        "source": str(source_path),
        "destination": str(dest_path),
    }


def diff_compare(
    source_a: str,
    source_b: str,
    mode: str = "unified",
    context_lines: int = 3,
) -> dict[str, Any]:
    """Compare two text sources and return diff information.

    Args:
        source_a: First text or file path to compare.
        source_b: Second text or file path to compare.
        mode: Diff output mode. One of:
            - "unified": Standard unified diff format (default)
            - "side_by_side": Side-by-side comparison
            - "summary": Only summary statistics
        context_lines: Number of context lines around changes (default 3).

    Returns:
        dict with comparison results:
        {
            "success": True,
            "diff": str,  # formatted diff output
            "summary": {"added": int, "removed": int, "changed": int},
            "similarity": float,  # 0-1 using SequenceMatcher.ratio()
        }
    """
    try:
        # Get content from sources (file paths or raw text)
        content_a = _get_content(source_a)
        content_b = _get_content(source_b)

        # Split into lines for comparison
        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)

        # Handle empty strings - ensure we have at least empty list
        if not lines_a and content_a == "":
            lines_a = []
        if not lines_b and content_b == "":
            lines_b = []

        # Calculate similarity
        matcher = difflib.SequenceMatcher(None, content_a, content_b)
        similarity = matcher.ratio()

        # Generate diff based on mode
        if mode == "unified":
            diff_output = _unified_diff(lines_a, lines_b, context_lines)
        elif mode == "side_by_side":
            diff_output = _side_by_side_diff(lines_a, lines_b)
        elif mode == "summary":
            diff_output = ""  # Summary mode focuses on statistics
        else:
            diff_output = _unified_diff(lines_a, lines_b, context_lines)

        # Calculate summary statistics
        summary = _calculate_summary(lines_a, lines_b)

        return {
            "success": True,
            "diff": diff_output,
            "summary": summary,
            "similarity": similarity,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_content(source: str) -> str:
    """Get content from a source (file path or raw text)."""
    # Check if source is a file path
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text()
    # Otherwise treat as raw text
    return source


def _unified_diff(lines_a: list[str], lines_b: list[str], context_lines: int) -> str:
    """Generate unified diff output."""
    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile="a",
        tofile="b",
        n=context_lines,
    )
    return "".join(diff)


def _side_by_side_diff(lines_a: list[str], lines_b: list[str]) -> str:
    """Generate side-by-side diff output."""
    # Use ndiff for detailed comparison, then format
    diff = list(difflib.ndiff(lines_a, lines_b))

    output_lines = []
    for line in diff:
        if line.startswith("- "):
            output_lines.append(f"< {line[2:]}")
        elif line.startswith("+ "):
            output_lines.append(f"> {line[2:]}")
        elif line.startswith("? "):
            # Skip hint lines
            continue
        else:
            output_lines.append(f"  {line[2:]}")

    return "".join(output_lines)


def _calculate_summary(lines_a: list[str], lines_b: list[str]) -> dict[str, int]:
    """Calculate diff summary statistics."""
    # Use SequenceMatcher to get opcodes
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

    added = 0
    removed = 0
    changed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            # Count the larger of the two as changes
            # and the difference as added or removed
            old_count = i2 - i1
            new_count = j2 - j1
            changed += min(old_count, new_count)
            if new_count > old_count:
                added += new_count - old_count
            elif old_count > new_count:
                removed += old_count - new_count

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }

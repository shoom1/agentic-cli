"""Read-only file operation tools.

Provides safe, read-only tools for file system access:
- read_file: Read file contents with optional offset/limit
- diff_compare: Compare two text sources

All tools in this module have PermissionLevel.SAFE.
"""

import difflib
from pathlib import Path
from typing import Any

from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    register_tool,
)


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="Read the contents of a file at the given path. Use this to examine source code, config files, or any text file. For finding files by name pattern use glob instead; for searching file contents use grep.",
)
def read_file(
    path: str,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Read file contents with optional offset and line limit.

    Use this tool when you know the exact file path and want to see its contents.
    For large files, use offset/limit to read specific sections.
    Prefer glob to find files by name, and grep to search contents.

    Args:
        path: Path to the file to read.
        offset: Line number to start reading from (0-indexed, default 0).
        limit: Maximum number of lines to read (default None = all lines).

    Returns:
        dict with:
        - success: True if file was read successfully
        - content: File contents (or lines if offset/limit specified)
        - path: Resolved file path
        - size: File size in bytes
        - lines_read: Number of lines returned (if offset/limit used)
        - total_lines: Total lines in file (if offset/limit used)
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {path}",
            "path": str(file_path),
        }

    if not file_path.is_file():
        return {
            "success": False,
            "error": f"Not a file: {path}",
            "path": str(file_path),
        }

    try:
        content = file_path.read_text()
        size = file_path.stat().st_size

        # If offset or limit specified, work with lines
        if offset > 0 or limit is not None:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)

            # Apply offset and limit
            end_idx = len(lines) if limit is None else offset + limit
            selected_lines = lines[offset:end_idx]
            content = "".join(selected_lines)

            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": size,
                "lines_read": len(selected_lines),
                "total_lines": total_lines,
                "offset": offset,
            }

        return {
            "success": True,
            "content": content,
            "path": str(file_path),
            "size": size,
        }
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"Cannot read file as text (binary file?): {path}",
            "path": str(file_path),
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {path}",
            "path": str(file_path),
        }


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="Compare two text sources (files or strings) and show differences. Use this to see what changed between two versions of content.",
)
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
        - success: True
        - diff: Formatted diff output
        - summary: {"added": int, "removed": int, "changed": int}
        - similarity: 0-1 ratio using SequenceMatcher
    """
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

"""Write/modify file operation tools.

Provides tools for file system modifications:
- write_file: Write content to a file (creates or overwrites)
- edit_file: Replace text in a file (sed-like operation)

All tools in this module have PermissionLevel.CAUTION.
Delete, move, and copy operations should use the shell tool.
"""

import re
from pathlib import Path
from typing import Any

from agentic_cli.persistence._utils import atomic_write_text
from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    register_tool,
)


@register_tool(
    category=ToolCategory.WRITE,
    permission_level=PermissionLevel.CAUTION,
    description="Write content to a file (creates or overwrites). Use this to create new files or replace entire file contents. For partial modifications use edit_file instead.",
)
def write_file(
    path: str,
    content: str,
    create_dirs: bool = True,
) -> dict[str, Any]:
    """Write content to a file, creating it if it doesn't exist.

    Use this when you need to create a new file or completely replace
    an existing file's contents. For targeted text replacements within
    an existing file, use edit_file instead.

    Args:
        path: Path to the file to write.
        content: Content to write to the file.
        create_dirs: Whether to create parent directories if they don't exist (default True).

    Returns:
        dict with:
        - success: True if file was written successfully
        - path: Resolved file path
        - size: File size in bytes after write
        - created: True if file was newly created, False if overwritten
    """
    file_path = Path(path).resolve()
    existed = file_path.exists()

    # Create parent directories if requested
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    elif not file_path.parent.exists():
        return {
            "success": False,
            "error": f"Parent directory does not exist: {file_path.parent}",
            "path": str(file_path),
        }

    try:
        atomic_write_text(file_path, content)
        size = file_path.stat().st_size

        return {
            "success": True,
            "path": str(file_path),
            "size": size,
            "created": not existed,
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {path}",
            "path": str(file_path),
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"Failed to write file: {e}",
            "path": str(file_path),
        }


@register_tool(
    category=ToolCategory.WRITE,
    permission_level=PermissionLevel.CAUTION,
    description="Replace specific text in an existing file. Use this for targeted edits (find-and-replace). For creating or fully rewriting files use write_file instead.",
)
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Replace text in a file (sed-like find-and-replace).

    Use this for precise, targeted edits within existing files. Replaces
    the first occurrence by default; use replace_all=True for global
    replacement. Supports regex patterns via use_regex=True.

    For creating new files or full rewrites, use write_file instead.

    Args:
        path: Path to the file to edit.
        old_text: Text to find and replace (or regex pattern if use_regex=True).
        new_text: Replacement text.
        replace_all: Replace all occurrences (default False = first only).
        use_regex: Treat old_text as a regex pattern (default False).

    Returns:
        dict with:
        - success: True if edit was successful
        - path: Resolved file path
        - replacements: Number of replacements made
        - size: File size in bytes after edit
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
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"Cannot read file as text (binary file?): {path}",
            "path": str(file_path),
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied reading file: {path}",
            "path": str(file_path),
        }

    # Perform replacement
    if use_regex:
        try:
            pattern = re.compile(old_text)
        except re.error as e:
            return {
                "success": False,
                "error": f"Invalid regex pattern: {e}",
            }

        if replace_all:
            new_content, count = pattern.subn(new_text, content)
        else:
            new_content, count = pattern.subn(new_text, content, count=1)
    else:
        # Plain text replacement
        if replace_all:
            count = content.count(old_text)
            new_content = content.replace(old_text, new_text)
        else:
            count = 1 if old_text in content else 0
            new_content = content.replace(old_text, new_text, 1)

    if count == 0:
        return {
            "success": False,
            "error": f"Text not found in file: {old_text[:50]}{'...' if len(old_text) > 50 else ''}",
            "path": str(file_path),
        }

    # Write the modified content
    try:
        atomic_write_text(file_path, new_content)
        size = file_path.stat().st_size

        return {
            "success": True,
            "path": str(file_path),
            "replacements": count,
            "size": size,
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied writing file: {path}",
            "path": str(file_path),
        }

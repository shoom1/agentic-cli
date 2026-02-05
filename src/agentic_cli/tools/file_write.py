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

from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    ToolError,
    ErrorCode,
    register_tool,
)


@register_tool(
    category=ToolCategory.WRITE,
    permission_level=PermissionLevel.CAUTION,
    description="Write content to a file",
)
def write_file(
    path: str,
    content: str,
    create_dirs: bool = True,
) -> dict[str, Any]:
    """Write content to a file.

    Creates the file if it doesn't exist, overwrites if it does.
    Optionally creates parent directories.

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

    Raises:
        ToolError: If write fails (permission denied, etc.).
    """
    file_path = Path(path).resolve()
    existed = file_path.exists()

    # Create parent directories if requested
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    elif not file_path.parent.exists():
        raise ToolError(
            message=f"Parent directory does not exist: {file_path.parent}",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=True,
            details={"path": str(file_path), "parent": str(file_path.parent)},
        )

    try:
        file_path.write_text(content)
        size = file_path.stat().st_size

        return {
            "success": True,
            "path": str(file_path),
            "size": size,
            "created": not existed,
        }
    except PermissionError as e:
        raise ToolError(
            message=f"Permission denied: {path}",
            error_code=ErrorCode.PERMISSION_DENIED,
            recoverable=False,
            details={"path": str(file_path), "error": str(e)},
        )
    except OSError as e:
        raise ToolError(
            message=f"Failed to write file: {e}",
            error_code=ErrorCode.INTERNAL_ERROR,
            recoverable=False,
            details={"path": str(file_path), "error": str(e)},
        )


@register_tool(
    category=ToolCategory.WRITE,
    permission_level=PermissionLevel.CAUTION,
    description="Replace text in a file (sed-like operation)",
)
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Replace text in a file.

    Performs sed-like text replacement in a file. By default replaces
    only the first occurrence; use replace_all=True for global replacement.

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

    Raises:
        ToolError: If file not found, text not found, or edit fails.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        raise ToolError(
            message=f"File not found: {path}",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=False,
            details={"path": str(file_path)},
        )

    if not file_path.is_file():
        raise ToolError(
            message=f"Not a file: {path}",
            error_code=ErrorCode.INVALID_INPUT,
            recoverable=False,
            details={"path": str(file_path)},
        )

    try:
        content = file_path.read_text()
    except UnicodeDecodeError as e:
        raise ToolError(
            message=f"Cannot read file as text (binary file?): {path}",
            error_code=ErrorCode.INVALID_INPUT,
            recoverable=False,
            details={"path": str(file_path), "error": str(e)},
        )
    except PermissionError as e:
        raise ToolError(
            message=f"Permission denied reading file: {path}",
            error_code=ErrorCode.PERMISSION_DENIED,
            recoverable=False,
            details={"path": str(file_path), "error": str(e)},
        )

    # Perform replacement
    if use_regex:
        try:
            pattern = re.compile(old_text)
        except re.error as e:
            raise ToolError(
                message=f"Invalid regex pattern: {e}",
                error_code=ErrorCode.INVALID_INPUT,
                recoverable=True,
                details={"pattern": old_text, "error": str(e)},
            )

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
        raise ToolError(
            message=f"Text not found in file: {old_text[:50]}{'...' if len(old_text) > 50 else ''}",
            error_code=ErrorCode.NOT_FOUND,
            recoverable=True,
            details={
                "path": str(file_path),
                "search_text": old_text[:100],
                "use_regex": use_regex,
            },
        )

    # Write the modified content
    try:
        file_path.write_text(new_content)
        size = file_path.stat().st_size

        return {
            "success": True,
            "path": str(file_path),
            "replacements": count,
            "size": size,
        }
    except PermissionError as e:
        raise ToolError(
            message=f"Permission denied writing file: {path}",
            error_code=ErrorCode.PERMISSION_DENIED,
            recoverable=False,
            details={"path": str(file_path), "error": str(e)},
        )

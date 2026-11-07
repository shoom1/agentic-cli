"""Write/modify file operation tools.

Provides tools for file system modifications:
- write_file: Write content to a file (creates or overwrites)
- edit_file: Replace text in a file (sed-like operation)

The framework ships no delete, move or copy tool.
"""

import contextlib
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agentic_cli.file_utils import atomic_write_text
from agentic_cli.paths import resolve_path
from agentic_cli.tools.registry import (
    ToolCategory,
    register_tool,
)
from agentic_cli.workflow.permissions import Capability


class _FileLock:
    """One file's lock, and how many calls hold it or wait for it: the entry
    is dropped when none does."""

    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


#: The files a call is writing, by resolved path.
_FILE_LOCKS: dict[Path, _FileLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _one_writer(file_path: Path) -> Iterator[None]:
    """Hold ``file_path``'s lock for one call.

    One model step's calls run together in worker threads (both backends),
    and ``edit_file`` reads a file, then writes it back: two calls on one
    file would each write over the other's change.
    """
    with _FILE_LOCKS_GUARD:
        entry = _FILE_LOCKS.setdefault(file_path, _FileLock())
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _FILE_LOCKS_GUARD:
            entry.users -= 1
            if not entry.users:
                del _FILE_LOCKS[file_path]


@register_tool(
    category=ToolCategory.WRITE,
    capabilities=[Capability("filesystem.write", target_arg="path")],
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
    file_path = resolve_path(path)
    with _one_writer(file_path):
        return _write(file_path, path, content, create_dirs)


def _write(file_path: Path, path: str, content: str, create_dirs: bool) -> dict[str, Any]:
    """``write_file``'s work, holding the file's lock."""
    existed = file_path.exists()

    if not create_dirs and not file_path.parent.exists():
        return {
            "success": False,
            "error": f"Parent directory does not exist: {file_path.parent}",
            "path": str(file_path),
        }

    try:
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(file_path, content, preserve_mode=True)
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
    capabilities=[Capability("filesystem.read", target_arg="path"), Capability("filesystem.write", target_arg="path")],
    description="Replace specific text in an existing file. old_text must appear exactly once unless replace_all is set. Use this for targeted edits (find-and-replace). For creating or fully rewriting files use write_file instead.",
)
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Replace text in a file (sed-like find-and-replace).

    Use this for precise, targeted edits within existing files. old_text
    must appear exactly once: include enough of the text around it to make
    it unique, or pass replace_all=True to replace every occurrence. Supports
    regex patterns via use_regex=True (the same rule applies to its matches).

    The file must be UTF-8 text. It is written back as UTF-8 with its own
    line endings: a file whose lines end in CRLF keeps them.

    For creating new files or full rewrites, use write_file instead.

    Args:
        path: Path to the file to edit.
        old_text: Text to find and replace (or regex pattern if use_regex=True).
        new_text: Replacement text.
        replace_all: Replace every occurrence (default False: old_text must
            appear exactly once).
        use_regex: Treat old_text as a regex pattern (default False).

    Returns:
        dict with:
        - success: True if edit was successful
        - path: Resolved file path
        - replacements: Number of replacements made
        - size: File size in bytes after edit
    """
    file_path = resolve_path(path)
    with _one_writer(file_path):
        return _edit(file_path, path, old_text, new_text, replace_all, use_regex)


def _edit(
    file_path: Path,
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
    use_regex: bool,
) -> dict[str, Any]:
    """``edit_file``'s work, holding the file's lock from its read to its write."""
    # An empty old_text matches between every two characters.
    if not old_text:
        return {
            "success": False,
            "error": "old_text is empty: give the text to replace",
            "path": str(file_path),
        }

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

    # UTF-8 whatever the locale: read with Windows' cp1252, a UTF-8 file came
    # back double-encoded. Bytes, not text mode, so no line ending is changed.
    try:
        content = file_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"Not a UTF-8 text file (binary, or another encoding): {path}",
            "path": str(file_path),
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied reading file: {path}",
            "path": str(file_path),
        }

    # A file whose every line ends in CRLF is matched with LF endings, as the
    # model writes text, and written back with CRLF. Any other file (LF, or
    # mixed) is edited as it is, so no line outside the edit changes.
    crlf = "\r\n" in content and content.count("\r\n") == content.count("\n")
    if crlf:
        content = content.replace("\r\n", "\n")
        old_text = old_text.replace("\r\n", "\n")
        new_text = new_text.replace("\r\n", "\n")

    if use_regex:
        try:
            pattern = re.compile(old_text)
        except re.error as e:
            return {
                "success": False,
                "error": f"Invalid regex pattern: {e}",
            }
        count = sum(1 for _ in pattern.finditer(content))
    else:
        count = content.count(old_text)

    if count == 0:
        return {
            "success": False,
            "error": f"Text not found in file: {old_text[:50]}{'...' if len(old_text) > 50 else ''}",
            "path": str(file_path),
        }

    # Replacing the first of several matches edited a place the model may not
    # have meant, and reported success.
    if count > 1 and not replace_all:
        return {
            "success": False,
            "error": (
                f"old_text appears {count} times in the file: include the text around "
                f"the one to change so it appears once, or pass replace_all=True to "
                f"change all {count}"
            ),
            "path": str(file_path),
        }

    if use_regex:
        try:
            new_content = pattern.sub(new_text, content)
        except re.error as e:
            return {
                "success": False,
                "error": f"Invalid regex replacement text: {e}",
                "path": str(file_path),
            }
    else:
        new_content = content.replace(old_text, new_text)
    if crlf:
        new_content = new_content.replace("\n", "\r\n")

    # Write the modified content
    try:
        atomic_write_text(file_path, new_content, preserve_mode=True)
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

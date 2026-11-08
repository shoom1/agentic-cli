"""Whether a file tool reads what the framework saved from outside: a fetched
page (``./.<app>/fetched/``) or a tool's full output (``./.<app>/tool-output/``).

Either can be read back in a later turn, or a later conversation, so reading
one brings outside content as the fetch did.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agentic_cli.config import get_settings
from agentic_cli.paths import resolve_path
from agentic_cli.workflow.permissions.matchers import _fold

_FOLDERS = ("fetched", "tool-output")


def _caseless(path: Path) -> Path:
    """``path`` as the permission engine compares a filesystem deny (NFC, then
    casefolded): on a case-insensitive filesystem every spelling of a folder is
    the folder, and the engine keeps the spelling a tool was given."""
    return Path(_fold(str(path)))


def reads_saved_output(*names: str, searching: bool = False) -> Callable[[dict], bool]:
    """A flag for ``register_tool(outside_content=...)``: true when one of the
    path arguments ``names`` resolves inside a saved-output folder.

    Args:
        *names: The path arguments to check.
        searching: The tool searches below its path (``grep``), so a path
            above a saved folder reaches it too, unless a hidden entry lies
            between them: such a tool searches no hidden entry below its path.
            ``./.<app>`` holds the saved folders, so ``.`` does not reach them
            and ``./.<app>`` does.
    """
    def check(args: dict) -> bool:
        base = Path.cwd() / f".{get_settings().app_name}"
        roots = [_caseless((base / folder).resolve()) for folder in _FOLDERS]
        for name in names:
            value = args.get(name)
            if not isinstance(value, str) or not value:
                continue
            try:
                target = _caseless(resolve_path(value))
            except ValueError:
                continue
            for root in roots:
                if target == root or target.is_relative_to(root):
                    return True
                if searching and root.is_relative_to(target):
                    between = root.relative_to(target).parts
                    if not any(part.startswith(".") for part in between):
                        return True
        return False

    return check

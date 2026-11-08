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

_FOLDERS = ("fetched", "tool-output")


def reads_saved_output(*names: str) -> Callable[[dict], bool]:
    """A flag for ``register_tool(outside_content=...)``: true when one of the
    path arguments ``names`` resolves inside a saved-output folder."""
    def check(args: dict) -> bool:
        base = Path.cwd() / f".{get_settings().app_name}"
        roots = [(base / folder).resolve() for folder in _FOLDERS]
        for name in names:
            value = args.get(name)
            if not isinstance(value, str) or not value:
                continue
            try:
                target = resolve_path(value)
            except ValueError:
                continue
            if any(target == root or target.is_relative_to(root) for root in roots):
                return True
        return False

    return check

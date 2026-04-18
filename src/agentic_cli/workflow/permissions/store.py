# src/agentic_cli/workflow/permissions/store.py
"""Context, JSON persistence, and builtin rules for the permission engine.

Only PermissionContext is implemented at this point — load/save helpers
and BUILTIN_RULES land in Tasks 10–12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PermissionContext:
    """Static per-run context exposed to matchers and substitution.

    Attributes:
        workdir: Absolute current working directory.
        home: Absolute home directory.
    """

    workdir: Path
    home: Path

    def substitute(self, s: str) -> str:
        """Expand ${workdir} and ${home} in a pattern string."""
        return (
            s.replace("${workdir}", str(self.workdir))
             .replace("${home}", str(self.home))
        )

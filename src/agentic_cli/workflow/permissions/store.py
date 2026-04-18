# src/agentic_cli/workflow/permissions/store.py
"""Context, JSON persistence, and builtin rules for the permission engine.

Only PermissionContext is implemented at this point — load/save helpers
and BUILTIN_RULES land in Tasks 10–12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource


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


BUILTIN_RULES: list[Rule] = [
    # Routine reads inside workdir: allowed without prompt.
    Rule("filesystem.read", "${workdir}/**", Effect.ALLOW, RuleSource.BUILTIN),

    # System locations — writes always denied.
    Rule("filesystem.write", "/etc/**",    Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "/usr/**",    Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "/bin/**",    Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "/sbin/**",   Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "/boot/**",   Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "/System/**", Effect.DENY, RuleSource.BUILTIN),  # macOS

    # Credential directories.
    Rule("filesystem.write", "${home}/.ssh/**",   Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "${home}/.aws/**",   Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "${home}/.gnupg/**", Effect.DENY, RuleSource.BUILTIN),
]

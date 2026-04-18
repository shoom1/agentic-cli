# src/agentic_cli/workflow/permissions/store.py
"""Context, JSON persistence, and builtin rules for the permission engine.

Only PermissionContext is implemented at this point — load/save helpers
and BUILTIN_RULES land in Tasks 10–12.
"""

from __future__ import annotations

import json
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


def load_rules(path: Path, source: RuleSource, ctx: PermissionContext) -> list[Rule]:
    """Load rules from a settings.json file's ``permissions`` section.

    Returns an empty list when the file is absent or has no ``permissions``
    key. Raises ``ValueError`` if the file is not valid JSON.
    """
    # Local import to avoid circular dependency: matchers.py imports PermissionContext from here.
    from agentic_cli.workflow.permissions.matchers import get_matcher  # noqa: PLC0415

    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    section = data.get("permissions") or {}
    rules: list[Rule] = []
    for effect_name, effect in (("allow", Effect.ALLOW), ("deny", Effect.DENY)):
        for entry in section.get(effect_name) or []:
            cap = entry["capability"]
            target_raw = entry["target"]
            target = get_matcher(cap).canonicalize(target_raw, ctx)
            rules.append(Rule(cap, target, effect, source))
    return rules

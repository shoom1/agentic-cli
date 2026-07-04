# src/agentic_cli/workflow/permissions/store.py
"""Context, JSON persistence, and builtin rules for the permission engine.

Only PermissionContext is implemented at this point — load/save helpers
and BUILTIN_RULES land in Tasks 10–12.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentic_cli.file_utils import atomic_write_text
from agentic_cli.settings_persistence import get_project_local_permissions_path
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource


@dataclass(frozen=True)
class PermissionContext:
    """Static per-run context exposed to matchers and substitution.

    Attributes:
        workdir: Absolute current working directory.
        home: Absolute home directory.
        app_name: Application name (drives the ``.{app_name}`` config dir).
    """

    workdir: Path
    home: Path
    app_name: str = "agentic_cli"

    def substitute(self, s: str) -> str:
        """Expand ${workdir}, ${home} and ${app_name} in a pattern string."""
        return (
            s.replace("${workdir}", str(self.workdir))
             .replace("${home}", str(self.home))
             .replace("${app_name}", self.app_name)
        )


BUILTIN_RULES: list[Rule] = [
    # Routine reads inside workdir: allowed without prompt.
    Rule("filesystem.read", "${workdir}/**", Effect.ALLOW, RuleSource.BUILTIN),

    # Agent-internal stores — reads and writes to the user's own knowledge base
    # and memory are routine workflow activity, not external side effects.
    Rule("memory.*", "*", Effect.ALLOW, RuleSource.BUILTIN),
    Rule("kb.*",     "*", Effect.ALLOW, RuleSource.BUILTIN),

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

    # The app's own config: writing it would let the agent rewrite permission
    # rules / disable the engine (self-escalation). Deny both the project- and
    # user-level config dirs. Deny-wins, so this beats any broadened grant.
    Rule("filesystem.write", "${workdir}/.${app_name}/**", Effect.DENY, RuleSource.BUILTIN),
    Rule("filesystem.write", "${home}/.${app_name}/**",    Effect.DENY, RuleSource.BUILTIN),
]


def load_rules(
    path: Path,
    source: RuleSource,
    ctx: PermissionContext,
    allowed_effects: frozenset[Effect] | None = None,
) -> list[Rule]:
    """Load rules from a settings.json file's ``permissions`` section.

    Returns an empty list when the file is absent or has no ``permissions``
    key. Raises ``ValueError`` if the file is not valid JSON.

    ``allowed_effects`` restricts which effects are honored from this source.
    The engine passes ``{Effect.DENY}`` for the untrusted PROJECT file so a
    cloned repo can tighten (deny) but never loosen (allow) the policy.
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
        if allowed_effects is not None and effect not in allowed_effects:
            continue
        for entry in section.get(effect_name) or []:
            cap = entry["capability"]
            target_raw = entry["target"]
            target = get_matcher(cap).canonicalize(target_raw, ctx)
            rules.append(Rule(cap, target, effect, source))
    return rules


def append_project_rule(app_name: str, rule: Rule) -> None:
    """Append ``rule`` to the project-local permissions file.

    Writes to ``./.{app}/permissions.local.json`` (NOT ``settings.json``) so
    the user's own "Allow always" grants are stored in a file that is loaded as
    trusted, separate from the repo-shippable ``settings.json`` whose allow
    rules are ignored. Creates the file if absent; dedupes by exact
    ``(capability, target)``; atomic rewrite via ``atomic_write_text``.

    Only ``Rule`` instances with ``source == RuleSource.PROJECT`` should be
    passed here — this helper doesn't validate (engine enforces the
    invariant).
    """
    path = get_project_local_permissions_path(app_name)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    key = "allow" if rule.effect is Effect.ALLOW else "deny"
    section = data.setdefault("permissions", {}).setdefault(key, [])
    entry = {"capability": rule.capability, "target": rule.target}
    if entry not in section:
        section.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2))

# src/agentic_cli/workflow/permissions/store.py
"""Context, JSON persistence, and builtin rules for the permission engine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from agentic_cli import protected_paths
from agentic_cli.file_utils import atomic_write_text
from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import get_user_project_grants_path
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.settings import PermissionRuleConfig, PermissionsConfig


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


def _pattern(entry: protected_paths.ProtectedPath) -> str:
    """A protected location as a rule pattern: a directory covers itself and
    everything below it (``<dir>/**``), a file only itself."""
    location = entry.location.replace("{app_name}", "${app_name}")
    if location.startswith("~/"):
        location = "${home}/" + location[2:]
    elif not location.startswith("/"):
        location = "${workdir}/" + location
    return f"{location}/**" if entry.is_dir else location


def _denies(
    capabilities: tuple[str, ...], entries: tuple[protected_paths.ProtectedPath, ...]
) -> list[Rule]:
    return [
        Rule(capability, _pattern(entry), Effect.DENY, RuleSource.BUILTIN)
        for entry in entries
        for capability in capabilities
    ]


_READ, _WRITE = "filesystem.read", "filesystem.write"

#: Routine reads inside the project: allowed without a prompt. The engine
#: leaves it out when the working directory is one a grant may not widen to
#: (the root, home or above it): there it would cover all of home.
WORKDIR_READ_ALLOW = Rule(_READ, "${workdir}/**", Effect.ALLOW, RuleSource.BUILTIN)

BUILTIN_RULES: list[Rule] = list(dict.fromkeys([  # each rule once (the lists overlap)
    WORKDIR_READ_ALLOW,
    # What the framework saves for read_file (workflow/tool_output.py and
    # tools/webfetch/saved.py, FOLDER_NAME), wherever the app was started.
    Rule(_READ, "${workdir}/.${app_name}/tool-output/**", Effect.ALLOW, RuleSource.BUILTIN),
    Rule(_READ, "${workdir}/.${app_name}/fetched/**", Effect.ALLOW, RuleSource.BUILTIN),

    # Agent-internal stores — reads and writes to the user's own knowledge base
    # and memory are routine workflow activity, not external side effects.
    Rule("memory.*", "*", Effect.ALLOW, RuleSource.BUILTIN),
    Rule("kb.*",     "*", Effect.ALLOW, RuleSource.BUILTIN),

    # The arXiv tools' two fixed endpoints (tools/arxiv_source.py:
    # ARXIV_API_URL, ARXIV_PDF_URL; download_pdf refuses any other URL). A URL
    # rule matches its exact path, so web_fetch of any other arxiv.org page
    # still asks.
    Rule("http.read", "https://export.arxiv.org/api/query", Effect.ALLOW, RuleSource.BUILTIN),
    Rule("http.read", "https://arxiv.org/pdf",              Effect.ALLOW, RuleSource.BUILTIN),

    # What the OS sandbox protects (agentic_cli.protected_paths), denied to
    # every tool. Deny wins, so no grant or allow rule reaches them.
    # Credential stores: neither read nor written.
    *_denies((_READ, _WRITE), protected_paths.CREDENTIAL_STORES),
    # Shell start-up files, git's config, the system's directories: never
    # written.
    *_denies((_WRITE,), protected_paths.NEVER_WRITABLE),
    # The project's .git (hooks and config git runs outside any sandbox) and
    # .${app_name} (its settings): never written. Its .env: never read.
    *_denies((_WRITE,), protected_paths.WORKDIR_NEVER_WRITABLE),
    *_denies((_READ,), protected_paths.WORKDIR_NEVER_READABLE),
    # And its env files at any depth (.env, .env.*, .envrc).
    *_denies((_READ,), protected_paths.WORKDIR_ENV_FILE_GLOBS),
    # The app's own config dir (keys, settings, grants, jobs, sessions):
    # writing it would let the agent rewrite permission rules or disable the
    # engine (self-escalation). Reading it is denied for the keys only: an app
    # may keep files there for its agent to read, and started from home it is
    # ./.${app_name}, where the framework saves tool outputs and pages. The
    # settings file is denied too (it may hold a key), and the engine also
    # denies reading every env file the settings read
    # (settings_env_file_denies).
    *_denies((_WRITE,), (protected_paths.APP_CONFIG_DIR,)),
    *_denies((_READ,), protected_paths.APP_NEVER_READABLE),
]))


def settings_env_file_denies(settings: object, ctx: PermissionContext) -> list[Rule]:
    """A builtin read deny for each env file the settings read.

    The files are ``BaseSettings.env_files`` (for settings composed from the
    mixins without it, their class's ``env_file``): the keys an app was given,
    wherever its ``env_file`` puts them. ``~`` expands to the process's home,
    as when the settings read the file, and a relative entry is anchored at
    the working directory. Each rule is the file's resolved path.
    """
    entries = getattr(settings, "env_files", None)
    if not isinstance(entries, (str, os.PathLike, list, tuple)):
        model_config = getattr(type(settings), "model_config", None)
        entries = model_config.get("env_file") if isinstance(model_config, dict) else None
    if entries is None:
        return []
    if isinstance(entries, (str, os.PathLike)):
        entries = (entries,)
    rules: list[Rule] = []
    for entry in entries:
        if entry is None:
            continue
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = ctx.workdir / path
        rule = Rule(_READ, str(path.resolve(strict=False)), Effect.DENY, RuleSource.BUILTIN)
        if rule not in rules:
            rules.append(rule)
    return rules


def settings_rules(settings: object, ctx: PermissionContext) -> list[Rule]:
    """The trusted rules: ``settings.permissions``, allow and deny.

    The setting holds every trusted source's rules together: the app's class
    default, the user's ``~/.{app}/settings.json``, the environment and the
    constructor (``WorkflowSettingsMixin._merge_shipped_entries``). The
    project file is not among them (its section sets no setting:
    ``load_project_denies`` reads it). The rules are
    ``RuleSource.USER``. A rule whose target cannot be canonicalized (a URL
    the HTTP client cannot send) is skipped with a warning
    (``permission_rule_skipped``). Settings without the field give none.
    """
    config = getattr(settings, "permissions", None)
    if not isinstance(config, PermissionsConfig):
        return []
    return _rules(
        ((Effect.ALLOW, config.allow), (Effect.DENY, config.deny)),
        RuleSource.USER,
        ctx,
        origin={"setting": "permissions"},
    )


def load_project_denies(path: Path, ctx: PermissionContext) -> list[Rule]:
    """The deny rules of the project file's ``permissions`` section.

    The project file (``./.{app}/settings.json``) is untrusted: a cloned
    repository can ship it, so it can tighten the policy but never loosen it,
    and its allow entries are not read. The settings pass the section over
    (``PROJECT_PRESERVED_KEYS``), so each deny entry is checked here with the
    settings' own schema (``PermissionRuleConfig``). Nothing here raises, so
    a typo in a cloned repository cannot stop the app from starting there:

    - an entry that is not a rule, or a section or ``deny`` list of the wrong
      shape, is skipped with one warning naming the file and the entry
      (``permission_project_rule_invalid``); the other entries still apply;
    - a file that is not a JSON object gives one warning
      (``permission_project_file_invalid``) and no rules;
    - a rule whose target cannot be canonicalized is skipped with a warning
      (``permission_rule_skipped``).

    Returns ``[]`` when the file is absent or has no ``permissions`` section.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # JSONDecodeError, UnicodeDecodeError
        _warn_project_file(path, f"not readable as JSON: {exc}")
        return []
    if not isinstance(data, dict):
        _warn_project_file(path, f"expected a JSON object, got {_json_type(data)}")
        return []

    section = data.get("permissions")
    if section is None:
        return []
    if not isinstance(section, dict):
        _warn_invalid(path, "permissions", f"expected an object, got {_json_type(section)}")
        return []
    entries = section.get("deny")
    if entries is None:
        return []
    if not isinstance(entries, list):
        _warn_invalid(path, "permissions.deny", f"expected a list, got {_json_type(entries)}")
        return []

    denies: list[PermissionRuleConfig] = []
    for index, entry in enumerate(entries):
        try:
            denies.append(PermissionRuleConfig.model_validate(entry))
        except ValidationError as exc:
            _warn_invalid(path, f"permissions.deny[{index}]", _summary(exc))
    return _rules(((Effect.DENY, denies),), RuleSource.PROJECT, ctx, origin={"path": str(path)})


def _rules(
    entries: tuple[tuple[Effect, list[PermissionRuleConfig]], ...],
    source: RuleSource,
    ctx: PermissionContext,
    origin: dict[str, str],
) -> list[Rule]:
    """Rules from checked entries, each target canonicalized as a pattern."""
    # Local import to avoid circular dependency: matchers.py imports PermissionContext from here.
    from agentic_cli.workflow.permissions.matchers import get_matcher  # noqa: PLC0415

    rules: list[Rule] = []
    for effect, configs in entries:
        for entry in configs:
            try:
                target = get_matcher(entry.capability).canonicalize_pattern(entry.target, ctx)
            except ValueError as exc:
                # A URL the HTTP client cannot send (an invalid IDNA host, a
                # port past 65535), a path with a NUL: no target can match
                # it, since such a target is itself denied as invalid.
                _log_skipped(entry.capability, entry.target, exc, origin)
                continue
            rules.append(Rule(entry.capability, target, effect, source))
    return rules


_JSON_TYPES = {
    dict: "an object", list: "a list", str: "a string", bool: "a boolean",
    int: "a number", float: "a number", type(None): "null",
}


def _json_type(value: object) -> str:
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _summary(exc: ValidationError) -> str:
    """A rule's validation errors on one line, without the entry's value."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'entry'}: {error['msg']}"
        for error in exc.errors()
    )


def _warn_project_file(path: Path, error: str) -> None:
    Loggers.workflow().warning("permission_project_file_invalid", path=str(path), error=error)


def _warn_invalid(path: Path, entry: str, error: str) -> None:
    Loggers.workflow().warning(
        "permission_project_rule_invalid", path=str(path), entry=entry, error=error
    )


def _warn_skipped(path: Path, capability: str, target: str, exc: ValueError) -> None:
    _log_skipped(capability, target, exc, {"path": str(path)})


def _log_skipped(capability: str, target: str, exc: ValueError, origin: dict[str, str]) -> None:
    Loggers.workflow().warning(
        "permission_rule_skipped",
        capability=capability,
        target=target,
        **origin,
        error=str(exc),
    )


def load_project_grants(app_name: str, ctx: PermissionContext) -> list[Rule]:
    """Load interactive 'Allow always' grants for the CURRENT project only.

    Reads ``~/.{app}/project_grants.json`` and returns the allow+deny rules
    stored under ``str(ctx.workdir.resolve())`` — trusted (``RuleSource.PROJECT``,
    both effects honored), since the user (not a repo) authored them. Three
    kinds of allow are dropped with a warning: a resource capability's
    wildcard (``permission_wildcard_grant_dropped``), one whose literal part
    holds a wildcard character (``permission_glob_grant_dropped``, see
    ``engine.is_literal_saved_grant``) and a filesystem grant on the root,
    home or a directory above it (``permission_broad_grant_dropped``, see
    ``engine.is_bounded_saved_grant``). A grant whose target cannot be
    canonicalized (a URL the HTTP client cannot send) is skipped with a
    warning (``permission_rule_skipped``). Returns ``[]`` when the file or the
    project's entry is absent. Raises ``ValueError`` on malformed JSON.
    """
    # Local imports to avoid cycles: matchers.py imports PermissionContext
    # here, and engine.py imports this module.
    from agentic_cli.workflow.permissions.capabilities import is_resource_capability  # noqa: PLC0415
    from agentic_cli.workflow.permissions.engine import (  # noqa: PLC0415
        is_bounded_saved_grant,
        is_literal_saved_grant,
    )
    from agentic_cli.workflow.permissions.matchers import get_matcher  # noqa: PLC0415

    path = get_user_project_grants_path(app_name)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")

    section = (data.get(str(ctx.workdir.resolve())) or {}).get("permissions") or {}
    rules: list[Rule] = []
    for effect_name, effect in (("allow", Effect.ALLOW), ("deny", Effect.DENY)):
        for entry in section.get(effect_name) or []:
            cap = entry["capability"]
            # Grants are written by the engine from already-resolved targets, so
            # they load as targets: expanding ${...} text here would turn a
            # directory name into a different location. "*" is the stored
            # wildcard of a targetless capability and stays one.
            raw = entry["target"]
            if raw == "*" and is_resource_capability(cap):
                # Only a declaration naming no target produces this, and it
                # approves the capability for every resource and every tool
                # (0.6.0 stored one for web_search). Never honour it.
                Loggers.workflow().warning(
                    "permission_wildcard_grant_dropped", capability=cap, path=str(path)
                )
                continue
            try:
                target = "*" if raw == "*" else get_matcher(cap).canonicalize_target(raw, ctx)
            except ValueError as exc:
                # Saved by an engine that judged URLs as spelled: one the HTTP
                # client cannot send matches nothing now (its target is denied).
                _warn_skipped(path, cap, raw, exc)
                continue
            if effect is Effect.ALLOW and not is_literal_saved_grant(cap, target):
                # Loaded as a target but matched as a pattern: a grant saved
                # with a wildcard character in its literal part would cover
                # more than was approved. A deny keeps its glob: it can only
                # deny more.
                Loggers.workflow().warning(
                    "permission_glob_grant_dropped",
                    capability=cap,
                    target=target,
                    path=str(path),
                )
                continue
            if effect is Effect.ALLOW and not is_bounded_saved_grant(cap, target, ctx.home):
                # A grant on the root, home or above it: the engine no longer
                # widens to them, so one saved by an older engine (or by hand)
                # would cover far more than a prompt now offers.
                Loggers.workflow().warning(
                    "permission_broad_grant_dropped",
                    capability=cap,
                    target=target,
                    path=str(path),
                )
                continue
            rules.append(Rule(cap, target, effect, RuleSource.PROJECT))
    return rules


def append_project_rule(app_name: str, rule: Rule, project_root: Path) -> None:
    """Persist an interactive 'Allow always' grant to the USER-side, path-keyed
    grants file (``~/.{app}/project_grants.json``), under ``project_root``'s
    resolved path.

    Kept out of the repo (P0-1) so a cloned workspace carries no grants: a clone
    at a different path has no matching entry and the user re-grants. Creates the
    file if absent; dedupes by exact ``(capability, target)`` within the
    project's section; atomic rewrite via ``atomic_write_text``.

    Only ``Rule`` instances with ``source == RuleSource.PROJECT`` should be
    passed here — this helper doesn't validate (engine enforces the invariant).
    """
    path = get_user_project_grants_path(app_name)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")

    proj_key = str(project_root.resolve())
    key = "allow" if rule.effect is Effect.ALLOW else "deny"
    section = (
        data.setdefault(proj_key, {})
        .setdefault("permissions", {})
        .setdefault(key, [])
    )
    entry = {"capability": rule.capability, "target": rule.target}
    if entry not in section:
        section.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2))

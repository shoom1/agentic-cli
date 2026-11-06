# src/agentic_cli/workflow/permissions/engine.py
"""Permission engine.

See docs/superpowers/specs/2026-04-18-permissions-system-design.md §4 for
the full decision flow. This file implements:

1. Rule loading from builtin + ``settings.permissions`` + the project
   file's deny rules + saved grants + in-memory session.
2. ``permissions_enabled=False`` short-circuit.
3. Full check() flow (Tasks 15–17).
"""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import Callable, Collection
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import (
    get_project_config_path,
    get_user_config_path,
)
from agentic_cli.urls import canonical_url
from agentic_cli.workflow.events import UserInputUnavailable
from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    ResolvedCapability,
    is_resource_capability,
)
from agentic_cli.workflow.permissions.matchers import _fold, get_matcher
from agentic_cli.workflow.permissions.rules import (
    AskScope,
    CheckResult,
    Effect,
    Rule,
    RuleSource,
)
from agentic_cli.workflow.permissions.store import (
    BUILTIN_RULES,
    WORKDIR_READ_ALLOW,
    PermissionContext,
    load_project_denies,
    load_project_grants,
    settings_env_file_denies,
    settings_rules,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

logger = Loggers.workflow()


def broaden_target_for_grant(cap: ResolvedCapability, home: Path | None = None) -> str:
    """Widen a resolved target before synthesising a session/persistent rule.

    A ``filesystem.*`` grant covers a whole directory, so one approval covers
    the files the agent works with there instead of prompting per file: a
    directory target covers itself (``dir/**``), a file target covers the
    directory it is in. A grant never widens to the filesystem root, to
    ``home`` itself or to a directory above it: there the exact target is
    granted instead, since ``~/**`` would cover the shell's start-up files and
    ``~/Library``, and ``/Users/**`` every user's home.

    An ``http.read`` grant covers the URL's site (see ``site_scope``), so one
    approval covers the pages read there; a URL with no site scope keeps its
    exact target.

    Other namespaces keep the exact resolved target (command, etc.), and
    the wildcard sentinel ``"*"`` passes through unchanged.

    Used by both the engine (when installing a rule) and the prompt
    builder (when describing the pending grant) so the displayed scope
    always matches what will actually be stored.
    """
    if cap.target == "*":
        return "*"
    if cap.name.startswith("filesystem."):
        scope = _grant_scope(cap, home)
        return cap.target if scope is None else f"{scope}/**"
    if cap.name == "http.read":
        return site_scope(cap.target) or cap.target
    return cap.target


def _grant_scope(cap: ResolvedCapability, home: Path | None) -> Path | None:
    """The directory a ``filesystem.*`` grant covers, or None when it may not
    widen (the grant is then the exact target)."""
    target = Path(cap.target)
    scope = target if target.is_dir() else target.parent
    return scope if _may_widen_to(scope, (home or Path.home()).resolve()) else None


def site_scope(url: str) -> str | None:
    """``https://host[:port]/**``: the site a URL grant may cover, if any.

    Only HTTPS, and only a DNS name with a dot: the exact host (no other
    subdomain) and port. A plain-HTTP URL, an IP address, ``localhost`` or a
    single-label name gets None, so its grant stays the exact URL. The host is
    the one the HTTP client sends (``canonical_url``), so ``localhost.`` is
    ``localhost`` and ``docs。example。org`` is ``docs.example.org``; a URL the
    client cannot send has no site.
    """
    try:
        parts = urlsplit(canonical_url(url))
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname or ""
    if parts.scheme != "https" or "." not in host or host.endswith(".localhost"):
        return None
    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        pass
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return f"https://{netloc}/**"


def _may_widen_to(directory: Path, home: Path) -> bool:
    """Whether a grant may cover all of ``directory``.

    Not the filesystem root, not ``home`` itself and not a directory above it
    (which includes the root): those would reach far beyond what was asked
    for. All of home holds the shell's start-up files, ``~/Library`` and every
    file no builtin rule names, so a file directly in home is granted exactly.
    Compared caseless, as a deny is: on a case-insensitive filesystem
    ``/USERS`` is ``/Users``, and a grant on ``/USERS/**`` covers every home
    spelled that way.
    """
    if directory == Path(directory.anchor):
        return False
    directory, home = Path(_fold(str(directory))), Path(_fold(str(home)))
    return not home.is_relative_to(directory)


#: The characters each matcher treats as wildcards in a rule's target:
#: ``_glob_to_regex`` (paths, URL paths) and ``fnmatch`` (shell commands,
#: strings, URL queries).
_PATH_WILDCARDS = "*?"
_GLOB_WILDCARDS = "*?["


class Unstorable(str, Enum):
    """Why an approval may not be remembered (see ``unstorable_reason``)."""

    #: A resource capability whose tool named no target: ``"*"``.
    TARGETLESS = "targetless"
    #: A target whose rule would hold one of its matcher's wildcard characters.
    WILDCARD = "wildcard"
    #: Code that would run on the host without a sandbox.
    UNCONTAINED = "uncontained"


def unstorable_reason(
    cap: ResolvedCapability,
    home: Path | None = None,
    uncontained: Collection[str] = (),
) -> Unstorable | None:
    """Why approving ``cap`` may not be remembered as a rule, or None when it may.

    The one decision: the engine asks it before storing a grant, and the prompt
    builder before offering one (``is_storable_grant`` is its yes/no).

    * ``UNCONTAINED``: ``cap`` is one of ``uncontained``, the code-execution
      capabilities its owner says would run on the host with the user's
      privileges (``tools/exec_policy.py``). CPython has no in-process
      sandbox, so such code is approved call by call: whatever approved the
      last call says nothing about the next one's code.
    * ``TARGETLESS``: a resource capability (``filesystem``/``http``/``shell``)
      whose target is the wildcard can only come from a declaration that names
      no target. Storing it would approve that capability for every resource
      and every tool that declares it.
    * ``WILDCARD``: the rule the grant would become matches more than the
      prompt showed. A target is a literal argument, but the stored rule is
      matched as a pattern, so a ``*`` in an approved ``ls *.py`` or
      ``/x/**/notes.txt`` would become a wildcard. The literal part of the
      rule (see ``broaden_target_for_grant``) must hold none of its matcher's
      wildcard characters:

      * ``filesystem.*``: the scope directory when the grant widens, else the
        exact target; ``*`` and ``?``.
      * ``http.*``: a site scope ``https://host[:port]/**`` is literal (hosts
        are compared exactly); an exact URL must have no ``*``, no ``[`` in
        its path or query, and no ``?`` in its query.
      * anything else (``shell.*``, the string fallback): the whole target;
        ``*``, ``?`` and ``[``.

    ``home`` is the home directory the grant scope is judged against (the
    process's by default), as in ``broaden_target_for_grant``.
    """
    if cap.name in uncontained:
        return Unstorable.UNCONTAINED
    if cap.target == "*":
        return Unstorable.TARGETLESS if is_resource_capability(cap.name) else None
    return None if _rule_is_literal(cap, home) else Unstorable.WILDCARD


def is_storable_grant(
    cap: ResolvedCapability,
    home: Path | None = None,
    uncontained: Collection[str] = (),
) -> bool:
    """Whether approving ``cap`` may be remembered as a rule: no
    ``unstorable_reason``."""
    return unstorable_reason(cap, home, uncontained) is None


def _rule_is_literal(cap: ResolvedCapability, home: Path | None) -> bool:
    """Whether the literal part of the rule a grant of ``cap`` would become
    holds none of its matcher's wildcard characters (see
    ``unstorable_reason``)."""
    if cap.name.startswith("filesystem."):
        scope = _grant_scope(cap, home)
        return _is_literal(cap.target if scope is None else str(scope), _PATH_WILDCARDS)
    if cap.name.startswith("http."):
        if cap.name == "http.read" and site_scope(cap.target):
            return True
        return _is_literal_url(cap.target)
    return _is_literal(cap.target, _GLOB_WILDCARDS)


def is_literal_saved_grant(capability: str, target: str) -> bool:
    """Whether a saved allow grant matches only what its literal part names.

    The engine writes grants from literal targets (``is_storable_grant``), so
    a saved one whose literal part holds a wildcard character came from the
    bug that stored ``/x/**/**`` for ``/x/**/notes.txt``, or from a hand edit.
    The literal part is the target less one trailing ``/**`` for
    ``filesystem.*`` (whether that directory may be covered at all is
    ``is_bounded_saved_grant``'s question); for ``http.*`` a site scope is
    literal, any other URL is checked as in ``is_storable_grant``.
    """
    if target == "*":
        return not is_resource_capability(capability)
    if capability.startswith("filesystem."):
        return _is_literal(target.removesuffix("/**"), _PATH_WILDCARDS)
    if capability.startswith("http."):
        return site_scope(target) == target or _is_literal_url(target)
    return _is_literal(target, _GLOB_WILDCARDS)


def is_bounded_saved_grant(capability: str, target: str, home: Path) -> bool:
    """Whether a saved allow grant covers no more than a grant may.

    A ``filesystem.*`` grant ``<dir>/**`` must name a directory a grant may
    widen to (``_may_widen_to``): not the filesystem root (``/**`` is the
    root's file named ``**``, which is granted as its exact target), not
    ``home`` and not a directory above it. One that does came from an engine
    that widened a file directly in home to all of it, or from a hand edit.
    Every other grant is bounded by its own target.
    """
    if not capability.startswith("filesystem.") or not target.endswith("/**"):
        return True
    scope = target.removesuffix("/**")
    return bool(scope) and _may_widen_to(Path(scope), home.resolve())


def _is_literal(text: str, wildcards: str) -> bool:
    return not any(ch in text for ch in wildcards)


def _is_literal_url(url: str) -> bool:
    """Whether an exact-URL rule matches only its own URL: no ``*`` anywhere,
    no ``[`` in its path or query, no ``?`` in its query (the path is matched
    as a glob, the query with ``fnmatch``). The host and port are compared
    exactly, so an IPv6 host's brackets are no wildcard."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return (
        "*" not in url
        and "[" not in parts.path
        and _is_literal(parts.query, _GLOB_WILDCARDS)
    )


def _same_file(a: Path, b: Path) -> bool:
    """Whether both paths name one existing file (another spelling, a link)."""
    try:
        return a.samefile(b)
    except OSError:
        return False


#: The warning logged when a client answers "for this session" or "always"
#: (which the prompt did not offer) for an approval that may not be stored.
_NOT_STORED_EVENTS = {
    Unstorable.TARGETLESS: "permission_wildcard_grant_not_stored",
    Unstorable.WILDCARD: "permission_glob_grant_not_stored",
    Unstorable.UNCONTAINED: "permission_uncontained_grant_not_stored",
}

#: The sources of a grant's allow rule: an interactive "Allow for this
#: session", and "Allow always", loaded from the user's
#: ``project_grants.json`` (a project file's allow rules are never loaded).
_GRANT_SOURCES = frozenset({RuleSource.SESSION, RuleSource.PROJECT})


class PermissionEngine:
    """Evaluate tool invocations against rules from four sources: the
    builtin rules, the user's (``settings.permissions``), the project's (the
    project file's deny rules and the saved grants) and the session's.

    Concurrency: one ``asyncio.Lock`` around the ask prompt only (rule
    matching is pure). See spec §4.4.

    ``uncontained`` is the owner's say on which code-execution capabilities
    would run on the host without a sandbox: called with the settings at each
    check, so a settings change applies at once (``BaseWorkflowManager``
    passes ``tools.exec_policy.uncontained_exec_capabilities``; the engine
    imports nothing from ``tools/``). Such a capability is approved call by
    call: the prompt offers no grant, and no grant's allow rule covers it,
    though one stored earlier, while the code ran contained, still exists. A
    rule the user wrote in their own settings still applies: that is their
    policy. None (the default): no capability is uncontained.
    """

    def __init__(
        self,
        settings: "BaseSettings",
        workflow: "BaseWorkflowManager",
        ctx: PermissionContext,
        *,
        uncontained: "Callable[[BaseSettings], Collection[str]] | None" = None,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._ctx = ctx
        self._uncontained = uncontained
        self._session_rules: list[Rule] = []
        self._ask_lock = asyncio.Lock()
        self._base_rules: list[Rule] = self._load_all_rules()

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------

    def _load_all_rules(self) -> list[Rule]:
        rules: list[Rule] = []
        # Started from home, above it or at the root, the working directory's
        # read allow would cover all of home (its start-up files, keychains)
        # without a prompt: there reads ask, as a grant there would.
        in_a_project = _may_widen_to(self._ctx.workdir.resolve(), self._ctx.home.resolve())
        for r in BUILTIN_RULES:
            if r == WORKDIR_READ_ALLOW and not in_a_project:
                continue
            # Canonicalise each builtin rule's target (which may use ${workdir}/${home}).
            rules.append(
                Rule(
                    capability=r.capability,
                    target=get_matcher(r.capability).canonicalize_pattern(r.target, self._ctx),
                    effect=r.effect,
                    source=r.source,
                )
            )
        # The keys the app was given, wherever its env_file puts them.
        rules += settings_env_file_denies(self._settings, self._ctx)
        # The trusted rules, allow and deny: settings.permissions, the union
        # of the app's class default, the user file, the environment and the
        # constructor. The user file is read once, by the settings.
        rules += settings_rules(self._settings, self._ctx)
        app = self._settings.app_name
        # PROJECT settings.json is untrusted (a cloned repo can ship it): honor
        # only deny-rules so a workspace can tighten but never loosen policy.
        # Started from home it is the user file, whose rules are the settings'.
        project_file = get_project_config_path(app)
        if not _same_file(project_file, get_user_config_path(app)):
            rules += load_project_denies(project_file, self._ctx)
        # Interactive "Allow always" grants live in USER config, keyed by the
        # resolved project path — trusted (allow+deny). A cloned repo carries
        # none (its path won't match), and a repo-shipped permissions.local.json
        # is no longer loaded at all.
        rules += load_project_grants(app, self._ctx)
        return rules

    @property
    def rules(self) -> list[Rule]:
        """All currently-active rules, in source order (builtin→user→project→session)."""
        return list(self._base_rules) + list(self._session_rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        tool_name: str,
        capabilities: list[Capability],
        args: dict,
    ) -> CheckResult:
        """Evaluate whether ``tool_name`` may run with ``args``."""
        if not self._settings.permissions_enabled:
            return CheckResult(True, "permissions disabled")

        try:
            resolved = self._resolve(capabilities, args)
        except ValueError as exc:
            # A target that cannot name a location (e.g. an embedded NUL) is
            # refused outright rather than raised into the turn or put to the
            # user: there is nothing meaningful to approve.
            logger.warning("permission_invalid_target", tool=tool_name, error=str(exc))
            return CheckResult(False, f"invalid target: {exc}")
        uncontained = self._uncontained_now()
        outcomes = self._evaluate(resolved, uncontained)

        # No capabilities to evaluate (e.g. every cap is optional and its target
        # arg was absent) → nothing to gate, allow.
        if not outcomes:
            return CheckResult(True, "no applicable capabilities")

        # DENY wins.
        deny_hits = [(c, r) for c, r in outcomes if r is not None and r.effect is Effect.DENY]
        if deny_hits:
            c, r = deny_hits[0]
            return CheckResult(False, self._fmt_rule_reason(r, c))

        # All allowed?
        if all(r is not None and r.effect is Effect.ALLOW for _, r in outcomes):
            any_c, any_r = outcomes[0]
            return CheckResult(True, self._fmt_rule_reason(any_r, any_c))

        # Ask flow lands in Task 16.
        return await self._ask_and_apply(tool_name, resolved, outcomes, args, uncontained)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(
        self, capabilities: list[Capability], args: dict
    ) -> list[ResolvedCapability]:
        """Resolve each capability's target from the call arguments.

        Arguments are *targets*, canonicalized with ``canonicalize_target``: no
        ``${...}`` placeholder is expanded and a path is resolved exactly as the
        tool resolves it, so the engine judges the location the tool acts on.

        Raises:
            ValueError: if an argument cannot be resolved to a target.
        """
        resolved: list[ResolvedCapability] = []
        for cap in capabilities:
            if cap.target is not None:
                matcher = get_matcher(cap.name)
                resolved.append(
                    ResolvedCapability(cap.name, matcher.canonicalize_target(cap.target, self._ctx))
                )
                continue
            if cap.target_arg is None:
                resolved.append(ResolvedCapability(cap.name, "*"))
                continue
            value = args.get(cap.target_arg, "")
            if cap.optional and (value is None or value == ""):
                # Optional target not supplied → the side effect isn't performed
                # this call, so don't resolve (and don't spuriously prompt) it.
                continue
            matcher = get_matcher(cap.name)
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                resolved.append(
                    ResolvedCapability(cap.name, matcher.canonicalize_target(str(item), self._ctx))
                )
        return resolved

    def _uncontained_now(self) -> frozenset[str]:
        """The capabilities whose code would run on the host, by the settings
        as they are now (see the class docstring)."""
        if self._uncontained is None:
            return frozenset()
        return frozenset(self._uncontained(self._settings))

    def _evaluate(
        self,
        resolved: list[ResolvedCapability],
        uncontained: Collection[str] = frozenset(),
    ) -> list[tuple[ResolvedCapability, Rule | None]]:
        """For each resolved capability, return the strongest matched rule or None.

        A grant's allow rule does not cover a capability in ``uncontained``:
        that code is approved per call.
        """
        from agentic_cli.workflow.permissions.matchers import _cap_matches
        all_rules = self.rules
        out: list[tuple[ResolvedCapability, Rule | None]] = []
        for cap in resolved:
            matcher = get_matcher(cap.name)
            per_call = cap.name in uncontained
            # A deny is matched by its own method: a filesystem deny ignores
            # case, so another spelling of a denied file is denied too.
            matched: list[Rule] = [
                r for r in all_rules
                if _cap_matches(r.capability, cap.name)
                and not (per_call and r.effect is Effect.ALLOW and r.source in _GRANT_SOURCES)
                and (
                    matcher.matches_deny(r.target, cap.target)
                    if r.effect is Effect.DENY
                    else matcher.matches(r.target, cap.target)
                )
            ]
            if not matched:
                out.append((cap, None))
                continue
            # DENY wins per capability; otherwise any ALLOW.
            deny = next((r for r in matched if r.effect is Effect.DENY), None)
            out.append((cap, deny or matched[0]))
        return out

    @staticmethod
    def _fmt_rule_reason(rule: Rule, cap: ResolvedCapability) -> str:
        return f"rule: {rule.source.value}/{rule.effect.value} {cap.name} {rule.target}"


    async def _ask_and_apply(
        self,
        tool_name: str,
        resolved: list[ResolvedCapability],
        outcomes: list[tuple[ResolvedCapability, Rule | None]],
        args: dict | None = None,
        uncontained: Collection[str] = frozenset(),
    ) -> CheckResult:
        from agentic_cli.workflow.permissions.prompt import build_request, parse_response
        from agentic_cli.workflow.permissions.store import append_project_rule

        unmatched = [cap for cap, r in outcomes if r is None]
        async with self._ask_lock:
            request = build_request(
                tool_name, resolved, args, home=self._ctx.home, uncontained=uncontained
            )
            # Fail closed: a prompt nobody can answer is a denial the model can
            # read, not an exception that aborts the turn. Cancellation is not
            # an answer and propagates (it is not an Exception).
            try:
                response = await self._workflow.request_user_input(request)
            except UserInputUnavailable:
                logger.warning("permission_no_approver", tool=tool_name)
                return CheckResult(False, "no rule + no interactive approver to ask")
            except Exception as exc:
                logger.warning(
                    "permission_prompt_failed", tool=tool_name, error=repr(exc),
                )
                return CheckResult(False, f"no rule + approval prompt failed: {exc}")
            scope = parse_response(response)

        if scope is AskScope.DENY:
            logger.info(
                "permission_denied_by_user",
                tool=tool_name,
                capabilities=[(c.name, c.target) for c in resolved],
            )
            return CheckResult(False, "no rule + user denied")

        if scope is AskScope.ONCE:
            return CheckResult(True, "no rule + user allowed (once)")

        source = RuleSource.SESSION if scope is AskScope.SESSION else RuleSource.PROJECT
        stored = 0
        for cap in unmatched:
            reason = unstorable_reason(cap, home=self._ctx.home, uncontained=uncontained)
            if reason is not None:
                logger.warning(
                    _NOT_STORED_EVENTS[reason], tool=tool_name, capability=cap.name
                )
                continue
            target = broaden_target_for_grant(cap, home=self._ctx.home)
            rule = Rule(cap.name, target, Effect.ALLOW, source)
            self._session_rules.append(rule)
            stored += 1
            if source is RuleSource.PROJECT:
                append_project_rule(self._settings.app_name, rule, self._ctx.workdir)

        if unmatched and not stored:
            return CheckResult(True, "no rule + user allowed (once: nothing to remember)")
        label = "session" if source is RuleSource.SESSION else "always, saved to project"
        return CheckResult(True, f"no rule + user allowed ({label})")

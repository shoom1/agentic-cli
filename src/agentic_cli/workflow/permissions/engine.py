# src/agentic_cli/workflow/permissions/engine.py
"""Permission engine.

See docs/superpowers/specs/2026-04-18-permissions-system-design.md §4 for
the full decision flow. This file implements:

1. Rule loading from builtin + user + project JSON + in-memory session.
2. ``permissions_enabled=False`` short-circuit.
3. Full check() flow (Tasks 15–17).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import (
    get_project_config_path,
    get_user_config_path,
)
from agentic_cli.workflow.events import UserInputUnavailable
from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    ResolvedCapability,
    is_resource_capability,
)
from agentic_cli.workflow.permissions.matchers import get_matcher
from agentic_cli.workflow.permissions.rules import (
    AskScope,
    CheckResult,
    Effect,
    Rule,
    RuleSource,
)
from agentic_cli.workflow.permissions.store import (
    BUILTIN_RULES,
    PermissionContext,
    load_project_grants,
    load_rules,
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
    directory it is in. A grant never widens to the filesystem root or to a
    directory above ``home``: there the exact target is granted instead, since
    ``/Users/**`` would cover every user's home.

    Other namespaces keep the exact resolved target (URL, command, etc.), and
    the wildcard sentinel ``"*"`` passes through unchanged.

    Used by both the engine (when installing a rule) and the prompt
    builder (when describing the pending grant) so the displayed scope
    always matches what will actually be stored.
    """
    if cap.target == "*":
        return "*"
    if cap.name.startswith("filesystem."):
        target = Path(cap.target)
        scope = target if target.is_dir() else target.parent
        if _may_widen_to(scope, (home or Path.home()).resolve()):
            return f"{scope}/**"
        return cap.target
    return cap.target


def _may_widen_to(directory: Path, home: Path) -> bool:
    """Whether a grant may cover all of ``directory``.

    Not the filesystem root, and not a proper ancestor of ``home`` (which
    includes the root): those would reach far beyond what was asked for.
    """
    if directory == Path(directory.anchor):
        return False
    return not (home != directory and home.is_relative_to(directory))


def is_storable_grant(cap: ResolvedCapability) -> bool:
    """Whether approving ``cap`` may be remembered as a rule.

    A resource capability (``filesystem``/``http``/``shell``) whose target is
    the wildcard can only come from a declaration that names no target. Storing
    it would approve that capability for every resource and every tool that
    declares it, so such an approval applies to the current call only.
    """
    return not (cap.target == "*" and is_resource_capability(cap.name))


class PermissionEngine:
    """Evaluate tool invocations against rules from four sources.

    Concurrency: one ``asyncio.Lock`` around the ask prompt only (rule
    matching is pure). See spec §4.4.
    """

    def __init__(
        self,
        settings: "BaseSettings",
        workflow: "BaseWorkflowManager",
        ctx: PermissionContext,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._ctx = ctx
        self._session_rules: list[Rule] = []
        self._ask_lock = asyncio.Lock()
        self._base_rules: list[Rule] = self._load_all_rules()

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------

    def _load_all_rules(self) -> list[Rule]:
        rules: list[Rule] = []
        for r in BUILTIN_RULES:
            # Canonicalise each builtin rule's target (which may use ${workdir}/${home}).
            rules.append(
                Rule(
                    capability=r.capability,
                    target=get_matcher(r.capability).canonicalize_pattern(r.target, self._ctx),
                    effect=r.effect,
                    source=r.source,
                )
            )
        app = self._settings.app_name
        rules += load_rules(get_user_config_path(app), RuleSource.USER, self._ctx)
        # PROJECT settings.json is untrusted (a cloned repo can ship it): honor
        # only deny-rules so a workspace can tighten but never loosen policy.
        rules += load_rules(
            get_project_config_path(app),
            RuleSource.PROJECT,
            self._ctx,
            allowed_effects=frozenset({Effect.DENY}),
        )
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
        outcomes = self._evaluate(resolved)

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
        return await self._ask_and_apply(tool_name, resolved, outcomes, args)

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

    def _evaluate(
        self, resolved: list[ResolvedCapability]
    ) -> list[tuple[ResolvedCapability, Rule | None]]:
        """For each resolved capability, return the strongest matched rule or None."""
        from agentic_cli.workflow.permissions.matchers import _cap_matches
        all_rules = self.rules
        out: list[tuple[ResolvedCapability, Rule | None]] = []
        for cap in resolved:
            matcher = get_matcher(cap.name)
            matched: list[Rule] = [
                r for r in all_rules
                if _cap_matches(r.capability, cap.name)
                and matcher.matches(r.target, cap.target)
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
    ) -> CheckResult:
        from agentic_cli.workflow.permissions.prompt import build_request, parse_response
        from agentic_cli.workflow.permissions.store import append_project_rule

        unmatched = [cap for cap, r in outcomes if r is None]
        async with self._ask_lock:
            request = build_request(tool_name, resolved, args, home=self._ctx.home)
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
            if not is_storable_grant(cap):
                logger.warning(
                    "permission_wildcard_grant_not_stored",
                    tool=tool_name,
                    capability=cap.name,
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

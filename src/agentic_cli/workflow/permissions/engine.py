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
from agentic_cli.workflow.permissions.capabilities import Capability, ResolvedCapability
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
    load_rules,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

logger = Loggers.workflow()


def broaden_target_for_grant(cap: ResolvedCapability) -> str:
    """Widen a resolved target before synthesising a session/persistent rule.

    For ``filesystem.*`` capabilities we broaden to the parent directory
    (glob) so one grant covers every file the agent writes/reads there —
    otherwise each new file in the same directory would prompt again.
    Other namespaces keep the exact resolved target (URL, command, etc.),
    and the wildcard sentinel ``"*"`` passes through unchanged.

    Used by both the engine (when installing a rule) and the prompt
    builder (when describing the pending grant) so the displayed scope
    always matches what will actually be stored.
    """
    if cap.target == "*":
        return "*"
    if cap.name.startswith("filesystem."):
        p = Path(cap.target)
        parent = p.parent
        # Already at the root: nothing to widen to.
        if str(p) == str(parent):
            return cap.target
        # Avoid "//**" when the parent is the filesystem root.
        if str(parent) == "/":
            return "/**"
        return f"{parent}/**"
    return cap.target


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
                    target=get_matcher(r.capability).canonicalize(r.target, self._ctx),
                    effect=r.effect,
                    source=r.source,
                )
            )
        app = self._settings.app_name
        rules += load_rules(get_user_config_path(app), RuleSource.USER, self._ctx)
        rules += load_rules(get_project_config_path(app), RuleSource.PROJECT, self._ctx)
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

        resolved = self._resolve(capabilities, args)
        outcomes = self._evaluate(resolved)

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
        return await self._ask_and_apply(tool_name, resolved, outcomes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(
        self, capabilities: list[Capability], args: dict
    ) -> list[ResolvedCapability]:
        resolved: list[ResolvedCapability] = []
        for cap in capabilities:
            raw = "*" if cap.target_arg is None else str(args.get(cap.target_arg, ""))
            target = "*" if cap.target_arg is None else get_matcher(cap.name).canonicalize(raw, self._ctx)
            resolved.append(ResolvedCapability(cap.name, target))
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
    ) -> CheckResult:
        from agentic_cli.workflow.permissions.prompt import build_request, parse_response
        from agentic_cli.workflow.permissions.store import append_project_rule

        unmatched = [cap for cap, r in outcomes if r is None]
        async with self._ask_lock:
            request = build_request(tool_name, resolved)
            response = await self._workflow.request_user_input(request)
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
        for cap in unmatched:
            target = broaden_target_for_grant(cap)
            rule = Rule(cap.name, target, Effect.ALLOW, source)
            self._session_rules.append(rule)
            if source is RuleSource.PROJECT:
                append_project_rule(self._settings.app_name, rule)

        label = "session" if source is RuleSource.SESSION else "always, saved to project"
        return CheckResult(True, f"no rule + user allowed ({label})")
